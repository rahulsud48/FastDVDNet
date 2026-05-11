"""
Trains FastDVDnet (single-frame + KV bank variant).

Key differences from the original training script:
  • Frames are fed ONE AT A TIME in temporal order.
  • A KVBank is maintained per-sequence and reset at sequence boundaries.
  • The dataloader yields sequences of length `temp_patch_size`; we iterate
    over frames within each sequence so the bank builds up naturally.
  • Gradients are accumulated across frames in a sequence and averaged before
    the optimizer step (equivalent to treating the whole sequence as one sample).
"""

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models import FastDVDnet, KVBank
from dataset import ValDataset
from simple_dataloader import train_simple_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import (resume_training, lr_scheduler, log_train_psnr,
                          validate_and_log, save_model_checkpoint)


# ---------------------------------------------------------------------------
# Validation helper (single-frame + bank)
# ---------------------------------------------------------------------------

def validate_and_log_singleframe(model, dataset_val, valnoisestd, temp_psz,
                                  writer, epoch, lr, logger, trainimg, bank_size):
    """
    Wrapper that runs validation frame-by-frame with a fresh KVBank per clip,
    then delegates logging to the original validate_and_log utility.
    We monkey-patch model.forward temporarily so validate_and_log still works.
    """
    # validate_and_log expects model(frames_stacked, noise_map) with frames_stacked
    # being (N, 3*temp_psz, H, W).  We wrap the model to unpack that signature.
    bank_size_val = bank_size

    class SequentialWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, frames_stacked, noise_map):
            N, _, H, W = frames_stacked.shape
            num_frames = frames_stacked.shape[1] // 3
            bank = KVBank(bank_size=bank_size_val)
            out = None
            for t in range(num_frames):
                ft = frames_stacked[:, 3*t:3*t+3, :, :]
                out = self.inner(ft, noise_map, bank)
            return out  # return denoised central (last) frame

    wrapped = SequentialWrapper(model)
    validate_and_log(
        model_temp=wrapped,
        dataset_val=dataset_val,
        valnoisestd=valnoisestd,
        temp_psz=temp_psz,
        writer=writer,
        epoch=epoch,
        lr=lr,
        logger=logger,
        trainimg=trainimg,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(**args):
    """Performs the main training loop."""

    # ── Load datasets ──────────────────────────────────────────────────────
    print('> Loading datasets ...')
    dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
    loader_train = train_simple_loader(
        batch_size=args['batch_size'],
        file_root=args['trainset_dir'],
        sequence_length=args['temp_patch_size'],
        crop_size=args['patch_size'],
        epoch_size=args['max_number_patches'],
        random_shuffle=True,
        temp_stride=3,
    )

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    ctrl_fr_idx     = (args['temp_patch_size'] - 1) // 2
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # ── Init loggers ───────────────────────────────────────────────────────
    writer, logger = init_logging(args)

    # ── Model ──────────────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    device_ids = [0]

    model = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
    )
    print("########### Model Architecture ###############")
    print(model)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # ── Loss & optimizer ───────────────────────────────────────────────────
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # ── Resume or start fresh ──────────────────────────────────────────────
    start_epoch, training_params = resume_training(args, model, optimizer)

    # ── Training ───────────────────────────────────────────────────────────
    start_time = time.time()

    for epoch in range(start_epoch, args['epochs']):

        # Learning rate schedule
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print('\nlearning rate %f' % current_lr)

        for i, data in enumerate(loader_train, 0):

            model.train()
            optimizer.zero_grad()

            # data['data']: (N, temp_patch_size, C, H, W) in [0, 255]
            # normalize_augment returns:
            #   img_train : (N, temp_patch_size*C, H, W) in [0, 1]
            #   gt_train  : (N, C, H, W)  — central frame clean
            img_train, gt_train = normalize_augment(data['data'], ctrl_fr_idx)
            N, _, H, W = img_train.size()
            num_frames = args['temp_patch_size']

            # Per-sequence noise std (same std for all frames in a sequence)
            stdn = torch.empty((N, 1, 1, 1), device='cpu').uniform_(
                args['noise_ival'][0], args['noise_ival'][1]
            )

            # ── Send ground truth to GPU ───────────────────────────────────
            gt_train   = gt_train.cuda(non_blocking=True)
            noise_map  = stdn.expand(N, 1, H, W).cuda(non_blocking=True)

            # ── Fresh KV bank for this batch of sequences ──────────────────
            # One bank is shared across the temporal dimension for the whole
            # mini-batch.  Each item in the batch is an independent sequence,
            # but they share the same bank object — this is fine because all
            # tensors are batched (N is the batch dim).
            bank = KVBank(bank_size=args['bank_size'])

            # ── Feed frames one by one through the sequence ────────────────
            loss = torch.tensor(0.0).cuda()
            out_train = None  # will hold the output of the central frame

            for t in range(num_frames):
                # Extract t-th frame: (N, 3, H, W) in [0, 1]
                ft = img_train[:, 3*t:3*t+3, :, :]

                # Add noise
                noise_t = torch.normal(
                    mean=torch.zeros_like(ft),
                    std=stdn.expand_as(ft)
                )
                ftn = (ft + noise_t).cuda(non_blocking=True)
                ft  = ft.cuda(non_blocking=True)

                # Forward pass (bank is updated inside model.forward)
                out_t = model(ftn, noise_map, bank)

                # Only compute loss for the central frame (match original behaviour)
                if t == ctrl_fr_idx:
                    loss = criterion(gt_train, out_t) / (N * 2)
                    out_train = out_t

            # ── Backprop ──────────────────────────────────────────────────
            loss.backward()
            optimizer.step()

            # ── Logging & regularization ──────────────────────────────────
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                log_train_psnr(
                    out_train, gt_train, loss,
                    writer, epoch, i, num_minibatches, training_params,
                )

            training_params['step'] += 1

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        validate_and_log_singleframe(
            model=model.module,          # unwrap DataParallel for clean forward
            dataset_val=dataset_val,
            valnoisestd=args['val_noiseL'],
            temp_psz=args['temp_patch_size'],
            writer=writer,
            epoch=epoch,
            lr=current_lr,
            logger=logger,
            trainimg=img_train,
            bank_size=args['bank_size'],
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FastDVDnet (single-frame + KV bank)")

    # Training
    parser.add_argument("--batch_size",          type=int,   default=64)
    parser.add_argument("--epochs", "--e",        type=int,   default=80)
    parser.add_argument("--resume_training", "--r", action='store_true')
    parser.add_argument("--milestone",            nargs=2, type=int, default=[50, 60])
    parser.add_argument("--lr",                   type=float, default=1e-3)
    parser.add_argument("--no_orthog",            action='store_true')
    parser.add_argument("--save_every",           type=int,   default=10)
    parser.add_argument("--save_every_epochs",    type=int,   default=5)
    parser.add_argument("--noise_ival",           nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_noiseL",           type=float, default=25)

    # Patch / sequence
    parser.add_argument("--patch_size", "--p",    type=int,   default=96)
    parser.add_argument("--temp_patch_size","--tp",type=int,  default=5,
                        help="Number of frames per training sequence")
    parser.add_argument("--max_number_patches","--m", type=int, default=256000)

    # KV bank
    parser.add_argument("--bank_size",            type=int,   default=10,
                        help="Max number of past frames stored in KV bank")
    parser.add_argument("--num_heads",            type=int,   default=4,
                        help="Attention heads in BottleneckCrossAttn")
    parser.add_argument("--pool_size",            type=int,   default=8,
                        help="Spatial pool size before attention (pool_size × pool_size tokens)")

    # Dirs
    parser.add_argument("--log_dir",              type=str,   default="logs")
    parser.add_argument("--trainset_dir",         type=str,   default=None)
    parser.add_argument("--valset_dir",           type=str,   default=None)

    argspar = parser.parse_args()

    # Normalise noise to [0, 1]
    argspar.val_noiseL      /= 255.
    argspar.noise_ival[0]   /= 255.
    argspar.noise_ival[1]   /= 255.

    print("\n### Training FastDVDnet (single-frame + KV bank) ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
