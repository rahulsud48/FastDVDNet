"""
Trains FastDVDnet (single RGB frame + KV bank).

Loss: MSE on the predicted residual vs true noise only.
  true_noise      = noisy_frame - clean_frame
  predicted_noise = model output (raw network residual)
  loss            = MSE(predicted_noise, true_noise)

This directly supervises on noise, so the model never sees clean image
texture in the loss — texture is invisible to the gradient and cannot
be suppressed.
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tutils
import time as _time

from models import FastDVDnet, KVBank
from dataset import ValDataset
from simple_dataloader import train_simple_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment, batch_psnr
from train_common import (resume_training, lr_scheduler, log_train_psnr,
                          save_model_checkpoint)
from fastdvdnet import denoise_seq_fastdvdnet


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_log_singleframe(model, dataset_val, valnoisestd, bank_size,
                                  writer, epoch, lr, logger, trainimg):
    """Returns psnr_val so the training loop can track the best model."""
    t1 = _time.time()
    psnr_val = 0.0
    model.eval()
    with torch.no_grad():
        for seq_val in dataset_val:
            noise       = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val    = (seq_val + noise).clamp(0., 1.).cuda()
            sigma_noise = torch.cuda.FloatTensor([valnoisestd])
            out_val = denoise_seq_fastdvdnet(
                seq=seqn_val, noise_std=sigma_noise,
                temp_psz=None, model_temporal=model, bank_size=bank_size,
            )
            psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), 1.)
        psnr_val /= len(dataset_val)
        t2 = _time.time()
        print("\n[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch + 1, psnr_val, (t2 - t1)))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', lr, epoch)
    try:
        idx = 0
        if epoch == 0:
            _, _, Ht, Wt = trainimg.size()
            img = tutils.make_grid(trainimg.view(-1, 3, Ht, Wt),
                                   nrow=8, normalize=True, scale_each=True)
            writer.add_image('Training patches', img, epoch)
        irecon = tutils.make_grid(out_val.data[idx].clamp(0., 1.),
                                  nrow=2, normalize=False, scale_each=False)
        writer.add_image('Reconstructed validation image {}'.format(idx), irecon, epoch)
    except Exception as e:
        logger.error("validate_and_log_singleframe(): {}".format(e))
    return psnr_val


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(**args):

    print('> Loading datasets ...')
    dataset_val  = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
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

    writer, logger = init_logging(args)

    torch.backends.cudnn.benchmark = True
    model = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
    )
    print("########### Model Architecture ###############")
    print(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # Residual-only loss: MSE on predicted noise vs true noise
    criterion = nn.MSELoss(reduction='mean').cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Initialise best PSNR tracker (preserved across resume)
    if 'best_psnr' not in training_params:
        training_params['best_psnr']  = 0.0
        training_params['best_epoch'] = 0

    start_time = time.time()

    for epoch in range(start_epoch, args['epochs']):

        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print('\nlearning rate %f' % current_lr)

        for i, data in enumerate(loader_train, 0):

            model.train()
            optimizer.zero_grad()

            img_train, gt_train = normalize_augment(data['data'], ctrl_fr_idx)
            N, _, H, W = img_train.size()
            num_frames = args['temp_patch_size']

            stdn      = torch.empty((N, 1, 1, 1)).uniform_(args['noise_ival'][0],
                                                            args['noise_ival'][1])
            gt_train  = gt_train.cuda(non_blocking=True)
            noise_map = stdn.expand(N, 1, H, W).cuda(non_blocking=True)

            bank      = KVBank(bank_size=args['bank_size'], detach=False)
            loss      = torch.tensor(0.0).cuda()
            out_train = None

            for t in range(num_frames):
                ft      = img_train[:, 3*t:3*t+3, :, :]
                noise_t = torch.normal(mean=torch.zeros_like(ft), std=stdn.expand_as(ft))
                ftn     = (ft + noise_t).clamp(0., 1.).cuda(non_blocking=True)
                ft_gpu  = ft.cuda(non_blocking=True)

                # true noise for this frame = noisy - clean
                true_noise = ftn - ft_gpu

                pred_residual, out_t = model(ftn, noise_map, bank)

                if t == ctrl_fr_idx:
                    # Loss: MSE between predicted residual and true noise
                    loss      = criterion(pred_residual, true_noise)
                    out_train = out_t

            loss.backward()
            optimizer.step()

            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)
                writer.add_scalar('loss/residual_mse', loss.item(), training_params['step'])
                log_train_psnr(out_train, gt_train, loss,
                               writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        psnr_val = validate_and_log_singleframe(
            model=model.module,
            dataset_val=dataset_val,
            valnoisestd=args['val_noiseL'],
            bank_size=args['bank_size'],
            writer=writer,
            epoch=epoch,
            lr=current_lr,
            logger=logger,
            trainimg=img_train,
        )

        # ── Best model tracking ───────────────────────────────────────────
        if psnr_val > training_params['best_psnr']:
            training_params['best_psnr']  = psnr_val
            training_params['best_epoch'] = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(args['log_dir'], 'net_best.pth')
            )
            print("  ★ New best PSNR: {:.4f} dB at epoch {} — saved net_best.pth".format(
                psnr_val, epoch + 1))
            logger.info("New best PSNR: {:.4f} dB at epoch {}".format(
                psnr_val, epoch + 1))
        else:
            print("  (best so far: {:.4f} dB at epoch {})".format(
                training_params['best_psnr'], training_params['best_epoch']))

        writer.add_scalar('Best PSNR', training_params['best_psnr'], epoch)

        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    elapsed = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FastDVDnet (residual MSE loss)")

    parser.add_argument("--batch_size",             type=int,   default=64)
    parser.add_argument("--epochs", "--e",           type=int,   default=80)
    parser.add_argument("--resume_training", "--r",  action='store_true')
    parser.add_argument("--milestone",               nargs=2, type=int, default=[50, 60])
    parser.add_argument("--lr",                      type=float, default=1e-3)
    parser.add_argument("--no_orthog",               action='store_true')
    parser.add_argument("--save_every",              type=int,   default=10)
    parser.add_argument("--save_every_epochs",       type=int,   default=5)
    parser.add_argument("--noise_ival",              nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_noiseL",              type=float, default=25)
    parser.add_argument("--patch_size", "--p",       type=int,   default=96)
    parser.add_argument("--temp_patch_size", "--tp", type=int,   default=5)
    parser.add_argument("--max_number_patches","--m",type=int,   default=256000)
    parser.add_argument("--bank_size",               type=int,   default=10)
    parser.add_argument("--num_heads",               type=int,   default=4)
    parser.add_argument("--pool_size",               type=int,   default=8)
    parser.add_argument("--log_dir",                 type=str,   default="logs")
    parser.add_argument("--trainset_dir",            type=str,   default=None)
    parser.add_argument("--valset_dir",              type=str,   default=None)

    argspar = parser.parse_args()
    argspar.val_noiseL    /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet (residual MSE loss) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
