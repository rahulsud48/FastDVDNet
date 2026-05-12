"""
Trains FastDVDnet (single-frame + KV bank variant).

Key differences from the original:
  - Frames fed ONE AT A TIME in temporal order via an inner loop.
  - A fresh KVBank is created per mini-batch; bank.push() is called inside
    model.forward() after attention, so temporal context builds up naturally.
  - Combined loss: Charbonnier + Frequency + Temporal consistency.
  - Noisy frames are clamped to [0, 1] before the forward pass.
  - During training, KV tensors in the bank are NOT detached so that gradients
    can flow back through the temporal attention path.
"""

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
# Loss functions
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((pred - target)^2 + eps^2)
    A smooth L1-like loss that is more robust than MSE — less over-smoothing,
    better edge preservation, slightly higher PSNR than MSE in practice.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class FrequencyLoss(nn.Module):
    """
    Frequency-domain loss: L1 on the 2D FFT magnitude spectrum.
    Penalises errors in high-frequency components (edges, textures) that
    pixel-space losses tend to under-weight, improving perceptual sharpness.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return torch.mean(torch.abs(pred_fft - target_fft))


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss: L1 between consecutive denoised frames.
    Penalises flicker — large changes between adjacent outputs that are not
    present in the clean signal. Applied only when a previous frame exists.
    """
    def forward(self, out_t: torch.Tensor, out_prev: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(out_t - out_prev))


class CombinedLoss(nn.Module):
    """
    Weighted sum of Charbonnier + Frequency + Temporal losses.

    Default weights:
        lambda_pixel  = 1.0   (anchor — drives PSNR)
        lambda_freq   = 0.1   (sharpness / texture recovery)
        lambda_temp   = 0.05  (flicker suppression)

    All three are individually logged to TensorBoard so you can tune weights
    by watching which component dominates.
    """
    def __init__(self,
                 lambda_pixel: float = 1.0,
                 lambda_freq:  float = 0.1,
                 lambda_temp:  float = 0.05,
                 charbonnier_eps: float = 1e-3):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_freq  = lambda_freq
        self.lambda_temp  = lambda_temp

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.frequency   = FrequencyLoss()
        self.temporal    = TemporalConsistencyLoss()

    def forward(self,
                pred:     torch.Tensor,
                target:   torch.Tensor,
                prev_out: torch.Tensor = None):
        """
        Args:
            pred    : (N, C, H, W) denoised output for the current frame
            target  : (N, C, H, W) clean ground truth
            prev_out: (N, C, H, W) denoised output of the previous frame,
                      or None for the very first frame in a sequence
        Returns:
            total   : scalar combined loss
            l_pixel : scalar Charbonnier component (for logging)
            l_freq  : scalar frequency component   (for logging)
            l_temp  : scalar temporal component    (for logging, 0 if prev_out is None)
        """
        l_pixel = self.charbonnier(pred, target)
        l_freq  = self.frequency(pred, target)
        l_temp  = self.temporal(pred, prev_out) if prev_out is not None \
                  else torch.tensor(0.0, device=pred.device)

        total = (self.lambda_pixel * l_pixel
                 + self.lambda_freq  * l_freq
                 + self.lambda_temp  * l_temp)

        return total, l_pixel, l_freq, l_temp


# ---------------------------------------------------------------------------
# Validation — runs frame-by-frame with a fresh KVBank per clip
# ---------------------------------------------------------------------------

def validate_and_log_singleframe(model, dataset_val, valnoisestd, bank_size,
                                  writer, epoch, lr, logger, trainimg):
    """Validation loop: feeds frames one at a time with a per-clip KVBank."""
    t1 = _time.time()
    psnr_val = 0.0

    model.eval()
    with torch.no_grad():
        for seq_val in dataset_val:
            noise    = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val = (seq_val + noise).clamp(0., 1.).cuda()
            sigma_noise = torch.cuda.FloatTensor([valnoisestd])

            out_val = denoise_seq_fastdvdnet(
                seq=seqn_val,
                noise_std=sigma_noise,
                temp_psz=None,
                model_temporal=model,
                bank_size=bank_size,
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
        logger.error("validate_and_log_singleframe(): Couldn't log results, {}".format(e))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(**args):
    """Performs the main training loop."""

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

    # Model
    torch.backends.cudnn.benchmark = True
    model = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
    )
    print("########### Model Architecture ###############")
    print(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # Combined loss
    criterion = CombinedLoss(
        lambda_pixel=args['lambda_pixel'],
        lambda_freq=args['lambda_freq'],
        lambda_temp=args['lambda_temp'],
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    start_epoch, training_params = resume_training(args, model, optimizer)
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

            # normalize_augment -> img_train: (N, temp_patch_size*C, H, W) in [0,1]
            #                      gt_train:  (N, C, H, W) central frame clean
            img_train, gt_train = normalize_augment(data['data'], ctrl_fr_idx)
            N, _, H, W = img_train.size()
            num_frames = args['temp_patch_size']

            stdn      = torch.empty((N, 1, 1, 1)).uniform_(args['noise_ival'][0],
                                                            args['noise_ival'][1])
            gt_train  = gt_train.cuda(non_blocking=True)
            noise_map = stdn.expand(N, 1, H, W).cuda(non_blocking=True)

            # Fresh KV bank per mini-batch (detach=False for gradient flow)
            bank = KVBank(bank_size=args['bank_size'], detach=False)

            loss      = torch.tensor(0.0).cuda()
            out_train = None
            out_prev  = None   # tracks previous frame output for temporal loss

            for t in range(num_frames):
                ft      = img_train[:, 3*t:3*t+3, :, :]
                noise_t = torch.normal(mean=torch.zeros_like(ft), std=stdn.expand_as(ft))
                ftn     = (ft + noise_t).clamp(0., 1.).cuda(non_blocking=True)

                out_t = model(ftn, noise_map, bank)

                # Compute combined loss on central frame
                if t == ctrl_fr_idx:
                    loss, l_pixel, l_freq, l_temp = criterion(
                        pred=out_t,
                        target=gt_train,
                        prev_out=out_prev,
                    )
                    out_train = out_t

                # Keep previous output for temporal loss (detached — we only
                # want to penalise the current frame's output, not backprop
                # through the previous frame's graph a second time)
                out_prev = out_t.detach()

            loss.backward()
            optimizer.step()

            # Logging & SVD regularisation
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log individual loss components to TensorBoard
                writer.add_scalar('loss/total',    loss.item(),    training_params['step'])
                writer.add_scalar('loss/pixel',    l_pixel.item(), training_params['step'])
                writer.add_scalar('loss/freq',     l_freq.item(),  training_params['step'])
                writer.add_scalar('loss/temporal', l_temp.item(),  training_params['step'])

                log_train_psnr(out_train, gt_train, loss,
                               writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # Validation
        validate_and_log_singleframe(
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

        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    elapsed = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train FastDVDnet (single-frame + KV bank)")

    # Training
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

    # Patch / sequence
    parser.add_argument("--patch_size", "--p",       type=int,   default=96)
    parser.add_argument("--temp_patch_size", "--tp", type=int,   default=5)
    parser.add_argument("--max_number_patches","--m",type=int,   default=256000)

    # KV bank
    parser.add_argument("--bank_size",               type=int,   default=10)
    parser.add_argument("--num_heads",               type=int,   default=4)
    parser.add_argument("--pool_size",               type=int,   default=8)

    # Loss weights
    parser.add_argument("--lambda_pixel",            type=float, default=1.0,
                        help="Weight for Charbonnier pixel loss")
    parser.add_argument("--lambda_freq",             type=float, default=0.1,
                        help="Weight for frequency domain loss")
    parser.add_argument("--lambda_temp",             type=float, default=0.05,
                        help="Weight for temporal consistency loss")

    # Dirs
    parser.add_argument("--log_dir",                 type=str,   default="logs")
    parser.add_argument("--trainset_dir",            type=str,   default=None)
    parser.add_argument("--valset_dir",              type=str,   default=None)

    argspar = parser.parse_args()

    argspar.val_noiseL    /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet (single-frame + KV bank) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
