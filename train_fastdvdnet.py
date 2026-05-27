"""
Trains FastDVDnet (Y-channel 3-frame input, no UV in network).

Training flow per mini-batch:
  1. Load RGB patch sequence (N, T*3, H, W) in [0, 1] from dataloader
  2. Extract central frame index ctrl_fr_idx  (e.g. 2 for T=5)
  3. For the window [ctrl_fr_idx-1, ctrl_fr_idx, ctrl_fr_idx+1]:
       - Convert each RGB frame to Y (clean)
       - Add Gaussian noise to each Y -> noisy Y
       - Stack noisy Y: y_frames (N, 3, H, W)
  4. Forward: model(y_frames, noise_map) -> y_pred  (N, 1, H, W)
  5. Loss: computed on y_pred vs y_clean (central frame Y only)

PSNR for validation is computed in RGB space (yuv422_to_rgb with clean UV)
so it stays comparable to RGB baselines.
"""

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tutils
import time as _time

from models import FastDVDnet, rgb_to_yuv422, yuv422_to_rgb
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
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return torch.mean(torch.abs(pred_fft - target_fft))


class CombinedLoss(nn.Module):
    """Charbonnier + Frequency — both applied in Y domain."""
    def __init__(self, lambda_pixel=1.0, lambda_freq=0.1, charbonnier_eps=1e-3):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_freq  = lambda_freq
        self.charbonnier  = CharbonnierLoss(eps=charbonnier_eps)
        self.frequency    = FrequencyLoss()

    def forward(self, y_pred, y_clean):
        """
        Args:
            y_pred  : (N, 1, H, W) — denoised Y (model output)
            y_clean : (N, 1, H, W) — ground truth Y (no noise)
        """
        l_pixel = self.charbonnier(y_pred, y_clean)
        l_freq  = self.frequency(y_pred, y_clean)
        total   = self.lambda_pixel * l_pixel + self.lambda_freq * l_freq
        return total, l_pixel, l_freq


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_log(model, dataset_val, valnoisestd, writer, epoch, lr, logger, trainimg):
    """
    Validation: for each val sequence, add noise to RGB, convert to Y,
    denoise Y, reconstruct RGB, compute PSNR in RGB space.
    """
    t1 = _time.time()
    psnr_val = 0.0

    model.eval()
    with torch.no_grad():
        for seq_val in dataset_val:
            # seq_val: (T, 3, H, W) clean RGB in [0, 1]
            noise    = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val = (seq_val + noise).clamp(0., 1.).cuda()
            noisestd = torch.cuda.FloatTensor([valnoisestd])

            out_val = denoise_seq_fastdvdnet(
                seq=seqn_val,
                noise_std=noisestd,
                temp_psz=None,
                model_temporal=model,
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
        logger.error("validate_and_log(): {}".format(e))


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
    ctrl_fr_idx     = (args['temp_patch_size'] - 1) // 2   # central frame index, e.g. 2 for T=5
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    writer, logger = init_logging(args)

    # ── Model ─────────────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    model = FastDVDnet(num_input_frames=3)
    print("########### Model Architecture ###############")
    print(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        lambda_pixel=args['lambda_pixel'],
        lambda_freq=args['lambda_freq'],
    ).cuda()

    optimizer   = optim.Adam(model.parameters(), lr=args['lr'])
    start_epoch, training_params = resume_training(args, model, optimizer)
    start_time  = time.time()

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

            # img_train: (N, T*3, H, W) in [0, 1]
            # gt_train:  (N, 3, H, W)   clean central RGB frame (unused for loss, kept for logging)
            img_train, gt_train = normalize_augment(data['data'], ctrl_fr_idx)
            N, _, H, W = img_train.size()

            # Sample noise std uniformly per sample
            stdn = torch.empty((N, 1, 1, 1)).uniform_(
                args['noise_ival'][0], args['noise_ival'][1]
            )

            # ── Build 3-frame Y input: [t-1, t, t+1] around central frame ──
            # Extract the 3 RGB frames that form the window
            t_prev = ctrl_fr_idx - 1
            t_curr = ctrl_fr_idx
            t_next = ctrl_fr_idx + 1

            rgb_prev = img_train[:, 3*t_prev:3*t_prev+3, :, :]   # (N, 3, H, W)
            rgb_curr = img_train[:, 3*t_curr:3*t_curr+3, :, :]   # (N, 3, H, W)
            rgb_next = img_train[:, 3*t_next:3*t_next+3, :, :]   # (N, 3, H, W)

            # Convert each to Y (clean)
            y_prev, _      = rgb_to_yuv422(rgb_prev)   # (N, 1, H, W)
            y_curr, uv_curr = rgb_to_yuv422(rgb_curr)  # (N, 1, H, W), (N, 2, H/2, W/2)
            y_next, _      = rgb_to_yuv422(rgb_next)   # (N, 1, H, W)

            # Add Gaussian noise to each Y frame
            noise_prev = torch.normal(mean=torch.zeros_like(y_prev), std=stdn.expand_as(y_prev))
            noise_curr = torch.normal(mean=torch.zeros_like(y_curr), std=stdn.expand_as(y_curr))
            noise_next = torch.normal(mean=torch.zeros_like(y_next), std=stdn.expand_as(y_next))

            yn_prev = (y_prev + noise_prev).clamp(0., 1.).cuda(non_blocking=True)
            yn_curr = (y_curr + noise_curr).clamp(0., 1.).cuda(non_blocking=True)
            yn_next = (y_next + noise_next).clamp(0., 1.).cuda(non_blocking=True)

            y_frames  = torch.cat([yn_prev, yn_curr, yn_next], dim=1)   # (N, 3, H, W)
            y_clean   = y_curr.cuda(non_blocking=True)                  # (N, 1, H, W) target
            noise_map = stdn.expand(N, 1, H, W).cuda(non_blocking=True) # (N, 1, H, W)

            # ── Forward ───────────────────────────────────────────────────
            y_pred = model(y_frames, noise_map)   # (N, 1, H, W)

            # ── Loss in Y domain ──────────────────────────────────────────
            loss, l_pixel, l_freq = criterion(y_pred, y_clean)
            loss.backward()
            optimizer.step()

            # ── Logging ───────────────────────────────────────────────────
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                writer.add_scalar('loss/total', loss.item(),    training_params['step'])
                writer.add_scalar('loss/pixel', l_pixel.item(), training_params['step'])
                writer.add_scalar('loss/freq',  l_freq.item(),  training_params['step'])

                # Reconstruct RGB for PSNR logging (clean UV, no loss computed on this)
                with torch.no_grad():
                    uv_clean  = uv_curr.cuda(non_blocking=True)
                    out_rgb   = yuv422_to_rgb(y_pred.detach(), uv_clean)   # (N, 3, H, W)
                    gt_rgb    = gt_train.cuda(non_blocking=True)           # (N, 3, H, W)

                log_train_psnr(out_rgb, gt_rgb, loss,
                               writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # ── Validation ────────────────────────────────────────────────────
        validate_and_log(
            model=model.module,
            dataset_val=dataset_val,
            valnoisestd=args['val_noiseL'],
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

    parser = argparse.ArgumentParser(description="Train FastDVDnet (Y-channel 3-frame)")

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
    parser.add_argument("--lambda_pixel",            type=float, default=1.0)
    parser.add_argument("--lambda_freq",             type=float, default=0.1)
    parser.add_argument("--log_dir",                 type=str,   default="logs")
    parser.add_argument("--trainset_dir",            type=str,   default=None)
    parser.add_argument("--valset_dir",              type=str,   default=None)

    argspar = parser.parse_args()
    argspar.val_noiseL    /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet (Y-channel 3-frame) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
