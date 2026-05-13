"""
Trains FastDVDnet (YUV422 single-frame + KV bank variant).

YUV422 training flow per frame:
  1. Load RGB patch (from dataloader, normalized to [0,1])
  2. Convert RGB -> Y (N,1,H,W) + UV (N,2,H/2,W/2)
  3. Add Gaussian noise to Y only (chroma noise modelled separately via noise_map)
  4. Forward: model(noisy_Y, UV, noise_map, bank) -> denoised_Y, denoised_UV
  5. Loss: computed on RGB reconstructed from denoised YUV vs clean RGB

Computing loss in RGB space ensures PSNR stays comparable to RGB baselines.
"""

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tutils
import time as _time

from models import FastDVDnet, KVBank, rgb_to_yuv422, yuv422_to_rgb
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


class TemporalConsistencyLoss(nn.Module):
    def forward(self, out_t, out_prev):
        return torch.mean(torch.abs(out_t - out_prev))


class CombinedLoss(nn.Module):
    """Charbonnier + Frequency + Temporal — applied in RGB space."""
    def __init__(self, lambda_pixel=1.0, lambda_freq=0.1,
                 lambda_temp=0.05, charbonnier_eps=1e-3):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_freq  = lambda_freq
        self.lambda_temp  = lambda_temp
        self.charbonnier  = CharbonnierLoss(eps=charbonnier_eps)
        self.frequency    = FrequencyLoss()
        self.temporal     = TemporalConsistencyLoss()

    def forward(self, pred_rgb, target_rgb, prev_rgb=None):
        l_pixel = self.charbonnier(pred_rgb, target_rgb)
        l_freq  = self.frequency(pred_rgb, target_rgb)
        l_temp  = self.temporal(pred_rgb, prev_rgb) if prev_rgb is not None \
                  else torch.tensor(0.0, device=pred_rgb.device)
        total   = (self.lambda_pixel * l_pixel
                   + self.lambda_freq  * l_freq
                   + self.lambda_temp  * l_temp)
        return total, l_pixel, l_freq, l_temp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_log_singleframe(model, dataset_val, valnoisestd, bank_size,
                                  writer, epoch, lr, logger, trainimg):
    """Validation: RGB -> YUV422 -> denoise -> RGB -> PSNR."""
    t1 = _time.time()
    psnr_val = 0.0

    model.eval()
    with torch.no_grad():
        for seq_val in dataset_val:
            # seq_val: (numframes, 3, H, W) RGB in [0,1]
            noise       = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val    = (seq_val + noise).clamp(0., 1.).cuda()
            sigma_noise = torch.cuda.FloatTensor([valnoisestd])

            # denoise_seq_fastdvdnet handles RGB->YUV->denoise->RGB internally
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
        logger.error("validate_and_log_singleframe(): {}".format(e))


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

    # ── Model ─────────────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    model = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
    )
    print("########### Model Architecture ###############")
    print(model)
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        lambda_pixel=args['lambda_pixel'],
        lambda_freq=args['lambda_freq'],
        lambda_temp=args['lambda_temp'],
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

            # img_train: (N, temp_patch_size*3, H, W) in [0,1] — all frames stacked
            # gt_train:  (N, 3, H, W) — clean central frame RGB
            img_train, gt_train = normalize_augment(data['data'], ctrl_fr_idx)
            N, _, H, W = img_train.size()
            num_frames = args['temp_patch_size']

            stdn      = torch.empty((N, 1, 1, 1)).uniform_(args['noise_ival'][0],
                                                            args['noise_ival'][1])
            gt_train  = gt_train.cuda(non_blocking=True)   # (N, 3, H, W) RGB

            # Noise map for Y channel (same std, same spatial size as Y)
            noise_map = stdn.expand(N, 1, H, W).cuda(non_blocking=True)

            # Fresh KV bank per mini-batch (detach=False for grad flow)
            bank      = KVBank(bank_size=args['bank_size'], detach=False)
            loss      = torch.tensor(0.0).cuda()
            out_train = None
            out_prev  = None   # previous denoised RGB for temporal loss

            for t in range(num_frames):
                # Clean RGB frame t
                ft_rgb = img_train[:, 3*t:3*t+3, :, :]   # (N, 3, H, W)

                # Convert clean frame to YUV422
                ft_y, ft_uv = rgb_to_yuv422(ft_rgb)       # (N,1,H,W), (N,2,H/2,W/2)

                # Add Gaussian noise to Y only
                noise_y = torch.normal(mean=torch.zeros_like(ft_y),
                                       std=stdn.expand_as(ft_y))
                noisy_y = (ft_y + noise_y).clamp(0., 1.).cuda(non_blocking=True)
                ft_uv   = ft_uv.cuda(non_blocking=True)

                # Forward — bank updated inside model
                den_y, den_uv = model(noisy_y, ft_uv, noise_map, bank)

                # Reconstruct denoised RGB for loss computation
                out_rgb = yuv422_to_rgb(den_y, den_uv)     # (N, 3, H, W)

                # Loss on central frame — computed in RGB space
                if t == ctrl_fr_idx:
                    loss, l_pixel, l_freq, l_temp = criterion(
                        pred_rgb=out_rgb,
                        target_rgb=gt_train,
                        prev_rgb=out_prev,
                    )
                    out_train = out_rgb

                out_prev = out_rgb.detach()

            loss.backward()
            optimizer.step()

            # ── Logging ───────────────────────────────────────────────────
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                writer.add_scalar('loss/total',    loss.item(),    training_params['step'])
                writer.add_scalar('loss/pixel',    l_pixel.item(), training_params['step'])
                writer.add_scalar('loss/freq',     l_freq.item(),  training_params['step'])
                writer.add_scalar('loss/temporal', l_temp.item(),  training_params['step'])

                log_train_psnr(out_train, gt_train, loss,
                               writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # ── Validation ────────────────────────────────────────────────────
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

    parser = argparse.ArgumentParser(description="Train FastDVDnet (YUV422 + KV bank)")

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
    parser.add_argument("--lambda_pixel",            type=float, default=1.0)
    parser.add_argument("--lambda_freq",             type=float, default=0.1)
    parser.add_argument("--lambda_temp",             type=float, default=0.05)
    parser.add_argument("--log_dir",                 type=str,   default="logs")
    parser.add_argument("--trainset_dir",            type=str,   default=None)
    parser.add_argument("--valset_dir",              type=str,   default=None)

    argspar = parser.parse_args()
    argspar.val_noiseL    /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet (YUV422 + KV bank) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
