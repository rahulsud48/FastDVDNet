"""
Trains FastDVDnet (single-frame + KV bank), full-scene scheme.

Two input paths selected by --yuv:
  RGB path: input = cat[noisy RGB(3), sigma_read(1), lambda_shot(1)] (5ch)
            loss = MSE(denoised_rgb, clean_rgb)
  YUV path: noise added in RGB, then RGB->YUV420 (UV at /4).
            model(y_noisy, uv_noise, bank_k, bank_v) -> y_res, uv_res
            loss = w_y*MSE(y_out, y_clean) + w_uv*MSE(uv_out, uv_clean)   [YUV space]
            reported PSNR (train log + validation) = RGB PSNR (YUV->RGB).

Full-scene scheme (both paths):
  - process scene_length consecutive frames sequentially
  - per-frame loss, loss.backward() each frame (grad accumulates), bank detached
  - one optimizer.step() per scene
  - sigma & lambda sampled once per scene
"""

import os, sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as tutils
import time as _time

from models import FastDVDnet
from dataset import ValDataset
from simple_dataloader import train_simple_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment, batch_psnr
from train_common import (resume_training, lr_scheduler, log_train_psnr,
                          save_model_checkpoint)
from fastdvdnet import denoise_seq_fastdvdnet
# QAT: lifecycle helper for prepare_qat.
from qat_utils import prepare_model_qat
from yuv_utils import rgb_to_yuv420, yuv420_to_rgb


# ---------------------------------------------------------------------------
# Validation — reports RGB PSNR in both modes.
# ---------------------------------------------------------------------------
def validate_and_log_singleframe(model, dataset_val, valnoisestd, val_lam,
                                  bank_size, writer, epoch, lr, logger, trainimg, yuv):
    t1 = _time.time()
    psnr_val = 0.0
    psnr_y_val = 0.0
    psnr_uv_val = 0.0
    model.eval()

    with torch.no_grad():
        for seq_idx, seq_val in enumerate(dataset_val):
            torch.manual_seed(epoch * 1000 + seq_idx)
            T, C, H, W = seq_val.shape

            # Build noisy RGB sequence (shot then read), same pipeline as train.
            noisy_frames = []
            for t in range(T):
                frame = seq_val[t].unsqueeze(0)
                photons = (frame * 255.0 / val_lam).clamp(1e-6)
                frame   = (torch.poisson(photons) * val_lam / 255.0).clamp(0., 1.)
                frame   = (frame + torch.randn_like(frame) * valnoisestd).clamp(0., 1.)
                noisy_frames.append(frame)
            seqn_val = torch.cat(noisy_frames, dim=0)           # CPU (T,3,H,W)

            sigma_noise = torch.FloatTensor([valnoisestd])
            # denoise_seq returns RGB in both modes.
            out_val = denoise_seq_fastdvdnet(
                seq=seqn_val,
                noise_std=sigma_noise,
                lambda_shot=val_lam,
                temp_psz=None,
                model_temporal=model,
                bank_size=bank_size,
                yuv=yuv,
            )
            clean_seq = seq_val.squeeze_()                       # (T,3,H,W) clean RGB
            psnr_val += batch_psnr(out_val.cpu(), clean_seq, 1.)

            # Diagnostic: Y and UV PSNR. Convert both denoised and clean RGB to
            # YUV420 and compare per channel. Tells us whether a low RGB PSNR is
            # driven by luma error, chroma error, or the YUV<->RGB conversion.
            y_den,  uv_den  = rgb_to_yuv420(out_val.cpu(),   uv_down=4)
            y_cln,  uv_cln  = rgb_to_yuv420(clean_seq.cpu(), uv_down=4)
            psnr_y_val  += batch_psnr(y_den,  y_cln,  1.)
            psnr_uv_val += batch_psnr(uv_den, uv_cln, 1.)

        nseq = len(dataset_val)
        psnr_val    /= nseq
        psnr_y_val  /= nseq
        psnr_uv_val /= nseq
        t2 = _time.time()
        print("\n[epoch %d] PSNR_val  RGB: %.4f | Y: %.4f | UV: %.4f  (%.2f sec)"
              % (epoch + 1, psnr_val, psnr_y_val, psnr_uv_val, t2 - t1))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('PSNR val Y',  psnr_y_val,  epoch)
        writer.add_scalar('PSNR val UV', psnr_uv_val, epoch)
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

    return psnr_val, psnr_y_val, psnr_uv_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(**args):
    yuv = args['yuv']
    w_y, w_uv = args['w_y'], args['w_uv']

    print('> Loading datasets ...')
    dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
    loader_train = train_simple_loader(
        batch_size=args['batch_size'],
        file_root=args['trainset_dir'],
        sequence_length=args['temp_patch_size'],
        crop_h=args['patch_h'],
        crop_w=args['patch_w'],
        epoch_size=args['max_number_patches'],
        random_shuffle=True,
        temp_stride=args['temp_stride'],
        num_workers=args['num_workers'],
    )

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    writer, logger = init_logging(args)

    torch.backends.cudnn.benchmark = True
    model = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
        train_mode=True,
        input_bits=args['input_bits'],   # QAT: deploy-time input bit-depth
    )
    print("########### Model Architecture ###############")
    print(model)
    # QAT: prepare for quantization-aware training (fuse -> qconfig -> prepare_qat).
    #      No DataParallel in QAT mode (eager-mode QAT + DataParallel don't mix).
    if args['qat']:
        print('> Preparing model for QAT ...')
        model = prepare_model_qat(model)
        model = model.cuda()
    else:
        model = nn.DataParallel(model, device_ids=[0]).cuda()

    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    start_epoch, training_params = resume_training(args, model, optimizer)
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

            img_train, gt_train = normalize_augment(data['data'])   # clean RGB stack, [0,1]
            N, _, H, W = img_train.size()
            num_frames = args['temp_patch_size']

            # sigma & lambda sampled ONCE per scene
            stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(
                args['noise_ival'][0], args['noise_ival'][1])
            lam = torch.empty((N, 1, 1, 1)).cuda().uniform_(
                args['lam_ival'][0], args['lam_ival'][1])

            sigma_read_map = stdn.expand(N, 1, H, W)    # RGB-path full-res map

            bank_k = torch.zeros(N, 10 * 64, 32).cuda()
            bank_v = torch.zeros(N, 10 * 64, 32).cuda()

            scene_loss_log = 0.0
            out_rgb_last, gt_rgb_last = None, None

            for t in range(num_frames):
                clean = img_train[:, 3*t:3*t+3, :, :].cuda(non_blocking=True)   # clean RGB

                # Noise in RGB (shot then read)
                photons = (clean * 255.0 / lam.expand_as(clean)).clamp(1e-6)
                shot    = (torch.poisson(photons) * lam.expand_as(clean) / 255.0).clamp(0., 1.)
                noise   = torch.normal(mean=torch.zeros_like(shot), std=stdn.expand_as(shot))
                noisy   = (shot + noise).clamp(0., 1.)

                if yuv:
                    # noisy & clean -> YUV420 (UV at /4)
                    y_n, uv_n = rgb_to_yuv420(noisy, uv_down=4)
                    y_c, uv_c = rgb_to_yuv420(clean, uv_down=4)

                    Hq, Wq = y_n.shape[-2] // 4, y_n.shape[-1] // 4
                    s_map = torch.full((N, 1, Hq, Wq), 0.0, device=y_n.device)
                    s_map = s_map + stdn.view(N, 1, 1, 1)          # flat sigma at /4
                    luma_q = F.avg_pool2d(y_n, 4)
                    l_map = (luma_q * lam.view(N, 1, 1, 1)).clamp(1e-6, 1.0)
                    uv_noise = torch.cat([uv_n, s_map, l_map], dim=1)   # (N,4,Hq,Wq)

                    y_res, uv_res, curr_k, curr_v = model(y_n, uv_noise, bank_k, bank_v)
                    y_out  = y_n  - y_res
                    uv_out = uv_n - uv_res

                    # Loss in YUV (weighted). reduction='sum' / (N*2*frames).
                    loss_y  = criterion(y_out,  y_c)  / (N * 2 * num_frames)
                    loss_uv = criterion(uv_out, uv_c) / (N * 2 * num_frames)
                    loss = w_y * loss_y + w_uv * loss_uv
                    loss.backward()

                    # RGB reconstruction for logging only (detached)
                    with torch.no_grad():
                        out_rgb_last = yuv420_to_rgb(y_out.detach().clamp(0,1),
                                                     uv_out.detach().clamp(0,1))
                        gt_rgb_last = clean
                else:
                    lambda_shot_map = (noisy.mean(dim=1, keepdim=True) *
                                       lam.expand(N, 1, H, W)).clamp(1e-6, 1.0)
                    input_data = torch.cat((noisy, sigma_read_map, lambda_shot_map), dim=1)
                    res_t, curr_k, curr_v = model(input_data, bank_k, bank_v)
                    out_t = noisy - res_t
                    loss = criterion(out_t, clean) / (N * 2 * num_frames)
                    loss.backward()
                    out_rgb_last, gt_rgb_last = out_t.detach(), clean

                bank_k = torch.cat([bank_k[:, 64:, :], curr_k.detach()], dim=1)
                bank_v = torch.cat([bank_v[:, 64:, :], curr_v.detach()], dim=1)
                scene_loss_log += float(loss.item())

            optimizer.step()

            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)
                loss_tensor = torch.tensor(scene_loss_log)
                log_train_psnr(out_rgb_last.clamp(0, 1), gt_rgb_last, loss_tensor,
                               writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # QAT: model is bare in QAT mode (no DataParallel), wrapped otherwise.
        val_model = model.module if isinstance(model, nn.DataParallel) else model
        psnr_val, psnr_y_val, psnr_uv_val = validate_and_log_singleframe(
            model=val_model,
            dataset_val=dataset_val,
            valnoisestd=args['val_noiseL'],
            val_lam=args['val_lam'],
            bank_size=args['bank_size'],
            writer=writer,
            epoch=epoch,
            lr=current_lr,
            logger=logger,
            trainimg=img_train,
            yuv=yuv,
        )

        # Selection score:
        #   YUV mode -> weighted Y+UV (same weights as the loss), since the
        #     model operates in YUV and RGB is just a derived view. This avoids
        #     letting YUV<->RGB conversion artifacts drive checkpoint choice.
        #   RGB mode -> plain RGB PSNR.
        if yuv:
            sel_score = w_y * psnr_y_val + w_uv * psnr_uv_val
            sel_name  = "Y+UV (w_y*Y + w_uv*UV)"
        else:
            sel_score = psnr_val
            sel_name  = "RGB"

        if sel_score > training_params['best_psnr']:
            training_params['best_psnr']  = sel_score
            training_params['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), os.path.join(args['log_dir'], 'net_best.pth'))
            print("  New best [{}]: {:.4f} at epoch {}  "
                  "(RGB {:.4f} | Y {:.4f} | UV {:.4f}) — saved net_best.pth"
                  .format(sel_name, sel_score, epoch + 1,
                          psnr_val, psnr_y_val, psnr_uv_val))
            logger.info("New best [{}]: {:.4f} at epoch {} (RGB {:.4f} | Y {:.4f} | UV {:.4f})"
                        .format(sel_name, sel_score, epoch + 1,
                                psnr_val, psnr_y_val, psnr_uv_val))
        else:
            print("  (best [{}]: {:.4f} at epoch {})".format(
                sel_name, training_params['best_psnr'], training_params['best_epoch']))

        writer.add_scalar('Best selection score', training_params['best_psnr'], epoch)
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    elapsed = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastDVDnet (KV bank, RGB/YUV)")

    parser.add_argument("--batch_size",             type=int,   default=64)
    parser.add_argument("--epochs", "--e",           type=int,   default=80)
    parser.add_argument("--resume_training", "--r",  action='store_true')
    parser.add_argument("--milestone",               nargs=2, type=int, default=[50, 60])
    parser.add_argument("--lr",                      type=float, default=1e-3)
    parser.add_argument("--no_orthog",               action='store_true')
    parser.add_argument("--save_every",              type=int,   default=10)
    parser.add_argument("--save_every_epochs",       type=int,   default=5)

    parser.add_argument("--noise_ival",   nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_noiseL",   type=float, default=25)
    parser.add_argument("--lam_ival",     nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_lam",      type=float, default=25)

    parser.add_argument("--patch_h", type=int, default=96)
    parser.add_argument("--patch_w", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=20)
    parser.add_argument("--max_number_patches", "--m", type=int, default=256000)
    parser.add_argument("--temp_stride", type=int, default=3)

    parser.add_argument("--bank_size",  type=int, default=10)
    parser.add_argument("--num_heads",  type=int, default=4)
    parser.add_argument("--pool_size",  type=int, default=8)

    # YUV mode
    parser.add_argument("--yuv", action='store_true',
                        help="Use YUV420 path (Y full-res + UV at /4). Default: RGB path.")
    parser.add_argument("--w_y",  type=float, default=1.0,  help="Y loss weight (YUV mode)")
    parser.add_argument("--w_uv", type=float, default=0.25, help="UV loss weight (YUV mode)")
    # QAT: quantization-aware training flags. Without --qat, training is FP32.
    parser.add_argument("--qat", action='store_true', help="QAT: enable quantization-aware training")
    parser.add_argument("--input_bits", type=int, default=8, help="QAT: deploy-time input bit-depth")

    parser.add_argument("--log_dir",        type=str, default="logs")
    parser.add_argument("--trainset_dir",   type=str, default=None)
    parser.add_argument("--valset_dir",     type=str, default=None)

    argspar = parser.parse_args()

    argspar.val_noiseL    /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.
    argspar.val_lam       /= 255.
    argspar.lam_ival[0]   /= 255.
    argspar.lam_ival[1]   /= 255.

    print("\n### Training FastDVDnet (%s) ###" % ("YUV420" if argspar.yuv else "RGB"))
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
