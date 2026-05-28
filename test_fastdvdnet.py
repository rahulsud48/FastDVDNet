#!/usr/bin/env python3
"""
Denoise sequences using FastDVDnet (Y-channel 3-frame input, no UV in network).

Pipeline per frame t:
  RGB -> Y (noisy) + UV (clean pass-through)
  model([y_{t-1}, y_t, y_{t+1}], noise_map) -> y_pred
  RGB = yuv444_to_rgb(y_pred, uv_t)
PSNR computed in RGB space for fair comparison with other methods.

Image saving (when --dont_save_results is NOT set):
  Results are written to:
    <save_path>/output_images_sigma<sigma_int>/
        gt/         <- clean RGB frames,   named 00000.png, 00001.png, ...
        noisy/      <- noisy RGB frames,   named 00000.png, 00001.png, ...
                       constructed as yuv444_to_rgb(y_noisy, uv_clean)
                       i.e. noise lives only in Y; UV is kept clean
        denoised/   <- denoised RGB frames, named 00000.png, 00001.png, ...

  One set of gt/noisy/denoised folders is shared across all sequences.
  Frame indices are global (continue incrementing across sequences).

  NOTE: The old per-sequence saving function save_out_seq() is preserved
  below but commented out so it can be re-enabled when merging into another
  environment.
"""

import os
import argparse
import time

import cv2
import torch
import torch.nn as nn

from models import FastDVDnet, rgb_to_yuv444, yuv444_to_rgb
from fastdvdnet import denoise_seq_fastdvdnet
from utils import (batch_psnr, init_logger_test,
                   variable_to_cv2_image, remove_dataparallel_wrapper,
                   open_sequence, close_logger)

OUTIMGEXT = '.png'


# ---------------------------------------------------------------------------
# OLD saving function — kept for reference, commented out.
# Re-enable by removing the block-comment markers and calling save_out_seq()
# inside the sequence loop (see comment in test_fastdvdnet()).
# ---------------------------------------------------------------------------
# def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
#     """Saves denoised (and optionally noisy) frames with sigma-prefixed names.
#     seqnoisy : (T, 3, H, W) noisy RGB tensor
#     seqclean : (T, 3, H, W) denoised RGB tensor  (named 'seqclean' in legacy code)
#     """
#     seq_len = seqnoisy.size()[0]
#     for idx in range(seq_len):
#         noisy_name = os.path.join(save_dir, 'n{}_{}'.format(sigmaval, idx) + OUTIMGEXT)
#         out_name   = os.path.join(save_dir,
#                                   ('n{}_FastDVDnet_{}'.format(sigmaval, idx)
#                                    if not suffix else
#                                    'n{}_FastDVDnet_{}_{}'.format(sigmaval, suffix, idx))
#                                   + OUTIMGEXT)
#         if save_noisy:
#             cv2.imwrite(noisy_name, variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.)))
#         cv2.imwrite(out_name, variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0)))


# ---------------------------------------------------------------------------
# NEW saving function
# ---------------------------------------------------------------------------

def save_image_set(seq_clean, seq_noisy_rgb, seq_denoised, dirs):
    """
    Saves one sequence worth of frames into the gt / noisy / denoised folders.
    Frame numbering resets to 00000 within each sequence (per-sequence layout).

    Args:
        seq_clean      : (T, 3, H, W) clean RGB tensor in [0, 1]
        seq_noisy_rgb  : (T, 3, H, W) noisy RGB tensor in [0, 1]
                         constructed as yuv444_to_rgb(y_noisy, uv_clean)
                         so noise lives in Y only, UV is chroma-clean
        seq_denoised   : (T, 3, H, W) denoised RGB tensor in [0, 1]
        dirs           : dict with keys 'gt', 'noisy', 'denoised' -> folder paths
    """
    T = seq_clean.size(0)

    for t in range(T):
        # Zero-padded filename, resets per sequence: 00000.png, 00001.png, ...
        fname = '{:05d}{}'.format(t, OUTIMGEXT)

        # ── Ground truth ──────────────────────────────────────────────────
        gt_img = variable_to_cv2_image(seq_clean[t].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(os.path.join(dirs['gt'], fname), gt_img)

        # ── Noisy (Y-noisy + UV-clean → RGB) ─────────────────────────────
        noisy_img = variable_to_cv2_image(seq_noisy_rgb[t].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(os.path.join(dirs['noisy'], fname), noisy_img)

        # ── Denoised ──────────────────────────────────────────────────────
        den_img = variable_to_cv2_image(seq_denoised[t].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(os.path.join(dirs['denoised'], fname), den_img)


def build_noisy_rgb(seq_clean, seq_noisy, device):
    """
    Constructs a 'visualisable' noisy RGB sequence where noise lives in Y only.

    For each frame t:
      1. Convert clean RGB -> Y_clean, UV_clean   (UV stays clean)
      2. Get Y_noisy from the noisy sequence:      Y_noisy, _ = rgb_to_yuv444(seq_noisy[t])
      3. Reconstruct: yuv444_to_rgb(Y_noisy, UV_clean)  → noisy-Y + clean-UV in RGB

    This gives a perceptually accurate noisy image (luma noise visible,
    chroma intact) and matches how noise is injected during training.

    Args:
        seq_clean : (T, 3, H, W) clean RGB in [0, 1]
        seq_noisy : (T, 3, H, W) noisy RGB in [0, 1]  (noise on all channels from torch.normal)
        device    : torch.device

    Returns:
        noisy_rgb : (T, 3, H, W) float32 in [0, 1]
    """
    T = seq_clean.size(0)
    noisy_rgb = torch.empty_like(seq_clean)

    for t in range(T):
        # Extract clean UV from the clean frame (chroma stays noise-free)
        _, uv_clean = rgb_to_yuv444(seq_clean[t].unsqueeze(0).to(device))   # (1, 2, H, W) full-res UV

        # Extract noisy Y from the noisy frame
        y_noisy, _  = rgb_to_yuv444(seq_noisy[t].unsqueeze(0).to(device))   # (1, 1, H, W)

        # Reconstruct RGB: noisy luma + clean chroma
        noisy_rgb[t] = yuv444_to_rgb(y_noisy, uv_clean).squeeze(0).cpu()    # (3,H,W)

    return noisy_rgb


def test_fastdvdnet(**args):
    start_time = time.time()

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')

    # ── Load model ────────────────────────────────────────────────────────
    print('Loading model ...')
    model_temp = FastDVDnet(num_input_frames=3)

    state_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        model_temp = nn.DataParallel(model_temp, device_ids=[0]).cuda()
    else:
        state_dict = remove_dataparallel_wrapper(state_dict)
    model_temp.load_state_dict(state_dict)
    model_temp.eval()

    # ── Find sequences ────────────────────────────────────────────────────
    seq_dirs = sorted([
        os.path.join(args['test_path'], d)
        for d in os.listdir(args['test_path'])
        if os.path.isdir(os.path.join(args['test_path'], d))
    ]) or [args['test_path']]

    # ── Create output folder structure ───────────────────────────────────
    # Folder: <save_path>/output_images_sigma<sigma_int>/gt|noisy|denoised
    # sigma_int is the integer sigma value (e.g. 25 for noise_sigma=25/255)
    if not args['dont_save_results']:
        sigma_int   = int(round(args['noise_sigma'] * 255))
        # Root: <save_path>/pred_sigma<sigma_int>/
        # Per-sequence: pred_sigma<sigma>/<seq_name>/{gt,noisy,denoised}/00000.png
        pred_root = os.path.join(args['save_path'], 'pred_sigma{}'.format(sigma_int))
        os.makedirs(pred_root, exist_ok=True)
        print('Saving per-sequence images to: {}'.format(pred_root))

    psnr_all         = []
    psnr_noisy_all   = []   # per-sequence noisy PSNR, for the summary table
    seq_names        = []   # sequence folder names, for the summary table

    with torch.no_grad():
        for seq_dir in seq_dirs:
            seq_start = time.time()
            seq_name  = os.path.basename(seq_dir.rstrip('/'))

            # Load clean RGB sequence
            seq, _, _ = open_sequence(seq_dir, args['gray'],
                                      expand_if_needed=False,
                                      max_num_fr=args['max_num_fr_per_seq'])
            seq    = torch.from_numpy(seq).to(device)    # (T, 3, H, W) RGB [0,1]
            load_t = time.time() - seq_start

            # Add noise to RGB — NOT clamped (preserve Gaussian stats, matches original)
            noise    = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma'])
            seqn     = (seq + noise)
            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            # Denoise: Y-channel 3-frame sliding window
            # CRITICAL: pass seq (the clean RGB) as seq_clean so UV is extracted
            # from the clean frame. Without this, UV defaults to the noisy frame
            # and caps RGB PSNR around 28 dB.
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                temp_psz=None,
                model_temporal=model_temp,
                seq_clean=seq,                                        # ← clean UV
            )

            run_t      = time.time() - seq_start - load_t
            psnr       = batch_psnr(denframes, seq, 1.)
            psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)

            # Accumulate per-sequence results for the summary table
            psnr_all.append(psnr)
            psnr_noisy_all.append(psnr_noisy)
            seq_names.append(seq_name)

            logger.info("Finished: {}".format(seq_dir))
            logger.info("\tFrames: {}  Load: {:.3f}s  Denoise: {:.3f}s".format(
                seq.size(0), load_t, run_t))
            logger.info("\tPSNR noisy: {:.4f} dB  PSNR denoised: {:.4f} dB".format(
                psnr_noisy, psnr))

            # ── Save results (per-sequence layout) ─────────────────────────
            if not args['dont_save_results']:

                # Build noisy RGB: noise in Y only, UV stays clean
                # This matches the training noise injection convention
                seq_noisy_rgb = build_noisy_rgb(seq.cpu(), seqn.cpu(), device)

                # Per-sequence folders: pred_sigma<>/<seq>/{gt,noisy,denoised}
                seq_dirs_out = {
                    'gt':       os.path.join(pred_root, seq_name, 'gt'),
                    'noisy':    os.path.join(pred_root, seq_name, 'noisy'),
                    'denoised': os.path.join(pred_root, seq_name, 'denoised'),
                }
                for d in seq_dirs_out.values():
                    os.makedirs(d, exist_ok=True)

                # Frame numbering resets to 00000 within each sequence
                save_image_set(
                    seq_clean=seq.cpu(),
                    seq_noisy_rgb=seq_noisy_rgb,
                    seq_denoised=denframes.cpu(),
                    dirs=seq_dirs_out,
                )

                # ── OLD per-sequence saving — commented out ───────────────
                # To restore: uncomment the block below and comment out the
                # save_image_set() call above.
                #
                # seq_save_dir = os.path.join(args['save_path'],
                #                             os.path.basename(seq_dir.rstrip('/')))
                # os.makedirs(seq_save_dir, exist_ok=True)
                # save_out_seq(seqn, denframes, seq_save_dir,
                #              int(args['noise_sigma'] * 255),
                #              args['suffix'], args['save_noisy'])

    # ── PSNR summary table ────────────────────────────────────────────────
    if psnr_all:
        sigma_int    = int(args['noise_sigma'] * 255)
        avg_psnr     = sum(psnr_all) / len(psnr_all)
        avg_noisy    = sum(psnr_noisy_all) / len(psnr_noisy_all)

        # Column widths: sequence name column is padded to the longest name
        col_seq   = max(len(n) for n in seq_names)
        col_seq   = max(col_seq, len('Sequence'))   # at least header width
        col_val   = 18                              # width for each PSNR column
        sep       = '-' * (col_seq + 2 * col_val + 4)

        header = '{:<{w}}  {:>{c}}  {:>{c}}'.format(
            'Sequence',
            'PSNR Noisy (dB)',
            'PSNR Denoised (dB)',
            w=col_seq, c=col_val,
        )
        avg_row = '{:<{w}}  {:>{c}.4f}  {:>{c}.4f}'.format(
            'Average',
            avg_noisy,
            avg_psnr,
            w=col_seq, c=col_val,
        )

        # Build all data rows first so we can log the full table at once
        rows = []
        for name, pn, pd in zip(seq_names, psnr_noisy_all, psnr_all):
            rows.append('{:<{w}}  {:>{c}.4f}  {:>{c}.4f}'.format(
                name, pn, pd, w=col_seq, c=col_val,
            ))

        table_lines = (
            ['\n=== PSNR Table (sigma={}) ==='.format(sigma_int)]
            + [header, sep]
            + rows
            + [sep, avg_row, sep]
        )
        table_str = '\n'.join(table_lines)

        # Log and print the full table
        logger.info(table_str)
        print(table_str)

    elapsed = time.time() - start_time
    logger.info("Total: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test FastDVDnet (Y-channel 3-frame)")

    parser.add_argument("--model_file",         type=str,   default="./model.pth")
    parser.add_argument("--test_path",          type=str,   default="./data/rgb/Kodak24")
    parser.add_argument("--save_path",          type=str,   default="./results")
    parser.add_argument("--suffix",             type=str,   default="")
    parser.add_argument("--max_num_fr_per_seq", type=int,   default=25)
    parser.add_argument("--noise_sigma",        type=float, default=25)
    parser.add_argument("--dont_save_results",  action='store_true')
    # --save_noisy is kept for API compatibility with the old save_out_seq() path
    parser.add_argument("--save_noisy",         action='store_true')
    parser.add_argument("--no_gpu",             action='store_true')
    parser.add_argument("--gray",               action='store_true')

    argspar = parser.parse_args()
    argspar.noise_sigma /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet (Y-channel 3-frame) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
