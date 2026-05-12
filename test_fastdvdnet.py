#!/usr/bin/env python3
"""
Denoise all sequences in a given folder using FastDVDnet (single-frame + KV bank).

Each sequence is denoised frame-by-frame in temporal order.
A fresh KVBank is created per sequence so context doesn't bleed across clips.
"""

import os
import argparse
import time

import cv2
import torch
import torch.nn as nn

from models import FastDVDnet, KVBank
from fastdvdnet import denoise_seq_fastdvdnet
from utils import (batch_psnr, init_logger_test,
                   variable_to_cv2_image, remove_dataparallel_wrapper,
                   open_sequence, close_logger)

OUTIMGEXT = '.png'


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    """Saves denoised (and optionally noisy) frames under save_dir."""
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        noisy_name = os.path.join(save_dir,
                                  'n{}_{}'.format(sigmaval, idx) + OUTIMGEXT)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir,
                                    'n{}_FastDVDnet_{}'.format(sigmaval, idx) + OUTIMGEXT)
        else:
            out_name = os.path.join(save_dir,
                                    'n{}_FastDVDnet_{}_{}'.format(sigmaval, suffix, idx) + OUTIMGEXT)

        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test_fastdvdnet(**args):
    """
    Denoises all sequences present in a given folder.
    Sequences must be stored as numbered image sequences in subfolders
    under args['test_path'].
    """
    start_time = time.time()

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')

    # ── Load model ────────────────────────────────────────────────────────
    print('Loading model ...')
    model_temp = FastDVDnet(
        bank_size=args['bank_size'],
        num_heads=args['num_heads'],
        pool_size=args['pool_size'],
    )

    state_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        model_temp = nn.DataParallel(model_temp, device_ids=[0]).cuda()
    else:
        state_dict = remove_dataparallel_wrapper(state_dict)
    model_temp.load_state_dict(state_dict)
    model_temp.eval()

    # ── Find all sequence subfolders ──────────────────────────────────────
    seq_dirs = sorted([
        os.path.join(args['test_path'], d)
        for d in os.listdir(args['test_path'])
        if os.path.isdir(os.path.join(args['test_path'], d))
    ])

    if not seq_dirs:
        # test_path itself is a single sequence
        seq_dirs = [args['test_path']]

    psnr_all = []

    with torch.no_grad():
        for seq_dir in seq_dirs:
            seq_start = time.time()

            # Load sequence
            seq, _, _ = open_sequence(
                seq_dir,
                args['gray'],
                expand_if_needed=False,
                max_num_fr=args['max_num_fr_per_seq'],
            )
            seq = torch.from_numpy(seq).to(device)   # (T, C, H, W) in [0, 1]
            seq_load_time = time.time() - seq_start

            # Add Gaussian noise
            noise  = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma'])
            seqn   = (seq + noise).clamp(0., 1.)
            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            # ── Denoise frame-by-frame with a fresh KV bank ───────────────
            # denoise_seq_fastdvdnet creates a new KVBank internally per call,
            # so each sequence gets independent temporal context.
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                temp_psz=None,                  # unused — bank handles temporal context
                model_temporal=model_temp,
                bank_size=args['bank_size'],
            )

            seq_run_time = time.time() - seq_start - seq_load_time

            # ── Metrics ───────────────────────────────────────────────────
            psnr       = batch_psnr(denframes, seq, 1.)
            psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
            psnr_all.append(psnr)

            seq_length = seq.size(0)
            logger.info("Finished denoising {}".format(seq_dir))
            logger.info("\tFrames: {}  |  Load: {:.3f}s  |  Denoise: {:.3f}s".format(
                seq_length, seq_load_time, seq_run_time))
            logger.info("\tPSNR noisy: {:.4f} dB  |  PSNR denoised: {:.4f} dB".format(
                psnr_noisy, psnr))

            # ── Save outputs ──────────────────────────────────────────────
            if not args['dont_save_results']:
                seq_save_dir = os.path.join(args['save_path'],
                                            os.path.basename(seq_dir.rstrip('/')))
                os.makedirs(seq_save_dir, exist_ok=True)
                save_out_seq(
                    seqn, denframes,
                    seq_save_dir,
                    int(args['noise_sigma'] * 255),
                    args['suffix'],
                    args['save_noisy'],
                )

    # ── Summary ───────────────────────────────────────────────────────────
    if psnr_all:
        avg_psnr = sum(psnr_all) / len(psnr_all)
        logger.info("\n=== Average PSNR over {} sequences: {:.4f} dB ===".format(
            len(psnr_all), avg_psnr))
        print("\n=== Average PSNR: {:.4f} dB ===".format(avg_psnr))

    elapsed = time.time() - start_time
    logger.info("Total elapsed time: {}".format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Denoise sequences with FastDVDnet (KV bank)")

    # I/O
    parser.add_argument("--model_file",         type=str, default="./model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_path",          type=str, default="./data/rgb/Kodak24",
                        help="Path to folder containing sequence subfolders")
    parser.add_argument("--save_path",          type=str, default="./results",
                        help="Where to save output images")
    parser.add_argument("--suffix",             type=str, default="",
                        help="Suffix to add to output filenames")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=25,
                        help="Max frames to load per sequence")

    # Noise
    parser.add_argument("--noise_sigma",        type=float, default=25,
                        help="Noise std used for testing (will be divided by 255)")

    # KV bank (must match training settings)
    parser.add_argument("--bank_size",          type=int, default=10,
                        help="KV bank capacity — must match training")
    parser.add_argument("--num_heads",          type=int, default=4,
                        help="Attention heads — must match training")
    parser.add_argument("--pool_size",          type=int, default=8,
                        help="Spatial pool size — must match training")

    # Misc
    parser.add_argument("--dont_save_results",  action='store_true',
                        help="Skip saving output images")
    parser.add_argument("--save_noisy",         action='store_true',
                        help="Also save noisy input frames")
    parser.add_argument("--no_gpu",             action='store_true',
                        help="Run on CPU")
    parser.add_argument("--gray",               action='store_true',
                        help="Denoise grayscale instead of RGB")

    argspar = parser.parse_args()

    argspar.noise_sigma /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet (single-frame + KV bank) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))