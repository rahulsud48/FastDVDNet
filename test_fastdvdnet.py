#!/usr/bin/env python3
"""
Denoise all sequences under --test_path using FastDVDnet (single RGB frame + KV bank).

Expected folder structure:
    test_path/
        blackswan/      <- each subfolder is one sequence
            00000.jpg
            00001.jpg
            ...
        camel/
            ...
        dance-twirl/
            ...

For each sequence, three subfolders are saved under --save_path:
    save_path/
        blackswan/
            groundtruth/    gt_sigma25_0000.png  ...
            noisy/          noisy_sigma25_0000.png  ...
            denoised/       denoised_sigma25_0000.png  ...
        camel/
            ...

A fixed random seed ensures identical noise across runs for fair comparison.

Per-sequence and summary PSNR are written to:
    save_path/psnr_log.txt   <- human-readable table
    save_path/log.txt        <- full run log
"""

import os
import argparse
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

from models import FastDVDnet, KVBank
from fastdvdnet import denoise_seq_fastdvdnet
from utils import (batch_psnr, init_logger_test,
                   variable_to_cv2_image, remove_dataparallel_wrapper,
                   open_sequence, close_logger)

OUTIMGEXT = '.png'
IMG_EXTS  = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Fix all random seeds for fully reproducible noise generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def find_sequence_dirs(root: str):
    """
    Return sorted list of subdirectories of root that contain image files.
    Handles the case where root itself contains images (single-sequence mode).
    """
    subdirs = sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    # Filter: keep only dirs that actually contain image files
    seq_dirs = []
    for d in subdirs:
        files = os.listdir(d)
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in files):
            seq_dirs.append(d)

    # Fallback: root itself is a single sequence
    if not seq_dirs:
        root_files = os.listdir(root)
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in root_files):
            seq_dirs = [root]

    return seq_dirs


def save_frames(frames: torch.Tensor, save_dir: str, prefix: str, sigmaval: int):
    """
    Save (T, C, H, W) tensor as numbered PNG files.

    Filenames: {prefix}_sigma{sigmaval}_{idx:04d}.png
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(frames.shape[0]):
        fname = os.path.join(
            save_dir,
            '{}_sigma{}_{:04d}{}'.format(prefix, sigmaval, idx, OUTIMGEXT)
        )
        img = variable_to_cv2_image(frames[idx].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(fname, img)


def write_psnr_table(psnr_log_path: str, rows: list, avg_noisy: float, avg_denoised: float):
    """
    Write a formatted PSNR table to psnr_log.txt.

    rows: list of (seq_name, num_frames, psnr_noisy, psnr_denoised, gain)
    """
    col_w = max(20, max(len(r[0]) for r in rows) + 2)
    header = '{:<{w}}  {:>7}  {:>12}  {:>12}  {:>10}'.format(
        'Sequence', 'Frames', 'PSNR noisy', 'PSNR denois', 'Gain',
        w=col_w
    )
    sep = '-' * len(header)

    lines = [sep, header, sep]
    for seq_name, nf, pn, pd, gain in rows:
        lines.append('{:<{w}}  {:>7}  {:>11.4f}  {:>11.4f}  {:>+10.4f}'.format(
            seq_name, nf, pn, pd, gain, w=col_w
        ))
    lines.append(sep)
    lines.append('{:<{w}}  {:>7}  {:>11.4f}  {:>11.4f}  {:>+10.4f}'.format(
        'AVERAGE', '-', avg_noisy, avg_denoised, avg_denoised - avg_noisy, w=col_w
    ))
    lines.append(sep)

    with open(psnr_log_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Also print to console
    print('\n' + '\n'.join(lines))


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test_fastdvdnet(**args):
    set_seed(args['seed'])

    os.makedirs(args['save_path'], exist_ok=True)
    logger       = init_logger_test(args['save_path'])
    psnr_log_path = os.path.join(args['save_path'], 'psnr_log.txt')

    sigmaval = int(args['noise_sigma'] * 255)

    logger.info("=" * 60)
    logger.info("FastDVDnet Test Run")
    logger.info("  model    : {}".format(args['model_file']))
    logger.info("  test_path: {}".format(args['test_path']))
    logger.info("  save_path: {}".format(args['save_path']))
    logger.info("  sigma    : {} ({}/255)".format(args['noise_sigma'], sigmaval))
    logger.info("  seed     : {}".format(args['seed']))
    logger.info("  bank_size: {}".format(args['bank_size']))
    logger.info("=" * 60)

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
    print('Model loaded.\n')

    # ── Discover sequences ────────────────────────────────────────────────
    seq_dirs = find_sequence_dirs(args['test_path'])
    if not seq_dirs:
        raise RuntimeError("No image sequences found under: {}".format(args['test_path']))
    print('Found {} sequence(s): {}\n'.format(
        len(seq_dirs), [os.path.basename(d) for d in seq_dirs]))

    psnr_all       = []
    psnr_noisy_all = []
    table_rows     = []

    with torch.no_grad():
        for seq_idx, seq_dir in enumerate(seq_dirs, 1):
            seq_name  = os.path.basename(seq_dir.rstrip('/'))
            seq_start = time.time()

            print('[{}/{}] Processing: {}'.format(seq_idx, len(seq_dirs), seq_name))

            # ── Load clean frames ──────────────────────────────────────────
            seq, _, _ = open_sequence(
                seq_dir,
                args['gray'],
                expand_if_needed=False,
                max_num_fr=args['max_num_fr_per_seq'],
            )
            seq    = torch.from_numpy(seq).to(device)   # (T, C, H, W) in [0, 1]
            load_t = time.time() - seq_start

            # ── Add reproducible noise ─────────────────────────────────────
            # Per-sequence seed: deterministic regardless of processing order
            seq_seed = args['seed'] + abs(hash(seq_name)) % (2 ** 16)
            torch.manual_seed(seq_seed)
            noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma'])
            seqn  = (seq + noise).clamp(0., 1.)

            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            # ── Denoise frame-by-frame ─────────────────────────────────────
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                temp_psz=None,
                model_temporal=model_temp,
                bank_size=args['bank_size'],
            )
            run_t = time.time() - seq_start - load_t

            # ── Per-sequence PSNR ──────────────────────────────────────────
            psnr       = batch_psnr(denframes.cpu(), seq.cpu(), 1.)
            psnr_noisy = batch_psnr(seqn.cpu(),      seq.cpu(), 1.)
            gain       = psnr - psnr_noisy

            psnr_all.append(psnr)
            psnr_noisy_all.append(psnr_noisy)
            table_rows.append((seq_name, seq.size(0), psnr_noisy, psnr, gain))

            # Log to file
            logger.info("")
            logger.info("Sequence : {}".format(seq_name))
            logger.info("  Frames : {}".format(seq.size(0)))
            logger.info("  Load   : {:.3f}s   Denoise: {:.3f}s".format(load_t, run_t))
            logger.info("  PSNR noisy    : {:.4f} dB".format(psnr_noisy))
            logger.info("  PSNR denoised : {:.4f} dB".format(psnr))
            logger.info("  PSNR gain     : {:+.4f} dB".format(gain))

            print('         noisy: {:.2f} dB  |  denoised: {:.2f} dB  |  gain: {:+.2f} dB'.format(
                psnr_noisy, psnr, gain))
            print('         load: {:.2f}s  denoise: {:.2f}s'.format(load_t, run_t))

            # ── Save images ────────────────────────────────────────────────
            if not args['dont_save_results']:
                base = os.path.join(args['save_path'], seq_name)

                save_frames(seq.cpu(),       os.path.join(base, 'groundtruth'), 'gt',       sigmaval)
                save_frames(seqn.cpu(),      os.path.join(base, 'noisy'),       'noisy',    sigmaval)
                save_frames(denframes.cpu(), os.path.join(base, 'denoised'),    'denoised', sigmaval)

                logger.info("  Saved to: {}".format(base))
                print('         Saved → {}'.format(base))

    # ── Summary & PSNR table ──────────────────────────────────────────────
    if psnr_all:
        avg_psnr       = sum(psnr_all)       / len(psnr_all)
        avg_psnr_noisy = sum(psnr_noisy_all) / len(psnr_noisy_all)

        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY — {} sequences".format(len(psnr_all)))
        logger.info("  Avg PSNR noisy    : {:.4f} dB".format(avg_psnr_noisy))
        logger.info("  Avg PSNR denoised : {:.4f} dB".format(avg_psnr))
        logger.info("  Avg PSNR gain     : {:+.4f} dB".format(avg_psnr - avg_psnr_noisy))
        logger.info("=" * 60)

        # Write the formatted PSNR table
        write_psnr_table(psnr_log_path, table_rows, avg_psnr_noisy, avg_psnr)
        logger.info("PSNR table written to: {}".format(psnr_log_path))

    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test FastDVDnet — iterates val sequences, saves GT/noisy/denoised"
    )

    parser.add_argument("--model_file",         type=str,   default="./model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_path",          type=str,   default="./data/val",
                        help="Root folder containing sequence subfolders (e.g. data/val/)")
    parser.add_argument("--save_path",          type=str,   default="./results",
                        help="Root folder for all outputs")
    parser.add_argument("--max_num_fr_per_seq", type=int,   default=25,
                        help="Max frames to load per sequence")
    parser.add_argument("--noise_sigma",        type=float, default=25,
                        help="Noise std — divided by 255 internally")
    parser.add_argument("--seed",               type=int,   default=42,
                        help="Global random seed for reproducible noise")
    parser.add_argument("--bank_size",          type=int,   default=10,
                        help="KV bank capacity — must match training")
    parser.add_argument("--num_heads",          type=int,   default=4,
                        help="Attention heads — must match training")
    parser.add_argument("--pool_size",          type=int,   default=8,
                        help="Pool size — must match training")
    parser.add_argument("--dont_save_results",  action='store_true',
                        help="Skip saving images (metrics only)")
    parser.add_argument("--no_gpu",             action='store_true')
    parser.add_argument("--gray",               action='store_true')

    argspar = parser.parse_args()
    argspar.noise_sigma /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### FastDVDnet Test (RGB + KV bank) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
