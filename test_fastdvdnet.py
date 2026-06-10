#!/usr/bin/env python3
"""
Test FastDVDnet (single-frame + KV bank + shot + read noise).

Noise args follow same [0,255] convention as training:
  --noise_sigma  AWGN sigma  (divided by 255 internally)
  --lam          Poisson lambda scale (divided by 255 internally)

Output structure:
    save_path/
        blackswan/
            groundtruth/   gt_s{sigma}_l{lam}_0000.png  ...
            noisy/         noisy_s{sigma}_l{lam}_0000.png  ...
            denoised/      denoised_s{sigma}_l{lam}_0000.png  ...
        psnr_log.txt
"""

import os
import argparse
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

from models import FastDVDnet#, KVBank
from fastdvdnet import denoise_seq_fastdvdnet
from utils import (batch_psnr, init_logger_test,
                   variable_to_cv2_image, remove_dataparallel_wrapper,
                   open_sequence, close_logger)

OUTIMGEXT = '.png'
IMG_EXTS  = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def find_sequence_dirs(root: str) -> list:
    subdirs = sorted([os.path.join(root, d) for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    seq_dirs = [d for d in subdirs
                if any(os.path.splitext(f)[1].lower() in IMG_EXTS
                       for f in os.listdir(d))]
    if not seq_dirs:
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in os.listdir(root)):
            seq_dirs = [root]
    return seq_dirs


def save_frames(frames: torch.Tensor, save_dir: str, prefix: str, tag: str):
    """Save (T, C, H, W) as {prefix}_{tag}_{idx:04d}.png"""
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(frames.shape[0]):
        fname = os.path.join(save_dir, f'{prefix}_{tag}_{idx:04d}{OUTIMGEXT}')
        img   = variable_to_cv2_image(frames[idx].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(fname, img)


def write_psnr_table(path: str, rows: list, avg_noisy: float, avg_den: float):
    col_w  = max(20, max(len(r[0]) for r in rows) + 2)
    header = '{:<{w}}  {:>7}  {:>12}  {:>12}  {:>10}'.format(
        'Sequence', 'Frames', 'PSNR noisy', 'PSNR denos.', 'Gain', w=col_w)
    sep    = '-' * len(header)
    lines  = [sep, header, sep]
    for name, nf, pn, pd, gain in rows:
        lines.append('{:<{w}}  {:>7}  {:>11.4f}  {:>11.4f}  {:>+10.4f}'.format(
            name, nf, pn, pd, gain, w=col_w))
    lines += [sep,
              '{:<{w}}  {:>7}  {:>11.4f}  {:>11.4f}  {:>+10.4f}'.format(
                  'AVERAGE', '-', avg_noisy, avg_den,
                  avg_den - avg_noisy, w=col_w),
              sep]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


def test_fastdvdnet(**args):
    set_seed(args['seed'])

    # Integer tag for filenames — human readable
    sigma_int = round(args['noise_sigma'] * 255)
    lam_int   = round(args['lam']         * 255)
    tag       = f's{sigma_int}_l{lam_int}'

    os.makedirs(args['save_path'], exist_ok=True)
    logger        = init_logger_test(args['save_path'])
    psnr_log_path = os.path.join(args['save_path'], 'psnr_log.txt')

    logger.info("FastDVDnet Test")
    logger.info(f"  model      : {args['model_file']}")
    logger.info(f"  noise_sigma: {sigma_int}/255 = {args['noise_sigma']:.5f}  (AWGN)")
    logger.info(f"  lam        : {lam_int}/255 = {args['lam']:.5f}  (Poisson)")
    logger.info(f"  tag        : {tag}")

    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')

    # Load model
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
    print(f'Model loaded.  sigma={sigma_int}/255  lam={lam_int}/255  tag={tag}\n')

    seq_dirs = find_sequence_dirs(args['test_path'])
    if not seq_dirs:
        raise RuntimeError(f"No sequences found: {args['test_path']}")
    print(f'Found {len(seq_dirs)} sequence(s): '
          f'{[os.path.basename(d) for d in seq_dirs]}\n')

    psnr_all       = []
    psnr_noisy_all = []
    table_rows     = []

    with torch.no_grad():
        for seq_idx, seq_dir in enumerate(seq_dirs, 1):
            seq_name  = os.path.basename(seq_dir.rstrip('/'))
            seq_start = time.time()
            print(f'[{seq_idx}/{len(seq_dirs)}] {seq_name}')

            seq, _, _ = open_sequence(seq_dir, args['gray'],
                                      expand_if_needed=False,
                                      max_num_fr=args['max_num_fr_per_seq'])
            seq    = torch.from_numpy(seq).to(device)
            load_t = time.time() - seq_start

            # Reproducible noise per sequence
            seq_seed = args['seed'] + abs(hash(seq_name)) % (2 ** 16)
            torch.manual_seed(seq_seed)

            # Shot noise then read noise — same order as training
            photons = (seq * 255.0 / args['lam']).clamp(1e-6)
            seqn    = (torch.poisson(photons) * args['lam'] / 255.0).clamp(0., 1.)
            noise   = torch.empty_like(seqn).normal_(mean=0, std=args['noise_sigma'])
            seqn    = (seqn + noise).clamp(0., 1.)

            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                lambda_shot=args['lam'],
                temp_psz=None,
                model_temporal=model_temp,
                bank_size=args['bank_size'],
            )
            run_t = time.time() - seq_start - load_t

            psnr       = batch_psnr(denframes.cpu(), seq.cpu(), 1.)
            psnr_noisy = batch_psnr(seqn.cpu(),      seq.cpu(), 1.)
            gain       = psnr - psnr_noisy
            psnr_all.append(psnr)
            psnr_noisy_all.append(psnr_noisy)
            table_rows.append((seq_name, seq.size(0), psnr_noisy, psnr, gain))

            logger.info(f"  {seq_name}: noisy={psnr_noisy:.4f} "
                        f"denoised={psnr:.4f} gain={gain:+.4f}")
            print(f'  noisy: {psnr_noisy:.2f} dB  denoised: {psnr:.2f} dB  '
                  f'gain: {gain:+.2f} dB  ({run_t:.1f}s)')

            if not args['dont_save_results']:
                base = os.path.join(args['save_path'], seq_name)
                save_frames(seq.cpu(),       os.path.join(base, 'groundtruth'), 'gt',       tag)
                save_frames(seqn.cpu(),      os.path.join(base, 'noisy'),       'noisy',    tag)
                save_frames(denframes.cpu(), os.path.join(base, 'denoised'),    'denoised', tag)
                print(f'  Saved → {base}')

    if psnr_all:
        avg_psnr       = sum(psnr_all)       / len(psnr_all)
        avg_psnr_noisy = sum(psnr_noisy_all) / len(psnr_noisy_all)
        logger.info(f"AVERAGE: noisy={avg_psnr_noisy:.4f} "
                    f"denoised={avg_psnr:.4f} gain={avg_psnr-avg_psnr_noisy:+.4f}")
        write_psnr_table(psnr_log_path, table_rows, avg_psnr_noisy, avg_psnr)
        print(f'\nPSNR table: {psnr_log_path}')

    close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test FastDVDnet — shot + read noise, dual noise maps"
    )
    parser.add_argument("--model_file",         type=str,   default="./model.pth")
    parser.add_argument("--test_path",          type=str,   default="./data/val")
    parser.add_argument("--save_path",          type=str,   default="./results")
    parser.add_argument("--max_num_fr_per_seq", type=int,   default=25)
    parser.add_argument("--seed",               type=int,   default=42)

    # Noise — integer [0,255] convention, same as training
    parser.add_argument("--noise_sigma",  type=float, default=25,
                        help="AWGN sigma in [0,255] (will be divided by 255)")
    parser.add_argument("--lam",          type=float, default=25,
                        help="Poisson lambda in [0,255] (will be divided by 255)")

    parser.add_argument("--bank_size",    type=int,   default=10)
    parser.add_argument("--num_heads",    type=int,   default=4)
    parser.add_argument("--pool_size",    type=int,   default=8)
    parser.add_argument("--dont_save_results", action='store_true')
    parser.add_argument("--no_gpu",       action='store_true')
    parser.add_argument("--gray",         action='store_true')

    argspar = parser.parse_args()

    # Normalize to [0,1] — same standard as original FastDVDnet
    argspar.noise_sigma /= 255.
    argspar.lam         /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet (shot + read noise) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
