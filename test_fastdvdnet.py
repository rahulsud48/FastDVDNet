#!/usr/bin/env python3
"""
Denoise sequences using FastDVDnet (YUV422 single-frame + KV bank).

Pipeline per frame:
  RGB -> YUV422 -> denoise Y (KV bank) -> pass-through UV -> RGB -> save
PSNR computed in RGB space for fair comparison with other methods.
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


def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        noisy_name = os.path.join(save_dir, 'n{}_{}'.format(sigmaval, idx) + OUTIMGEXT)
        out_name   = os.path.join(save_dir,
                                  ('n{}_FastDVDnet_{}'.format(sigmaval, idx)
                                   if not suffix else
                                   'n{}_FastDVDnet_{}_{}'.format(sigmaval, suffix, idx))
                                  + OUTIMGEXT)
        if save_noisy:
            cv2.imwrite(noisy_name, variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.)))
        cv2.imwrite(out_name, variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0)))


def test_fastdvdnet(**args):
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

    # ── Find sequences ────────────────────────────────────────────────────
    seq_dirs = sorted([
        os.path.join(args['test_path'], d)
        for d in os.listdir(args['test_path'])
        if os.path.isdir(os.path.join(args['test_path'], d))
    ]) or [args['test_path']]

    psnr_all = []

    with torch.no_grad():
        for seq_dir in seq_dirs:
            seq_start = time.time()

            # Load RGB sequence
            seq, _, _ = open_sequence(seq_dir, args['gray'],
                                      expand_if_needed=False,
                                      max_num_fr=args['max_num_fr_per_seq'])
            seq      = torch.from_numpy(seq).to(device)   # (T, 3, H, W) RGB [0,1]
            load_t   = time.time() - seq_start

            # Add noise to RGB (matches training: noise on full RGB)
            noise    = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma'])
            seqn     = (seq + noise).clamp(0., 1.)
            noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

            # denoise_seq_fastdvdnet handles RGB->YUV422->denoise->RGB internally
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                temp_psz=None,
                model_temporal=model_temp,
                bank_size=args['bank_size'],
            )

            run_t      = time.time() - seq_start - load_t
            # PSNR in RGB space
            psnr       = batch_psnr(denframes, seq, 1.)
            psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
            psnr_all.append(psnr)

            logger.info("Finished: {}".format(seq_dir))
            logger.info("\tFrames: {}  Load: {:.3f}s  Denoise: {:.3f}s".format(
                seq.size(0), load_t, run_t))
            logger.info("\tPSNR noisy: {:.4f} dB  PSNR denoised: {:.4f} dB".format(
                psnr_noisy, psnr))

            if not args['dont_save_results']:
                seq_save_dir = os.path.join(args['save_path'],
                                            os.path.basename(seq_dir.rstrip('/')))
                os.makedirs(seq_save_dir, exist_ok=True)
                save_out_seq(seqn, denframes, seq_save_dir,
                             int(args['noise_sigma'] * 255),
                             args['suffix'], args['save_noisy'])

    if psnr_all:
        avg_psnr = sum(psnr_all) / len(psnr_all)
        logger.info("\n=== Average PSNR over {} sequences: {:.4f} dB ===".format(
            len(psnr_all), avg_psnr))
        print("\n=== Average PSNR: {:.4f} dB ===".format(avg_psnr))

    elapsed = time.time() - start_time
    logger.info("Total: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    close_logger(logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test FastDVDnet (YUV422 + KV bank)")

    parser.add_argument("--model_file",         type=str,   default="./model.pth")
    parser.add_argument("--test_path",          type=str,   default="./data/rgb/Kodak24")
    parser.add_argument("--save_path",          type=str,   default="./results")
    parser.add_argument("--suffix",             type=str,   default="")
    parser.add_argument("--max_num_fr_per_seq", type=int,   default=25)
    parser.add_argument("--noise_sigma",        type=float, default=25)
    parser.add_argument("--bank_size",          type=int,   default=10)
    parser.add_argument("--num_heads",          type=int,   default=4)
    parser.add_argument("--pool_size",          type=int,   default=8)
    parser.add_argument("--dont_save_results",  action='store_true')
    parser.add_argument("--save_noisy",         action='store_true')
    parser.add_argument("--no_gpu",             action='store_true')
    parser.add_argument("--gray",               action='store_true')

    argspar = parser.parse_args()
    argspar.noise_sigma /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet (YUV422 + KV bank) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
