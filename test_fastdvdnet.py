#!/usr/bin/env python3
"""
Denoise sequences with blind FastDVDnet (single RGB frame + KV bank).

Noise is fully controlled via a JSON config file.
See test_config_example.json for all available options.

Usage:
    python test_fastdvdnet.py \\
        --model_file logs/net_best.pth \\
        --test_path  data/val \\
        --save_path  results \\
        --config     test_config.json

Output structure:
    save_path/
        blackswan/
            groundtruth/   0000.png  0001.png  ...
            noisy/         0000.png  0001.png  ...
            denoised/      0000.png  0001.png  ...
        psnr_log.txt
        noise_log.txt
        config_used.json   <- exact config used for this run (for reproducibility)
"""

import os
import json
import argparse
import time
import random
import math
import copy

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
# Default config — used when JSON fields are missing
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "seed": 42,

    # Fraction of sequences that are entirely clean (0.0 = all noisy, 1.0 = all clean)
    "p_clean": 0.3,

    # Which noise types to apply (subset of: shot, read, flicker)
    "noise_types": ["shot", "read", "flicker"],

    "shot": {
        "enabled": True,
        # Parameters sampled from [scale_min, scale_max] per frame (log-uniform)
        "scale_min": 0.1,
        "scale_max": 1.5,
        # Set fixed_scale to a number to disable randomness, e.g. 0.5
        "fixed_scale": None
    },

    "read": {
        "enabled": True,
        # sigma as fraction of [0,1] range, sampled log-uniform
        "sigma_min": 0.005,
        "sigma_max": 0.04,
        # Set fixed_sigma to disable randomness, e.g. 0.02
        "fixed_sigma": None
    },

    "flicker": {
        "enabled": True,
        # Amplitude sampled from Beta(2,8) * amp_max
        "amp_min": 0.0,
        "amp_max": 0.08,
        # Set fixed_amp to disable randomness, e.g. 0.05
        "fixed_amp": None,
        # Frequency randomly chosen from freq_choices weighted by freq_weights
        "freq_choices": [50, 60, 100, 120],
        "freq_weights": [0.15, 0.15, 0.35, 0.35],
        # Set fixed_freq to force a specific frequency, e.g. 50
        "fixed_freq": None
    },

    "variability": {
        # True  = each frame independently decides to be noisy (more realistic)
        # False = all frames get the same noise level (simpler, consistent)
        "per_frame": True,
        # Probability each frame gets noise when per_frame=True
        # 1.0 = all frames noisy, 0.0 = all frames clean
        "p_noisy_per_frame": 0.7
    }
}


# ---------------------------------------------------------------------------
# Config loading and merging
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load JSON config and merge with defaults for any missing fields."""
    with open(path, 'r') as f:
        user_cfg = json.load(f)

    # Deep merge: user values override defaults, missing fields use defaults
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, val in user_cfg.items():
        if key.startswith('_'):
            continue   # skip comment fields
        if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    return cfg


def save_config(cfg: dict, path: str):
    """Save the exact config used for this run."""
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)


# ---------------------------------------------------------------------------
# Parameter sampling from config
# ---------------------------------------------------------------------------

def _log_uniform(low: float, high: float) -> float:
    return math.exp(random.uniform(math.log(low), math.log(high)))


def sample_shot_params(cfg: dict) -> float:
    s = cfg['shot']
    if s['fixed_scale'] is not None:
        return float(s['fixed_scale'])
    return _log_uniform(s['scale_min'], s['scale_max'])


def sample_read_params(cfg: dict) -> float:
    r = cfg['read']
    if r['fixed_sigma'] is not None:
        return float(r['fixed_sigma'])
    return _log_uniform(r['sigma_min'], r['sigma_max'])


def sample_flicker_params(cfg: dict) -> tuple:
    fl = cfg['flicker']
    amp  = float(fl['fixed_amp'])  if fl['fixed_amp']  is not None \
           else float(np.random.beta(2, 8)) * fl['amp_max']
    freq = float(fl['fixed_freq']) if fl['fixed_freq'] is not None \
           else float(random.choices(fl['freq_choices'],
                                     weights=fl['freq_weights'])[0])
    return amp, freq


# ---------------------------------------------------------------------------
# Noise application
# ---------------------------------------------------------------------------

def apply_noise_to_frame(x: torch.Tensor,
                         enabled: set,
                         cfg: dict,
                         t: int) -> tuple:
    """
    Apply enabled noise types to a single frame tensor (1, C, H, W) or (C, H, W).
    Returns (noisy_frame, params_used).
    """
    params = {}

    if 'shot' in enabled and cfg['shot']['enabled']:
        scale = sample_shot_params(cfg)
        photons = (x * 255.0 / scale).clamp(1e-6)
        x = (torch.poisson(photons) * scale / 255.0).clamp(0., 1.)
        params['shot_scale'] = round(scale, 5)

    if 'read' in enabled and cfg['read']['enabled']:
        sigma = sample_read_params(cfg)
        x = (x + torch.randn_like(x) * sigma).clamp(0., 1.)
        params['read_sigma'] = round(sigma, 5)

    if 'flicker' in enabled and cfg['flicker']['enabled']:
        amp, freq = sample_flicker_params(cfg)
        mod = 1.0 + amp * math.sin(2 * math.pi * freq * t / 30.0)
        x = (x * mod).clamp(0., 1.)
        params['flicker_amp']  = round(amp, 5)
        params['flicker_freq'] = freq

    return x, params


# ---------------------------------------------------------------------------
# Per-sequence noise with per-frame variability
# ---------------------------------------------------------------------------

def apply_sequence_noise(seq: torch.Tensor,
                         seq_name: str,
                         cfg: dict) -> tuple:
    """
    Apply noise to an entire sequence according to the config.

    Returns:
        seqn      : (T, C, H, W) noisy sequence
        frame_log : list of dicts, one per frame
        enabled   : set of noise types applied
    """
    seq_seed = cfg['seed'] + abs(hash(seq_name)) % (2 ** 16)
    random.seed(seq_seed)
    np.random.seed(seq_seed % (2 ** 31))
    torch.manual_seed(seq_seed)

    T         = seq.shape[0]
    var_cfg   = cfg['variability']
    enabled   = set(cfg['noise_types'])

    # Filter to only those with enabled=True in their sub-config
    enabled = {n for n in enabled
               if n in cfg and cfg[n].get('enabled', True)}

    # Sequence-level clean decision
    if not enabled or random.random() < cfg['p_clean']:
        frame_log = [{'frame': t, 'noisy': False, 'noise_type': 'clean',
                      'params': {}} for t in range(T)]
        return seq.clone(), frame_log, set()

    noisy_frames = []
    frame_log    = []

    for t in range(T):
        x = seq[t].clone()

        # Per-frame or all-frames noise decision
        if var_cfg['per_frame']:
            is_noisy = random.random() < var_cfg['p_noisy_per_frame']
        else:
            is_noisy = True   # all frames noisy when per_frame=False

        if not is_noisy:
            noisy_frames.append(x)
            frame_log.append({'frame': t, 'noisy': False,
                               'noise_type': 'clean', 'params': {}})
        else:
            x_noisy, params = apply_noise_to_frame(x, enabled, cfg, t)
            noisy_frames.append(x_noisy)
            frame_log.append({'frame': t, 'noisy': True,
                               'noise_type': ', '.join(sorted(enabled)),
                               'params': params})

    return torch.stack(noisy_frames, dim=0), frame_log, enabled


# ---------------------------------------------------------------------------
# Save / log helpers
# ---------------------------------------------------------------------------

def save_frames(frames: torch.Tensor, save_dir: str):
    """Save (T, C, H, W) as 0000.png, 0001.png, ..."""
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(frames.shape[0]):
        fname = os.path.join(save_dir, f'{idx:04d}{OUTIMGEXT}')
        img   = variable_to_cv2_image(frames[idx].unsqueeze(0).clamp(0., 1.))
        cv2.imwrite(fname, img)


def find_sequence_dirs(root: str) -> list:
    subdirs = sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    return [d for d in subdirs
            if any(os.path.splitext(f)[1].lower() in IMG_EXTS
                   for f in os.listdir(d))] or \
           ([root] if any(os.path.splitext(f)[1].lower() in IMG_EXTS
                          for f in os.listdir(root)) else [])


def write_psnr_table(path: str, rows: list,
                     avg_noisy: float, avg_den: float):
    col_w  = max(20, max(len(r[0]) for r in rows) + 2)
    header = '{:<{w}}  {:>7}  {:>8}  {:>8}  {:>12}  {:>12}  {:>10}'.format(
        'Sequence', 'Frames', 'Noisy%', 'Noisy#',
        'PSNR noisy', 'PSNR denos.', 'Gain', w=col_w)
    sep   = '-' * len(header)
    lines = [sep, header, sep]

    for row in rows:
        name, nf, noisy_pct, pn, pd, gain, n_noisy_fr, total_fr = row
        pn_s   = f'{pn:>11.4f}'   if pn   == pn   else '        n/a'
        gain_s = f'{gain:>+10.4f}' if gain == gain else '       n/a'
        lines.append('{:<{w}}  {:>7}  {:>7.1f}%  {:>7}  {}  {:>11.4f}  {}'.format(
            name, nf, noisy_pct, f'{n_noisy_fr}/{total_fr}',
            pn_s, pd, gain_s, w=col_w))

    lines.append(sep)
    if avg_noisy == avg_noisy:
        lines.append('{:<{w}}  {:>7}  {:>8}  {:>8}  {:>11.4f}  {:>11.4f}  {:>+10.4f}'.format(
            'AVERAGE', '-', '-', '-', avg_noisy, avg_den,
            avg_den - avg_noisy, w=col_w))
    else:
        lines.append('AVERAGE: no sequences with noise')
    lines.append(sep)

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_fastdvdnet(**args):
    # Load and finalise config
    if args['config'] and os.path.isfile(args['config']):
        cfg = load_config(args['config'])
        print(f'> Loaded config: {args["config"]}')
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        print('> No config file — using defaults')

    # CLI seed overrides config seed if explicitly provided
    if args.get('seed') is not None:
        cfg['seed'] = args['seed']

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    os.makedirs(args['save_path'], exist_ok=True)

    # Save exact config used
    save_config(cfg, os.path.join(args['save_path'], 'config_used.json'))

    logger         = init_logger_test(args['save_path'])
    psnr_log_path  = os.path.join(args['save_path'], 'psnr_log.txt')
    noise_log_path = os.path.join(args['save_path'], 'noise_log.txt')

    logger.info("Blind FastDVDnet Test")
    logger.info(f"  model    : {args['model_file']}")
    logger.info(f"  test_path: {args['test_path']}")
    logger.info(f"  config   : {args.get('config', 'defaults')}")
    logger.info(f"  seed     : {cfg['seed']}")
    logger.info(f"  p_clean  : {cfg['p_clean']}")
    logger.info(f"  noise    : {cfg['noise_types']}")
    logger.info(f"  per_frame: {cfg['variability']['per_frame']}  "
                f"p_noisy={cfg['variability']['p_noisy_per_frame']}")

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

    seq_dirs = find_sequence_dirs(args['test_path'])
    if not seq_dirs:
        raise RuntimeError(f"No sequences found: {args['test_path']}")
    print(f'Found {len(seq_dirs)} sequence(s): '
          f'{[os.path.basename(d) for d in seq_dirs]}\n')

    psnr_all       = []
    psnr_noisy_all = []
    table_rows     = []
    noise_log      = open(noise_log_path, 'w')
    noise_log.write('Per-frame noise log\n')
    noise_log.write('=' * 60 + '\n\n')

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

            # Apply noise from config
            seqn, frame_log, enabled = apply_sequence_noise(seq, seq_name, cfg)
            seqn = seqn.to(device)

            n_noisy   = sum(1 for f in frame_log if f['noisy'])
            n_clean   = len(frame_log) - n_noisy
            noisy_pct = 100.0 * n_noisy / len(frame_log)
            noise_str = ', '.join(sorted(enabled)) if enabled else 'clean'

            print(f'  noise: [{noise_str}]  '
                  f'noisy frames: {n_noisy}/{len(frame_log)} ({noisy_pct:.0f}%)')

            # Write noise log
            noise_log.write(f'[{seq_name}]\n')
            noise_log.write(f'  Noise type    : {noise_str}\n')
            noise_log.write(f'  Noisy frames  : {n_noisy}/{len(frame_log)}\n')
            for fl in frame_log:
                if fl['noisy']:
                    ps = '  '.join(f'{k}={v}' for k, v in fl['params'].items())
                    noise_log.write(
                        f"  frame {fl['frame']:04d}: NOISY  {ps}\n")
                else:
                    noise_log.write(f"  frame {fl['frame']:04d}: clean\n")
            noise_log.write('\n')

            # Denoise
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=torch.FloatTensor([0.0]).to(device),
                temp_psz=None,
                model_temporal=model_temp,
                bank_size=args['bank_size'],
            )
            run_t = time.time() - seq_start - load_t

            # Metrics — only on noisy frames
            noisy_idx = [f['frame'] for f in frame_log if f['noisy']]
            if noisy_idx:
                idx_t      = torch.tensor(noisy_idx)
                psnr       = batch_psnr(denframes[idx_t].cpu(), seq[idx_t].cpu(), 1.)
                psnr_noisy = batch_psnr(seqn[idx_t].cpu(),      seq[idx_t].cpu(), 1.)
                gain       = psnr - psnr_noisy
                psnr_all.append(psnr)
                psnr_noisy_all.append(psnr_noisy)
            else:
                psnr       = batch_psnr(denframes.cpu(), seq.cpu(), 1.)
                psnr_noisy = float('nan')
                gain       = float('nan')

            table_rows.append((seq_name, seq.size(0), noisy_pct,
                                psnr_noisy, psnr, gain,
                                n_noisy, seq.size(0)))

            def _f(v): return f'{v:.4f}' if v == v else 'n/a'
            logger.info(f"  {seq_name}: noisy={n_noisy}/{len(frame_log)}  "
                        f"PSNR noisy={_f(psnr_noisy)}  "
                        f"denoised={psnr:.4f}  gain={_f(gain)}")
            print(f'  noisy: {_f(psnr_noisy)} dB  '
                  f'denoised: {psnr:.4f} dB  '
                  f'gain: {_f(gain)} dB  ({run_t:.1f}s)')

            if not args['dont_save_results']:
                base = os.path.join(args['save_path'], seq_name)
                save_frames(seq.cpu(),       os.path.join(base, 'groundtruth'))
                save_frames(seqn.cpu(),      os.path.join(base, 'noisy'))
                save_frames(denframes.cpu(), os.path.join(base, 'denoised'))
                print(f'  Saved → {base}')

    noise_log.close()

    if table_rows:
        avg_psnr       = sum(psnr_all) / len(psnr_all) \
                         if psnr_all else float('nan')
        avg_psnr_noisy = sum(psnr_noisy_all) / len(psnr_noisy_all) \
                         if psnr_noisy_all else float('nan')
        write_psnr_table(psnr_log_path, table_rows, avg_psnr_noisy, avg_psnr)
        print(f'\nConfig    : {os.path.join(args["save_path"], "config_used.json")}')
        print(f'Noise log : {noise_log_path}')
        print(f'PSNR table: {psnr_log_path}')

    close_logger(logger)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test blind FastDVDnet with JSON noise config"
    )
    parser.add_argument("--model_file",         type=str, default="./logs/net_best.pth")
    parser.add_argument("--test_path",          type=str, default="./data/val")
    parser.add_argument("--save_path",          type=str, default="./results")
    parser.add_argument("--config",             type=str, default=None,
                        help="Path to JSON noise config file. "
                             "Uses defaults if not provided.")
    parser.add_argument("--seed",               type=int, default=None,
                        help="Override seed from config")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=25)
    parser.add_argument("--bank_size",          type=int, default=10)
    parser.add_argument("--num_heads",          type=int, default=4)
    parser.add_argument("--pool_size",          type=int, default=8)
    parser.add_argument("--dont_save_results",  action='store_true')
    parser.add_argument("--no_gpu",             action='store_true')
    parser.add_argument("--gray",               action='store_true')

    argspar = parser.parse_args()
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Blind FastDVDnet Test (JSON config) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
