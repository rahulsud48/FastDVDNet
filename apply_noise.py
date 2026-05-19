#!/usr/bin/env python3
"""
apply_noise.py

Apply any combination of synthetic noise types to a video dataset
(DAVIS or any folder of image sequences) and save the results.

Folder structure expected:
    input_dir/
        blackswan/
            00000.jpg  00001.jpg  ...
        camel/
            ...

Output structure:
    output_dir/
        blackswan/
            00000.png  00001.png  ...   <- noisy frames
        camel/
            ...
        noise_config.txt                <- log of exact parameters used

Noise types available (pass any combination via --noise):
    shot          -- Poisson shot noise (signal-dependent)
    read          -- Gaussian read noise (signal-independent)
    fpn           -- Fixed pattern noise (column/row stripes)
    prnu          -- Pixel response non-uniformity (multiplicative)
    correlated    -- Temporally correlated noise across frames (AR(1))
    flicker       -- Periodic intensity modulation (50/60 Hz lighting)
    drift         -- Thermal gain/offset drift across sequence
    shake         -- Global camera motion blur
    object_motion -- Local object motion blur (per-region)
    rolling       -- Rolling shutter row-shift effect

Examples:
    # All noise types
    python apply_noise.py --input data/val --output data/val_noisy

    # Only flicker + motion blur
    python apply_noise.py --input data/val --output data/val_noisy \\
        --noise flicker shake

    # Shot + read + correlated temporal only
    python apply_noise.py --input data/val --output data/val_noisy \\
        --noise shot read correlated

    # Strong indoor low-light preset
    python apply_noise.py --input data/val --output data/val_noisy \\
        --preset indoor_lowlight

    # Custom parameters
    python apply_noise.py --input data/val --output data/val_noisy \\
        --noise shot read \\
        --shot_scale 1.5 \\
        --read_sigma 0.04 \\
        --seed 42
"""

import os
import argparse
import random
import math
import time

import cv2
import numpy as np
import subprocess
import shutil
import torch
import torch.nn.functional as F

# ── Import noise primitives from noise_model.py ──────────────────────────────
from noise_model import (
    NoiseConfig,
    add_shot_noise,
    make_fpn_map,
    effective_sigma_map,
    apply_camera_shake,
    apply_object_motion,
    apply_rolling_shutter,
    make_flicker_modulation,
    _smooth_field,
)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}

ALL_NOISE_TYPES = [
    'shot', 'read', 'fpn', 'prnu',
    'correlated', 'flicker', 'drift',
    'shake', 'object_motion', 'rolling',
]

PRESETS = {
    'all': ALL_NOISE_TYPES,
    'spatial_only': ['shot', 'read', 'fpn', 'prnu'],
    'temporal_only': ['correlated', 'flicker', 'drift'],
    'motion_only': ['shake', 'object_motion', 'rolling'],
    'indoor_lowlight': ['shot', 'read', 'fpn', 'prnu', 'correlated', 'flicker'],
    'outdoor_bright': ['shot', 'read', 'shake', 'object_motion'],
    'sensor_only': ['shot', 'read', 'fpn', 'prnu', 'correlated'],
    'none': [],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_sequence_dirs(root: str):
    """Return sorted subdirs of root that contain image files."""
    subdirs = sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    seq_dirs = [
        d for d in subdirs
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS
               for f in os.listdir(d))
    ]
    if not seq_dirs:
        # root itself may be a single sequence
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS
               for f in os.listdir(root)):
            seq_dirs = [root]
    return seq_dirs


def load_sequence(seq_dir: str, max_frames: int) -> list:
    """Load frames from a sequence directory as float32 tensors in [0,1]."""
    files = sorted([
        os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])[:max_frames]

    frames = []
    for fpath in files:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] Could not read {fpath}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img.astype(np.float32) / 255.0)\
                   .permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frames.append(t)
    return frames, [os.path.basename(f) for f in files]


def save_frame(tensor: torch.Tensor, path: str):
    """Save (1, 3, H, W) float32 [0,1] tensor as PNG."""
    img = tensor.squeeze(0).permute(1, 2, 0).clamp(0., 1.)
    img = (img.numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Per-sequence noise application
# ---------------------------------------------------------------------------

def apply_noise_to_sequence(frames: list,
                             enabled: set,
                             cfg: NoiseConfig,
                             args,
                             device: torch.device,
                             seq_name: str) -> list:
    """
    Apply the selected noise types to a list of frames.

    Args:
        frames  : list of T tensors (1, 3, H, W) in [0,1]
        enabled : set of noise type strings to apply
        cfg     : NoiseConfig with parameter ranges
        args    : parsed CLI args (for override values)
        device  : torch device
        seq_name: used for per-sequence seed

    Returns:
        noisy_frames: list of T tensors (1, 3, H, W) in [0,1]
    """
    T         = len(frames)
    N, C, H, W = 1, 3, frames[0].shape[2], frames[0].shape[3]

    # Per-sequence seed for reproducibility
    seq_seed = args.seed + abs(hash(seq_name)) % (2 ** 16)
    random.seed(seq_seed)
    torch.manual_seed(seq_seed)

    # ── Sample sequence-level parameters ──────────────────────────────────
    shot_scale  = args.shot_scale  if args.shot_scale  is not None \
                  else random.uniform(*cfg.shot_scale_range)
    read_sigma  = args.read_sigma  if args.read_sigma  is not None \
                  else random.uniform(*cfg.read_sigma_range)
    fpn_str     = args.fpn_strength if args.fpn_strength is not None \
                  else random.uniform(*cfg.fpn_strength_range)
    prnu_str    = args.prnu_strength if args.prnu_strength is not None \
                  else random.uniform(*cfg.prnu_strength_range)
    alpha       = args.temporal_alpha if args.temporal_alpha is not None \
                  else random.uniform(*cfg.temporal_alpha_range)
    shake_len   = args.shake_kernel if args.shake_kernel is not None \
                  else random.randint(*cfg.shake_kernel_range)
    shake_angle = args.shake_angle  if args.shake_angle  is not None \
                  else random.uniform(*cfg.shake_angle_range)
    obj_kernel  = args.obj_kernel   if args.obj_kernel   is not None \
                  else random.randint(*cfg.obj_motion_kernel_range)
    obj_regions = args.obj_regions  if args.obj_regions  is not None \
                  else random.randint(*cfg.obj_motion_num_regions)
    rolling_shift = args.rolling_shift if args.rolling_shift is not None \
                    else random.randint(1, cfg.rolling_shutter_max_shift)
    flicker_freq  = args.flicker_freq  if args.flicker_freq  is not None \
                    else random.uniform(*cfg.flicker_freq_range)
    flicker_amp   = args.flicker_amp   if args.flicker_amp   is not None \
                    else random.uniform(*cfg.flicker_amp_range)
    drift_gain    = args.drift_gain    if args.drift_gain    is not None \
                    else random.uniform(*cfg.drift_gain_range)
    drift_offset  = args.drift_offset  if args.drift_offset  is not None \
                    else random.uniform(*cfg.drift_offset_range)

    # Fixed maps — generated once, same for all frames in sequence
    fpn_add  = None
    prnu_mul = None

    # Flicker modulation list
    flicker_mods = make_flicker_modulation(T, flicker_freq, fps=30.0,
                                            amplitude=flicker_amp) \
                   if 'flicker' in enabled else [1.0] * T

    # AR(1) state
    prev_noise = torch.zeros(N, C, H, W, device=device)

    noisy_frames = []

    for t, frame in enumerate(frames):
        x = frame.to(device)
        _, _, fH, fW = x.shape

        # ── Motion blur (before noise) ─────────────────────────────────
        if 'shake' in enabled:
            x = apply_camera_shake(x, shake_len, shake_angle)

        if 'object_motion' in enabled:
            x = apply_object_motion(x, obj_regions, obj_kernel)

        if 'rolling' in enabled:
            x = apply_rolling_shutter(x, rolling_shift)

        # Defensive size reset
        x = x[:, :, :fH, :fW]

        # ── Thermal drift ──────────────────────────────────────────────
        if 'drift' in enabled:
            g_t = drift_gain   ** t
            o_t = drift_offset  * t
            x = (x * g_t + o_t).clamp(0., 1.)

        # ── Lazy FPN/PRNU map ──────────────────────────────────────────
        if ('fpn' in enabled or 'prnu' in enabled):
            cH, cW = x.shape[2], x.shape[3]
            if fpn_add is None or fpn_add.shape[2] != cH or fpn_add.shape[3] != cW:
                fpn_add, prnu_mul = make_fpn_map(N, C, cH, cW,
                                                  fpn_str, fpn_str,
                                                  prnu_str, device)

        # ── PRNU ──────────────────────────────────────────────────────
        if 'prnu' in enabled:
            x = (x * prnu_mul).clamp(0., 1.)

        # ── Shot noise ─────────────────────────────────────────────────
        if 'shot' in enabled:
            x = add_shot_noise(x, shot_scale)

        # ── Temporal correlated read noise (AR(1)) ─────────────────────
        if 'correlated' in enabled:
            fresh      = torch.randn(N, C, x.shape[2], x.shape[3],
                                     device=device) * read_sigma
            corr_noise = alpha * prev_noise[:, :, :x.shape[2], :x.shape[3]] \
                         + math.sqrt(max(1 - alpha ** 2, 0)) * fresh
            prev_noise = torch.zeros_like(corr_noise)
            prev_noise[:, :, :corr_noise.shape[2], :corr_noise.shape[3]] = \
                corr_noise.detach()
            x = (x + corr_noise).clamp(0., 1.)
        elif 'read' in enabled:
            # Independent read noise (no temporal correlation)
            x = (x + torch.randn_like(x) * read_sigma).clamp(0., 1.)

        # ── Fixed pattern noise ────────────────────────────────────────
        if 'fpn' in enabled and fpn_add is not None:
            cH, cW = x.shape[2], x.shape[3]
            x = (x + fpn_add[:, :, :cH, :cW]).clamp(0., 1.)

        # ── Flicker ────────────────────────────────────────────────────
        if 'flicker' in enabled:
            x = (x * flicker_mods[t]).clamp(0., 1.)

        noisy_frames.append(x.cpu())

    return noisy_frames, {
        'shot_scale':   shot_scale,
        'read_sigma':   read_sigma,
        'fpn_strength': fpn_str,
        'prnu_strength':prnu_str,
        'alpha':        alpha,
        'shake_len':    shake_len,
        'shake_angle':  shake_angle,
        'obj_kernel':   obj_kernel,
        'obj_regions':  obj_regions,
        'rolling_shift':rolling_shift,
        'flicker_freq': flicker_freq,
        'flicker_amp':  flicker_amp,
        'drift_gain':   drift_gain,
        'drift_offset': drift_offset,
    }



# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def has_ffmpeg() -> bool:
    return shutil.which('ffmpeg') is not None


def frames_to_video(frame_dirs: list, labels: list,
                    out_path: str, fps: int = 30,
                    side_by_side: bool = True):
    """
    Create a video from one or more frame directories.

    If side_by_side=True and multiple dirs given, frames are stacked
    horizontally with a label overlay per column.
    Falls back to OpenCV if ffmpeg is not available.

    frame_dirs : list of directories, each containing PNG frames in order
    labels     : list of label strings (same length as frame_dirs)
    out_path   : output .mp4 path
    fps        : frames per second
    """
    # Collect sorted frame lists from each directory
    frame_lists = []
    for d in frame_dirs:
        fl = sorted([
            os.path.join(d, f) for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        frame_lists.append(fl)

    if not frame_lists or not frame_lists[0]:
        print(f'  [WARN] No frames found for video export')
        return

    T = min(len(fl) for fl in frame_lists)
    if T == 0:
        return

    # Read first frame to get dimensions
    sample = cv2.imread(frame_lists[0][0])
    H, W   = sample.shape[:2]

    # ── Try ffmpeg first ──────────────────────────────────────────────────
    if has_ffmpeg() and side_by_side and len(frame_dirs) > 1:
        # Build hstack filter
        inputs  = []
        filters = []
        for i, d in enumerate(frame_dirs):
            inputs  += ['-framerate', str(fps), '-i',
                        os.path.join(d, '%05d.png')]
            filters.append(f'[{i}:v]drawtext=text=\'{labels[i]}\':fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v{i}]')

        filter_str = ';'.join(filters) + ';'
        filter_str += ''.join(f'[v{i}]' for i in range(len(frame_dirs)))
        filter_str += f'hstack=inputs={len(frame_dirs)}[out]'

        cmd = ['ffmpeg', '-y'] + inputs + [
            '-filter_complex', filter_str,
            '-map', '[out]',
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            out_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f'  Video (ffmpeg): {out_path}')
            return
        else:
            print(f'  [WARN] ffmpeg failed, falling back to OpenCV')
            print('  ' + result.stderr.decode('utf-8', errors='replace')[:200])

    elif has_ffmpeg() and len(frame_dirs) == 1:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frame_dirs[0], '%05d.png'),
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            out_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f'  Video (ffmpeg): {out_path}')
            return

    # ── OpenCV fallback ───────────────────────────────────────────────────
    total_W = W * len(frame_dirs) if side_by_side else W
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(out_path, fourcc, fps, (total_W, H))

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2

    for idx in range(T):
        cols = []
        for i, fl in enumerate(frame_lists):
            if idx < len(fl):
                frame = cv2.imread(fl[idx])
                if frame is None:
                    frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                frame = np.zeros((H, W, 3), dtype=np.uint8)

            # Draw label
            if labels and i < len(labels):
                cv2.rectangle(frame, (5, 5), (5 + len(labels[i]) * 14, 35),
                              (0, 0, 0), -1)
                cv2.putText(frame, labels[i], (10, 28),
                            font, font_scale, (255, 255, 255), thickness)
            cols.append(frame)

        row = np.concatenate(cols, axis=1) if side_by_side else cols[0]
        writer.write(row)

    writer.release()
    print(f'  Video (OpenCV): {out_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Apply synthetic noise to a video dataset (DAVIS or similar)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise types:
  shot          Poisson shot noise (signal-dependent, dominant at normal light)
  read          Gaussian read noise (signal-independent, dominant at low light)
  fpn           Fixed pattern noise (column/row stripes, same every frame)
  prnu          Pixel response non-uniformity (multiplicative per-pixel)
  correlated    Temporally correlated read noise via AR(1) process
  flicker       Periodic intensity modulation from 50/60Hz mains lighting
  drift         Thermal gain/offset drift across the sequence
  shake         Global camera motion blur (linear kernel, random angle)
  object_motion Local object motion blur (per random region)
  rolling       Rolling shutter row-shift effect (CMOS sensor model)

Presets:
  all             All noise types
  spatial_only    shot + read + fpn + prnu
  temporal_only   correlated + flicker + drift
  motion_only     shake + object_motion + rolling
  indoor_lowlight shot + read + fpn + prnu + correlated + flicker
  outdoor_bright  shot + read + shake + object_motion
  sensor_only     shot + read + fpn + prnu + correlated
  none            No noise (copy only)

Examples:
  python apply_noise.py --input data/val --output data/val_noisy
  python apply_noise.py --input data/val --output data/val_noisy --noise flicker shake
  python apply_noise.py --input data/val --output data/val_noisy --preset indoor_lowlight
  python apply_noise.py --input data/val --output data/val_noisy --noise shot read --shot_scale 1.5
        """
    )

    # I/O
    parser.add_argument('--input',          type=str, required=True,
                        help='Root folder containing sequence subfolders')
    parser.add_argument('--output',         type=str, required=True,
                        help='Root folder for noisy output sequences')
    parser.add_argument('--max_frames',     type=int, default=None,
                        help='Max frames per sequence (default: all)')
    parser.add_argument('--seed',           type=int, default=42,
                        help='Global random seed for reproducibility')
    parser.add_argument('--no_gpu',         action='store_true',
                        help='Force CPU even if CUDA available')
    parser.add_argument('--make_video',     action='store_true',
                        help='Export comparison video after processing each sequence')
    parser.add_argument('--fps',            type=int, default=30,
                        help='Frame rate for exported videos (default: 30)')
    parser.add_argument('--video_dir',      type=str, default=None,
                        help='Directory to save videos (default: same as --output)')

    # Noise selection
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument('--noise',  nargs='+', choices=ALL_NOISE_TYPES,
                              metavar='TYPE',
                              help='Noise types to apply (space-separated). '
                                   'Choices: ' + ' '.join(ALL_NOISE_TYPES))
    noise_group.add_argument('--preset', choices=list(PRESETS.keys()),
                              default='all',
                              help='Named noise preset (default: all)')

    # Parameter overrides (optional — if not set, sampled from NoiseConfig ranges)
    parser.add_argument('--shot_scale',     type=float, default=None,
                        help='Shot noise scale [0.2-2.0]. Low=bright, high=dark/highISO')
    parser.add_argument('--read_sigma',     type=float, default=None,
                        help='Read noise std as fraction of [0,1] [0.005-0.05]')
    parser.add_argument('--fpn_strength',   type=float, default=None,
                        help='FPN column/row stripe amplitude [0.0-0.02]')
    parser.add_argument('--prnu_strength',  type=float, default=None,
                        help='PRNU per-pixel gain std [0.0-0.01]')
    parser.add_argument('--temporal_alpha', type=float, default=None,
                        help='AR(1) temporal correlation [0.0-0.9]')
    parser.add_argument('--flicker_freq',   type=float, default=None,
                        help='Flicker frequency in Hz [40-70]')
    parser.add_argument('--flicker_amp',    type=float, default=None,
                        help='Flicker amplitude [0.0-0.03]')
    parser.add_argument('--drift_gain',     type=float, default=None,
                        help='Per-frame thermal gain drift [0.99-1.01]')
    parser.add_argument('--drift_offset',   type=float, default=None,
                        help='Per-frame thermal offset drift [-0.005-0.005]')
    parser.add_argument('--shake_kernel',   type=int,   default=None,
                        help='Camera shake kernel length in pixels [3-15]')
    parser.add_argument('--shake_angle',    type=float, default=None,
                        help='Camera shake angle in degrees [0-180]')
    parser.add_argument('--obj_kernel',     type=int,   default=None,
                        help='Object motion blur kernel length [3-9]')
    parser.add_argument('--obj_regions',    type=int,   default=None,
                        help='Number of object motion blur regions [1-4]')
    parser.add_argument('--rolling_shift',  type=int,   default=None,
                        help='Rolling shutter max row shift in pixels [1-3]')

    args = parser.parse_args()

    # ── Resolve enabled noise types ────────────────────────────────────────
    if args.noise is not None:
        enabled = set(args.noise)
    else:
        enabled = set(PRESETS[args.preset])

    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device('cpu') if args.no_gpu or not torch.cuda.is_available() \
             else torch.device('cuda')
    cfg    = NoiseConfig()

    os.makedirs(args.output, exist_ok=True)

    # ── Print config ───────────────────────────────────────────────────────
    print('\n=== apply_noise.py ===')
    print(f'  Input  : {args.input}')
    print(f'  Output : {args.output}')
    print(f'  Device : {device}')
    print(f'  Seed   : {args.seed}')
    if enabled:
        print(f'  Noise  : {", ".join(sorted(enabled))}')
    else:
        print('  Noise  : NONE (copy only)')

    # Print any overrides
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in
                 {'input','output','max_frames','seed','no_gpu','noise','preset'}}
    if overrides:
        print('  Overrides:')
        for k, v in overrides.items():
            print(f'    --{k} = {v}')
    print()

    # ── Discover sequences ─────────────────────────────────────────────────
    seq_dirs = find_sequence_dirs(args.input)
    if not seq_dirs:
        raise RuntimeError(f'No image sequences found under: {args.input}')
    print(f'Found {len(seq_dirs)} sequence(s): '
          f'{[os.path.basename(d) for d in seq_dirs]}\n')

    # ── Config log file ────────────────────────────────────────────────────
    config_log_path = os.path.join(args.output, 'noise_config.txt')
    with open(config_log_path, 'w') as log:
        log.write('apply_noise.py — run configuration\n')
        log.write('=' * 50 + '\n')
        log.write(f'Input  : {args.input}\n')
        log.write(f'Output : {args.output}\n')
        log.write(f'Seed   : {args.seed}\n')
        log.write(f'Noise  : {", ".join(sorted(enabled)) if enabled else "none"}\n')
        log.write(f'Overrides:\n')
        for k, v in overrides.items():
            log.write(f'  {k} = {v}\n')
        log.write('\nPer-sequence sampled parameters:\n')
        log.write('-' * 50 + '\n')

    # ── Process sequences ──────────────────────────────────────────────────
    total_frames = 0
    total_start  = time.time()

    for seq_idx, seq_dir in enumerate(seq_dirs, 1):
        seq_name = os.path.basename(seq_dir.rstrip('/'))
        out_dir  = os.path.join(args.output, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f'[{seq_idx}/{len(seq_dirs)}] {seq_name}')
        t0 = time.time()

        # Load
        frames, fnames = load_sequence(seq_dir, args.max_frames or 99999)
        if not frames:
            print(f'  [SKIP] No readable frames found')
            continue

        print(f'  Loaded {len(frames)} frames  '
              f'({frames[0].shape[2]}x{frames[0].shape[3]})')

        # Apply noise
        if enabled:
            noisy_frames, params = apply_noise_to_sequence(
                frames, enabled, cfg, args, device, seq_name
            )
        else:
            noisy_frames = frames
            params = {}

        # Save
        for noisy, fname in zip(noisy_frames, fnames):
            stem    = os.path.splitext(fname)[0]
            outpath = os.path.join(out_dir, stem + '.png')
            save_frame(noisy, outpath)

        elapsed = time.time() - t0
        total_frames += len(frames)
        print(f'  Saved {len(frames)} frames -> {out_dir}  ({elapsed:.1f}s)')

        # ── Export video ──────────────────────────────────────────────────
        if args.make_video:
            video_out_dir = args.video_dir or args.output
            os.makedirs(video_out_dir, exist_ok=True)
            video_path = os.path.join(video_out_dir, f'{seq_name}_comparison.mp4')

            # Build list of directories and labels to include in video
            dirs_to_show   = []
            labels_to_show = []

            # Always include noisy output
            dirs_to_show.append(out_dir)
            labels_to_show.append('Noisy')

            # Include clean input if it has matching frame names
            clean_frames_exist = all(
                os.path.exists(os.path.join(seq_dir, fn))
                for fn in fnames[:3]
            )
            if clean_frames_exist:
                dirs_to_show.insert(0, seq_dir)
                labels_to_show.insert(0, 'Clean')

            # Check if denoised results exist (from test_fastdvdnet.py output)
            denoised_dir = os.path.join(
                os.path.dirname(args.output), 'results', seq_name, 'denoised'
            )
            if os.path.isdir(denoised_dir):
                dirs_to_show.append(denoised_dir)
                labels_to_show.append('Denoised')

            frames_to_video(
                frame_dirs=dirs_to_show,
                labels=labels_to_show,
                out_path=video_path,
                fps=args.fps,
                side_by_side=True,
            )

        # Log sampled params for this sequence
        with open(config_log_path, 'a') as log:
            log.write(f'\n[{seq_name}]\n')
            for k, v in params.items():
                log.write(f'  {k:<20s} = {v:.6f}\n')

    total_elapsed = time.time() - total_start
    print(f'\nDone. {total_frames} frames across {len(seq_dirs)} sequences '
          f'in {total_elapsed:.1f}s')
    print(f'Config log: {config_log_path}')


if __name__ == '__main__':
    main()
