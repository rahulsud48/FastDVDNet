"""
make_video.py — 3-way comparison video: Clean | Noisy | Denoised

Handles new filename format: {prefix}_s{sigma}_l{lam}_{idx:04d}.png
Auto-detects available sigma/lambda tags if not specified.

Usage:
    # Auto-detect all tags, all sequences
    python make_video.py --results results_kvbank

    # Specific tag
    python make_video.py --results results_kvbank --tag s0.0200_l0.5000

    # Specific sequence
    python make_video.py --results results_kvbank --seq blackswan

    # Custom fps
    python make_video.py --results results_kvbank --fps 5
"""

import cv2
import glob
import os
import re
import argparse


def get_frames(folder: str, tag: str = None) -> list:
    """
    Get sorted frames from folder.
    If tag given: match *_{tag}_*.png
    If no tag:    match all *.png (sorted)
    """
    if tag:
        pattern = os.path.join(folder, f'*_{tag}_*.png')
    else:
        pattern = os.path.join(folder, '*.png')
    return sorted(glob.glob(pattern))


def detect_tags(folder: str) -> list:
    """
    Scan a folder and return all unique s{sigma}_l{lam} tags found.
    e.g. ['s0.0200_l0.5000', 's0.0500_l1.0000']
    """
    files = glob.glob(os.path.join(folder, '*.png'))
    tags  = set()
    for f in files:
        m = re.search(r'(s\d+_l\d+)', os.path.basename(f))
        if m:
            tags.add(m.group(1))
    return sorted(tags)


def find_sequences(results_dir: str) -> list:
    return sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and os.path.isdir(os.path.join(results_dir, d, 'groundtruth'))
        and os.path.isdir(os.path.join(results_dir, d, 'denoised'))
    ])


def make_video(results_dir: str, seq_name: str,
               tag: str, fps: int = 5, out_dir: str = None):
    """Create Clean | Noisy | Denoised video for one sequence and one tag."""
    base         = os.path.join(results_dir, seq_name)
    gt_dir       = os.path.join(base, 'groundtruth')
    noisy_dir    = os.path.join(base, 'noisy')
    denoised_dir = os.path.join(base, 'denoised')

    for d in [gt_dir, noisy_dir, denoised_dir]:
        if not os.path.isdir(d):
            print(f'  [SKIP] Missing folder: {d}')
            return

    gt_frames       = get_frames(gt_dir,       tag)
    noisy_frames    = get_frames(noisy_dir,     tag)
    denoised_frames = get_frames(denoised_dir,  tag)

    if not gt_frames:
        print(f'  [SKIP] No frames for tag={tag} in {gt_dir}')
        available = detect_tags(gt_dir)
        if available:
            print(f'  Available tags: {available}')
        return

    T = min(len(gt_frames), len(noisy_frames), len(denoised_frames))
    if T == 0:
        print(f'  [SKIP] Empty frame list')
        return

    sample = cv2.imread(gt_frames[0])
    if sample is None:
        print(f'  [SKIP] Could not read {gt_frames[0]}')
        return
    H, W = sample.shape[:2]

    out_dir  = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{seq_name}_{tag}_3way.mp4')

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (W * 3, H)
    )

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    # Parse tag for label
    m = re.match(r's(\d+)_l(\d+)', tag)
    if m:
        sigma_lbl = f'sigma={m.group(1)}/255'
        lam_lbl   = f'lam={m.group(2)}/255'
        noisy_lbl = f'Noisy ({sigma_lbl}, {lam_lbl})'
    else:
        noisy_lbl = f'Noisy ({tag})'

    labels = ['Clean', noisy_lbl, 'Denoised']

    for i in range(T):
        cols = []
        for frame_path, label in zip(
            [gt_frames[i], noisy_frames[i], denoised_frames[i]], labels
        ):
            f = cv2.imread(frame_path)
            if f is None:
                f = sample.copy() * 0
            (tw, tth), _ = cv2.getTextSize(label, font, fs, th)
            cv2.rectangle(f, (8, 6), (tw + 16, tth + 16), (0, 0, 0), -1)
            cv2.putText(f, label, (12, tth + 10), font, fs, (255, 255, 255), th)
            cols.append(f)
        writer.write(cv2.hconcat(cols))

    writer.release()
    print(f'  Saved ({T} frames, tag={tag}, fps={fps}): {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='3-way comparison video — Clean | Noisy | Denoised'
    )
    parser.add_argument('--results', type=str, required=True,
                        help='Results root folder')
    parser.add_argument('--seq',     type=str, default=None, nargs='+',
                        help='Sequence name(s). Default: all')
    parser.add_argument('--tag',     type=str, default=None, nargs='+',
                        help='Noise tag(s) e.g. s0.0200_l0.5000. Default: auto-detect all')
    parser.add_argument('--fps',     type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=None)

    args = parser.parse_args()

    sequences = args.seq or find_sequences(args.results)
    if not sequences:
        print(f'No valid sequence folders found under: {args.results}')
        return

    print(f'Results   : {args.results}')
    print(f'Sequences : {sequences}')
    print(f'FPS       : {args.fps}')
    print()

    for seq in sequences:
        print(f'[{seq}]')

        # Resolve tags — auto-detect if not specified
        if args.tag is None:
            gt_dir = os.path.join(args.results, seq, 'groundtruth')
            tags   = detect_tags(gt_dir) if os.path.isdir(gt_dir) else []
            if not tags:
                print(f'  [SKIP] No tags found in {gt_dir}')
                continue
            print(f'  Auto-detected tags: {tags}')
        else:
            tags = args.tag

        for tag in tags:
            make_video(args.results, seq, tag,
                       fps=args.fps, out_dir=args.out_dir)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
