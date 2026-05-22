"""
make_video.py

Create 3-way comparison videos (Clean | Noisy | Denoised) from
test_fastdvdnet.py output folders.

Updated for new filename format: 0000.png, 0001.png, ...
(no sigma tag in filenames)

Usage:
    # All sequences
    python make_video.py --results results_kvbank

    # Specific sequence
    python make_video.py --results results_kvbank --seq blackswan

    # Multiple sequences
    python make_video.py --results results_kvbank --seq blackswan camel

    # Custom fps and output folder
    python make_video.py --results results_kvbank --fps 5 --out_dir videos/
"""

import cv2
import glob
import os
import argparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_frames(folder: str) -> list:
    """
    Get sorted list of PNG frames from a folder.
    Expects simple numbered filenames: 0000.png, 0001.png, ...
    """
    frames = sorted(glob.glob(os.path.join(folder, '*.png')))
    return frames


def find_sequences(results_dir: str) -> list:
    """Return all sequence subdirectories that have the expected subfolders."""
    seqs = []
    for d in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, d)
        if not os.path.isdir(full):
            continue
        # Must have at least groundtruth and denoised subfolders
        has_gt       = os.path.isdir(os.path.join(full, 'groundtruth'))
        has_denoised = os.path.isdir(os.path.join(full, 'denoised'))
        if has_gt and has_denoised:
            seqs.append(d)
    return seqs


# ---------------------------------------------------------------------------
# Video creation
# ---------------------------------------------------------------------------

def make_video(results_dir: str, seq_name: str,
               fps: int = 5, out_dir: str = None):
    """
    Create a 3-way comparison video: Clean | Noisy | Denoised.

    Reads frames from:
        results_dir/seq_name/groundtruth/0000.png ...
        results_dir/seq_name/noisy/0000.png ...
        results_dir/seq_name/denoised/0000.png ...
    """
    base         = os.path.join(results_dir, seq_name)
    gt_dir       = os.path.join(base, 'groundtruth')
    noisy_dir    = os.path.join(base, 'noisy')
    denoised_dir = os.path.join(base, 'denoised')

    # Check folders
    missing = [d for d in [gt_dir, noisy_dir, denoised_dir]
               if not os.path.isdir(d)]
    if missing:
        print(f'  [SKIP] Missing: {[os.path.basename(d) for d in missing]}')
        return

    gt_frames       = get_frames(gt_dir)
    noisy_frames    = get_frames(noisy_dir)
    denoised_frames = get_frames(denoised_dir)

    if not gt_frames:
        print(f'  [SKIP] No frames in {gt_dir}')
        return

    T = min(len(gt_frames), len(noisy_frames), len(denoised_frames))
    if T == 0:
        print(f'  [SKIP] Empty frame list')
        return

    # Read first frame for dimensions
    sample = cv2.imread(gt_frames[0])
    if sample is None:
        print(f'  [SKIP] Could not read {gt_frames[0]}')
        return
    H, W = sample.shape[:2]

    out_dir  = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{seq_name}_3way.mp4')

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W * 3, H)
    )

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    labels     = ['Clean', 'Noisy', 'Denoised']

    for i in range(T):
        cols = []
        for frame_path, label in zip(
            [gt_frames[i], noisy_frames[i], denoised_frames[i]], labels
        ):
            f = cv2.imread(frame_path)
            if f is None:
                f = sample.copy() * 0

            # Draw label
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(f, (8, 6), (tw + 16, th + 16), (0, 0, 0), -1)
            cv2.putText(f, label, (12, th + 10),
                        font, font_scale, (255, 255, 255), thickness)
            cols.append(f)

        writer.write(cv2.hconcat(cols))

    writer.release()
    print(f'  Saved ({T} frames, fps={fps}): {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create 3-way comparison videos — Clean | Noisy | Denoised'
    )
    parser.add_argument('--results', type=str, required=True,
                        help='Results root folder (e.g. results_kvbank_flicker_shot_read)')
    parser.add_argument('--seq',     type=str, default=None, nargs='+',
                        help='Sequence name(s). Default: all sequences found')
    parser.add_argument('--fps',     type=int, default=5,
                        help='Video frame rate (default: 5)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output folder for videos (default: same as --results)')

    args = parser.parse_args()

    # Resolve sequences
    if args.seq is None:
        sequences = find_sequences(args.results)
        if not sequences:
            print(f'No valid sequence folders found under: {args.results}')
            return
        print(f'Found {len(sequences)} sequence(s): {sequences}')
    else:
        sequences = args.seq

    print(f'Results : {args.results}')
    print(f'FPS     : {args.fps}')
    print()

    for seq in sequences:
        print(f'[{seq}]')
        make_video(
            results_dir=args.results,
            seq_name=seq,
            fps=args.fps,
            out_dir=args.out_dir,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
