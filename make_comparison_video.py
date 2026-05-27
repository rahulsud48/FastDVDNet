#!/usr/bin/env python3
"""
Creates a side-by-side comparison video from the output_images_sigma<N> folder
produced by test_fastdvdnet.py.

Layout (left to right):
    [ Clean GT  |  Noisy  |  Denoised ]

Each panel is labelled with a text overlay at the top-left.

Usage:
    python make_comparison_video.py --result_dir ./results --sigma 25
    python make_comparison_video.py --result_dir ./results  # auto-detects sigma folder

Output:
    <result_dir>/output_images_sigma<N>/comparison_sigma<N>.mp4
"""

import os
import argparse
import glob
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

def list_images(folder):
    """Returns a sorted list of image paths in folder."""
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()
    return paths


def draw_label(frame, text,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.9,
               thickness=2,
               color=(255, 255, 255),
               shadow_color=(0, 0, 0),
               margin=10):
    """
    Draws a text label with a thin shadow for legibility on any background.
    Shadow is drawn first (offset by 1 px), then the main text on top.
    """
    pos = (margin, margin + int(font_scale * 30))   # rough baseline offset

    # Shadow
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                font, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    # Label
    cv2.putText(frame, text, pos,
                font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def load_frame(path):
    """Reads an image as BGR uint8. Raises if path is missing or unreadable."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Could not read image: {}".format(path))
    return img


def make_side_by_side(gt, noisy, denoised, divider_px=2, divider_color=(180, 180, 180)):
    """
    Concatenates three BGR frames horizontally with thin dividers.

    Args:
        gt, noisy, denoised : (H, W, 3) uint8 BGR frames — must be same size
        divider_px          : width of the divider stripe in pixels
        divider_color       : BGR colour of the divider

    Returns:
        (H, 3*W + 2*divider_px, 3) uint8 composite frame
    """
    H, W = gt.shape[:2]
    divider = np.full((H, divider_px, 3), divider_color, dtype=np.uint8)
    return np.concatenate([gt, divider, noisy, divider, denoised], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_video(result_dir, sigma, fps, codec):

    # ── Locate output_images_sigma<N> folder ────────────────────────────
    if sigma is not None:
        img_root = os.path.join(result_dir, 'output_images_sigma{}'.format(sigma))
        if not os.path.isdir(img_root):
            sys.exit("ERROR: Folder not found: {}".format(img_root))
    else:
        # Auto-detect: pick the first matching folder
        candidates = sorted(glob.glob(
            os.path.join(result_dir, 'output_images_sigma*')
        ))
        if not candidates:
            sys.exit("ERROR: No output_images_sigma* folder found in: {}".format(result_dir))
        img_root = candidates[0]
        sigma    = img_root.split('sigma')[-1]   # string, used only for naming
        print("Auto-detected image folder: {}".format(img_root))

    gt_dir       = os.path.join(img_root, 'gt')
    noisy_dir    = os.path.join(img_root, 'noisy')
    denoised_dir = os.path.join(img_root, 'denoised')

    for d in (gt_dir, noisy_dir, denoised_dir):
        if not os.path.isdir(d):
            sys.exit("ERROR: Expected sub-folder not found: {}".format(d))

    # ── Collect and validate frame lists ────────────────────────────────
    gt_frames       = list_images(gt_dir)
    noisy_frames    = list_images(noisy_dir)
    denoised_frames = list_images(denoised_dir)

    if not gt_frames:
        sys.exit("ERROR: No images found in: {}".format(gt_dir))

    n_frames = len(gt_frames)
    if not (len(noisy_frames) == len(denoised_frames) == n_frames):
        sys.exit(
            "ERROR: Frame count mismatch — gt:{} noisy:{} denoised:{}".format(
                n_frames, len(noisy_frames), len(denoised_frames)
            )
        )

    print("Found {} frames in each of gt / noisy / denoised".format(n_frames))

    # ── Determine frame size from first frame ────────────────────────────
    sample = load_frame(gt_frames[0])
    H, W   = sample.shape[:2]
    # Side-by-side width: 3 panels + 2 dividers (2 px each)
    out_W  = W * 3 + 2 * 2
    print("Frame size: {}x{}  →  composite: {}x{}".format(W, H, out_W, H))

    # ── Set up VideoWriter ───────────────────────────────────────────────
    out_path = os.path.join(img_root, 'comparison_sigma{}.mp4'.format(sigma))
    fourcc   = cv2.VideoWriter_fourcc(*codec)
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (out_W, H))

    if not writer.isOpened():
        sys.exit("ERROR: VideoWriter failed to open. Try a different --codec.")

    # ── Write frames ─────────────────────────────────────────────────────
    print("Writing video: {}".format(out_path))
    for idx, (gp, np_, dp) in enumerate(zip(gt_frames, noisy_frames, denoised_frames)):

        gt_bgr  = load_frame(gp)
        n_bgr   = load_frame(np_)
        den_bgr = load_frame(dp)

        # Sanity-check all three frames have the same spatial size
        if gt_bgr.shape[:2] != (H, W):
            print("WARNING: frame {} gt size mismatch {}, resizing".format(
                idx, gt_bgr.shape[:2]))
            gt_bgr = cv2.resize(gt_bgr, (W, H))
        if n_bgr.shape[:2] != (H, W):
            print("WARNING: frame {} noisy size mismatch {}, resizing".format(
                idx, n_bgr.shape[:2]))
            n_bgr = cv2.resize(n_bgr, (W, H))
        if den_bgr.shape[:2] != (H, W):
            print("WARNING: frame {} denoised size mismatch {}, resizing".format(
                idx, den_bgr.shape[:2]))
            den_bgr = cv2.resize(den_bgr, (W, H))

        # Add text labels to each panel
        draw_label(gt_bgr,  'Clean GT')
        draw_label(n_bgr,   'Noisy  (sigma={})'.format(sigma))
        draw_label(den_bgr, 'Denoised')

        # Composite
        composite = make_side_by_side(gt_bgr, n_bgr, den_bgr)
        writer.write(composite)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_frames:
            print("  [{}/{}] frames written".format(idx + 1, n_frames))

    writer.release()
    print("\nDone. Video saved to: {}".format(out_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build side-by-side comparison video from FastDVDnet results."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the results folder (contains output_images_sigma<N>/)."
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=None,
        help="Sigma value used during testing (e.g. 25). "
             "If omitted, the script auto-detects the first output_images_sigma* folder."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the output video (default: 10)."
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec string for VideoWriter (default: mp4v). "
             "Use 'avc1' or 'H264' if mp4v produces unplayable files on your system."
    )

    args = parser.parse_args()

    print("\n### FastDVDnet comparison video ###")
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))
    print()

    make_video(
        result_dir=args.result_dir,
        sigma=args.sigma,
        fps=args.fps,
        codec=args.codec,
    )
