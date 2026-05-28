#!/usr/bin/env python3
"""
Builds LOSSLESS side-by-side comparison videos from the per-sequence test output.

Reads the PNGs saved by test_fastdvdnet.py under:
    <results>/pred_sigma<sigma>/<seq_name>/{gt,noisy,denoised}/00000.png ...

Produces, per sequence, a side-by-side video laid out left → right as:
    [ Noisy | Clean (GT) | Denoised ]

Output (default): lossless MP4
    <results>/pred_sigma<sigma>/<seq_name>/comparison_<seq_name>.mp4

Encoding:
  - Preferred: ffmpeg with libx264 -crf 0 (mathematically lossless H.264 in MP4).
    Plays everywhere — default Ubuntu video player, browsers, Quicktime, VLC, mpv.
  - Fallback (if ffmpeg is not found on PATH): FFV1 in an MKV container via
    cv2.VideoWriter (also lossless, but needs VLC/mpv to play). The script warns
    and switches automatically.

Both paths are pixel-identical to the source PNGs — safe for frame-by-frame analysis.

Usage:
    # all sequences under a pred_sigma folder
    python make_comparison_video.py --pred_dir results/pred_sigma25

    # a single sequence folder
    python make_comparison_video.py --seq_dir results/pred_sigma25/blackswan

    # options
    python make_comparison_video.py --pred_dir results/pred_sigma25 --fps 15 --no_labels
    python make_comparison_video.py --pred_dir results/pred_sigma25 --force_mkv
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess

import cv2
import numpy as np


IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
PANELS   = ('noisy', 'gt', 'denoised')          # left → right order
LABELS   = {'noisy': 'Noisy', 'gt': 'Clean (GT)', 'denoised': 'Denoised'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def have_ffmpeg():
    """True if an ffmpeg binary is available on PATH."""
    return shutil.which('ffmpeg') is not None


def list_images(folder):
    """Sorted list of image paths in a folder."""
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()
    return paths


def load_frame(path):
    """Reads an image as BGR uint8; raises if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not read image: {}".format(path))
    return img


def draw_label(frame, text, font_scale=0.8, thickness=2,
               color=(255, 255, 255), shadow=(0, 0, 0), margin=10):
    """Draws a text label with a 1px shadow for legibility on any background."""
    pos = (margin, margin + int(font_scale * 30))
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def make_side_by_side(panels, divider_px=2, divider_color=(128, 128, 128)):
    """Concatenates panels horizontally with thin grey dividers.
    panels: list of (H, W, 3) uint8 BGR frames, all the same size."""
    H = panels[0].shape[0]
    divider = np.full((H, divider_px, 3), divider_color, dtype=np.uint8)
    out = []
    for i, p in enumerate(panels):
        out.append(p)
        if i != len(panels) - 1:
            out.append(divider)
    return np.concatenate(out, axis=1)


def build_composites(seq_dir, add_labels):
    """Loads frames, validates, and yields composite BGR frames one at a time.
    Returns (generator, n_frames, out_W, H) or (None, 0, 0, 0) on skip."""
    seq_name = os.path.basename(seq_dir.rstrip('/'))

    panel_dirs = {p: os.path.join(seq_dir, p) for p in PANELS}
    for p, d in panel_dirs.items():
        if not os.path.isdir(d):
            print("  [skip] {} — missing '{}' subfolder".format(seq_name, p))
            return None, 0, 0, 0

    frames = {p: list_images(panel_dirs[p]) for p in PANELS}
    counts = {p: len(frames[p]) for p in PANELS}
    n = counts[PANELS[0]]
    if n == 0:
        print("  [skip] {} — no frames found".format(seq_name))
        return None, 0, 0, 0
    if len(set(counts.values())) != 1:
        print("  [skip] {} — frame count mismatch {}".format(seq_name, counts))
        return None, 0, 0, 0

    sample = load_frame(frames['gt'][0])
    H, W   = sample.shape[:2]
    out_W  = W * len(PANELS) + 2 * (len(PANELS) - 1)

    def gen():
        for t in range(n):
            panels = []
            for p in PANELS:
                img = load_frame(frames[p][t])
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
                if add_labels:
                    img = draw_label(img.copy(), LABELS[p])
                panels.append(img)
            yield make_side_by_side(panels)

    return gen(), n, out_W, H


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def write_mp4_ffmpeg(seq_dir, composites, n, out_W, H, fps, player_safe=False):
    """Lossless MP4 via ffmpeg. Pipes raw BGR24 frames to ffmpeg.

    Two modes:
      - Default (player_safe=False): libx264rgb -crf 0. Encodes RGB directly with
        NO colour-space conversion → bit-exact (pixel-identical to source PNGs).
        Plays in VLC/mpv and most modern players.
      - player_safe=True: libx264 -crf 0 -pix_fmt yuv444p. Goes through an RGB→YUV
        conversion so it's "visually lossless" (±1-2 per channel) rather than
        bit-exact, but is compatible with the widest range of basic players.

    Returns the output path on success, raises on failure.
    """
    seq_name = os.path.basename(seq_dir.rstrip('/'))
    out_path = os.path.join(seq_dir, 'comparison_{}.mp4'.format(seq_name))

    if player_safe:
        codec_args = ['-vcodec', 'libx264', '-crf', '0', '-pix_fmt', 'yuv444p']
    else:
        codec_args = ['-vcodec', 'libx264rgb', '-crf', '0']   # bit-exact, no RGB→YUV

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',                    # matches OpenCV's channel order
        '-s', '{}x{}'.format(out_W, H),
        '-r', str(fps),
        '-i', '-',                              # read frames from stdin
        '-an',                                  # no audio
    ] + codec_args + [
        '-preset', 'veryslow',
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in composites:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
    except BrokenPipeError:
        ret = proc.wait()

    if ret != 0:
        err = proc.stderr.read().decode('utf-8', errors='ignore')[-1500:]
        raise RuntimeError("ffmpeg failed (code {}):\n{}".format(ret, err))

    mode = "visually lossless yuv444p" if player_safe else "bit-exact libx264rgb"
    print("  [ok]  {} → {}  ({} frames, {}x{}, {})".format(
        seq_name, os.path.basename(out_path), n, out_W, H, mode))
    return out_path


def write_mkv_ffv1(seq_dir, composites, n, out_W, H, fps):
    """Fallback: lossless FFV1 in MKV via cv2.VideoWriter (no ffmpeg needed)."""
    seq_name = os.path.basename(seq_dir.rstrip('/'))
    out_path = os.path.join(seq_dir, 'comparison_{}.mkv'.format(seq_name))

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_W, H))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter (FFV1) failed to open for {}".format(seq_name))

    for frame in composites:
        writer.write(frame)
    writer.release()

    print("  [ok]  {} → {}  ({} frames, {}x{}, lossless FFV1/MKV)".format(
        seq_name, os.path.basename(out_path), n, out_W, H))
    return out_path


def build_video_for_sequence(seq_dir, fps, add_labels, use_ffmpeg, player_safe=False):
    """Builds one lossless comparison video for a single sequence folder."""
    composites, n, out_W, H = build_composites(seq_dir, add_labels)
    if composites is None:
        return False

    try:
        if use_ffmpeg:
            write_mp4_ffmpeg(seq_dir, composites, n, out_W, H, fps, player_safe=player_safe)
        else:
            write_mkv_ffv1(seq_dir, composites, n, out_W, H, fps)
        return True
    except Exception as e:
        print("  [error] {} — {}".format(os.path.basename(seq_dir.rstrip('/')), e))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build lossless side-by-side [noisy|clean|denoised] videos from test PNGs."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred_dir", type=str,
                   help="pred_sigma<sigma> folder containing per-sequence subfolders")
    g.add_argument("--seq_dir",  type=str,
                   help="single sequence folder containing gt/ noisy/ denoised/")

    parser.add_argument("--fps",        type=float, default=10.0, help="Output frames per second")
    parser.add_argument("--no_labels",  action='store_true',      help="Don't overlay panel labels")
    parser.add_argument("--force_mkv",  action='store_true',
                        help="Force FFV1/MKV output even if ffmpeg is available")
    parser.add_argument("--player_safe", action='store_true',
                        help="MP4: use yuv444p (visually lossless, ±1-2) for max player "
                             "compatibility instead of bit-exact libx264rgb")

    args = parser.parse_args()
    add_labels = not args.no_labels

    # Decide encoder
    if args.force_mkv:
        use_ffmpeg = False
        print("Encoder: FFV1/MKV (forced via --force_mkv)")
    elif have_ffmpeg():
        use_ffmpeg = True
        if args.player_safe:
            print("Encoder: ffmpeg libx264 -crf 0 -pix_fmt yuv444p (visually lossless MP4)")
        else:
            print("Encoder: ffmpeg libx264rgb -crf 0 (bit-exact lossless MP4)")
    else:
        use_ffmpeg = False
        print("WARNING: ffmpeg not found on PATH — falling back to lossless FFV1/MKV.")
        print("         Install ffmpeg for MP4 output:  sudo apt install ffmpeg")

    if args.seq_dir:
        seq_dirs = [args.seq_dir]
        print("Building video for sequence: {}".format(args.seq_dir))
    else:
        if not os.path.isdir(args.pred_dir):
            sys.exit("ERROR: pred_dir not found: {}".format(args.pred_dir))
        seq_dirs = sorted([
            os.path.join(args.pred_dir, d)
            for d in os.listdir(args.pred_dir)
            if os.path.isdir(os.path.join(args.pred_dir, d))
        ])
        if not seq_dirs:
            sys.exit("ERROR: no sequence subfolders found in {}".format(args.pred_dir))
        print("Building videos for {} sequences under: {}".format(len(seq_dirs), args.pred_dir))

    n_ok = 0
    for sd in seq_dirs:
        if build_video_for_sequence(sd, fps=args.fps, add_labels=add_labels,
                                    use_ffmpeg=use_ffmpeg, player_safe=args.player_safe):
            n_ok += 1

    if use_ffmpeg:
        fmt = "MP4 (yuv444p visually lossless)" if args.player_safe else "MP4 (libx264rgb bit-exact)"
    else:
        fmt = "MKV (FFV1 lossless)"
    print("\nDone. {}/{} videos written as {}.".format(n_ok, len(seq_dirs), fmt))


if __name__ == "__main__":
    main()
