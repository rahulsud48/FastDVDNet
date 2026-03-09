#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def copy_sequence(src_seq: Path, dst_seq: Path):
    dst_seq.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([p for p in src_seq.iterdir() if p.suffix.lower() in exts])

    if not frames:
        print(f"[WARN] No frames found in {src_seq}")
        return

    for frame in frames:
        shutil.copy2(frame, dst_seq / frame.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--davis_root",
        type=str,
        default="DAVIS/JPEGImages/480p",
        help="Path to DAVIS sequence folders"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data",
        help="Output root directory"
    )
    parser.add_argument(
        "--val_seqs",
        nargs="+",
        default=["blackswan", "camel", "dance-twirl", "kite-surf", "soapbox"],
        help="Sequence names to place in validation set"
    )
    args = parser.parse_args()

    davis_root = Path(args.davis_root)
    out_root = Path(args.out_root)
    train_root = out_root / "train"
    val_root = out_root / "val"

    if not davis_root.exists():
        raise FileNotFoundError(f"DAVIS root not found: {davis_root}")

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    val_set = set(args.val_seqs)

    all_sequences = sorted([p for p in davis_root.iterdir() if p.is_dir()])
    if not all_sequences:
        raise RuntimeError(f"No sequence folders found under {davis_root}")

    print(f"Found {len(all_sequences)} DAVIS sequences")

    for seq_dir in all_sequences:
        seq_name = seq_dir.name
        if seq_name in val_set:
            dst = val_root / seq_name
            split = "val"
        else:
            dst = train_root / seq_name
            split = "train"

        print(f"[{split}] {seq_name}")
        copy_sequence(seq_dir, dst)

    print("\nDone.")
    print(f"Train dir: {train_root}")
    print(f"Val dir:   {val_root}")

if __name__ == "__main__":
    main()