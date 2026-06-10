import os
import random
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

# CHANGED_TO: fixed full-HD canvas. Production ISP inference is locked at
# 1080x1920 full-frame, and the bottleneck pooling conv has a static kernel
# whose output token grid depends on input size — so train resolution must
# match inference resolution exactly. Larger frames are random-cropped to
# this size; smaller frames are reflect-padded up to it.
TARGET_H, TARGET_W = 1080, 1920


def _list_frames(seq_dir):
    frames = []
    for ext in IMG_EXTS:
        frames.extend(glob.glob(os.path.join(seq_dir, ext)))
    return sorted(frames)


class SimpleVideoDataset(Dataset):
    # CHANGED_TO: defaults updated for the full-scene scheme — long scenes of
    # consecutive frames (temp_stride=1). crop_size kept in the signature for
    # backward compatibility but is no longer used; the spatial size is fixed
    # to (TARGET_H, TARGET_W) via crop-or-pad below.
    def __init__(self, file_root, sequence_length=12, crop_size=None,
                 epoch_size=256000, temp_stride=1):
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.epoch_size = epoch_size
        self.temp_stride = temp_stride

        self.seq_dirs = [
            os.path.join(file_root, d)
            for d in os.listdir(file_root)
            if os.path.isdir(os.path.join(file_root, d))
        ]
        self.sequences = []

        # CHANGED_TO: forward-window sampling (was symmetric-around-center).
        # The new scheme processes a window starting at `start` going forward
        # for sequence_length frames; the temporal-length filter is unchanged.
        for seq_dir in self.seq_dirs:
            frames = _list_frames(seq_dir)
            min_needed = 1 + (sequence_length - 1) * temp_stride
            if len(frames) >= min_needed:
                max_start = len(frames) - 1 - (sequence_length - 1) * temp_stride
                valid_starts = list(range(0, max_start + 1))
                self.sequences.append((seq_dir, frames, valid_starts))

        if not self.sequences:
            raise RuntimeError(
                f"No valid training sequences found with sequence_length={sequence_length}, "
                f"temp_stride={temp_stride}. Each scene needs at least "
                f"{1 + (sequence_length - 1) * temp_stride} frames."
            )

    def __len__(self):
        return self.epoch_size

    # CHANGED_TO: new helper. Forces every frame in a scene onto the fixed
    # (TARGET_H, TARGET_W) canvas. Crop offset (for larger frames) and pad
    # amount (for smaller frames) are decided ONCE per scene and applied
    # identically to all frames so the temporal stack stays spatially aligned.
    def _fit_to_canvas(self, imgs):
        h, w, _ = imgs[0].shape

        # Crop if larger than target (single random offset for the scene)
        if h > TARGET_H:
            top = random.randint(0, h - TARGET_H)
            imgs = [im[top:top + TARGET_H, :, :] for im in imgs]
            h = TARGET_H
        if w > TARGET_W:
            left = random.randint(0, w - TARGET_W)
            imgs = [im[:, left:left + TARGET_W, :] for im in imgs]
            w = TARGET_W

        # Pad if smaller than target. Reflect when the deficit is smaller
        # than the dimension (otherwise reflection would wrap past the edge);
        # fall back to replicate for large deficits.
        pad_b = max(0, TARGET_H - h)
        pad_r = max(0, TARGET_W - w)
        if pad_b or pad_r:
            use_reflect = (pad_b < h) and (pad_r < w)
            border = cv2.BORDER_REFLECT_101 if use_reflect else cv2.BORDER_REPLICATE
            imgs = [cv2.copyMakeBorder(im, 0, pad_b, 0, pad_r, border) for im in imgs]

        return imgs

    def __getitem__(self, idx):
        _, frames, valid_starts = random.choice(self.sequences)
        start = random.choice(valid_starts)

        # CHANGED_TO: forward window from a random start.
        indices = [start + i * self.temp_stride for i in range(self.sequence_length)]

        imgs = []
        for i in indices:
            img = cv2.imread(frames[i], cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read {frames[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        # CHANGED_TO: fixed-canvas crop-or-pad replaces the old fixed
        # random crop to crop_size.
        imgs = self._fit_to_canvas(imgs)

        # [F, H, W, C] -> [F, C, H, W], float32
        arr = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float()
            for img in imgs
        ], dim=0)

        return {"data": arr}


def train_simple_loader(batch_size, file_root, sequence_length, crop_size=None,
                        epoch_size=256000, random_shuffle=True, temp_stride=1):
    # CHANGED_TO: temp_stride default 1; crop_size optional/unused (fixed canvas).
    # num_workers kept low: each sample is sequence_length x 3 x 1080 x 1920
    # fp32 (~hundreds of MB), so high worker counts can blow up host RAM.
    ds = SimpleVideoDataset(
        file_root=file_root,
        sequence_length=sequence_length,
        crop_size=crop_size,
        epoch_size=epoch_size,
        temp_stride=temp_stride
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=random_shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
