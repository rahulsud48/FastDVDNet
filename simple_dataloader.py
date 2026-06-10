import os
import random
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _list_frames(seq_dir):
    frames = []
    for ext in IMG_EXTS:
        frames.extend(glob.glob(os.path.join(seq_dir, ext)))
    return sorted(frames)


class SimpleVideoDataset(Dataset):
    # CHANGED_TO: spatial size is now parameterized via crop_h x crop_w instead
    # of a hardcoded full-HD canvas. Pass crop_h=crop_w=96 for fast patch
    # training, or crop_h=1080, crop_w=1920 for full-frame training. Two
    # independent dims (not a single square int) so non-square full-HD is
    # expressible. Frames larger than the target are randomly cropped; frames
    # smaller are reflect/replicate padded. The crop offset and pad are decided
    # ONCE per scene and applied to all frames so the temporal stack stays
    # spatially aligned.
    def __init__(self, file_root, sequence_length=20, crop_h=96, crop_w=96,
                 epoch_size=256000, temp_stride=1):
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.epoch_size = epoch_size
        self.temp_stride = temp_stride

        self.seq_dirs = [
            os.path.join(file_root, d)
            for d in os.listdir(file_root)
            if os.path.isdir(os.path.join(file_root, d))
        ]
        self.sequences = []

        # CHANGED_TO: forward-window sampling (was symmetric-around-center).
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

    # CHANGED_TO: generalized crop-or-pad to (crop_h, crop_w) instead of a
    # fixed 1080x1920 canvas. Single random crop offset per scene; reflect
    # pad when the deficit is smaller than the dimension, else replicate.
    def _fit_to_canvas(self, imgs):
        target_h, target_w = self.crop_h, self.crop_w
        h, w, _ = imgs[0].shape

        # Crop if larger than target (single random offset, same for all frames)
        if h > target_h:
            top = random.randint(0, h - target_h)
            imgs = [im[top:top + target_h, :, :] for im in imgs]
            h = target_h
        if w > target_w:
            left = random.randint(0, w - target_w)
            imgs = [im[:, left:left + target_w, :] for im in imgs]
            w = target_w

        # Pad if smaller than target
        pad_b = max(0, target_h - h)
        pad_r = max(0, target_w - w)
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

        # CHANGED_TO: parameterized crop-or-pad to (crop_h, crop_w).
        imgs = self._fit_to_canvas(imgs)

        # [F, H, W, C] -> [F, C, H, W], float32
        arr = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float()
            for img in imgs
        ], dim=0)

        return {"data": arr}


def train_simple_loader(batch_size, file_root, sequence_length,
                        crop_h=96, crop_w=96, epoch_size=256000,
                        random_shuffle=True, temp_stride=1, num_workers=2):
    # CHANGED_TO: crop_h/crop_w replace the single crop_size; num_workers
    # exposed (set 0 if shared-memory limited). temp_stride default 1.
    ds = SimpleVideoDataset(
        file_root=file_root,
        sequence_length=sequence_length,
        crop_h=crop_h,
        crop_w=crop_w,
        epoch_size=epoch_size,
        temp_stride=temp_stride,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=random_shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=True,
    )
