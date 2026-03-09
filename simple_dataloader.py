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
    def __init__(self, file_root, sequence_length=5, crop_size=96,
                 epoch_size=256000, temp_stride=3):
        self.file_root = file_root
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.epoch_size = epoch_size
        self.temp_stride = temp_stride

        self.seq_dirs = [
            os.path.join(file_root, d)
            for d in os.listdir(file_root)
            if os.path.isdir(os.path.join(file_root, d))
        ]
        self.sequences = []
        half = sequence_length // 2

        for seq_dir in self.seq_dirs:
            frames = _list_frames(seq_dir)
            min_needed = 1 + (sequence_length - 1) * temp_stride
            if len(frames) >= min_needed:
                valid_centers = list(range(
                    half * temp_stride,
                    len(frames) - half * temp_stride
                ))
                self.sequences.append((seq_dir, frames, valid_centers))

        if not self.sequences:
            raise RuntimeError("No valid training sequences found.")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        _, frames, valid_centers = random.choice(self.sequences)
        center = random.choice(valid_centers)

        half = self.sequence_length // 2
        indices = [
            center + (i - half) * self.temp_stride
            for i in range(self.sequence_length)
        ]

        imgs = []
        for i in indices:
            img = cv2.imread(frames[i], cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read {frames[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        h, w, _ = imgs[0].shape
        if h < self.crop_size or w < self.crop_size:
            raise RuntimeError(
                f"Image too small for crop: {(h, w)} < {self.crop_size}"
            )

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        imgs = [
            img[top:top+self.crop_size, left:left+self.crop_size, :]
            for img in imgs
        ]

        # [F, H, W, C] -> [F, C, H, W], float32
        arr = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float()
            for img in imgs
        ], dim=0)

        return {"data": arr}

def train_simple_loader(batch_size, file_root, sequence_length, crop_size,
                        epoch_size, random_shuffle=True, temp_stride=3):
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
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )