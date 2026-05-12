"""
FastDVDnet — Single-frame input with KV Bank cross-attention at bottleneck.

Changes from the original:
  1. Single-frame input (frame_t + noise_map) instead of 3-frame stacked input.
  2. KV bank cross-attention at the bottleneck — current frame queries past frames'
     compressed features stored in a rolling ring-buffer (KVBank).
  3. All DSConv (depthwise 3x3 + pointwise 1x1) replaced with standard Conv2d(3x3).
  4. PixelShuffle replaced with ConvTranspose2d(kernel=2, stride=2).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class CvBlock(nn.Module):
    """(Conv2d 3x3 => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """(Conv2d 3x3 => BN => ReLU) x 2 — accepts single frame + noise map (4 channels)."""
    def __init__(self, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        # 3 (RGB) + 1 (noise map) = 4 input channels
        self.convblock = nn.Sequential(
            nn.Conv2d(4, self.interm_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Stride-2 Conv2d => BN => ReLU => CvBlock"""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """CvBlock => ConvTranspose2d x2"""
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.cvblock  = CvBlock(in_ch, in_ch)
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.upsample(self.cvblock(x))


class OutputCvBlock(nn.Module):
    """Conv2d 3x3 => BN => ReLU => Conv2d 3x3"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


# ---------------------------------------------------------------------------
# KV Bank — ring buffer, lives outside the model
# ---------------------------------------------------------------------------

class KVBank:
    """
    Rolling buffer that stores (key, value) tensors from past bottleneck frames.

    Usage:
        bank = KVBank(bank_size=10)
        bank.reset()                 # call at sequence start or scene cut
        bank.push(k, v)              # store current frame's KV after forward pass
        keys, vals = bank.get()      # returns concatenated past KVs, or (None, None)

    Tensors stay on whatever device they were computed on — no explicit .to() needed.
    """

    def __init__(self, bank_size: int = 10, detach: bool = True):
        self.bank_size = bank_size
        # detach=True  -> inference/validation (no grad needed, saves memory)
        # detach=False -> training (gradients flow through temporal attention)
        self.detach = detach
        self._keys:   list = []   # each entry: (N, S, C)
        self._values: list = []   # each entry: (N, S, C)

    def reset(self):
        self._keys.clear()
        self._values.clear()

    def push(self, k: torch.Tensor, v: torch.Tensor):
        """Store current frame's projected key and value tokens."""
        self._keys.append(k.detach() if self.detach else k)
        self._values.append(v.detach() if self.detach else v)
        if len(self._keys) > self.bank_size:
            self._keys.pop(0)
            self._values.pop(0)

    def get(self):
        """Returns (keys, values) shaped (N, T*S, C), or (None, None) if empty."""
        if not self._keys:
            return None, None
        return torch.cat(self._keys, dim=1), torch.cat(self._values, dim=1)

    def __len__(self):
        return len(self._keys)


# ---------------------------------------------------------------------------
# Bottleneck cross-attention with KV bank
# ---------------------------------------------------------------------------

class BottleneckCrossAttn(nn.Module):
    """
    Cross-attention at the bottleneck:
      Q  <- current frame's bottleneck features (spatially pooled to pool_size x pool_size)
      KV <- concatenated past frames from KVBank

    When the bank is empty (first frame of a sequence), passes features through unchanged.

    Args:
        ch        : bottleneck channel count (128 by default)
        num_heads : number of attention heads
        pool_size : spatial size after AdaptiveAvgPool2d before attention.
                    e.g. pool_size=8 gives 64 tokens regardless of input resolution,
                    keeping attention cost O(1) w.r.t. spatial resolution.
    """
    def __init__(self, ch: int = 128, num_heads: int = 4, pool_size: int = 8):
        super(BottleneckCrossAttn, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.to_q = nn.Linear(ch, ch, bias=False)
        self.to_k = nn.Linear(ch, ch, bias=False)
        self.to_v = nn.Linear(ch, ch, bias=False)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

    def _to_tokens(self, feat):
        """(N, C, H, W) -> (N, S, C)  where S = pool_size^2"""
        return self.pool(feat).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, bank: KVBank):
        """
        Args:
            x    : bottleneck feature map (N, C, H, W)
            bank : KVBank (may be empty on the first frame)
        Returns:
            x_out  : (N, C, H, W) — temporally enriched bottleneck features
            k_cur  : (N, S, C)    — current frame's key   (caller pushes to bank)
            v_cur  : (N, S, C)    — current frame's value (caller pushes to bank)
        """
        N, C, H, W = x.shape

        tokens = self._to_tokens(x)       # (N, S, C)
        q_cur  = self.to_q(tokens)
        k_cur  = self.to_k(tokens)
        v_cur  = self.to_v(tokens)

        bank_k, bank_v = bank.get()
        if bank_k is None:
            # First frame — no past context, pass through unchanged
            return x, k_cur, v_cur

        # Cross-attention: Q=current, KV=past frames
        attn_out, _ = self.attn(q_cur, bank_k, bank_v)   # (N, S, C)

        # Channel-wise gate applied to full-res feature map (residual add)
        gate  = self.gate(attn_out.mean(dim=1)).view(N, C, 1, 1)
        x_out = x * gate + x

        return x_out, k_cur, v_cur


# ---------------------------------------------------------------------------
# Denoising block
# ---------------------------------------------------------------------------

class DenBlock(nn.Module):
    """Definition of the denoising block of FastDVDnet (single-frame + KV bank)."""
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        # Encoder
        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)

        # Bottleneck KV-bank cross-attention
        self.kv_attn = BottleneckCrossAttn(ch=self.chs_lyr2,
                                           num_heads=num_heads,
                                           pool_size=pool_size)

        # Decoder
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, frame_t: torch.Tensor, noise_map: torch.Tensor, bank: KVBank):
        """
        Args:
            frame_t   : (N, 3, H, W)  noisy current frame in [0, 1]
            noise_map : (N, 1, H, W)  per-image noise std map
            bank      : KVBank        rolling buffer of past K/V tensors
        Returns:
            denoised  : (N, 3, H, W)
        """
        # Encoder
        x0 = self.inc(torch.cat((frame_t, noise_map), dim=1))  # (N, 32, H,   W  )
        x1 = self.downc0(x0)                                   # (N, 64, H/2, W/2)
        x2 = self.downc1(x1)                                   # (N,128, H/4, W/4)

        # Bottleneck cross-attention
        x2, k_cur, v_cur = self.kv_attn(x2, bank)

        # Push current frame's KV into bank (available to next frame)
        bank.push(k_cur, v_cur)

        # Decoder — plain residual skip connections (same as original)
        x2 = self.upc2(x2)          # (N, 64, H/2, W/2)
        x1 = self.upc1(x1 + x2)    # (N, 32, H,   W  )
        x  = self.outc(x0 + x1)    # (N,  3, H,   W  )

        # Residual denoising: predict noise residual, subtract from input
        return frame_t - x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FastDVDnet(nn.Module):
    """
    FastDVDnet with single-frame input and temporal KV bank.

    The model is stateless — KVBank is passed in at each forward call so the
    training loop fully controls temporal state (resets at sequence boundaries).
    """
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = 1
        self.temp = DenBlock(bank_size=bank_size, num_heads=num_heads, pool_size=pool_size)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, frame_t: torch.Tensor, noise_map: torch.Tensor, bank: KVBank):
        """
        Args:
            frame_t   : (N, 3, H, W)
            noise_map : (N, 1, H, W)
            bank      : KVBank (shared across all frames in a sequence)
        Returns:
            denoised  : (N, 3, H, W)
        """
        return self.temp(frame_t, noise_map, bank)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from torchinfo import summary

    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=4, pool_size=8)
    print(model)

    # Simulate a short sequence of 5 frames
    bank = KVBank(bank_size=bank_size)
    model.eval()
    with torch.no_grad():
        for t in range(5):
            frame     = torch.randn(1, 3, 96, 96)
            noise_map = torch.randn(1, 1, 96, 96)
            out = model(frame, noise_map, bank)
            print(f"t={t}  bank_len={len(bank)}  out={out.shape}")

    summary(
        model,
        input_data=(
            torch.randn(1, 3, 96, 96),
            torch.randn(1, 1, 96, 96),
            KVBank(bank_size=bank_size),
        ),
        col_names=["input_size", "output_size", "num_params"],
        depth=5,
    )
