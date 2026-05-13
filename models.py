"""
FastDVDnet — YUV422 single-frame input with KV Bank cross-attention at bottleneck.

YUV422 architecture:
  - Input: Y channel (N, 1, H, W) + noise map (N, 1, H, W)  → 2 channels at full res
  - After first DownBlock (H/2, W/2): U and V channels (N, 2, H/2, W/2) are
    concatenated with the feature map — exactly matching their native YUV422 resolution
  - Decoder outputs Y channel (N, 1, H, W) only — U/V passed through with bilinear upsample
  - Final output: reconstructed RGB (N, 3, H, W) via YUV→RGB conversion

Benefits:
  - InputCvBlock and downc0 operate on 2ch instead of 4ch → ~2x cheaper at full resolution
  - U/V injected at H/2 where they naturally live in YUV422 — no artificial upsampling
  - KV bank at bottleneck sees Y+UV enriched features — same concept, richer representation
  - PSNR measured on RGB after YUV→RGB reconstruction for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """
    First encoder block for YUV422 input.
    Accepts Y channel + noise map only: (N, 2, H, W)
    U/V are injected later at H/2 in DownBlock0.
    """
    def __init__(self, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        # Y (1) + noise map (1) = 2 input channels
        self.convblock = nn.Sequential(
            nn.Conv2d(2, self.interm_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlockWithUV(nn.Module):
    """
    First downsampling block — injects U and V after stride-2 conv.

    Flow:
      x (N, in_ch, H, W)
        → stride-2 conv → (N, out_ch, H/2, W/2)
        → concat U, V   → (N, out_ch+2, H/2, W/2)
        → fusion conv   → (N, out_ch, H/2, W/2)
        → CvBlock       → (N, out_ch, H/2, W/2)

    U and V arrive at (N, 1, H/2, W/2) each — their native YUV422 resolution.
    No upsampling needed; they slot in exactly at the right spatial scale.
    """
    def __init__(self, in_ch, out_ch):
        super(DownBlockWithUV, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Fusion: merges downsampled features with U and V (2 extra channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch + 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cvblock = CvBlock(out_ch, out_ch)

    def forward(self, x, uv):
        """
        Args:
            x  : (N, in_ch, H, W)     — Y feature map at full res
            uv : (N, 2, H/2, W/2)     — U and V channels at half res
        Returns:
            out: (N, out_ch, H/2, W/2)
        """
        x = self.down(x)              # (N, out_ch, H/2, W/2)
        x = torch.cat([x, uv], dim=1) # (N, out_ch+2, H/2, W/2)
        x = self.fusion(x)            # (N, out_ch, H/2, W/2)
        return self.cvblock(x)


class DownBlock(nn.Module):
    """Standard downsampling block (used after UV injection)."""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
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
    """
    Final decoder block.
    Outputs Y channel only (N, 1, H, W) — U/V are handled separately.
    """
    def __init__(self, in_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=False),  # Y only
        )

    def forward(self, x):
        return self.convblock(x)


# ---------------------------------------------------------------------------
# KV Bank — ring buffer, lives outside the model
# ---------------------------------------------------------------------------

class KVBank:
    """
    Rolling buffer storing (key, value) tensors from past bottleneck frames.

    Each entry: (N, S, C) where S = pool_size^2.
    bank.get() returns (N, T*S, C) across T stored frames.

    detach=True  -> inference (no grad, saves memory)
    detach=False -> training  (gradients flow through temporal attention)
    """

    def __init__(self, bank_size: int = 10, detach: bool = True):
        self.bank_size = bank_size
        self.detach    = detach
        self._keys:   list = []
        self._values: list = []

    def reset(self):
        self._keys.clear()
        self._values.clear()

    def push(self, k: torch.Tensor, v: torch.Tensor):
        self._keys.append(k.detach() if self.detach else k)
        self._values.append(v.detach() if self.detach else v)
        if len(self._keys) > self.bank_size:
            self._keys.pop(0)
            self._values.pop(0)

    def get(self):
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
    Cross-attention at the bottleneck (unchanged from original design):
      Q  <- current frame bottleneck tokens (AdaptiveAvgPool -> flatten)
      KV <- past T frames from KVBank

    First frame falls through unchanged (empty bank).

    Args:
        ch        : bottleneck channels (128)
        num_heads : attention heads
        pool_size : output size of AdaptiveAvgPool2d (pool_size x pool_size tokens)
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
        """(N, C, H, W) -> (N, S, C)"""
        return self.pool(feat).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, bank: KVBank):
        N, C, H, W = x.shape

        tokens = self._to_tokens(x)
        q_cur  = self.to_q(tokens)
        k_cur  = self.to_k(tokens)
        v_cur  = self.to_v(tokens)

        bank_k, bank_v = bank.get()
        if bank_k is None:
            return x, k_cur, v_cur

        attn_out, _ = self.attn(q_cur, bank_k, bank_v)
        gate  = self.gate(attn_out.mean(dim=1)).view(N, C, 1, 1)
        x_out = x * gate + x

        return x_out, k_cur, v_cur


# ---------------------------------------------------------------------------
# Denoising block — YUV422 input
# ---------------------------------------------------------------------------

class DenBlock(nn.Module):
    """
    FastDVDnet denoising block with YUV422 input.

    Input flow:
      Y  : (N, 1, H, W)   — full resolution luma
      UV : (N, 2, H/2, W/2) — half-res chroma (YUV422 native)
      noise_map: (N, 1, H, W)

    Encoder:
      L0: InputCvBlock(Y + noise_map)          → (N, 32, H, W)
      L1: DownBlockWithUV(x0, UV)              → (N, 64, H/2, W/2)  ← UV injected here
      L2: DownBlock(x1)                        → (N, 128, H/4, W/4) ← bottleneck

    KV bank cross-attention at L2 (bottleneck).

    Decoder:
      upc2: UpBlock(x2)        + skip x1 → (N, 64, H/2, W/2)
      upc1: UpBlock(x1+x2)     + skip x0 → (N, 32, H, W)
      outc: OutputCvBlock(x0+x1)         → (N, 1, H, W)  ← Y residual

    Output:
      denoised_Y = Y - predicted_residual   (N, 1, H, W)
      denoised_UV = UV (passed through, optionally light denoising)
      → convert back to RGB in forward()
    """

    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        # Encoder
        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)           # Y + noise_map
        self.downc0 = DownBlockWithUV(self.chs_lyr0, self.chs_lyr1)  # injects UV at H/2
        self.downc1 = DownBlock(self.chs_lyr1, self.chs_lyr2)     # bottleneck

        # Bottleneck KV-bank cross-attention
        self.kv_attn = BottleneckCrossAttn(
            ch=self.chs_lyr2,
            num_heads=num_heads,
            pool_size=pool_size,
        )

        # Decoder
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0)  # outputs 1ch Y

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, y: torch.Tensor, uv: torch.Tensor,
                noise_map: torch.Tensor, bank: KVBank):
        """
        Args:
            y         : (N, 1, H, W)     — noisy Y channel in [0, 1]
            uv        : (N, 2, H/2, W/2) — noisy U, V channels in [0, 1]
            noise_map : (N, 1, H, W)     — noise std map for Y
            bank      : KVBank

        Returns:
            denoised_y  : (N, 1, H, W)
            denoised_uv : (N, 2, H/2, W/2)  — UV passed through (model focuses on Y)
        """
        # Encoder
        x0 = self.inc(torch.cat([y, noise_map], dim=1))  # (N, 32, H,   W  )
        x1 = self.downc0(x0, uv)                         # (N, 64, H/2, W/2)
        x2 = self.downc1(x1)                             # (N,128, H/4, W/4)

        # Bottleneck KV attention
        x2, k_cur, v_cur = self.kv_attn(x2, bank)
        bank.push(k_cur, v_cur)

        # Decoder
        x2 = self.upc2(x2)          # (N, 64, H/2, W/2)
        x1 = self.upc1(x1 + x2)    # (N, 32, H,   W  )
        x  = self.outc(x0 + x1)    # (N,  1, H,   W  ) — Y residual

        denoised_y  = (y - x).clamp(0., 1.)
        # UV: pass through (chroma noise is much lower energy, no separate denoiser needed)
        denoised_uv = uv.clamp(0., 1.)

        return denoised_y, denoised_uv


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FastDVDnet(nn.Module):
    """
    FastDVDnet with YUV422 single-frame input and temporal KV bank.

    forward() accepts Y and UV separately, returns denoised Y and UV.
    RGB conversion is handled outside the model (in fastdvdnet.py / training loop)
    so the model stays format-agnostic.
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

    def forward(self, y: torch.Tensor, uv: torch.Tensor,
                noise_map: torch.Tensor, bank: KVBank):
        """
        Args:
            y         : (N, 1, H, W)
            uv        : (N, 2, H/2, W/2)
            noise_map : (N, 1, H, W)
            bank      : KVBank
        Returns:
            denoised_y  : (N, 1, H, W)
            denoised_uv : (N, 2, H/2, W/2)
        """
        return self.temp(y, uv, noise_map, bank)


# ---------------------------------------------------------------------------
# YUV <-> RGB utilities
# ---------------------------------------------------------------------------

def rgb_to_yuv422(rgb: torch.Tensor):
    """
    Converts an RGB tensor to YUV422.

    Args:
        rgb : (N, 3, H, W) float32 in [0, 1]  — R, G, B channels

    Returns:
        y   : (N, 1, H, W)     float32 in [0, 1]
        uv  : (N, 2, H/2, W/2) float32 in [0, 1]  — U then V, downsampled 2x
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    # BT.601 coefficients (standard for video)
    y =  0.299 * r + 0.587 * g + 0.114 * b               # (N, 1, H, W)
    u = -0.169 * r - 0.331 * g + 0.500 * b + 0.5         # (N, 1, H, W) shifted to [0,1]
    v =  0.500 * r - 0.419 * g - 0.081 * b + 0.5         # (N, 1, H, W) shifted to [0,1]

    # Downsample U, V by 2x (YUV422: full H, half W; here we do H/2 W/2 for simplicity)
    u_ds = F.avg_pool2d(u, kernel_size=2, stride=2)       # (N, 1, H/2, W/2)
    v_ds = F.avg_pool2d(v, kernel_size=2, stride=2)       # (N, 1, H/2, W/2)

    uv = torch.cat([u_ds, v_ds], dim=1)                   # (N, 2, H/2, W/2)
    return y, uv


def yuv422_to_rgb(y: torch.Tensor, uv: torch.Tensor):
    """
    Converts YUV422 back to RGB.

    Args:
        y   : (N, 1, H, W)     float32 in [0, 1]
        uv  : (N, 2, H/2, W/2) float32 in [0, 1]

    Returns:
        rgb : (N, 3, H, W) float32 in [0, 1]
    """
    # Upsample U, V back to full resolution
    u = F.interpolate(uv[:, 0:1], scale_factor=2, mode='bilinear', align_corners=False)
    v = F.interpolate(uv[:, 1:2], scale_factor=2, mode='bilinear', align_corners=False)

    # Undo the [0,1] shift
    u = u - 0.5
    v = v - 0.5

    # BT.601 inverse
    r = y + 1.402  * v
    g = y - 0.344  * u - 0.714 * v
    b = y + 1.772  * u

    return torch.cat([r, g, b], dim=1).clamp(0., 1.)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torchinfo import summary

    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=4, pool_size=8)
    print(model)

    bank = KVBank(bank_size=bank_size)
    model.eval()
    with torch.no_grad():
        for t in range(5):
            rgb       = torch.rand(1, 3, 96, 96)
            y, uv     = rgb_to_yuv422(rgb)
            noise_map = torch.zeros(1, 1, 96, 96)
            den_y, den_uv = model(y, uv, noise_map, bank)
            den_rgb = yuv422_to_rgb(den_y, den_uv)
            print(f"t={t}  bank={len(bank)}  y={den_y.shape}  uv={den_uv.shape}  rgb={den_rgb.shape}")

    print("\nSanity check passed!")
