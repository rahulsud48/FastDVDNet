"""
FastDVDnet — Single-frame input with KV Bank cross-attention at bottleneck.

Architecture:
  • Each frame is encoded independently through the U-Net encoder.
  • At the bottleneck, a cross-attention layer uses the current frame's features
    as Q and retrieves context from a rolling KV bank of up to `bank_size` past
    bottleneck feature maps.
  • The decoder then reconstructs the denoised frame as usual.

Key design choices:
  • Spatial pooling before attention keeps cost manageable at high resolutions
    (e.g. 4K → H/4 × W/4 at bottleneck → pooled to pool_size × pool_size for attn).
  • KV bank is a simple ring-buffer (KVBank) managed outside the model so that
    the training loop controls when to reset (scene cuts, start of sequence, etc.).
  • Bank is updated AFTER attention so the current frame's KV is available for
    the NEXT frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks (depthwise-separable)
# ---------------------------------------------------------------------------

class DSConv(nn.Module):
    """Depthwise 3×3 + Pointwise 1×1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


class CvBlock(nn.Module):
    """(DSConv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            DSConv(out_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """First encoder block — accepts a single frame + noise map (4 channels)."""
    def __init__(self, out_ch):
        super().__init__()
        self.interm_ch = 30
        # Input: 3 (RGB) + 1 (noise map) = 4 channels
        self.convblock = nn.Sequential(
            nn.Conv2d(4, self.interm_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.interm_ch), nn.ReLU(inplace=True),
            DSConv(self.interm_ch, out_ch),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Stride-2 DSConv → CvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch, stride=2),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """CvBlock → DSConv → PixelShuffle ×2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            DSConv(in_ch, out_ch * 4),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """DSConv → BN → ReLU → DSConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, in_ch), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            DSConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)


class CBAM(nn.Module):
    def __init__(self, ch, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False), nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel_attn(x)
        sa = self.spatial_attn(torch.cat([x.mean(1, keepdim=True),
                                           x.max(1, keepdim=True).values], dim=1))
        return x * sa


# ---------------------------------------------------------------------------
# KV Bank  (ring buffer, lives outside the model)
# ---------------------------------------------------------------------------

class KVBank:
    """
    Rolling buffer that stores (key, value) tensors from past bottleneck frames.

    Usage:
        bank = KVBank(bank_size=10)
        bank.reset()                      # call at sequence start / scene cut
        kv = bank.get()                   # returns list of (k, v) or [] if empty
        bank.push(k, v)                   # store current frame's KV after attention
    
    Tensors are kept on the same device as the model — no explicit .to() needed
    because we store whatever device the tensors arrive on.
    """

    def __init__(self, bank_size: int = 10):
        self.bank_size = bank_size
        self._keys: list = []    # list of tensors (N, S, C)
        self._values: list = []  # list of tensors (N, S, C)

    def reset(self):
        self._keys.clear()
        self._values.clear()

    def push(self, k: torch.Tensor, v: torch.Tensor):
        """k, v: (N, S, C) — spatially pooled & projected bottleneck features."""
        self._keys.append(k.detach())
        self._values.append(v.detach())
        if len(self._keys) > self.bank_size:
            self._keys.pop(0)
            self._values.pop(0)

    def get(self):
        """Returns (keys, values) each of shape (N, T*S, C), or None if empty."""
        if not self._keys:
            return None, None
        keys = torch.cat(self._keys, dim=1)    # (N, T*S, C)
        values = torch.cat(self._values, dim=1)
        return keys, values

    def __len__(self):
        return len(self._keys)


# ---------------------------------------------------------------------------
# Bottleneck cross-attention with KV bank
# ---------------------------------------------------------------------------

class BottleneckCrossAttn(nn.Module):
    """
    Cross-attention at the bottleneck:
      Q  ← current frame's bottleneck features  (spatially pooled)
      KV ← concatenated past frames from KVBank (spatially pooled)

    When the bank is empty (first frame of a sequence) we fall through with
    a simple identity (no temporal context yet).

    Args:
        ch        : number of bottleneck channels (default 128)
        num_heads : attention heads
        pool_size : spatial size after AdaptiveAvgPool2d before attention
                    (reduces H/4 × W/4 → pool_size × pool_size)
    """

    def __init__(self, ch: int = 128, num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        self.ch = ch
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.to_q  = nn.Linear(ch, ch, bias=False)
        self.to_k  = nn.Linear(ch, ch, bias=False)
        self.to_v  = nn.Linear(ch, ch, bias=False)

        self.attn  = nn.MultiheadAttention(ch, num_heads, batch_first=True)

        # Gate: blend attended context into the full-res bottleneck feature map
        self.gate  = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

        self.pool_size = pool_size

    def _pool_to_tokens(self, feat):
        """(N, C, H, W) → (N, S, C) where S = pool_size²"""
        return self.pool(feat).flatten(2).transpose(1, 2)   # (N, S, C)

    def forward(self, x: torch.Tensor, bank: KVBank):
        """
        Args:
            x    : bottleneck feature map  (N, C, H, W)
            bank : KVBank instance (may be empty for the first frame)
        Returns:
            x_out : (N, C, H, W) — temporally enriched bottleneck features
            k, v  : (N, S, C)   — current frame's projected K and V
                                   (caller should push these to the bank)
        """
        N, C, H, W = x.shape

        # ── Project current frame ──────────────────────────────────────────
        q_tokens = self._pool_to_tokens(x)          # (N, S, C)
        k_cur    = self.to_k(q_tokens)              # (N, S, C)
        v_cur    = self.to_v(q_tokens)              # (N, S, C)
        q_cur    = self.to_q(q_tokens)              # (N, S, C)

        # ── If bank is empty, skip cross-attention ─────────────────────────
        bank_k, bank_v = bank.get()
        if bank_k is None:
            # No past context; return features unchanged
            return x, k_cur, v_cur

        # ── Cross-attention: Q=current, KV=past ───────────────────────────
        attn_out, _ = self.attn(q_cur, bank_k, bank_v)  # (N, S, C)

        # ── Gate: modulate full-res feature map ───────────────────────────
        gate = self.gate(attn_out.mean(dim=1))      # (N, C)
        gate = gate.view(N, C, 1, 1)               # broadcast over spatial dims
        x_out = x * gate + x                       # residual gating

        return x_out, k_cur, v_cur


# ---------------------------------------------------------------------------
# Main denoising block (single-frame input)
# ---------------------------------------------------------------------------

class DenBlock(nn.Module):
    """
    U-Net denoising block with:
      • Single-frame input  (frame_t + noise_map → 4 channels)
      • KV bank cross-attention at the bottleneck
      • CBAM on skip connections
    """
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        # ── Encoder ───────────────────────────────────────────────────────
        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(self.chs_lyr0, self.chs_lyr1)
        self.downc1 = DownBlock(self.chs_lyr1, self.chs_lyr2)

        # ── Bottleneck cross-attention ────────────────────────────────────
        self.kv_attn = BottleneckCrossAttn(
            ch=self.chs_lyr2,
            num_heads=num_heads,
            pool_size=pool_size,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.upc2 = UpBlock(self.chs_lyr2, self.chs_lyr1)
        self.upc1 = UpBlock(self.chs_lyr1, self.chs_lyr0)
        self.outc = OutputCvBlock(self.chs_lyr0, 3)

        # ── Skip-connection attention (CBAM) ──────────────────────────────
        self.cbam1 = CBAM(self.chs_lyr1)
        self.cbam0 = CBAM(self.chs_lyr0)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
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
        # ── Encoder ───────────────────────────────────────────────────────
        x0 = self.inc(torch.cat([frame_t, noise_map], dim=1))  # (N, 32, H,   W  )
        x1 = self.downc0(x0)                                   # (N, 64, H/2, W/2)
        x2 = self.downc1(x1)                                   # (N,128, H/4, W/4)

        # ── Bottleneck cross-attention ────────────────────────────────────
        x2, k_cur, v_cur = self.kv_attn(x2, bank)

        # Update bank with current frame's KV (available for next frame)
        bank.push(k_cur, v_cur)

        # ── Decoder ───────────────────────────────────────────────────────
        x2 = self.upc2(x2)                                     # (N, 64, H/2, W/2)
        x1 = self.upc1(self.cbam1(x1) + x2)                   # (N, 32, H,   W  )
        x  = self.outc(self.cbam0(x0) + x1)                   # (N,  3, H,   W  )

        # Residual learning: predict noise, subtract from input
        return frame_t - x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FastDVDnet(nn.Module):
    """
    Single-frame-input FastDVDnet with temporal KV bank.

    The model itself is stateless — the KVBank is passed in at each forward
    call so the training loop fully controls temporal state (resets, batching).
    """
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        self.bank_size = bank_size
        self.temp = DenBlock(bank_size=bank_size, num_heads=num_heads, pool_size=pool_size)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, frame_t: torch.Tensor, noise_map: torch.Tensor, bank: KVBank):
        """
        Args:
            frame_t   : (N, 3, H, W)
            noise_map : (N, 1, H, W)
            bank      : KVBank  (shared across frames in a sequence)
        Returns:
            denoised  : (N, 3, H, W)
        """
        return self.temp(frame_t, noise_map, bank)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torchinfo import summary

    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=4, pool_size=8)
    print(model)

    # Simulate a short sequence
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
