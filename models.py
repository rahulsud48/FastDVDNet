"""
FastDVDnet — Single-frame RGB input with KV Bank cross-attention at bottleneck.

Key change from previous versions:
  - DenBlock.forward() now returns BOTH the predicted residual (noise) AND
    the denoised image, so the training loop can supervise directly on the
    residual for better texture preservation.
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """First encoder block — RGB frame + noise map: (N, 4, H, W)."""
    def __init__(self, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(4, self.interm_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
    """Conv2d 3x3 => BN => ReLU => Conv2d 3x3"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.convblock(x)


# ---------------------------------------------------------------------------
# KV Bank
# ---------------------------------------------------------------------------

class KVBank:
    """
    Rolling ring-buffer storing (key, value) bottleneck tokens from past frames.

    Each entry: (N, S, C) where S = pool_size^2.
    bank.get() returns (N, T*S, C) across T stored frames.

    detach=True  -> inference (no grad needed)
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
    Cross-attention at the bottleneck:
      Q  <- current frame tokens (AdaptiveAvgPool -> flatten)
      KV <- past T frames from KVBank

    First frame falls through unchanged (empty bank).
    """
    def __init__(self, ch: int = 128, num_heads: int = 4, pool_size: int = 8):
        super(BottleneckCrossAttn, self).__init__()
        self.pool  = nn.AdaptiveAvgPool2d(pool_size)
        self.to_q  = nn.Linear(ch, ch, bias=False)
        self.to_k  = nn.Linear(ch, ch, bias=False)
        self.to_v  = nn.Linear(ch, ch, bias=False)
        self.attn  = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.gate  = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

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
# Denoising block
# ---------------------------------------------------------------------------

class DenBlock(nn.Module):
    """
    FastDVDnet denoising block (single RGB frame + KV bank).

    forward() returns:
        predicted_residual : (N, 3, H, W) — raw network output (predicted noise)
        denoised           : (N, 3, H, W) — frame_t - predicted_residual, clamped

    Returning both allows the training loop to supervise directly on the
    residual (noise), which prevents texture being treated as noise.
    """

    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(self.chs_lyr0, self.chs_lyr1)
        self.downc1 = DownBlock(self.chs_lyr1, self.chs_lyr2)

        self.kv_attn = BottleneckCrossAttn(
            ch=self.chs_lyr2, num_heads=num_heads, pool_size=pool_size
        )

        self.upc2 = UpBlock(self.chs_lyr2, self.chs_lyr1)
        self.upc1 = UpBlock(self.chs_lyr1, self.chs_lyr0)
        self.outc = OutputCvBlock(self.chs_lyr0, 3)

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
            frame_t   : (N, 3, H, W)  noisy RGB frame in [0, 1]
            noise_map : (N, 1, H, W)  noise std map
            bank      : KVBank

        Returns:
            predicted_residual : (N, 3, H, W)  — predicted noise (raw net output)
            denoised           : (N, 3, H, W)  — frame_t - residual, clamped [0,1]
        """
        x0 = self.inc(torch.cat((frame_t, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        x2, k_cur, v_cur = self.kv_attn(x2, bank)
        bank.push(k_cur, v_cur)

        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        predicted_residual = self.outc(x0 + x1)   # raw network output = predicted noise

        denoised = (frame_t - predicted_residual).clamp(0., 1.)

        return predicted_residual, denoised


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FastDVDnet(nn.Module):
    """
    FastDVDnet with single RGB frame input and temporal KV bank.
    Returns (predicted_residual, denoised) — both (N, 3, H, W).
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
        return self.temp(frame_t, noise_map, bank)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=4, pool_size=8)
    bank  = KVBank(bank_size=bank_size)
    model.eval()
    with torch.no_grad():
        for t in range(5):
            frame     = torch.rand(1, 3, 96, 96)
            noise_map = torch.zeros(1, 1, 96, 96)
            residual, denoised = model(frame, noise_map, bank)
            print(f"t={t}  bank={len(bank)}  residual={residual.shape}  denoised={denoised.shape}")
    print("Sanity check passed!")
