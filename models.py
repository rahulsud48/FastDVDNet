"""
FastDVDnet — Y-channel 3-frame input (t-1, t, t+1), no UV in the network.

Architecture:
  Input : 3 noisy Y frames + noise map → (N, 6, H, W)  [y0,nm, y1,nm, y2,nm]
  Encoder:
    L0: InputCvBlock  → (N, 32,  H,   W  )
    L1: DownBlock     → (N, 64,  H/2, W/2)
    L2: DownBlock     → (N, 128, H/4, W/4)  ← bottleneck (plain CvBlock, no attention)
  Decoder:
    upc2: UpBlock(L2)        + skip L1 → (N, 64,  H/2, W/2)
    upc1: UpBlock(L1+L2)     + skip L0 → (N, 32,  H,   W  )
    outc: OutputCvBlock(L0+L1)         → (N, 1,   H,   W  )  ← Y residual

  y_pred = y_noisy_t - residual  (unclamped, clamp happens outside after loss)

UV stays clean throughout and is never touched by the model.
Loss is computed in Y domain: L(y_pred, y_clean_t).
RGB reconstruction (for PSNR logging only) uses YUV444 — full spatial resolution,
no chroma subsampling, lossless roundtrip.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basic building blocks  (unchanged from original)
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
    First encoder block.
    Input: 3 noisy Y frames + 1 noise map = 4 channels  (N, 4, H, W)
    Each frame gets the same noise map, matching models_old.py convention.
    """
    def __init__(self, num_in_frames: int = 3, out_ch: int = 32):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        in_ch = num_in_frames * (1 + 1)   # 1 Y + 1 noise map per frame = 2 * 3 = 6
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, num_in_frames * self.interm_ch,
                      kernel_size=3, padding=1,
                      groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Stride-2 conv => BN => ReLU => CvBlock"""
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
    """Conv => BN => ReLU => Conv  (outputs 1-channel Y residual)"""
    def __init__(self, in_ch: int):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.convblock(x)


# ---------------------------------------------------------------------------
# Denoising block — Y-only, 3-frame input
# ---------------------------------------------------------------------------

class DenBlock(nn.Module):
    """
    FastDVDnet denoising block.

    Input:
      y_frames  : (N, 3, H, W)  — noisy Y at t-1, t, t+1
      noise_map : (N, 1, H, W)  — noise std map (same std broadcast to all frames)

    The noise map is replicated per-frame and interleaved before InputCvBlock,
    matching the models_old.py grouped-conv convention:
        cat([y_{t-1}, nm, y_t, nm, y_{t+1}, nm])  → (N, 6, H, W)

    Output:
      residual  : (N, 1, H, W)
      y_pred    : y_noisy_t - residual  (central frame denoised)
    """

    def __init__(self, num_input_frames: int = 3):
        super(DenBlock, self).__init__()
        self.num_input_frames = num_input_frames
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc    = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(self.chs_lyr0, self.chs_lyr1)
        self.downc1 = DownBlock(self.chs_lyr1, self.chs_lyr2)   # bottleneck

        self.upc2   = UpBlock(self.chs_lyr2, self.chs_lyr1)
        self.upc1   = UpBlock(self.chs_lyr1, self.chs_lyr0)
        self.outc   = OutputCvBlock(self.chs_lyr0)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, y_frames: torch.Tensor, noise_map: torch.Tensor):
        """
        Args:
            y_frames  : (N, 3, H, W)  — noisy Y for frames t-1, t, t+1
            noise_map : (N, 1, H, W)  — noise std map

        Returns:
            y_pred : (N, 1, H, W)  — denoised central Y frame
        """
        # Interleave noise map with each Y frame: (y0, nm, y1, nm, y2, nm) → (N,6,H,W)
        chunks = []
        for i in range(self.num_input_frames):
            chunks.append(y_frames[:, i:i+1, :, :])
            chunks.append(noise_map)
        inp = torch.cat(chunks, dim=1)   # (N, 6, H, W)

        # Encoder
        x0 = self.inc(inp)       # (N,  32, H,   W  )
        x1 = self.downc0(x0)     # (N,  64, H/2, W/2)
        x2 = self.downc1(x1)     # (N, 128, H/4, W/4)

        # Decoder with skip connections
        x2 = self.upc2(x2)          # (N,  64, H/2, W/2)
        x1 = self.upc1(x1 + x2)    # (N,  32, H,   W  )
        residual = self.outc(x0 + x1)  # (N, 1, H, W)

        y_center = y_frames[:, 1:2, :, :]          # central frame (t)
        y_pred   = (y_center - residual)#.clamp(0., 1.)
        return y_pred


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class FastDVDnet(nn.Module):
    """
    FastDVDnet: 3-frame Y-only denoiser.

    forward(y_frames, noise_map) → y_pred
    UV handling and RGB reconstruction are done outside the model.
    """

    def __init__(self, num_input_frames: int = 3):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = DenBlock(num_input_frames=num_input_frames)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, y_frames: torch.Tensor, noise_map: torch.Tensor):
        """
        Args:
            y_frames  : (N, 3, H, W)  — noisy Y frames [t-1, t, t+1]
            noise_map : (N, 1, H, W)  — noise std map

        Returns:
            y_pred : (N, 1, H, W)  — denoised Y (central frame)
        """
        return self.temp(y_frames, noise_map)


# ---------------------------------------------------------------------------
# YUV444 <-> RGB utilities
# Used outside the model for PSNR logging and image saving only.
# YUV444 keeps full spatial resolution for U and V — no chroma subsampling,
# lossless roundtrip (no avg_pool / interpolate), so RGB PSNR is meaningful.
# BT.601 coefficients.
# ---------------------------------------------------------------------------

def rgb_to_yuv444(rgb: torch.Tensor):
    """
    Converts an RGB tensor to YUV444 (full resolution, no subsampling).

    Args:
        rgb : (N, 3, H, W) float32 in [0, 1]

    Returns:
        y  : (N, 1, H, W)   luma
        uv : (N, 2, H, W)   chroma U and V at full resolution
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    y =  0.299  * r + 0.587  * g + 0.114  * b
    u = -0.169  * r - 0.331  * g + 0.500  * b + 0.5
    v =  0.500  * r - 0.419  * g - 0.081  * b + 0.5

    uv = torch.cat([u, v], dim=1)   # (N, 2, H, W) — full resolution, no pooling
    return y, uv


def yuv444_to_rgb(y: torch.Tensor, uv: torch.Tensor):
    """
    Converts YUV444 back to RGB.

    Args:
        y  : (N, 1, H, W)
        uv : (N, 2, H, W)   chroma at full resolution

    Returns:
        rgb : (N, 3, H, W) float32 in [0, 1]
    """
    u = uv[:, 0:1] - 0.5   # undo the +0.5 offset applied in rgb_to_yuv444
    v = uv[:, 1:2] - 0.5

    r = y + 1.402  * v
    g = y - 0.344  * u - 0.714 * v
    b = y + 1.772  * u

    return torch.cat([r, g, b], dim=1).clamp(0., 1.)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torchinfo import summary

    model = FastDVDnet(num_input_frames=3)
    print(model)

    model.eval()
    with torch.no_grad():
        seq_rgb   = torch.rand(5, 3, 96, 96)
        noise_std = 25. / 255.

        for t in range(1, 4):
            rgb_t     = seq_rgb[t-1:t+2]
            y_t, uv_t = zip(*[rgb_to_yuv444(rgb_t[i:i+1]) for i in range(3)])
            y_frames  = torch.cat(y_t, dim=1)            # (1, 3, 96, 96)
            noise_map = torch.full((1, 1, 96, 96), noise_std)
            y_pred    = model(y_frames, noise_map)        # (1, 1, 96, 96)

            uv_clean  = uv_t[1]                          # central frame UV (N,2,H,W)
            rgb_pred  = yuv444_to_rgb(y_pred.clamp(0., 1.), uv_clean)
            print(f"t={t}  y_pred={y_pred.shape}  rgb_pred={rgb_pred.shape}")

    print("\nSanity check passed!")
