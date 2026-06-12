"""
YUV420 helpers (BT.601, full-range [0,1]).

Shared by train and test so the conversion is IDENTICAL on both sides.

Convention (Option 1 — UV at quarter res to match the bottleneck):
  rgb_to_yuv420 : RGB(N,3,H,W) -> Y(N,1,H,W), UV(N,2,H/uv_down,W/uv_down)
                  UV downsampled by uv_down (default 4) via average pooling,
                  so it injects directly at the model's H/4 x W/4 bottleneck.
  yuv420_to_rgb : inverse; UV bilinearly upsampled to full res, then BT.601^-1.

U,V carry a +0.5 offset so they live in [0,1] (image-friendly).
"""
import torch
import torch.nn.functional as F

_M_RGB2YUV = torch.tensor([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312],
], dtype=torch.float32)

_M_YUV2RGB = torch.tensor([
    [1.0,  0.0,       1.402],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772,     0.0],
], dtype=torch.float32)


def rgb_to_yuv420(rgb, uv_down=4):
    """RGB(N,3,H,W) in [0,1] -> Y(N,1,H,W), UV(N,2,H/uv_down,W/uv_down)."""
    m = _M_RGB2YUV.to(rgb.device, rgb.dtype)
    yuv = torch.einsum('ij,njhw->nihw', m, rgb).clone()
    yuv[:, 1:3] = yuv[:, 1:3] + 0.5
    y = yuv[:, 0:1]
    uv = F.avg_pool2d(yuv[:, 1:3], kernel_size=uv_down, stride=uv_down)
    return y, uv


def yuv420_to_rgb(y, uv):
    """Y(N,1,H,W) + UV(N,2,h,w) -> RGB(N,3,H,W) in [0,1]. UV upsampled to full."""
    N, _, H, W = y.shape
    uv_full = F.interpolate(uv, size=(H, W), mode='bilinear', align_corners=False)
    yuv = torch.cat([y, uv_full - 0.5], dim=1)
    m = _M_YUV2RGB.to(y.device, y.dtype)
    rgb = torch.einsum('ij,njhw->nihw', m, yuv)
    return rgb.clamp(0.0, 1.0)
