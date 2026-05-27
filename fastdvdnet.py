"""
FastDVDnet denoising — Y-channel 3-frame input (t-1, t, t+1), no UV in network.

Pipeline per frame t:
  1. Extract Y frames for [t-1, t, t+1] from the noisy sequence
  2. model(y_frames, noise_map) -> y_pred  (denoised central Y)
  3. RGB reconstruction for PSNR logging: yuv422_to_rgb(y_pred, uv_clean_t)
     where uv_clean_t is derived from the clean (noiseless) central frame.

Boundary frames (t=0 and t=T-1) use frame replication:
  t=0   -> window is [0, 0, 1]
  t=T-1 -> window is [T-2, T-1, T-1]
"""

import torch
import torch.nn.functional as F
from models import rgb_to_yuv422, yuv422_to_rgb


def temp_denoise(model, y_frames, noise_map):
    """
    Pads y_frames and noise_map to multiples of 4 (two stride-2 downsamples),
    runs the model, then removes padding from y_pred.

    Args:
        model     : FastDVDnet instance
        y_frames  : (N, 3, H, W)  — noisy Y for [t-1, t, t+1]
        noise_map : (N, 1, H, W)

    Returns:
        y_pred : (N, 1, H, W)  — denoised central Y
    """
    _, _, H, W = y_frames.shape

    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4

    if pad_h or pad_w:
        y_frames  = F.pad(y_frames,  (0, pad_w, 0, pad_h), mode='reflect')
        noise_map = F.pad(noise_map, (0, pad_w, 0, pad_h), mode='reflect')

    y_pred = model(y_frames, noise_map)

    if pad_h:
        y_pred = y_pred[:, :, :-pad_h, :]
    if pad_w:
        y_pred = y_pred[:, :, :, :-pad_w]

    return y_pred


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal, bank_size=None):
    """
    Denoises a sequence of RGB frames using Y-channel 3-frame sliding window.

    For each frame t:
      - Converts [t-1, t, t+1] RGB frames to Y (noisy) and UV (clean, from seq directly)
      - Denoises Y with the model
      - Reconstructs RGB using denoised Y + clean UV of central frame

    Args:
        seq           : (T, 3, H, W) RGB float32 in [0, 1] — NOISY input
        noise_std     : scalar Tensor — noise std
        temp_psz      : unused, kept for API compatibility
        model_temporal: FastDVDnet instance
        bank_size     : unused, kept for API compatibility

    Returns:
        denframes : (T, 3, H, W) RGB float32 in [0, 1]
    """
    T, C, H, W = seq.shape
    denframes  = torch.empty((T, 3, H, W), device=seq.device)
    noise_map  = noise_std.expand((1, 1, H, W))

    for t in range(T):
        # Boundary replication
        t_prev = max(t - 1, 0)
        t_next = min(t + 1, T - 1)

        # Convert each of the 3 RGB frames to Y (noisy)
        y_prev, _    = rgb_to_yuv422(seq[t_prev].unsqueeze(0))   # (1,1,H,W)
        y_curr, uv_t = rgb_to_yuv422(seq[t].unsqueeze(0))        # (1,1,H,W), (1,2,H/2,W/2)
        y_next, _    = rgb_to_yuv422(seq[t_next].unsqueeze(0))   # (1,1,H,W)

        y_frames = torch.cat([y_prev, y_curr, y_next], dim=1)    # (1, 3, H, W)

        y_pred = temp_denoise(model_temporal, y_frames, noise_map)  # (1, 1, H, W)

        # RGB reconstruction: denoised Y + clean UV of central frame
        # UV comes from the noisy frame (clean UV assumption — chroma noise is low energy)
        denframes[t] = yuv422_to_rgb(y_pred, uv_t).squeeze(0)    # (3, H, W)

    torch.cuda.empty_cache()
    return denframes
