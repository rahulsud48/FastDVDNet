"""
FastDVDnet denoising — Y-channel 3-frame input (t-1, t, t+1), no UV in network.

Pipeline per frame t:
  1. Extract noisy Y frames for [t-1, t, t+1] via rgb_to_yuv444
  2. model(y_frames, noise_map) -> y_pred  (denoised central Y, unclamped)
  3. RGB reconstruction: yuv444_to_rgb(y_pred.clamp(0,1), uv_clean_t)
     UV is extracted from seq_clean at full spatial resolution (YUV444 —
     no chroma subsampling, lossless roundtrip, correct PSNR).

Boundary frames use replication:
  t=0   -> window [0, 0, 1]
  t=T-1 -> window [T-2, T-1, T-1]
"""

import torch
import torch.nn.functional as F
from models import rgb_to_yuv444, yuv444_to_rgb


def temp_denoise(model, y_frames, noise_map):
    """
    Pads y_frames and noise_map to multiples of 4 (two stride-2 downsamples),
    runs the model, then removes padding from y_pred.

    Args:
        model     : FastDVDnet instance
        y_frames  : (N, 3, H, W)  — noisy Y for [t-1, t, t+1]
        noise_map : (N, 1, H, W)

    Returns:
        y_pred : (N, 1, H, W)  — denoised central Y (unclamped)
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


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal,
                           seq_clean=None, bank_size=None):
    """
    Denoises a sequence of RGB frames using Y-channel 3-frame sliding window.

    For each frame t:
      - Extracts noisy Y from [t-1, t, t+1] via rgb_to_yuv444
      - Denoises Y with the model
      - Reconstructs RGB via yuv444_to_rgb(y_pred, uv_clean_t)
        where uv_clean_t comes from seq_clean at full resolution (no subsampling)

    Args:
        seq           : (T, 3, H, W) RGB float32 in [0, 1] — noisy input
        noise_std     : scalar Tensor — noise std
        temp_psz      : unused, kept for API compatibility
        model_temporal: FastDVDnet instance
        seq_clean     : (T, 3, H, W) clean RGB — UV extracted from here.
                        Pass during validation/test for correct PSNR.
                        If None, UV falls back to the noisy frame.
        bank_size     : unused, kept for API compatibility

    Returns:
        denframes : (T, 3, H, W) RGB float32 in [0, 1]
    """
    T, C, H, W = seq.shape
    denframes  = torch.empty((T, 3, H, W), device=seq.device)
    noise_map  = noise_std.expand((1, 1, H, W))

    # UV source: clean frames preferred; noisy frames as fallback
    uv_source = seq_clean if seq_clean is not None else seq

    for t in range(T):
        # Boundary replication
        t_prev = max(t - 1, 0)
        t_next = min(t + 1, T - 1)

        # Extract noisy Y from each of the 3 frames (UV discarded here)
        y_prev, _ = rgb_to_yuv444(seq[t_prev].unsqueeze(0))   # (1, 1, H, W)
        y_curr, _ = rgb_to_yuv444(seq[t].unsqueeze(0))        # (1, 1, H, W)
        y_next, _ = rgb_to_yuv444(seq[t_next].unsqueeze(0))   # (1, 1, H, W)

        y_frames = torch.cat([y_prev, y_curr, y_next], dim=1) # (1, 3, H, W)

        y_pred = temp_denoise(model_temporal, y_frames, noise_map)  # (1, 1, H, W)

        # UV from clean (or fallback noisy) central frame — full resolution (YUV444)
        _, uv_t = rgb_to_yuv444(uv_source[t].unsqueeze(0))    # (1, 2, H, W)

        # Clamp y_pred at output boundary (not inside model, so loss gets raw gradients)
        denframes[t] = yuv444_to_rgb(y_pred.clamp(0., 1.), uv_t).squeeze(0)

    torch.cuda.empty_cache()
    return denframes
