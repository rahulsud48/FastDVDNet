"""
FastDVDnet denoising — YUV path + KV bank. Memory-optimized full-HD inference.
Bank dim is 32 (channels_layer2), slot 64 (pool 8x8). Matches models.py.
"""
import torch
import torch.nn.functional as F

from yuv_utils import rgb_to_yuv420, yuv420_to_rgb


def temp_denoise_yuv(model, y_noisy, uv_noisy, sigma, lam, bank_k, bank_v):
    """Single-frame YUV denoise. Returns (y_out, uv_out, k, v)."""
    sh = y_noisy.size()
    eh = (4 - sh[-2] % 4) % 4
    ew = (4 - sh[-1] % 4) % 4
    y_pad = F.pad(y_noisy, (0, ew, 0, eh), mode='reflect')

    Hq, Wq = y_pad.shape[-2] // 4, y_pad.shape[-1] // 4
    if uv_noisy.shape[-2] != Hq or uv_noisy.shape[-1] != Wq:
        uv_pad = F.interpolate(uv_noisy, size=(Hq, Wq), mode='bilinear', align_corners=False)
    else:
        uv_pad = uv_noisy

    s_map = torch.full((1, 1, Hq, Wq), float(sigma), device=y_pad.device, dtype=y_pad.dtype)
    l_map = (F.avg_pool2d(y_pad, 4) * float(lam)).clamp(1e-6, 1.0)
    uv_noise = torch.cat([uv_pad, s_map, l_map], dim=1)

    y_res, uv_res, k, v = model(y_pad, uv_noise, bank_k, bank_v)
    y_out = torch.clamp(y_pad - y_res, 0., 1.)
    uv_out = torch.clamp(uv_pad - uv_res, 0., 1.)

    if eh:
        y_out = y_out[:, :, :-eh, :]
    if ew:
        y_out = y_out[:, :, :, :-ew]
    return y_out, uv_out, k, v


def denoise_seq_fastdvdnet(seq, noise_std, lambda_shot, temp_psz,
                           model_temporal, bank_size=10, yuv=True, device='cuda'):
    """
    Denoise a noisy-RGB sequence (CPU) via the YUV path, return denoised RGB (CPU).
    One frame on GPU at a time; bank dim 32, slot 64.
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, 3, H, W), dtype=torch.float32, device='cpu')
    sigma = float(noise_std.item()) if torch.is_tensor(noise_std) else float(noise_std)

    bank_k = torch.zeros(1, 10 * 64, 32, device=device)
    bank_v = torch.zeros(1, 10 * 64, 32, device=device)

    for fridx in range(numframes):
        frame = seq[fridx].unsqueeze(0).to(device, non_blocking=True)
        y_n, uv_n = rgb_to_yuv420(frame, uv_down=4)
        y_out, uv_out, k, v = temp_denoise_yuv(
            model_temporal, y_n, uv_n, sigma, lambda_shot, bank_k, bank_v)
        out = yuv420_to_rgb(y_out, uv_out)
        bank_k = torch.cat([bank_k[:, 64:, :], k], dim=1)
        bank_v = torch.cat([bank_v[:, 64:, :], v], dim=1)
        denframes[fridx] = out.squeeze(0).cpu()
        del frame, out, k, v

    if device == 'cuda':
        torch.cuda.empty_cache()
    return denframes
