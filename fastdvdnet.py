"""
FastDVDnet denoising — single-frame + KV bank.

Two paths, selected by `yuv`:
  RGB path (yuv=False): model(input5ch, bank_k, bank_v) -> res, k, v
  YUV path (yuv=True) : model(y, uv_noise, bank_k, bank_v) -> y_res, uv_res, k, v
     uv_noise = cat[ UV(2), sigma_read_map(1), lambda_shot_map(1) ] at H/4 x W/4

Memory-optimized for full-HD inference: seq stays on CPU, one frame on GPU at
a time, results moved off-GPU immediately, bank kept hardcoded.
"""
import torch
import torch.nn.functional as F

from yuv_utils import rgb_to_yuv420, yuv420_to_rgb


# ---------------------------------------------------------------------------
# RGB path (unchanged)
# ---------------------------------------------------------------------------
def temp_denoise(model, noisyframe, sigma_read_map, lambda_shot_map, bank_k, bank_v):
    """RGB path. Handles padding, calls model, strips padding."""
    sh_im      = noisyframe.size()
    expanded_h = sh_im[-2] % 4
    if expanded_h:
        expanded_h = 4 - expanded_h
    expanded_w = sh_im[-1] % 4
    if expanded_w:
        expanded_w = 4 - expanded_w
    padexp = (0, expanded_w, 0, expanded_h)

    noisyframe      = F.pad(noisyframe,       padexp, mode='reflect')
    sigma_read_map  = F.pad(sigma_read_map,   padexp, mode='reflect')
    lambda_shot_map = F.pad(lambda_shot_map,  padexp, mode='reflect')
    input_data = torch.cat((noisyframe, sigma_read_map, lambda_shot_map), dim=1)

    res, curr_k, curr_v = model(input_data, bank_k, bank_v)
    out = torch.clamp(noisyframe - res, 0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]
    return out, curr_k, curr_v


# ---------------------------------------------------------------------------
# YUV path
# ---------------------------------------------------------------------------
def temp_denoise_yuv(model, y_noisy, uv_noisy, sigma, lam, bank_k, bank_v):
    """
    YUV path for a single frame.
      y_noisy  : (1,1,H,W)   full-res noisy Y in [0,1]
      uv_noisy : (1,2,H/4,W/4) noisy UV (with +0.5 offset)
      sigma    : float  read-noise std
      lam      : float  shot-noise scale
    Returns: (y_out(1,1,H,W), uv_out(1,2,H/4,W/4), curr_k, curr_v)
    """
    # Pad Y so H,W are multiples of 4 (so the /4 bottleneck is integer).
    sh = y_noisy.size()
    eh = sh[-2] % 4
    if eh:
        eh = 4 - eh
    ew = sh[-1] % 4
    if ew:
        ew = 4 - ew
    padexp = (0, ew, 0, eh)
    y_pad = F.pad(y_noisy, padexp, mode='reflect')

    Hq, Wq = y_pad.shape[-2] // 4, y_pad.shape[-1] // 4
    # UV is already quarter-res; pad it to match the (possibly padded) bottleneck.
    if uv_noisy.shape[-2] != Hq or uv_noisy.shape[-1] != Wq:
        uv_pad = F.interpolate(uv_noisy, size=(Hq, Wq), mode='bilinear', align_corners=False)
    else:
        uv_pad = uv_noisy

    # Noise maps at quarter res: sigma flat, lambda from quarter-res luma * lam.
    s_map = torch.full((1, 1, Hq, Wq), float(sigma), device=y_pad.device, dtype=y_pad.dtype)
    luma_q = F.avg_pool2d(y_pad, 4)                       # (1,1,Hq,Wq)
    l_map = (luma_q * float(lam)).clamp(1e-6, 1.0)
    uv_noise = torch.cat([uv_pad, s_map, l_map], dim=1)   # (1,4,Hq,Wq)

    y_res, uv_res, curr_k, curr_v = model(y_pad, uv_noise, bank_k, bank_v)
    y_out  = torch.clamp(y_pad - y_res, 0., 1.)
    uv_out = torch.clamp(uv_pad - uv_res, 0., 1.)

    # Strip Y padding (UV stays quarter-res; caller upsamples for RGB).
    if eh:
        y_out = y_out[:, :, :-eh, :]
    if ew:
        y_out = y_out[:, :, :, :-ew]
    return y_out, uv_out, curr_k, curr_v


# ---------------------------------------------------------------------------
# Sequence driver
# ---------------------------------------------------------------------------
def denoise_seq_fastdvdnet(seq, noise_std, lambda_shot, temp_psz,
                           model_temporal, bank_size=10, yuv=False):
    """
    Denoises a sequence frame-by-frame with a fresh KV bank per sequence.

    Args:
        seq          : (T, 3, H, W) noisy RGB in [0,1] on CPU
        noise_std    : scalar Tensor — AWGN std (sigma_read)
        lambda_shot  : scalar float  — Poisson lambda scale
        temp_psz     : unused (API compat)
        model_temporal: FastDVDnet
        bank_size    : KV bank capacity (hardcoded shapes below)
        yuv          : if True, run the YUV420 path and return denoised RGB
                       (model operates in YUV, output converted back to RGB).
    Returns:
        denframes    : (T, 3, H, W) denoised RGB on CPU
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, 3, H, W), dtype=torch.float32, device='cpu')

    sigma = float(noise_std.item()) if torch.is_tensor(noise_std) else float(noise_std)

    bank_k = torch.zeros(1, 10 * 64, 32).cuda()
    bank_v = torch.zeros(1, 10 * 64, 32).cuda()

    if not yuv:
        sigma_read_map = noise_std.expand((1, 1, H, W))

    for fridx in range(numframes):
        frame_t = seq[fridx].unsqueeze(0).cuda(non_blocking=True)   # (1,3,H,W) noisy RGB

        if yuv:
            # noisy RGB -> noisy YUV420 (UV at /4)
            y_n, uv_n = rgb_to_yuv420(frame_t, uv_down=4)
            y_out, uv_out, curr_k, curr_v = temp_denoise_yuv(
                model_temporal, y_n, uv_n, sigma, lambda_shot, bank_k, bank_v)
            out = yuv420_to_rgb(y_out, uv_out)              # back to RGB (1,3,H,W)
        else:
            lambda_shot_map = (frame_t.mean(dim=1, keepdim=True) * lambda_shot)\
                              .clamp(1e-6, 1.0)
            out, curr_k, curr_v = temp_denoise(
                model_temporal, frame_t, sigma_read_map, lambda_shot_map, bank_k, bank_v)

        bank_k = torch.cat([bank_k[:, 64:, :], curr_k], dim=1)
        bank_v = torch.cat([bank_v[:, 64:, :], curr_v], dim=1)

        denframes[fridx] = out.squeeze(0).cpu()
        del frame_t, out, curr_k, curr_v

    torch.cuda.empty_cache()
    return denframes
