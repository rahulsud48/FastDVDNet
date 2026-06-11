"""
FastDVDnet denoising — single-frame + KV bank + dual noise maps.

CHANGED_TO: memory-optimized for full-HD inference (bank kept hardcoded).
  - denframes kept on CPU; each denoised frame moved off-GPU immediately.
  - one input frame moved to GPU at a time (seq stays on CPU).
  - intermediates deleted each iteration.
"""
import torch
import torch.nn.functional as F


def temp_denoise(model, noisyframe, sigma_read_map, lambda_shot_map, bank_k, bank_v):
    """Handles padding, calls model, strips padding. Returns (out, curr_k, curr_v)."""
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
    input_data = torch.cat((noisyframe, sigma_read_map, lambda_shot_map), dim = 1)

    res, curr_k, curr_v = model(input_data, bank_k, bank_v)
    out = torch.clamp(noisyframe - res, 0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]
    return out, curr_k, curr_v


def denoise_seq_fastdvdnet(seq, noise_std, lambda_shot, temp_psz,
                           model_temporal, bank_size=10):
    """
    Denoises a sequence frame-by-frame with a fresh KV bank per sequence.

    Args:
        seq          : (T, C, H, W) noisy RGB in [0,1] — kept on CPU; each
                       frame is moved to GPU one at a time.
        noise_std    : scalar Tensor — AWGN std (sigma_read)
        lambda_shot  : scalar float  — Poisson lambda scale
        temp_psz     : unused (API compat)
        model_temporal: FastDVDnet
        bank_size    : KVBank capacity
    Returns:
        denframes    : (T, C, H, W) on CPU
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, C, H, W), dtype=torch.float32, device='cpu')

    # sigma_read map: flat scalar broadcast — AWGN is spatially uniform
    sigma_read_map = noise_std.expand((1, 1, H, W))

    bank_k = torch.zeros(1,10*64,64).cuda()
    bank_v = torch.zeros(1,10*64,64).cuda()

    for fridx in range(numframes):
        # CHANGED_TO: move ONE frame to GPU at a time (seq stays on CPU).
        frame_t = seq[fridx].unsqueeze(0).cuda(non_blocking=True)   # (1, C, H, W)

        # lambda_shot map: spatially varying — proportional to pixel intensity
        lambda_shot_map = (frame_t.mean(dim=1, keepdim=True) * lambda_shot)\
                          .clamp(1e-6, 1.0)

        out, curr_k, curr_v = temp_denoise(
            model_temporal, frame_t,
            sigma_read_map, lambda_shot_map, bank_k, bank_v
        )
        bank_k = torch.cat([bank_k[:, 64:, :], curr_k], dim=1)
        bank_v = torch.cat([bank_v[:, 64:, :], curr_v], dim=1)

        # CHANGED_TO: move result to CPU immediately, free the GPU tensors.
        denframes[fridx] = out.squeeze(0).cpu()
        del frame_t, out, curr_k, curr_v, lambda_shot_map

    torch.cuda.empty_cache()
    return denframes
