"""
FastDVDnet denoising — single-frame + KV bank + dual noise maps.
"""
import torch
import torch.nn.functional as F
from models import KVBank


def temp_denoise(model, noisyframe, sigma_read_map, lambda_shot_map, bank):
    """Handles padding, calls model, strips padding."""
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

    out = torch.clamp(model(noisyframe, sigma_read_map, lambda_shot_map, bank), 0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]
    return out


def denoise_seq_fastdvdnet(seq, noise_std, lambda_shot, temp_psz,
                           model_temporal, bank_size=10):
    """
    Denoises a sequence frame-by-frame with a fresh KVBank per sequence.

    Args:
        seq          : (T, C, H, W) noisy RGB in [0,1]
        noise_std    : scalar Tensor — AWGN std (sigma_read)
        lambda_shot  : scalar float  — Poisson lambda scale
        temp_psz     : unused (API compat)
        model_temporal: FastDVDnet
        bank_size    : KVBank capacity
    Returns:
        denframes    : (T, C, H, W)
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    # sigma_read map: flat scalar broadcast — AWGN is spatially uniform
    sigma_read_map = noise_std.expand((1, 1, H, W))

    bank = KVBank(bank_size=bank_size)

    for fridx in range(numframes):
        frame_t = seq[fridx].unsqueeze(0)   # (1, C, H, W)

        # lambda_shot map: spatially varying — proportional to pixel intensity
        # sigma_shot(h,w) = sqrt(intensity(h,w) * lambda_shot)
        lambda_shot_map = (frame_t.mean(dim=1, keepdim=True) * lambda_shot)\
                          .clamp(1e-6, 1.0)

        denframes[fridx] = temp_denoise(
            model_temporal, frame_t,
            sigma_read_map, lambda_shot_map, bank
        )

    torch.cuda.empty_cache()
    return denframes
