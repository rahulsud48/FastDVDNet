"""
FastDVDnet denoising — single RGB frame + KV bank.
model() returns (predicted_residual, denoised); this file uses denoised only.
"""
import torch
import torch.nn.functional as F
from models import KVBank


def temp_denoise(model, noisyframe, sigma_noise, bank):
    """Handles padding, calls model, strips padding. Returns denoised frame."""
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2] % 4
    if expanded_h:
        expanded_h = 4 - expanded_h
    expanded_w = sh_im[-1] % 4
    if expanded_w:
        expanded_w = 4 - expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe  = F.pad(noisyframe,  padexp, mode='reflect')
    sigma_noise = F.pad(sigma_noise, padexp, mode='reflect')

    # model returns (residual, denoised) — use denoised for inference
    _, out = model(noisyframe, sigma_noise, bank)
    out = out.clamp(0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]
    return out


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal, bank_size=10):
    """
    Denoises a sequence frame-by-frame with a fresh KVBank per sequence.

    Args:
        seq           : (numframes, C, H, W) noisy RGB in [0, 1]
        noise_std     : scalar noise std Tensor
        temp_psz      : unused (kept for API compatibility)
        model_temporal: FastDVDnet instance
        bank_size     : KVBank capacity
    Returns:
        denframes     : (numframes, C, H, W)
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)
    noise_map = noise_std.expand((1, 1, H, W))
    bank      = KVBank(bank_size=bank_size)

    for fridx in range(numframes):
        frame_t = seq[fridx].unsqueeze(0)
        denframes[fridx] = temp_denoise(model_temporal, frame_t, noise_map, bank)

    torch.cuda.empty_cache()
    return denframes
