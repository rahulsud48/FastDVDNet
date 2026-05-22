"""
FastDVDnet denoising — blind single RGB frame + KV bank.
No noise_map — model is blind to noise parameters.
"""
import torch
import torch.nn.functional as F
from models import KVBank


def temp_denoise(model, noisyframe, bank):
    """Handles padding, calls blind model, strips padding."""
    sh_im      = noisyframe.size()
    expanded_h = sh_im[-2] % 4
    if expanded_h:
        expanded_h = 4 - expanded_h
    expanded_w = sh_im[-1] % 4
    if expanded_w:
        expanded_w = 4 - expanded_w
    padexp     = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(noisyframe, padexp, mode='reflect')

    # Blind model — no noise_map argument
    _, out = model(noisyframe, bank)
    out = out.clamp(0., 1.)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]
    return out


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal, bank_size=10):
    """
    Denoises a sequence frame-by-frame. Blind — noise_std ignored by model
    (kept for API compatibility with test/validation code).
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)
    bank      = KVBank(bank_size=bank_size)

    for fridx in range(numframes):
        frame_t = seq[fridx].unsqueeze(0)
        denframes[fridx] = temp_denoise(model_temporal, frame_t, bank)

    torch.cuda.empty_cache()
    return denframes
