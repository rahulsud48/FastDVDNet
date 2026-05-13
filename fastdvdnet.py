"""
FastDVDnet denoising — YUV422 single-frame + KV bank variant.
"""
import torch
import torch.nn.functional as F
from models import KVBank, rgb_to_yuv422, yuv422_to_rgb


def temp_denoise(model, y, uv, sigma_noise, bank):
    """
    Handles padding and calls the model.
    Pads Y to multiple of 4; pads UV to multiple of 2 (half of Y's multiple).
    """
    _, _, H, W = y.shape

    # Y padding — multiple of 4 (two stride-2 downsamples)
    expanded_h = H % 4
    if expanded_h:
        expanded_h = 4 - expanded_h
    expanded_w = W % 4
    if expanded_w:
        expanded_w = 4 - expanded_w

    if expanded_h or expanded_w:
        y           = F.pad(y,           (0, expanded_w, 0, expanded_h),   mode='reflect')
        sigma_noise = F.pad(sigma_noise, (0, expanded_w, 0, expanded_h),   mode='reflect')
        # UV padding is half of Y padding
        uv          = F.pad(uv,          (0, expanded_w//2, 0, expanded_h//2), mode='reflect')

    den_y, den_uv = model(y, uv, sigma_noise, bank)

    # Remove padding from Y
    if expanded_h:
        den_y = den_y[:, :, :-expanded_h, :]
    if expanded_w:
        den_y = den_y[:, :, :, :-expanded_w]

    # Remove padding from UV (half amounts)
    if expanded_h:
        den_uv = den_uv[:, :, :-expanded_h//2, :] if expanded_h//2 > 0 else den_uv
    if expanded_w:
        den_uv = den_uv[:, :, :, :-expanded_w//2] if expanded_w//2 > 0 else den_uv

    return den_y, den_uv


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal, bank_size=10):
    """
    Denoises a sequence of RGB frames using YUV422 single-frame + KV bank.

    Internally:
      1. Converts each frame RGB -> YUV422
      2. Denoises Y with KV bank; UV passed through
      3. Converts denoised YUV422 -> RGB

    Args:
        seq           : Tensor [numframes, 3, H, W] RGB in [0, 1]
        noise_std     : scalar noise std Tensor
        temp_psz      : unused (kept for API compatibility)
        model_temporal: FastDVDnet instance
        bank_size     : KVBank capacity
    Returns:
        denframes     : Tensor [numframes, 3, H, W] RGB in [0, 1]
    """
    numframes, C, H, W = seq.shape
    denframes = torch.empty((numframes, 3, H, W)).to(seq.device)

    noise_map = noise_std.expand((1, 1, H, W))
    bank      = KVBank(bank_size=bank_size)

    for fridx in range(numframes):
        rgb   = seq[fridx].unsqueeze(0)          # (1, 3, H, W)
        y, uv = rgb_to_yuv422(rgb)               # (1,1,H,W), (1,2,H/2,W/2)

        den_y, den_uv = temp_denoise(
            model_temporal, y, uv, noise_map, bank
        )

        # Convert back to RGB
        denframes[fridx] = yuv422_to_rgb(den_y, den_uv).squeeze(0)

    torch.cuda.empty_cache()
    return denframes
