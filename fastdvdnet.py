"""
FastDVDnet denoising algorithm — single-frame + KV bank variant.
"""
import torch
import torch.nn.functional as F
from models import KVBank


def temp_denoise(model, noisyframe, sigma_noise, bank):
	"""Encapsulates call to denoising model and handles padding.
	Expects noisyframe to be normalised in [0., 1.]
	"""
	# Make spatial dims a multiple of 4 (two stride-2 downsamples in the UNet)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2] % 4
	if expanded_h:
		expanded_h = 4 - expanded_h
	expanded_w = sh_im[-1] % 4
	if expanded_w:
		expanded_w = 4 - expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# Denoise — model updates bank internally
	out = torch.clamp(model(noisyframe, sigma_noise, bank), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out


def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal, bank_size=10):
	r"""Denoises a sequence of frames with FastDVDnet (single-frame + KV bank).

	Args:
		seq           : Tensor [numframes, C, H, W] -- noisy input frames in [0, 1]
		noise_std     : Tensor -- scalar noise std
		temp_psz      : kept for API compatibility (unused -- bank handles temporal context)
		model_temporal: FastDVDnet model instance
		bank_size     : KVBank capacity (default 10)
	Returns:
		denframes     : Tensor [numframes, C, H, W]
	"""
	numframes, C, H, W = seq.shape
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# Build noise map from noise std
	noise_map = noise_std.expand((1, 1, H, W))

	# Fresh bank per sequence
	bank = KVBank(bank_size=bank_size)

	for fridx in range(numframes):
		frame_t = seq[fridx].unsqueeze(0)   # (1, C, H, W)
		denframes[fridx] = temp_denoise(model_temporal, frame_t, noise_map, bank)

	torch.cuda.empty_cache()
	return denframes
