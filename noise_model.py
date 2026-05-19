"""
noise_model.py

Physically-motivated synthetic noise pipeline for ISP video denoising training.

Models three real-world degradation sources:

1. SPATIAL NOISE (per-frame, independent across frames)
   a. Shot noise    — Poisson, proportional to signal intensity (photon counting)
   b. Read noise    — Gaussian, signal-independent (ADC + amplifier)
   c. Fixed Pattern — column/row offsets + PRNU, repeatable per sequence
   d. Spatially varying sigma map — smooth field, not uniform

2. TEMPORAL NOISE (correlated across frames)
   a. Correlated noise — partial noise persistence frame-to-frame (AR(1) process)
   b. Flicker          — periodic intensity modulation (50/60 Hz lighting)
   c. Thermal drift    — slow gain/offset change across a sequence

3. MOTION BLUR (optical, applied before noise)
   a. Camera shake  — global linear/curved motion kernel
   b. Object motion — per-region local blur (approximated via random local kernels)
   c. Rolling shutter — row-dependent horizontal shift (CMOS sensor model)

Usage:
    augmentor = SequenceNoiseAugmentor(
        noise_cfg=NoiseConfig(),
        temporal_mode='correlated',   # 'correlated' | 'independent' | 'random'
        spatial_mode='varying',        # 'uniform' | 'varying' | 'random'
    )
    noisy_frames, noise_maps = augmentor(clean_frames)
    # clean_frames : list of T tensors (N, C, H, W) in [0,1]
    # noisy_frames : same shape, degraded
    # noise_maps   : list of T tensors (N, 1, H, W) effective per-pixel sigma

The noise_map returned is the per-pixel effective sigma estimate — pass this
directly to the model as the noise conditioning input.
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration dataclass — all noise parameters in one place
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """
    All noise parameters with physically motivated defaults.

    Indoor low-light  : shot_scale=1.5, read_sigma=(0.02,0.05), fpn_strength=0.02
    Outdoor bright    : shot_scale=0.3, read_sigma=(0.005,0.02), fpn_strength=0.005
    Mixed (default)   : covers both regimes by sampling from wide ranges
    """

    # ── Shot noise (Poisson) ──────────────────────────────────────────────
    # Higher = more shot noise. Physical: inversely proportional to exposure/gain
    shot_scale_range: tuple = (0.2, 2.0)      # sampled per sequence

    # ── Read noise (Gaussian) ─────────────────────────────────────────────
    read_sigma_range: tuple = (0.005, 0.05)   # fraction of [0,1] range

    # ── Fixed pattern noise ───────────────────────────────────────────────
    fpn_strength_range: tuple  = (0.0, 0.02)  # column/row stripe amplitude
    prnu_strength_range: tuple = (0.0, 0.01)  # pixel response non-uniformity

    # ── Temporal correlation ──────────────────────────────────────────────
    # alpha=0 -> independent frames, alpha=0.9 -> strong persistence
    temporal_alpha_range: tuple = (0.3, 0.8)

    # ── Flicker ───────────────────────────────────────────────────────────
    flicker_prob: float         = 0.3         # probability of flicker in a sequence
    flicker_freq_range: tuple   = (40, 70)    # Hz (50/60 Hz mains + harmonics)
    flicker_amp_range: tuple    = (0.0, 0.03) # amplitude of intensity modulation

    # ── Thermal drift ─────────────────────────────────────────────────────
    drift_prob: float           = 0.2
    drift_gain_range: tuple     = (0.99, 1.01) # multiplicative gain drift
    drift_offset_range: tuple   = (-0.005, 0.005) # additive offset drift

    # ── Motion blur — camera shake ────────────────────────────────────────
    shake_prob: float           = 0.5
    shake_kernel_range: tuple   = (3, 15)     # kernel length in pixels (forced odd in apply_camera_shake)
    shake_angle_range: tuple    = (0, 180)    # degrees

    # ── Motion blur — object motion ───────────────────────────────────────
    obj_motion_prob: float      = 0.3
    obj_motion_kernel_range: tuple = (3, 9)
    obj_motion_num_regions: tuple  = (1, 4)   # number of locally-blurred regions

    # ── Rolling shutter ───────────────────────────────────────────────────
    rolling_shutter_prob: float  = 0.2
    rolling_shutter_max_shift: int = 3        # max pixel shift between rows


# ---------------------------------------------------------------------------
# Spatial noise helpers
# ---------------------------------------------------------------------------

def _smooth_field(N: int, H: int, W: int,
                  mean: float, std: float,
                  device: torch.device) -> torch.Tensor:
    """
    Generate a smooth spatially-varying scalar field (N, 1, H, W).
    Created by bilinearly upsampling a small random map — cheap and smooth.
    Used for spatially-varying sigma maps and PRNU.
    """
    small_h = max(H // 8, 4)
    small_w = max(W // 8, 4)
    raw     = torch.randn(N, 1, small_h, small_w, device=device) * std + mean
    return F.interpolate(raw, size=(H, W), mode='bilinear', align_corners=False)\
             .clamp(1e-6, 1.0)


def add_shot_noise(frame: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Poisson shot noise: variance proportional to signal.
    Approximated as Gaussian with std = sqrt(pixel * scale) / scale.
    True Poisson sampling via torch.poisson at integer photon counts.

    scale: inverse of photon count scaling. Higher = more noise.
    """
    photons = (frame * 255.0 / scale).clamp(1e-6)
    noisy   = torch.poisson(photons) * scale / 255.0
    return noisy.clamp(0., 1.)


def add_read_noise(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    """Signal-independent Gaussian read noise."""
    return (frame + torch.randn_like(frame) * sigma).clamp(0., 1.)


def make_fpn_map(N: int, C: int, H: int, W: int,
                 col_strength: float, row_strength: float,
                 prnu_strength: float,
                 device: torch.device) -> torch.Tensor:
    """
    Fixed Pattern Noise map (N, C, H, W) — same per sequence, independent per item in batch.

    Components:
      - Column stripes : (N, C, 1, W) — same value down each column
      - Row stripes    : (N, C, H, 1) — same value across each row
      - PRNU           : (N, C, H, W) — per-pixel multiplicative gain variation
    """
    col_noise = torch.randn(N, C, 1, W, device=device) * col_strength
    row_noise = torch.randn(N, C, H, 1, device=device) * row_strength
    prnu      = 1.0 + torch.randn(N, C, H, W, device=device) * prnu_strength
    return col_noise.expand(N, C, H, W) + row_noise.expand(N, C, H, W), prnu


def effective_sigma_map(frame: torch.Tensor,
                        shot_scale: float,
                        read_sigma: float,
                        spatial_mode: str = 'varying') -> torch.Tensor:
    """
    Compute per-pixel effective noise std map (N, 1, H, W).

    sigma_eff(x,y) = sqrt(sigma_shot²(x,y) + sigma_read²)

    where sigma_shot(x,y) = sqrt(pixel_intensity * shot_scale) / 255
    This is the heteroscedastic noise model used in real ISP calibration.
    """
    N, C, H, W = frame.shape
    device     = frame.device

    # Shot noise std: sqrt(intensity * scale) — average across channels
    luma        = frame.mean(dim=1, keepdim=True).clamp(1e-6)    # (N,1,H,W)
    sigma_shot  = (luma * shot_scale / 255.0).sqrt()

    # Read noise std: spatially uniform or smooth varying
    if spatial_mode == 'varying':
        sigma_read = _smooth_field(N, H, W, read_sigma, read_sigma * 0.3, device)
    else:
        sigma_read = torch.full((N, 1, H, W), read_sigma, device=device)

    # Combined effective sigma
    sigma_eff = (sigma_shot ** 2 + sigma_read ** 2).sqrt()
    return sigma_eff.clamp(1e-4, 1.0)


# ---------------------------------------------------------------------------
# Motion blur helpers
# ---------------------------------------------------------------------------

def _linear_motion_kernel(length: int, angle: float,
                           device: torch.device) -> torch.Tensor:
    """
    Build a 1D linear motion blur kernel of given length and angle.
    Returns (length, length) normalized kernel tensor.
    """
    kernel = torch.zeros(length, length, device=device)
    center = length // 2
    angle_rad = math.radians(angle)
    for i in range(length):
        offset = i - center
        x = center + round(offset * math.cos(angle_rad))
        y = center + round(offset * math.sin(angle_rad))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    s = kernel.sum()
    return kernel / s if s > 0 else kernel.fill_(0).fill_diagonal_(1.0 / length)


def apply_camera_shake(frame: torch.Tensor,
                       kernel_size: int,
                       angle: float) -> torch.Tensor:
    """Apply global linear motion blur (camera shake) to all channels."""
    # Force odd kernel so padding=kernel_size//2 exactly preserves H and W
    if kernel_size % 2 == 0:
        kernel_size += 1
    N, C, H, W = frame.shape
    device     = frame.device
    kernel     = _linear_motion_kernel(kernel_size, angle, device)
    k          = kernel.view(1, 1, kernel_size, kernel_size)\
                       .expand(C, 1, kernel_size, kernel_size)
    pad        = kernel_size // 2
    out = F.conv2d(frame, k, padding=pad, groups=C)
    # Defensive crop to exact input size (handles any remaining off-by-one)
    return out[:, :, :H, :W].clamp(0., 1.)


def apply_object_motion(frame: torch.Tensor,
                        num_regions: int,
                        kernel_size: int) -> torch.Tensor:
    """
    Apply local motion blur to random rectangular regions (moving objects).
    Each region gets an independently oriented blur kernel.
    """
    N, C, H, W = frame.shape
    device     = frame.device
    result     = frame.clone()

    # Force odd kernel size so padding = kernel_size//2 always preserves spatial dims
    if kernel_size % 2 == 0:
        kernel_size += 1

    for _ in range(num_regions):
        # Random region — ensure minimum size to avoid degenerate kernels
        rh    = max(random.randint(H // 8, H // 3), kernel_size + 2)
        rw    = max(random.randint(W // 8, W // 3), kernel_size + 2)
        rh    = min(rh, H)
        rw    = min(rw, W)
        top   = random.randint(0, H - rh)
        left  = random.randint(0, W - rw)
        angle = random.uniform(0, 180)

        kernel = _linear_motion_kernel(kernel_size, angle, device)
        k      = kernel.view(1, 1, kernel_size, kernel_size)\
                       .expand(C, 1, kernel_size, kernel_size)
        pad    = kernel_size // 2   # safe: odd kernel -> output same size as input

        region  = frame[:, :, top:top+rh, left:left+rw]   # (N, C, rh, rw)
        blurred = F.conv2d(region, k, padding=pad, groups=C)  # (N, C, rh, rw)

        # Guarantee exact size match (defensive, should already match with odd kernel)
        bh, bw = blurred.shape[2], blurred.shape[3]
        rh2, rw2 = min(rh, bh), min(rw, bw)

        # Smooth blend mask — use a proper feathered gradient mask
        # that fades from 1 in the centre to 0 at the edges
        # This completely eliminates visible bounding box artefacts
        margin = max(8, min(rh2 // 6, rw2 // 6, 24))  # feather width in pixels

        # Build 1D fade ramps for H and W
        ramp_h = torch.ones(rh2, device=device)
        ramp_w = torch.ones(rw2, device=device)
        for m in range(margin):
            alpha = m / margin                  # 0 at edge -> 1 at margin
            ramp_h[m]        = alpha
            ramp_h[rh2-1-m]  = alpha
            ramp_w[m]        = alpha
            ramp_w[rw2-1-m]  = alpha

        # 2D mask = outer product of H and W ramps
        mask2d = ramp_h.unsqueeze(1) * ramp_w.unsqueeze(0)  # (rh2, rw2)
        mask   = mask2d.view(1, 1, rh2, rw2).expand(N, C, rh2, rw2)

        result[:, :, top:top+rh2, left:left+rw2] = (
            mask * blurred[:, :, :rh2, :rw2]
            + (1 - mask) * region[:, :, :rh2, :rw2]
        )

    return result.clamp(0., 1.)


def apply_rolling_shutter(frame: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    Rolling shutter: each row is shifted horizontally by a different amount,
    simulating the row-by-row readout of a CMOS sensor during scene motion.
    """
    N, C, H, W = frame.shape
    device     = frame.device
    result     = frame.clone()

    # Linear shift profile: top row has 0 shift, bottom row has max_shift
    for row in range(H):
        shift = int(round(max_shift * row / H))
        if shift == 0:
            continue
        result[:, :, row, :] = torch.roll(frame[:, :, row, :], shift, dims=-1)

    return result


# ---------------------------------------------------------------------------
# Temporal noise helpers
# ---------------------------------------------------------------------------

def make_flicker_modulation(T: int, freq: float, fps: float,
                            amplitude: float) -> list:
    """
    Generate per-frame scalar intensity modulation from flicker.
    Returns list of T floats in [1-amp, 1+amp].
    """
    return [1.0 + amplitude * math.sin(2 * math.pi * freq * t / fps)
            for t in range(T)]


# ---------------------------------------------------------------------------
# Main sequence noise augmentor
# ---------------------------------------------------------------------------

class SequenceNoiseAugmentor:
    """
    Full physically-motivated noise pipeline for a sequence of video frames.

    Args:
        cfg           : NoiseConfig dataclass
        temporal_mode : 'correlated' | 'independent' | 'random'
            correlated  — AR(1) noise persistence across frames
            independent — each frame gets fresh independent noise
            random      — randomly pick one per sequence
        spatial_mode  : 'varying' | 'uniform' | 'random'
            varying     — smooth spatially-varying sigma field
            uniform     — flat sigma map (original behaviour)
            random      — randomly pick one per sequence

    Usage:
        augmentor = SequenceNoiseAugmentor()
        noisy_frames, noise_maps = augmentor(clean_frames)
    """

    def __init__(self,
                 cfg: NoiseConfig = None,
                 temporal_mode: str = 'random',
                 spatial_mode: str = 'random'):
        self.cfg           = cfg or NoiseConfig()
        self.temporal_mode = temporal_mode
        self.spatial_mode  = spatial_mode

    def _pick(self, mode: str, options: list) -> str:
        return random.choice(options) if mode == 'random' else mode

    def __call__(self, frames: list) -> tuple:
        """
        Args:
            frames : list of T tensors (N, C, H, W) in [0,1] — clean frames

        Returns:
            noisy_frames : list of T tensors (N, C, H, W) in [0,1]
            noise_maps   : list of T tensors (N, 1, H, W) — effective sigma per pixel
        """
        T             = len(frames)
        N, C, H, W    = frames[0].shape
        device        = frames[0].device
        cfg           = self.cfg

        temporal_mode = self._pick(self.temporal_mode, ['correlated', 'independent'])
        spatial_mode  = self._pick(self.spatial_mode,  ['varying', 'uniform'])

        # ── Sample sequence-level parameters ──────────────────────────────
        shot_scale  = random.uniform(*cfg.shot_scale_range)
        read_sigma  = random.uniform(*cfg.read_sigma_range)
        fpn_col_str = random.uniform(*cfg.fpn_strength_range)
        fpn_row_str = random.uniform(*cfg.fpn_strength_range)
        prnu_str    = random.uniform(*cfg.prnu_strength_range)
        alpha       = random.uniform(*cfg.temporal_alpha_range) \
                      if temporal_mode == 'correlated' else 0.0

        # Fixed pattern noise maps are generated lazily on first frame
        # using actual frame dimensions (safer than pre-computing at H,W
        # since motion blur steps may alter effective size)
        fpn_add  = None
        prnu_mul = None

        # Flicker modulation
        use_flicker  = random.random() < cfg.flicker_prob
        flicker_mods = make_flicker_modulation(
            T,
            freq=random.uniform(*cfg.flicker_freq_range),
            fps=30.0,
            amplitude=random.uniform(*cfg.flicker_amp_range),
        ) if use_flicker else [1.0] * T

        # Thermal drift — slow per-frame gain/offset change
        use_drift    = random.random() < cfg.drift_prob
        drift_gain   = random.uniform(*cfg.drift_gain_range)   if use_drift else 1.0
        drift_offset = random.uniform(*cfg.drift_offset_range) if use_drift else 0.0

        # Motion blur decisions — sampled once per sequence
        use_shake   = random.random() < cfg.shake_prob
        shake_len   = random.randint(*cfg.shake_kernel_range)
        shake_angle = random.uniform(*cfg.shake_angle_range)

        use_obj_motion  = random.random() < cfg.obj_motion_prob
        obj_kernel      = random.randint(*cfg.obj_motion_kernel_range)
        obj_num_regions = random.randint(*cfg.obj_motion_num_regions)

        use_rolling      = random.random() < cfg.rolling_shutter_prob
        rolling_shift    = random.randint(1, cfg.rolling_shutter_max_shift)

        # ── Per-frame correlated noise state ──────────────────────────────
        prev_noise = torch.zeros(N, C, H, W, device=device)

        noisy_frames = []
        noise_maps   = []

        for t, frame in enumerate(frames):

            # ── Step 1: Motion blur (optical — before noise) ───────────────
            x = frame.clone()

            if use_shake:
                x = apply_camera_shake(x, shake_len, shake_angle)

            if use_obj_motion:
                x = apply_object_motion(x, obj_num_regions, obj_kernel)

            if use_rolling:
                x = apply_rolling_shutter(x, rolling_shift)

            # ── Defensive size reset after motion blur ────────────────────
            # Clamp to original (H, W) in case any blur op had off-by-one
            x = x[:, :, :H, :W]

            # ── Step 2: Thermal drift ──────────────────────────────────────
            if use_drift:
                # Cumulative drift: gain and offset change slowly over time
                frame_gain   = drift_gain   ** t
                frame_offset = drift_offset  * t
                x = (x * frame_gain + frame_offset).clamp(0., 1.)

            # ── Step 3: PRNU (multiplicative fixed pattern) ────────────────
            # Regenerate FPN/PRNU lazily using actual frame size
            fH, fW = x.shape[2], x.shape[3]
            if fpn_add is None or fpn_add.shape[2] != fH or fpn_add.shape[3] != fW:
                fpn_add, prnu_mul = make_fpn_map(N, C, fH, fW,
                                                  fpn_col_str, fpn_row_str,
                                                  prnu_str, device)
            x = (x * prnu_mul).clamp(0., 1.)

            # ── Step 4: Shot noise ─────────────────────────────────────────
            x = add_shot_noise(x, shot_scale)

            # ── Step 5: Temporal correlation ──────────────────────────────
            # AR(1): new_noise = alpha * prev_noise + sqrt(1-alpha²) * fresh_noise
            fresh_noise = torch.randn(N, C, H, W, device=device) * read_sigma
            if temporal_mode == 'correlated':
                corr_noise  = alpha * prev_noise + \
                              math.sqrt(1 - alpha ** 2) * fresh_noise
            else:
                corr_noise  = fresh_noise
            prev_noise  = corr_noise.detach()

            # ── Step 6: Read noise (correlated Gaussian) ──────────────────
            x = (x + corr_noise).clamp(0., 1.)

            # ── Step 7: Additive fixed pattern noise ──────────────────────
            # fpn_add already guaranteed correct size from Step 3
            x = (x + fpn_add).clamp(0., 1.)

            # ── Step 8: Flicker ───────────────────────────────────────────
            x = (x * flicker_mods[t]).clamp(0., 1.)

            # ── Step 9: Compute effective per-pixel sigma map ──────────────
            # sigma_eff = sqrt(sigma_shot² + sigma_read²) — spatially varying
            sigma_map = effective_sigma_map(frame, shot_scale, read_sigma,
                                            spatial_mode)

            noisy_frames.append(x)
            noise_maps.append(sigma_map)

        return noisy_frames, noise_maps
