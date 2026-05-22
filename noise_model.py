"""
noise_model.py  —  Blind denoiser noise pipeline

Implements only three noise types relevant for blind training:
  1. Shot noise  — Poisson, signal-dependent (dominant at normal/bright light)
  2. Read noise  — Gaussian, signal-independent (dominant at low light)
  3. Flicker     — periodic multiplicative intensity modulation (50/60Hz lighting)

Design choices for blind training:
  - All sequences receive noise — no clean pass-through.
    Training focuses purely on denoising rather than identity mapping.
  - Noise type selection per sequence:
      30% clean
      ~23% shot only
      ~23% read only
      ~12% shot + read
      ~12% shot + read + flicker
    Probabilities drawn to match natural image statistics.
  - Parameters sampled from distributions that reflect real camera behaviour:
      shot_scale : LogUniform[0.1, 1.5]  (log scale — most sensors are low-noise)
      read_sigma : LogUniform[0.005, 0.04]
      flicker_amp: Beta(2, 8) * 0.08     (right-skewed — usually subtle)
      flicker_freq: discrete {50, 60, 100, 120} Hz (real mains frequencies)
  - No noise_map returned — blind model doesn't receive sigma.

Returns:
    noisy_frames : list of T tensors (N, C, H, W) in [0,1]
    (no noise_maps — blind model doesn't use them)
"""

import math
import random
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Realistic parameter sampling
# ---------------------------------------------------------------------------

def _log_uniform(low: float, high: float) -> float:
    """
    Sample from log-uniform distribution.
    Most camera noise parameters are better modelled in log space —
    the difference between sigma=0.005 and 0.01 is perceptually similar
    to the difference between 0.02 and 0.04.
    """
    return math.exp(random.uniform(math.log(low), math.log(high)))


def _sample_noise_params() -> dict:
    """
    Sample noise parameters from physically motivated distributions.

    shot_scale  ~ LogUniform[0.1, 1.5]
        0.1  = well-lit indoor scene (low ISO, low noise)
        0.5  = typical indoor video
        1.5  = dark scene / high ISO

    read_sigma  ~ LogUniform[0.005, 0.04]
        0.005 = clean modern sensor
        0.02  = typical consumer camera
        0.04  = older/cheaper sensor or high ISO

    flicker_amp ~ Beta(2, 8) scaled to [0, 0.08]
        Mean ~0.018, right-skewed — most flicker is subtle
        Occasional stronger flicker modelled by tail of distribution

    flicker_freq: discrete {50, 60, 100, 120} Hz
        50/60 Hz  = fundamental mains frequency
        100/120 Hz = second harmonic (more common in LED lighting)
        Weighted: 100/120 Hz more common in modern LED environments
    """
    shot_scale   = _log_uniform(0.1, 1.5)
    read_sigma   = _log_uniform(0.005, 0.04)

    # Beta(2,8) — mode near 0, long right tail, capped at 0.08
    flicker_amp  = float(np.random.beta(2, 8)) * 0.08

    # Weighted discrete: 100/120Hz (LED) more common than 50/60Hz (fluorescent)
    flicker_freq = random.choices(
        [50, 60, 100, 120],
        weights=[0.15, 0.15, 0.35, 0.35]
    )[0]

    return {
        'shot_scale':   shot_scale,
        'read_sigma':   read_sigma,
        'flicker_amp':  flicker_amp,
        'flicker_freq': flicker_freq,
    }


def _select_noise_types() -> set:
    """
    Select which noise types to apply to a sequence.
    No clean pass-through — every sequence receives some form of noise.

    Probabilities:
        ~30% shot only
        ~30% read only
        ~25% shot + read
        ~15% shot + read + flicker

    Rationale:
        - No clean sequences — model trained exclusively on noisy data
        - Shot-only and read-only teach each noise type independently
        - Combined modes match realistic camera conditions
        - Flicker always combined with shot+read (never standalone)
    """
    r = random.random()

    if r < 0.30:
        return {'shot'}                        # shot only  (30%)

    elif r < 0.60:
        return {'read'}                        # read only  (30%)

    elif r < 0.85:
        return {'shot', 'read'}               # shot + read  (25%)

    else:
        return {'shot', 'read', 'flicker'}    # shot + read + flicker  (15%)


# ---------------------------------------------------------------------------
# Noise application functions
# ---------------------------------------------------------------------------

def _add_shot_noise(frame: torch.Tensor, shot_scale: float) -> torch.Tensor:
    """
    Poisson shot noise — signal-dependent.
    Higher shot_scale = lower photon count = more noise.
    """
    photons = (frame * 255.0 / shot_scale).clamp(1e-6)
    noisy   = torch.poisson(photons) * shot_scale / 255.0
    return noisy.clamp(0., 1.)


def _add_read_noise(frame: torch.Tensor, read_sigma: float) -> torch.Tensor:
    """Gaussian read noise — signal-independent."""
    return (frame + torch.randn_like(frame) * read_sigma).clamp(0., 1.)


def _add_flicker(frame: torch.Tensor,
                 t: int,
                 flicker_amp: float,
                 flicker_freq: float,
                 fps: float = 30.0) -> torch.Tensor:
    """
    Multiplicative periodic intensity modulation.
    f(t) = 1 + amp * sin(2*pi * freq * t / fps)
    """
    mod = 1.0 + flicker_amp * math.sin(2 * math.pi * flicker_freq * t / fps)
    return (frame * mod).clamp(0., 1.)


# ---------------------------------------------------------------------------
# Main augmentor
# ---------------------------------------------------------------------------

class SequenceNoiseAugmentor:
    """
    Blind noise augmentor for video denoising training.

    Applies shot, read and/or flicker noise to a sequence of frames.
    30% of sequences are passed through clean (no noise).
    Parameters are sampled from physically realistic distributions.

    No noise_map is returned — this is a blind denoiser.

    Usage:
        augmentor = SequenceNoiseAugmentor()
        noisy_frames = augmentor(clean_frames)
        # clean_frames : list of T tensors (N, C, H, W) in [0,1]
        # noisy_frames : list of T tensors (N, C, H, W) in [0,1]
    """

    def __init__(self):
        pass

    def __call__(self, frames: list) -> list:
        """
        Args:
            frames : list of T tensors (N, C, H, W) in [0,1]
        Returns:
            noisy_frames : list of T tensors (N, C, H, W) in [0,1]
                          (may be identical to frames if clean was selected)
        """
        enabled = _select_noise_types()

        # Clean pass-through — no noise
        if not enabled:
            return [f.clone() for f in frames]

        params = _sample_noise_params()

        noisy_frames = []
        for t, frame in enumerate(frames):
            x = frame.clone()

            if 'shot' in enabled:
                x = _add_shot_noise(x, params['shot_scale'])

            if 'read' in enabled:
                x = _add_read_noise(x, params['read_sigma'])

            if 'flicker' in enabled:
                x = _add_flicker(x, t,
                                  params['flicker_amp'],
                                  params['flicker_freq'])

            noisy_frames.append(x)

        return noisy_frames
