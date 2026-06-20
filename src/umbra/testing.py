# testing.py

"""Test utilities for validating the Umbra pipeline without the UI."""

from __future__ import annotations

import logging

import numpy as np

from umbra.codec import decode_waveform_to_image, encode_image_to_waveform
from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.metrics import ReconstructionMetrics, compute_metrics
from umbra.reconstruction import suggest_sample_rate, suggest_transmission_profile

logger = logging.getLogger(__name__)


def run_smoke_test(
    *,
    seed: int = 1234,
    size: int = 128,
    sigma: float = 0.25,
    denoise_sigma: float = 0.9,
) -> ReconstructionMetrics:
    """Execute a minimal encode/decode cycle on a synthetic pattern.

    The function generates a simple gradient image, runs it through the
    stochastic encoder/decoder pair, and returns the reconstruction metrics.
    It can be used from tests or the CLI to ensure the core pipeline works
    without launching the Streamlit dashboard.
    """

    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")

    size = int(max(8, size))
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gradient = np.outer(coords, coords)
    gradient_rgb = np.stack([gradient, gradient, gradient], axis=-1)

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)

    packet = encoder.encode(gradient_rgb, seed)
    reconstruction = decoder.decode(packet, seed)
    if reconstruction.ndim == 2:
        reconstruction_rgb = np.stack([reconstruction] * 3, axis=-1)
    elif reconstruction.ndim == 3 and reconstruction.shape[2] == 1:
        reconstruction_rgb = np.repeat(reconstruction, 3, axis=-1)
    else:
        reconstruction_rgb = reconstruction
    reconstruction_rgb = np.clip(reconstruction_rgb, 0.0, 1.0)
    gradient_rgb = np.clip(gradient_rgb, 0.0, 1.0)
    packet_metrics = compute_metrics(gradient_rgb, reconstruction_rgb)

    sample_rate = suggest_sample_rate(reconstruction_rgb)
    segments, marker_duration = suggest_transmission_profile(reconstruction_rgb)
    waveform = encode_image_to_waveform(
        reconstruction_rgb,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    waveform_image, _ = decode_waveform_to_image(
        waveform,
        sample_rate=sample_rate,
        resolution=reconstruction_rgb.shape[:2],
        segments=segments,
        marker_duration=marker_duration,
        return_metadata=True,
    )
    waveform_gray = np.clip(np.asarray(waveform_image, dtype=np.float32)[..., 0], 0.0, 1.0)
    wav_metrics = compute_metrics(gradient, waveform_gray)

    def _normalize_psnr(value: float) -> float:
        return float(np.clip((value - 20.0) / 40.0, 0.0, 1.0))

    def _normalize_ssim(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    packet_score = 0.3 * _normalize_psnr(packet_metrics.psnr) + 0.3 * _normalize_ssim(
        packet_metrics.ssim
    )
    wav_score = 0.3 * _normalize_psnr(wav_metrics.psnr) + 0.3 * _normalize_ssim(
        wav_metrics.ssim
    )
    if abs(packet_score - wav_score) > 0.05:
        logger.warning("Significant discrepancy between packet and WAV scores: %.3f vs %.3f", packet_score, wav_score)

    return packet_metrics


__all__ = ["run_smoke_test"]