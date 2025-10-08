"""High-level helpers for the Project Umbra toy pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .metrics import ReconstructionMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    packet_path: Path | None
    reconstruction_path: Path | None
    metrics: ReconstructionMetrics


def run_pipeline(
    image_path: str | Path,
    seed: int,
    sigma: float = 0.2,
    packet_path: str | Path | None = None,
    reconstruction_path: str | Path | None = None,
    denoise_sigma: float | None = 1.0,
) -> PipelineResult:
    """Execute the encode/decode process and return reconstruction metrics."""

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)

    logger.info(
        "Running pipeline with seed=%d sigma=%.3f denoise=%.3f", seed, sigma, denoise_sigma
    )

    original = encoder.load_image(image_path)
    packet = encoder.encode(original, seed)

    if packet_path is not None:
        packet.to_file(packet_path)
        logger.debug("Saved encoded packet to %s", packet_path)

    reconstructed = decoder.decode(packet, seed)

    if reconstruction_path is not None:
        decoder.save_image(reconstructed, reconstruction_path)
        logger.debug("Saved reconstruction to %s", reconstruction_path)

    metrics = compute_metrics(original, reconstructed)
    logger.info(
        "Pipeline metrics for %s: PSNR %.2f SSIM %.3f",
        image_path,
        metrics.psnr,
        metrics.ssim,
    )
    return PipelineResult(
        packet_path=Path(packet_path) if packet_path else None,
        reconstruction_path=Path(reconstruction_path) if reconstruction_path else None,
        metrics=metrics,
    )


def replay_packet(packet_path: str | Path, seed: int, denoise_sigma: float | None = 1.0) -> np.ndarray:
    """Decode an existing packet."""
    logger.info("Replaying packet from %s with seed=%d", packet_path, seed)
    packet = NoisePacket.from_file(packet_path)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)
    return decoder.decode(packet, seed)


__all__ = ["run_pipeline", "replay_packet", "PipelineResult"]
