# pipeline.py

"""High-level helpers for the Project Umbra toy pipeline."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decoding import DiffusionInpainter, NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .metrics import ReconstructionMetrics, compute_metrics
from .sound import messy_key_hash_from_overlap

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    packet_path: Path | None
    reconstruction_path: Path | None
    metrics: ReconstructionMetrics
    messy_key: str | None


def run_pipeline(
    image_path: str | Path,
    seed: int,
    sigma: float = 0.2,
    packet_path: str | Path | None = None,
    reconstruction_path: str | Path | None = None,
    denoise_sigma: float | None = 1.0,
) -> PipelineResult:
    """Execute the encode/decode process and return reconstruction metrics."""

    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if denoise_sigma is not None and denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma, inpainter=DiffusionInpainter())

    logger.info(
        "Running pipeline with seed=%d sigma=%.3f denoise=%.3f", seed, sigma, denoise_sigma
    )

    try:
        original = encoder.load_image(image_path)
    except Exception as exc:
        logger.error(f"Failed to load image from {image_path}: {exc}")
        raise

    packet = encoder.encode(original, seed)

    if packet_path is not None:
        Path(packet_path).parent.mkdir(parents=True, exist_ok=True)
        packet.to_file(packet_path)
        logger.debug("Saved encoded packet to %s", packet_path)

    reconstructed = decoder.decode(packet, seed)

    if reconstruction_path is not None:
        Path(reconstruction_path).parent.mkdir(parents=True, exist_ok=True)
        decoder.save_image(reconstructed, reconstruction_path)
        logger.debug("Saved reconstruction to %s", reconstruction_path)

    metrics = compute_metrics(original, reconstructed)
    messy_key = None
    if packet.messy_latent is not None:
        messy_key = messy_key_hash_from_overlap(reconstructed)
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
        messy_key=messy_key,
    )


def replay_packet(packet_path: str | Path, seed: int, denoise_sigma: float | None = 1.0) -> np.ndarray:
    """Decode an existing packet."""
    if denoise_sigma is not None and denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")
    logger.info("Replaying packet from %s with seed=%d", packet_path, seed)
    try:
        packet = NoisePacket.from_file(packet_path)
    except Exception as exc:
        logger.error(f"Failed to load packet from {packet_path}: {exc}")
        raise
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)
    return decoder.decode(packet, seed)


def run_pipeline_async(
    jobs: list[tuple[str | Path, int]],
    *,
    sigma: float = 0.2,
    denoise_sigma: float | None = 1.0,
    max_workers: int | None = None,
) -> list[PipelineResult]:
    """Evaluate multiple encode/decode tasks asynchronously."""

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    results: list[PipelineResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                run_pipeline,
                image_path,
                seed,
                sigma=sigma,
                denoise_sigma=denoise_sigma,
            ): (image_path, seed)
            for image_path, seed in jobs
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive logging
                image_path, seed = future_map[future]
                logger.exception("Pipeline job failed for %s seed=%s: %s", image_path, seed, exc)
                raise  # Propagate to caller
    return results


__all__ = ["run_pipeline", "replay_packet", "run_pipeline_async", "PipelineResult"]