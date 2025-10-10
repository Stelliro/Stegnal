"""Test utilities for validating the Umbra pipeline without the UI."""

from __future__ import annotations

import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.metrics import ReconstructionMetrics, compute_metrics


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

    size = int(max(8, size))
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gradient = np.outer(coords, coords)

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)

    packet = encoder.encode(gradient, seed)
    reconstruction = decoder.decode(packet, seed)
    return compute_metrics(gradient, reconstruction)


__all__ = ["run_smoke_test"]
