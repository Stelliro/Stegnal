"""Evaluation utilities for Project Umbra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class ReconstructionMetrics:
    psnr: float
    ssim: float

    def as_dict(self) -> dict[str, float]:
        return {"psnr": self.psnr, "ssim": self.ssim}


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> ReconstructionMetrics:
    """Compute PSNR and SSIM between two images in [0, 1]."""
    if reference.shape != candidate.shape:
        raise ValueError("Input images must share the same shape")

    channel_axis = -1 if reference.ndim == 3 else None
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    ssim = float(
        structural_similarity(
            reference,
            candidate,
            data_range=1.0,
            channel_axis=channel_axis,
        )
    )
    return ReconstructionMetrics(psnr=psnr, ssim=ssim)


__all__ = ["compute_metrics", "ReconstructionMetrics"]
