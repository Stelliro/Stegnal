"""Visualization helpers for the Project Umbra UI."""

from __future__ import annotations

import numpy as np


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary float array to the [0, 1] range for visualization."""
    arr = np.asarray(array, dtype=np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def multiplicative_overlap(
    original: np.ndarray, reconstructed: np.ndarray
) -> tuple[np.ndarray, float]:
    """Compute a multiplicative overlap map and its mean percentage score.

    Parameters
    ----------
    original:
        Reference image normalized to the [0, 1] range.
    reconstructed:
        Candidate image normalized to the [0, 1] range.

    Returns
    -------
    overlap_map:
        Element-wise product of ``original`` and ``reconstructed`` clipped to [0, 1].
    score:
        Mean value of the overlap map expressed as a percentage in [0, 100].
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Images must share the same shape to compute overlap")

    overlap = np.clip(np.asarray(original) * np.asarray(reconstructed), 0.0, 1.0)
    score = float(overlap.mean() * 100.0)
    return overlap.astype(np.float32), score


def to_uint8_image(array: np.ndarray) -> np.ndarray:
    """Convert a normalized float image to uint8 grayscale."""
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


__all__ = ["normalize_for_display", "multiplicative_overlap", "to_uint8_image"]
