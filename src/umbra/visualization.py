# visualization.py

"""Visualization helpers for the Project Umbra UI."""

from __future__ import annotations

import logging

import numpy as np

try:  # pragma: no cover - optional acceleration
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cp = None

logger = logging.getLogger(__name__)


def _block_average(channel: np.ndarray, block_size: int) -> np.ndarray:
    """Average pixels within ``block_size`` square regions.

    Vectorized implementation that handles trailing edges by averaging over
    their partial blocks.
    """

    if block_size <= 1:
        return np.asarray(channel, dtype=np.float32)

    arr = np.asarray(channel, dtype=np.float32)

    if arr.ndim not in (2, 3):
        raise ValueError("Expected a 2D array or an RGB image for block averaging")

    h, w = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    pad_width = ((0, pad_h), (0, pad_w)) + (() if arr.ndim == 2 else ((0, 0),))
    if pad_h or pad_w:
        pad = np.pad(arr, pad_width, mode="edge")
    else:
        pad = arr

    ph, pw = pad.shape[:2]
    if arr.ndim == 2:
        reshaped = pad.reshape(ph // block_size, block_size, pw // block_size, block_size)
        block_means = reshaped.mean(axis=(1, 3))
        expanded = np.repeat(np.repeat(block_means, block_size, axis=0), block_size, axis=1)
        result = expanded[:h, :w]
        return result.astype(np.float32)

    reshaped = pad.reshape(
        ph // block_size,
        block_size,
        pw // block_size,
        block_size,
        channels,
    )
    block_means = reshaped.mean(axis=(1, 3))
    expanded = np.repeat(np.repeat(block_means, block_size, axis=0), block_size, axis=1)
    result = expanded[:h, :w, :]
    return result.astype(np.float32)


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary float array to the [0, 1] range for visualization."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_val = float(np.min(arr))
    ptp = float(np.ptp(arr)) or 1.0
    normalized = (arr - min_val) / ptp
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _feature_weight_map(reference: np.ndarray) -> np.ndarray:
    """Build a weight map that emphasizes high-contrast features."""

    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref = np.tensordot(ref, luma_weights, axes=([-1], [0]))

    grad_x = np.abs(np.diff(ref, axis=1, prepend=ref[:, :1]))
    grad_y = np.abs(np.diff(ref, axis=0, prepend=ref[:1, :]))
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)

    weights = 0.5 + 0.5 * edge_strength / (np.max(edge_strength) + 1e-8)
    return weights.astype(np.float32)


def _weighted_overlap_score(overlap: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean of an overlap map."""

    total_weight = float(np.sum(weights))
    if total_weight < 1e-12:
        return float(np.mean(overlap))
    return float(np.sum(overlap * weights) / total_weight)


def _overlap_against_constant(reference: np.ndarray, value: float) -> np.ndarray:
    """Compute overlap map between reference and a constant image."""

    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref = np.tensordot(ref, luma_weights, axes=([-1], [0]))
    constant = np.full_like(ref, value)
    return np.clip(1.0 - np.abs(ref - constant), 0.0, 1.0)


def build_overlap_cache(reference: np.ndarray) -> dict:
    """Pre-compute reusable data for repeated overlap calls against the same reference."""

    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref_luma = np.tensordot(ref, luma_weights, axes=([-1], [0]))
    elif ref.ndim == 2:
        ref_luma = ref
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs")

    weights = _feature_weight_map(ref_luma)
    baseline_zero = _weighted_overlap_score(
        _overlap_against_constant(ref_luma, 0.0), weights,
    )
    baseline_one = _weighted_overlap_score(
        _overlap_against_constant(ref_luma, 1.0), weights,
    )
    return {
        "ref_luma": ref_luma,
        "weights": weights,
        "baseline": max(baseline_zero, baseline_one),
    }


def multiplicative_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    cache: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Return the pixel-wise overlap map and a baseline-adjusted score in [0, 100]."""

    ref = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
    cand = np.clip(np.asarray(candidate, dtype=np.float32), 0.0, 1.0)

    if ref.shape != cand.shape:
        raise ValueError("Reference and candidate images must share the same shape")

    if ref.ndim == 2:
        ref_luma = ref
    elif ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref_luma = np.tensordot(ref, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs for reference")

    if cand.ndim == 2:
        cand_luma = cand
    elif cand.ndim == 3 and cand.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        cand_luma = np.tensordot(cand, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs for candidate")

    overlap_map = np.clip(1.0 - np.abs(ref_luma - cand_luma), 0.0, 1.0)

    if cache is not None:
        weights = cache["weights"]
        baseline = cache["baseline"]
    else:
        weights = _feature_weight_map(ref_luma)
        baseline_zero = _weighted_overlap_score(
            _overlap_against_constant(ref_luma, 0.0), weights,
        )
        baseline_one = _weighted_overlap_score(
            _overlap_against_constant(ref_luma, 1.0), weights,
        )
        baseline = max(baseline_zero, baseline_one)

    raw_score = _weighted_overlap_score(overlap_map, weights)
    adjusted = max(raw_score - baseline, 0.0) / max(1.0 - baseline, 1e-6) * 100.0
    score = float(np.clip(adjusted, 0.0, 100.0))

    return overlap_map.astype(np.float32), score


def to_uint8_image(array: np.ndarray) -> np.ndarray:
    """Convert a float array in [0, 1] to a uint8 image suitable for display."""

    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        return (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return (arr * 255.0).round().astype(np.uint8)
    raise ValueError("Expected 2D grayscale or 3-channel color array for conversion")


def colorize_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    block_size: int = 8,
) -> np.ndarray:
    """Create a color overlay highlighting agreement and disagreement.

    The resulting RGB image uses the following colour coding:

    * Overlapping signal (agreement) – rendered using the grayscale intensity
      from ``reference`` to preserve the look of the predicted image.
    * Candidate-only signal – highlighted in red.
    * Reference-only signal – highlighted in blue.

    The ``block_size`` parameter controls how aggressively pixels are averaged
    into larger shapes, making subtle differences easier to spot.
    """

    ref = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
    cand = np.clip(np.asarray(candidate, dtype=np.float32), 0.0, 1.0)

    if ref.shape != cand.shape:
        raise ValueError("Reference and candidate images must share the same shape")

    if ref.ndim == 2:
        ref_luma = ref
    elif ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref_luma = np.tensordot(ref, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs for reference")

    if cand.ndim == 2:
        cand_luma = cand
    elif cand.ndim == 3 and cand.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        cand_luma = np.tensordot(cand, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs for candidate")

    overlap = np.minimum(ref_luma, cand_luma)
    ref_only = np.clip(ref_luma - overlap, 0.0, 1.0)
    cand_only = np.clip(cand_luma - overlap, 0.0, 1.0)

    color = np.repeat(overlap[..., None], 3, axis=-1)
    color[..., 0] = np.clip(color[..., 0] + cand_only, 0.0, 1.0)
    color[..., 2] = np.clip(color[..., 2] + ref_only, 0.0, 1.0)

    if block_size > 1:
        color = _block_average(color, block_size)

    return np.clip(color, 0.0, 1.0).astype(np.float32)


__all__ = [
    "_feature_weight_map",
    "_overlap_against_constant",
    "_weighted_overlap_score",
    "build_overlap_cache",
    "colorize_comparison",
    "normalize_for_display",
    "multiplicative_overlap",
    "to_uint8_image",
]