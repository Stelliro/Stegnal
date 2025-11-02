"""Visualization helpers for the Project Umbra UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional acceleration
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlapCache:
    """Reusable weights and baselines for overlap scoring."""

    reference_shape: tuple[int, ...]
    weights: np.ndarray
    baseline: float
    denominator: float


def build_overlap_cache(reference: np.ndarray) -> OverlapCache:
    """Return cached weights and normalization factors for ``reference``."""

    ref = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
    weights = _feature_weight_map(ref)
    baseline_zero = _weighted_overlap_score(_overlap_against_constant(ref, 0.0), weights)
    baseline_one = _weighted_overlap_score(_overlap_against_constant(ref, 1.0), weights)
    baseline = float(max(baseline_zero, baseline_one))
    denominator = float(max(1.0 - baseline, 1e-6))
    return OverlapCache(
        reference_shape=tuple(ref.shape),
        weights=np.asarray(weights, dtype=np.float32),
        baseline=baseline,
        denominator=denominator,
    )


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
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def _feature_weight_map(reference: np.ndarray, xp_module: Any | None = None) -> Any:
    """Return a weight map that emphasises informative regions of ``reference``."""

    xp = np if xp_module is None else xp_module
    ref = xp.asarray(reference, dtype=xp.float32)

    if ref.ndim == 3 and ref.shape[-1] >= 3:
        luma_weights = xp.asarray([0.2126, 0.7152, 0.0722], dtype=ref.dtype)
        luma = xp.tensordot(ref[..., :3], luma_weights, axes=([-1], [0]))
    else:
        luma = ref

    gradients = xp.gradient(luma)
    gradient_energy = xp.zeros_like(luma)
    for component in gradients:
        gradient_energy = gradient_energy + component ** 2
    gradient_magnitude = xp.sqrt(gradient_energy)

    max_gradient = float(xp.max(gradient_magnitude))
    if not np.isfinite(max_gradient) or max_gradient <= 1e-6:
        gradient_norm = xp.zeros_like(gradient_magnitude)
    else:
        gradient_norm = gradient_magnitude / max_gradient

    weights = 0.2 + 0.8 * gradient_norm
    return xp.clip(weights, 0.05, 1.0)


def _overlap_against_constant(reference: np.ndarray, value: float, xp_module: Any | None = None) -> Any:
    """Compute the overlap map between ``reference`` and a constant image."""

    xp = np if xp_module is None else xp_module
    ref = xp.asarray(reference, dtype=xp.float32)
    diff = xp.abs(ref - value)
    return xp.clip(1.0 - diff, 0.0, 1.0)


def _weighted_overlap_score(overlap_map: Any, weights: Any, xp_module: Any | None = None) -> float:
    """Return the weighted mean of ``overlap_map`` using ``weights``."""

    xp = np if xp_module is None else xp_module

    weight_map = weights
    if overlap_map.ndim == weight_map.ndim + 1:
        weight_map = weight_map[..., None]

    weighted_sum = xp.sum(overlap_map * weight_map)
    total_weight = xp.sum(weight_map)

    total_weight_value = float(total_weight)
    if not np.isfinite(total_weight_value) or total_weight_value <= 1e-6:
        return 0.0

    weighted_value = float(weighted_sum) if xp is np else float(weighted_sum.get())
    return weighted_value / total_weight_value


def multiplicative_overlap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    cache: OverlapCache | None = None,
) -> tuple[np.ndarray, float]:
    """Compute an agreement map and a baseline-adjusted percentage score.

    The raw overlap map is still defined as ``1 - |original - reconstructed|`` so
    perfect reconstructions yield ones everywhere.  However, we now discount the
    score by subtracting the best overlap achievable by a blank image (all black
    or all white).  This prevents large uniform regions from dominating the
    result and artificially inflating the reward of early generations that only
    reproduce the background.
    """

    if original.shape != reconstructed.shape:
        raise ValueError("Images must share the same shape to compute overlap")

    if cache is not None and cache.reference_shape != original.shape:
        cache = None

    if cache is not None:
        orig_cpu = np.asarray(original, dtype=np.float32)
        recon_cpu = np.asarray(reconstructed, dtype=np.float32)
        diff = np.abs(orig_cpu - recon_cpu)
        overlap = np.clip(1.0 - diff, 0.0, 1.0).astype(np.float32)
        raw_score = _weighted_overlap_score(overlap, cache.weights, np)
        adjusted = max(raw_score - cache.baseline, 0.0)
        score = float(np.clip(adjusted / cache.denominator, 0.0, 1.0) * 100.0)
        return overlap, score

    if cp is not None and original.size >= 65_536:  # pragma: no branch - runtime check
        try:
            xp = cp
            orig_gpu = xp.asarray(original, dtype=xp.float32)
            recon_gpu = xp.asarray(reconstructed, dtype=xp.float32)
            diff_gpu = xp.abs(orig_gpu - recon_gpu)
            overlap_gpu = xp.clip(1.0 - diff_gpu, 0.0, 1.0)
            weights_gpu = _feature_weight_map(orig_gpu, xp)
            raw_score = _weighted_overlap_score(overlap_gpu, weights_gpu, xp)
            baseline_zero = _weighted_overlap_score(
                _overlap_against_constant(orig_gpu, 0.0, xp), weights_gpu, xp
            )
            baseline_one = _weighted_overlap_score(
                _overlap_against_constant(orig_gpu, 1.0, xp), weights_gpu, xp
            )
            baseline = max(baseline_zero, baseline_one)
            adjusted = max(raw_score - baseline, 0.0)
            denominator = max(1.0 - baseline, 1e-6)
            score = float(np.clip(adjusted / denominator, 0.0, 1.0) * 100.0)
            overlap = xp.asnumpy(overlap_gpu)
            return overlap.astype(np.float32), score
        except Exception:  # pragma: no cover - GPU fallback
            logger.debug("Falling back to NumPy overlap after CuPy failure", exc_info=True)

    xp = np
    orig_cpu = xp.asarray(original, dtype=xp.float32)
    recon_cpu = xp.asarray(reconstructed, dtype=xp.float32)
    diff = xp.abs(orig_cpu - recon_cpu)
    overlap = xp.clip(1.0 - diff, 0.0, 1.0)
    weights = _feature_weight_map(orig_cpu, xp)
    raw_score = _weighted_overlap_score(overlap, weights, xp)
    baseline_zero = _weighted_overlap_score(_overlap_against_constant(orig_cpu, 0.0, xp), weights, xp)
    baseline_one = _weighted_overlap_score(_overlap_against_constant(orig_cpu, 1.0, xp), weights, xp)
    baseline = max(baseline_zero, baseline_one)
    adjusted = max(raw_score - baseline, 0.0)
    denominator = max(1.0 - baseline, 1e-6)
    score = float(np.clip(adjusted / denominator, 0.0, 1.0) * 100.0)
    return overlap.astype(np.float32), score


def to_uint8_image(array: np.ndarray) -> np.ndarray:
    """Convert a normalized float image to uint8 grayscale."""
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        return (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
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
        cand_luma = cand
    elif ref.ndim == 3 and ref.shape[2] == 3:
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref_luma = np.tensordot(ref, luma_weights, axes=([-1], [0]))
        cand_luma = np.tensordot(cand, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs")

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
    "build_overlap_cache",
    "colorize_comparison",
    "normalize_for_display",
    "multiplicative_overlap",
    "OverlapCache",
    "to_uint8_image",
]
