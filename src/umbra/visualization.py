"""Visualization helpers for the Project Umbra UI."""

from __future__ import annotations

import logging

import numpy as np

try:  # pragma: no cover - optional acceleration
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
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
    h, w = arr.shape

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h or pad_w:
        pad = np.pad(
            arr,
            ((0, pad_h), (0, pad_w)),
            mode="edge",
        )
    else:
        pad = arr

    ph, pw = pad.shape
    reshaped = pad.reshape(ph // block_size, block_size, pw // block_size, block_size)
    block_means = reshaped.mean(axis=(1, 3))
    expanded = np.repeat(np.repeat(block_means, block_size, axis=0), block_size, axis=1)
    result = expanded[:h, :w]
    return result.astype(np.float32)


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
    """Compute an agreement map and percentage score for two normalized images.

    The previous implementation multiplied both images together which caused even
    perfect reconstructions to cap well below 100% for natural scenes.  The new
    formulation treats the overlap map as ``1 - |original - reconstructed|`` so a
    perfect match yields ones everywhere and therefore a 100% score.  Values are
    clipped to the [0, 1] interval which keeps the behaviour stable for inputs
    that may slightly exceed the nominal range due to noise.

    Parameters
    ----------
    original:
        Reference image normalized to the [0, 1] range.
    reconstructed:
        Candidate image normalized to the [0, 1] range.

    Returns
    -------
    overlap_map:
        Agreement intensity for each pixel expressed in [0, 1].
    score:
        Mean value of the overlap map expressed as a percentage in [0, 100].
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Images must share the same shape to compute overlap")

    if cp is not None and original.size >= 65_536:  # pragma: no branch - runtime check
        try:
            orig_gpu = cp.asarray(original, dtype=cp.float32)
            recon_gpu = cp.asarray(reconstructed, dtype=cp.float32)
            diff_gpu = cp.abs(orig_gpu - recon_gpu)
            overlap_gpu = cp.clip(1.0 - diff_gpu, 0.0, 1.0)
            score = float(cp.mean(overlap_gpu).get() * 100.0)
            overlap = cp.asnumpy(overlap_gpu)
            return overlap.astype(np.float32), score
        except Exception:  # pragma: no cover - GPU fallback
            logger.debug("Falling back to NumPy overlap after CuPy failure", exc_info=True)

    diff = np.abs(np.asarray(original, dtype=np.float32) - np.asarray(reconstructed, dtype=np.float32))
    overlap = np.clip(1.0 - diff, 0.0, 1.0)
    score = float(overlap.mean() * 100.0)
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
        spatial_shape = ref.shape
        ref_luma = ref
        cand_luma = cand
    elif ref.ndim == 3 and ref.shape[2] == 3:
        spatial_shape = ref.shape[:2]
        luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref_luma = np.tensordot(ref, luma_weights, axes=([-1], [0]))
        cand_luma = np.tensordot(cand, luma_weights, axes=([-1], [0]))
    else:
        raise ValueError("Expected 2D grayscale or 3-channel RGB inputs")

    overlap = np.minimum(ref_luma, cand_luma)
    ref_only = np.clip(ref_luma - overlap, 0.0, 1.0)
    cand_only = np.clip(cand_luma - overlap, 0.0, 1.0)

    color = np.zeros((*spatial_shape, 3), dtype=np.float32)
    color[..., :] = overlap[..., None]
    color[..., 0] += cand_only  # red channel emphasises candidate-only content
    color[..., 2] += ref_only   # blue channel emphasises reference-only content
    color = np.clip(color, 0.0, 1.0)

    if block_size > 1:
        for channel in range(3):
            color[..., channel] = _block_average(color[..., channel], block_size)

    return color.astype(np.float32)


__all__ = [
    "colorize_comparison",
    "normalize_for_display",
    "multiplicative_overlap",
    "to_uint8_image",
]
