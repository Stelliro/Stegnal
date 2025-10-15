"""Evaluation utilities for Project Umbra."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

logger = logging.getLogger(__name__)

# Baseline PSNR thresholds shared by the UI and evolution subsystems when
# converting perceptual metrics into aggregate scores. The baseline reflects a
# "barely acceptable" reconstruction while the target approximates the
# high-quality regime we expect the evolution process to seek.
AI_PSNR_BASELINE = 20.0
AI_PSNR_TARGET = 60.0


def _validate_shapes(reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape:
        raise ValueError("Input images must share the same shape")


def _ensure_channel_axis(array: np.ndarray, channel_axis: int | None) -> int | None:
    if channel_axis is None:
        return None
    if channel_axis not in (-1, array.ndim - 1):
        raise ValueError("channel_axis must be None or the last axis")
    return array.ndim - 1

def _dct_basis(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=np.float32)
    basis = np.cos(np.pi / (2 * size) * (2 * indices[:, None] + 1) * indices[None, :])
    basis[0, :] *= np.sqrt(1.0 / size)
    basis[1:, :] *= np.sqrt(2.0 / size)
    return basis


def _dct2(image: np.ndarray) -> np.ndarray:
    basis_row = _dct_basis(image.shape[0])
    basis_col = _dct_basis(image.shape[1])
    return basis_row @ image @ basis_col.T


@dataclass
class ReconstructionMetrics:
    psnr: float
    ssim: float

    def as_dict(self) -> dict[str, float]:
        return {"psnr": self.psnr, "ssim": self.ssim}


def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> ReconstructionMetrics:
    """Compute PSNR and SSIM between two images in [0, 1]."""
    _validate_shapes(reference, candidate)

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
    logger.debug("Computed metrics: PSNR=%.2f SSIM=%.3f", psnr, ssim)
    return ReconstructionMetrics(psnr=psnr, ssim=ssim)


def composite_score(overlap_pct: float, psnr: float, ssim: float) -> float:
    """Combine overlap, PSNR, and SSIM into a 0–100 aggregate score."""

    overlap_value = float(np.nan_to_num(overlap_pct, nan=0.0, posinf=0.0, neginf=0.0))
    psnr_value = float(
        np.nan_to_num(psnr, nan=AI_PSNR_BASELINE, posinf=AI_PSNR_BASELINE, neginf=AI_PSNR_BASELINE)
    )
    ssim_value = float(np.nan_to_num(ssim, nan=0.0, posinf=0.0, neginf=0.0))

    overlap_norm = float(np.clip(overlap_value / 100.0, 0.0, 1.0))
    psnr_span = max(AI_PSNR_TARGET - AI_PSNR_BASELINE, 1e-6)
    psnr_norm = float(np.clip((psnr_value - AI_PSNR_BASELINE) / psnr_span, 0.0, 1.0))
    ssim_norm = float(np.clip(ssim_value, 0.0, 1.0))

    composite = float(np.clip(0.4 * overlap_norm + 0.3 * psnr_norm + 0.3 * ssim_norm, 0.0, 1.0))
    return composite * 100.0


def readability_score(overlap_pct: float, psnr: float, ssim: float) -> float:
    """Derive a readability score emphasising consistency between reconstructions."""

    overlap_value = float(np.nan_to_num(overlap_pct, nan=0.0, posinf=0.0, neginf=0.0))
    psnr_value = float(
        np.nan_to_num(psnr, nan=AI_PSNR_BASELINE, posinf=AI_PSNR_BASELINE, neginf=AI_PSNR_BASELINE)
    )
    ssim_value = float(np.nan_to_num(ssim, nan=0.0, posinf=0.0, neginf=0.0))

    overlap_norm = float(np.clip(overlap_value / 100.0, 0.0, 1.0))
    psnr_span = max(AI_PSNR_TARGET - AI_PSNR_BASELINE, 1e-6)
    psnr_norm = float(np.clip((psnr_value - AI_PSNR_BASELINE) / psnr_span, 0.0, 1.0))
    ssim_norm = float(np.clip(ssim_value, 0.0, 1.0))

    readability = float(np.clip((overlap_norm + psnr_norm + ssim_norm) / 3.0, 0.0, 1.0))
    return readability * 100.0


def compute_ms_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    channel_axis: int | None,
) -> float:
    """Compute a lightweight multi-scale SSIM between two images."""

    reference_f = np.asarray(reference, dtype=np.float32)
    candidate_f = np.asarray(candidate, dtype=np.float32)
    _validate_shapes(reference_f, candidate_f)
    axis = _ensure_channel_axis(reference_f, channel_axis)

    if axis is None:
        spatial_dims = reference_f.shape[:2]
    else:
        spatial_dims = tuple(
            reference_f.shape[idx] for idx in range(reference_f.ndim) if idx != axis
        )
    min_spatial = min(spatial_dims)

    # The default Gaussian-weighted SSIM uses an 11x11 window. Guard against
    # requesting that configuration on smaller tensors by falling back to a
    # single-scale SSIM with an explicit window size.
    MIN_GAUSSIAN_EXTENT = 11
    if min_spatial < MIN_GAUSSIAN_EXTENT:
        fallback_win = min_spatial
        if fallback_win % 2 == 0:
            fallback_win -= 1
        if fallback_win < 3:
            if np.allclose(reference_f, candidate_f):
                return 1.0
            diff = float(np.mean(np.abs(reference_f - candidate_f)))
            return float(np.clip(1.0 - diff, 0.0, 1.0))

        return float(
            structural_similarity(
                reference_f,
                candidate_f,
                data_range=1.0,
                channel_axis=axis,
                gaussian_weights=False,
                win_size=fallback_win,
            )
        )

    ref_scale = reference_f
    cand_scale = candidate_f
    scores: list[float] = []
    for _ in range(3):
        spatial_min = min(ref_scale.shape[0], ref_scale.shape[1])
        if spatial_min < MIN_GAUSSIAN_EXTENT:
            break
        score = float(
            structural_similarity(
                ref_scale,
                cand_scale,
                data_range=1.0,
                channel_axis=axis,
                gaussian_weights=True,
            )
        )
        scores.append(score)

        if spatial_min <= MIN_GAUSSIAN_EXTENT:
            break

        new_spatial = (
            max(1, ref_scale.shape[0] // 2),
            max(1, ref_scale.shape[1] // 2),
        )
        if axis is None:
            ref_scale = resize(
                ref_scale,
                new_spatial,
                order=1,
                anti_aliasing=True,
                preserve_range=True,
                mode="reflect",
            ).astype(np.float32)
            cand_scale = resize(
                cand_scale,
                new_spatial,
                order=1,
                anti_aliasing=True,
                preserve_range=True,
                mode="reflect",
            ).astype(np.float32)
        else:
            channel_count = ref_scale.shape[axis]
            ref_channels: list[np.ndarray] = []
            cand_channels: list[np.ndarray] = []
            for idx in range(channel_count):
                ref_channel = np.take(ref_scale, idx, axis=axis)
                cand_channel = np.take(cand_scale, idx, axis=axis)
                ref_resized = resize(
                    ref_channel,
                    new_spatial,
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                    mode="reflect",
                ).astype(np.float32)
                cand_resized = resize(
                    cand_channel,
                    new_spatial,
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                    mode="reflect",
                ).astype(np.float32)
                ref_channels.append(ref_resized)
                cand_channels.append(cand_resized)
            ref_scale = np.stack(ref_channels, axis=-1)
            cand_scale = np.stack(cand_channels, axis=-1)

    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def dct_band_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    bands: tuple[int, int] = (2, 12),
) -> float:
    """Correlate mid-frequency DCT magnitudes between two images."""

    if len(bands) != 2:
        raise ValueError("bands must be a (start, end) tuple")
    start, end = bands
    if start < 0 or end <= start:
        raise ValueError("bands must define a positive frequency range")

    reference_f = np.asarray(reference, dtype=np.float32)
    candidate_f = np.asarray(candidate, dtype=np.float32)
    _validate_shapes(reference_f, candidate_f)

    if reference_f.ndim == 2:
        ref_channels = [reference_f]
        cand_channels = [candidate_f]
    elif reference_f.ndim == 3 and reference_f.shape[-1] <= 4:
        ref_channels = [reference_f[..., idx] for idx in range(reference_f.shape[-1])]
        cand_channels = [candidate_f[..., idx] for idx in range(candidate_f.shape[-1])]
    else:
        raise ValueError("Input images must be 2D or 3-channel")

    correlations: list[float] = []
    for ref_channel, cand_channel in zip(ref_channels, cand_channels):
        ref_dct = _dct2(ref_channel)
        cand_dct = _dct2(cand_channel)
        ref_band = np.abs(ref_dct[start:end, start:end]).ravel()
        cand_band = np.abs(cand_dct[start:end, start:end]).ravel()
        if ref_band.size == 0:
            continue
        if np.allclose(ref_band, 0.0) and np.allclose(cand_band, 0.0):
            correlations.append(1.0)
            continue
        if np.allclose(ref_band, 0.0) or np.allclose(cand_band, 0.0):
            correlations.append(0.0)
            continue
        corr_matrix = np.corrcoef(ref_band, cand_band)
        corr = float(corr_matrix[0, 1])
        if np.isfinite(corr):
            correlations.append(float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0)))

    if not correlations:
        return 0.0
    return float(np.clip(np.mean(correlations), 0.0, 1.0))


__all__ = [
    "compute_metrics",
    "ReconstructionMetrics",
    "compute_ms_ssim",
    "dct_band_correlation",
]
