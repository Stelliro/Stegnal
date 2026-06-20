# metrics.py

"""
Evaluation utilities for Project Umbra.
Using Sigmoid gating to encourage climbing out of local minima.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
AI_PSNR_BASELINE = 15.0
AI_PSNR_TARGET = 40.0

@dataclass
class ReconstructionMetrics:
    psnr: float
    ssim: float
    fft_score: float = 0.0
    edge_score: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "fft": self.fft_score,
            "edge": self.edge_score
        }

def compute_fft_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_gray = rgb2gray(reference) if reference.ndim == 3 else reference
    cand_gray = rgb2gray(candidate) if candidate.ndim == 3 else candidate
    
    fft_ref = np.fft.fft2(ref_gray)
    fft_cand = np.fft.fft2(cand_gray)
    
    spec_ref = np.log(np.abs(np.fft.fftshift(fft_ref)) + 1e-8)
    spec_cand = np.log(np.abs(np.fft.fftshift(fft_cand)) + 1e-8)
    
    spec_ref = spec_ref / (np.max(spec_ref) + 1e-6)
    spec_cand = spec_cand / (np.max(spec_cand) + 1e-6)
    
    diff = np.mean(np.abs(spec_ref - spec_cand))
    return float(np.clip(1.0 - (diff * 2.0), 0.0, 1.0))

def compute_edge_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_gray = rgb2gray(reference) if reference.ndim == 3 else reference
    cand_gray = rgb2gray(candidate) if candidate.ndim == 3 else candidate
    
    edge_ref = sobel(ref_gray)
    edge_cand = sobel(cand_gray)
    
    diff = np.mean(np.abs(edge_ref - edge_cand))
    return float(np.clip(1.0 - (diff * 6.0), 0.0, 1.0))

def compute_metrics(reference: np.ndarray, candidate: np.ndarray) -> ReconstructionMetrics:
    if reference.ndim < 2 or candidate.ndim < 2:
        raise ValueError("Reference and candidate must be at least 2-dimensional arrays")
    if reference.shape != candidate.shape:
        # Auto-resize candidate if needed (defensive)
        from skimage.transform import resize
        candidate = resize(candidate, reference.shape, anti_aliasing=True)

    channel_axis = -1 if reference.ndim == 3 else None
    
    psnr = float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))
    ssim = float(structural_similarity(reference, candidate, channel_axis=channel_axis, data_range=1.0))
    
    fft = compute_fft_score(reference, candidate)
    edge = compute_edge_score(reference, candidate)
    
    return ReconstructionMetrics(psnr=psnr, ssim=ssim, fft_score=fft, edge_score=edge)

def _sigmoid_gate(value: float, threshold: float, steepness: float = 20.0) -> float:
    """Soft gating function. Returns 0.0 to 1.0."""
    return 1.0 / (1.0 + np.exp(-steepness * (value - threshold)))

def composite_score(overlap: float, psnr: float, ssim: float) -> float:
    """Compute a weighted composite score from overlap, PSNR, and SSIM."""

    psnr_score = float(np.clip(
        (psnr - AI_PSNR_BASELINE) / (AI_PSNR_TARGET - AI_PSNR_BASELINE), 0.0, 1.0
    ))
    overlap_norm = float(np.clip(overlap / 100.0, 0.0, 1.0))

    w_overlap = 0.40
    w_ssim = 0.35
    w_psnr = 0.25

    raw = overlap_norm * w_overlap + ssim * w_ssim + psnr_score * w_psnr
    structure_gate = _sigmoid_gate(ssim, 0.35)
    final = raw * (0.2 + 0.8 * structure_gate)
    return float(np.clip(final, 0.0, 1.0))


def compute_ms_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    channel_axis: int | None = None,
) -> float:
    """Multi-scale SSIM approximation using Gaussian pyramid levels."""

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)

    mssim = 1.0
    for i, w in enumerate(weights):
        s = float(structural_similarity(
            ref, cand, data_range=1.0, channel_axis=channel_axis,
        ))
        mssim *= s ** w
        if i < len(weights) - 1:
            if channel_axis is None:
                ref = 0.25 * (ref[::2, ::2] + ref[1::2, ::2] + ref[::2, 1::2] + ref[1::2, 1::2])
                cand = 0.25 * (cand[::2, ::2] + cand[1::2, ::2] + cand[::2, 1::2] + cand[1::2, 1::2])
            else:
                ref = 0.25 * (ref[::2, ::2, :] + ref[1::2, ::2, :] + ref[::2, 1::2, :] + ref[1::2, 1::2, :])
                cand = 0.25 * (cand[::2, ::2, :] + cand[1::2, ::2, :] + cand[::2, 1::2, :] + cand[1::2, 1::2, :])
            if ref.shape[0] < 7 or ref.shape[1] < 7:
                break
    return float(mssim)


def dct_band_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute DCT band correlation between two images."""

    from scipy.fft import dctn

    ref_gray = rgb2gray(reference) if reference.ndim == 3 else np.asarray(reference, dtype=np.float64)
    cand_gray = rgb2gray(candidate) if candidate.ndim == 3 else np.asarray(candidate, dtype=np.float64)

    ref_dct = dctn(ref_gray, norm="ortho")
    cand_dct = dctn(cand_gray, norm="ortho")

    ref_flat = ref_dct.ravel()
    cand_flat = cand_dct.ravel()

    norm_ref = np.linalg.norm(ref_flat)
    norm_cand = np.linalg.norm(cand_flat)
    if norm_ref < 1e-12 or norm_cand < 1e-12:
        return 0.0
    corr = float(np.dot(ref_flat, cand_flat) / (norm_ref * norm_cand))
    return float(np.clip(corr, 0.0, 1.0))


def partial_alignment_fraction(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    threshold: float = 0.1,
) -> float:
    """Fraction of pixels where the candidate is aligned with the reference."""

    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.ndim == 3:
        ref = rgb2gray(ref)
    if cand.ndim == 3:
        cand = rgb2gray(cand)

    diff = np.abs(ref - cand)
    aligned = diff < threshold
    return float(np.mean(aligned))


def audio_fidelity_score(
    overlap: float,
    psnr: float,
    ssim: float,
    partial_credit: float = 0.0,
) -> float:
    """Score audio-mediated reconstruction fidelity.

    Returns 0.0 when metrics are below baseline thresholds, unless
    *partial_credit* is provided to boost borderline results.
    """

    psnr_ok = psnr >= 20.0
    ssim_ok = ssim >= 0.05

    if not (psnr_ok and ssim_ok):
        if partial_credit > 0.0:
            return float(np.clip(partial_credit * overlap * 0.01, 0.0, 100.0))
        return 0.0

    psnr_norm = float(np.clip((psnr - AI_PSNR_BASELINE) / (AI_PSNR_TARGET - AI_PSNR_BASELINE), 0.0, 1.0))
    overlap_norm = float(np.clip(overlap / 100.0, 0.0, 1.0))

    raw = 0.4 * overlap_norm + 0.35 * ssim + 0.25 * psnr_norm
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def readability_score(overlap: float, psnr: float, ssim: float) -> float:
    """Score how readable the reconstruction is for a human observer."""

    psnr_norm = float(np.clip((psnr - AI_PSNR_BASELINE) / (AI_PSNR_TARGET - AI_PSNR_BASELINE), 0.0, 1.0))
    overlap_norm = float(np.clip(overlap / 100.0, 0.0, 1.0))

    raw = 0.5 * overlap_norm + 0.3 * ssim + 0.2 * psnr_norm
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def team_cohesion_score(
    overlap: float,
    psnr: float,
    ssim: float,
    *,
    sound_reference_overlap: float = 0.0,
    sound_reference_psnr: float = 0.0,
    sound_reference_ssim: float = 0.0,
    sound_alignment_overlap: float = 0.0,
    sound_alignment_psnr: float = 0.0,
    sound_alignment_ssim: float = 0.0,
    sound_reference_partial: float = 0.0,
    sound_alignment_partial: float = 0.0,
    readability: float = 0.0,
) -> float:
    """Holistic team score that combines visual and audio-mediated metrics."""

    visual = composite_score(overlap, psnr, ssim)
    sound_ref = audio_fidelity_score(
        sound_reference_overlap, sound_reference_psnr, sound_reference_ssim,
        partial_credit=sound_reference_partial,
    )
    sound_align = audio_fidelity_score(
        sound_alignment_overlap, sound_alignment_psnr, sound_alignment_ssim,
        partial_credit=sound_alignment_partial,
    )
    read_norm = float(np.clip(readability / 100.0, 0.0, 1.0))

    combined = (
        0.35 * visual
        + 0.25 * (sound_ref / 100.0)
        + 0.20 * (sound_align / 100.0)
        + 0.20 * read_norm
    )
    return float(np.clip(combined * 100.0, 0.0, 100.0))