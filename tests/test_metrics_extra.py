import numpy as np
import pytest

from umbra.metrics import (
    audio_fidelity_score,
    compute_ms_ssim,
    dct_band_correlation,
    partial_alignment_fraction,
)


def test_compute_ms_ssim_identical_is_one() -> None:
    image = np.random.default_rng(0).random((32, 32), dtype=np.float32)
    score = compute_ms_ssim(image, image, channel_axis=None)
    assert score == pytest.approx(1.0, rel=1e-6)


def test_dct_band_correlation_prefers_identical() -> None:
    rng = np.random.default_rng(1)
    reference = rng.random((16, 16), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = 0.0

    identical = dct_band_correlation(reference, reference)
    altered = dct_band_correlation(reference, candidate)

    assert 0.0 <= identical <= 1.0
    assert identical >= altered


def test_audio_fidelity_score_zero_when_below_baseline() -> None:
    assert audio_fidelity_score(90.0, 10.0, 0.9) == 0.0
    assert audio_fidelity_score(90.0, 30.0, 0.01) == 0.0


def test_audio_fidelity_score_rewards_balanced_metrics() -> None:
    score = audio_fidelity_score(85.0, 45.0, 0.82)
    assert 0.0 < score < 100.0


def test_partial_alignment_fraction_detects_stripe() -> None:
    size = 64
    reference = np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32), (size, 1))
    candidate = np.zeros_like(reference)
    candidate[10:22, :] = reference[30:42, :]

    fraction = partial_alignment_fraction(reference, candidate)
    assert 0.05 < fraction < 0.6


def test_audio_fidelity_score_uses_partial_credit() -> None:
    baseline = audio_fidelity_score(10.0, 18.0, 0.04)
    boosted = audio_fidelity_score(10.0, 18.0, 0.04, partial_credit=0.5)
    assert baseline == 0.0
    assert boosted > baseline
