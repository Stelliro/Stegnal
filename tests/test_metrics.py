"""Tests for umbra.metrics — edge cases and validation."""

from __future__ import annotations

import numpy as np
import pytest

from umbra.metrics import compute_fft_score, compute_edge_score, compute_metrics


def test_compute_metrics_identical_images():
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    m = compute_metrics(img, img)
    assert m.ssim == pytest.approx(1.0, abs=1e-4)
    assert m.fft_score == pytest.approx(1.0, abs=1e-2)
    assert m.edge_score == pytest.approx(1.0, abs=1e-2)


def test_compute_metrics_rejects_1d_input():
    ref = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2-dimensional"):
        compute_metrics(ref, ref)


def test_compute_metrics_auto_resizes_candidate():
    rng = np.random.default_rng(5)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    cand = rng.random((16, 16, 3)).astype(np.float32)
    m = compute_metrics(ref, cand)
    # Should not crash; metrics should be finite
    assert np.isfinite(m.psnr)
    assert np.isfinite(m.ssim)


def test_fft_score_constant_images():
    """Two constant images should have near-perfect FFT similarity."""
    a = np.full((16, 16), 0.5)
    b = np.full((16, 16), 0.5)
    score = compute_fft_score(a, b)
    assert score >= 0.9


def test_edge_score_blank_images():
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    score = compute_edge_score(a, b)
    assert score == pytest.approx(1.0)


def test_compute_metrics_as_dict_keys():
    img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    m = compute_metrics(img, img)
    d = m.as_dict()
    assert set(d.keys()) == {"psnr", "ssim", "fft", "edge"}
