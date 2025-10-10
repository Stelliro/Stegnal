import numpy as np
import pytest

from umbra.metrics import compute_ms_ssim, dct_band_correlation


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
