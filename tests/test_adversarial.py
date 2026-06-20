"""Tests for umbra.adversarial — generator transforms and co-evolution."""

from __future__ import annotations

import numpy as np
import pytest

from umbra.adversarial import (
    AdversarialManager,
    CoevolutionState,
    GeneratorParams,
    apply_generator,
)


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((32, 32, 3), dtype=np.float32)


def test_apply_generator_identity():
    """No-op params should leave the image nearly unchanged."""
    img = _sample_image()
    params = GeneratorParams(blur_sigma=0.0, contrast=1.0, brightness=0.0)
    out = apply_generator(img, params)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_apply_generator_clips_output():
    img = np.ones((8, 8, 3), dtype=np.float32) * 0.9
    params = GeneratorParams(blur_sigma=0.0, contrast=2.0, brightness=0.5)
    out = apply_generator(img, params)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_adversarial_step_returns_improved_or_equal_score():
    """A single co-evolution step should not regress below its own base score."""
    img = _sample_image(1)
    target = _sample_image(2)
    mgr = AdversarialManager()
    _, score, _ = mgr.step(img, target)
    assert score >= 0.0


def test_adversarial_step_updates_state():
    img = _sample_image(1)
    target = _sample_image(2)
    mgr = AdversarialManager()
    assert mgr.state.step_count == 0
    mgr.step(img, target)
    assert mgr.state.step_count == 1
    mgr.step(img, target)
    assert mgr.state.step_count == 2


def test_adversarial_step_rejects_shape_mismatch():
    mgr = AdversarialManager()
    a = np.zeros((8, 8, 3), dtype=np.float32)
    b = np.zeros((16, 16, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        mgr.step(a, b)


def test_inject_burst_noise_preserves_bounds():
    mgr = AdversarialManager()
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    noisy = mgr.inject_burst_noise(img, severity=0.5)
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0
    assert noisy.shape == img.shape


def test_inject_burst_noise_changes_some_pixels():
    mgr = AdversarialManager()
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    noisy = mgr.inject_burst_noise(img, severity=0.8)
    assert not np.allclose(noisy, img)


def test_decoder_sigma_stays_within_bounds():
    """Repeated steps should keep decoder_sigma inside [0.05, 2.5]."""
    mgr = AdversarialManager()
    img = _sample_image(10)
    target = _sample_image(20)
    for _ in range(20):
        _, _, sigma = mgr.step(img, target)
    assert 0.05 <= sigma <= 2.5
