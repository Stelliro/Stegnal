"""Tests for the noise reconstruction helpers."""

from __future__ import annotations

import numpy as np

from umbra.reconstruction import (
    create_variations,
    generate_shape_collage,
    run_reconstruction_cycle,
    waveform_to_wav_bytes,
)


def test_generate_shape_collage_produces_multiple_shapes() -> None:
    collage, shapes = generate_shape_collage(1234, resolution=(96, 128))
    assert collage.shape == (96, 128, 3)
    assert np.all(collage >= 0.0) and np.all(collage <= 1.0)
    assert 3 <= len(shapes) <= 15


def test_create_variations_preserves_bounds() -> None:
    base, _ = generate_shape_collage(42, resolution=(64, 64), shape_count=7)
    rng = np.random.default_rng(1)
    variations = create_variations(
        base,
        variation_count=5,
        noise_sigma=0.4,
        dropout_probability=0.25,
        rng=rng,
    )
    assert variations.shape == (5, 64, 64, 3)
    assert np.all(variations >= 0.0) and np.all(variations <= 1.0)


def test_run_reconstruction_cycle_returns_audio_and_hybrid() -> None:
    result = run_reconstruction_cycle(
        77,
        resolution=(48, 48),
        variation_count=4,
        noise_sigma=0.25,
        dropout_probability=0.3,
        sample_rate=24_000,
    )
    assert result.base_image.shape == (48, 48, 3)
    assert result.ensemble_prediction.shape == (48, 48, 3)
    assert result.hybrid_prediction.shape == (48, 48, 3)
    assert result.coverage.shape == (48, 48)
    assert result.variations.shape[0] == 4
    assert np.all(result.coverage >= 0.0) and np.all(result.coverage <= 1.0)

    wav_bytes = waveform_to_wav_bytes(result.waveform, result.sample_rate)
    assert wav_bytes[:4] == b"RIFF"

