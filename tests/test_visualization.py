import numpy as np
import pytest

from umbra.visualization import (
    colorize_comparison,
    multiplicative_overlap,
    normalize_for_display,
    to_uint8_image,
)


def test_normalize_for_display_scales_between_zero_and_one():
    arr = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    normalized = normalize_for_display(arr)
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_normalize_for_display_handles_constant_array():
    arr = np.full((2, 2), 0.5, dtype=np.float32)
    normalized = normalize_for_display(arr)
    assert np.allclose(normalized, 0.0)


def test_multiplicative_overlap_returns_map_and_score():
    reference = np.array([[0.2, 0.5], [0.8, 1.0]], dtype=np.float32)
    candidate = np.array([[0.2, 0.3], [0.4, 0.9]], dtype=np.float32)
    overlap, score = multiplicative_overlap(reference, candidate)
    expected_overlap = np.clip(1.0 - np.abs(reference - candidate), 0.0, 1.0)
    assert np.allclose(overlap, expected_overlap)
    assert score == pytest.approx(expected_overlap.mean() * 100.0)


def test_multiplicative_overlap_reaches_hundred_for_identical_images():
    reference = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3)
    overlap, score = multiplicative_overlap(reference, reference.copy())
    assert np.allclose(overlap, 1.0)
    assert score == pytest.approx(100.0)


def test_multiplicative_overlap_requires_matching_shapes():
    with pytest.raises(ValueError):
        multiplicative_overlap(np.zeros((2, 2)), np.zeros((3, 3)))


def test_to_uint8_image_converts_range():
    normalized = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    converted = to_uint8_image(normalized)
    assert converted.dtype == np.uint8
    assert converted[0, 0] == 0
    assert converted[1, 0] == 255


def test_colorize_comparison_rgb_overlay_has_expected_shape():
    height, width = 4, 5
    reference = np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32).reshape(height, width, 3)
    candidate = reference[::-1]

    overlay = colorize_comparison(reference, candidate)

    assert overlay.shape == (height, width, 3)
    assert overlay.dtype == np.float32
    assert np.isfinite(overlay).all()


def test_colorize_comparison_block_average_preserves_rgb_shape():
    height, width = 6, 6
    reference = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    candidate = reference.T

    overlay = colorize_comparison(reference, candidate, block_size=3)

    assert overlay.shape == (height, width, 3)
    assert overlay.dtype == np.float32
    assert np.isfinite(overlay).all()
