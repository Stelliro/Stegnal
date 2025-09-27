import numpy as np
import pytest

from umbra.visualization import multiplicative_overlap, normalize_for_display, to_uint8_image


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
    expected_overlap = np.clip(reference * candidate, 0.0, 1.0)
    assert np.allclose(overlap, expected_overlap)
    assert score == pytest.approx(expected_overlap.mean() * 100.0)


def test_multiplicative_overlap_requires_matching_shapes():
    with pytest.raises(ValueError):
        multiplicative_overlap(np.zeros((2, 2)), np.zeros((3, 3)))


def test_to_uint8_image_converts_range():
    normalized = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    converted = to_uint8_image(normalized)
    assert converted.dtype == np.uint8
    assert converted[0, 0] == 0
    assert converted[1, 0] == 255
