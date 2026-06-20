"""Tests for umbra.payload — DataPayloadCodec encode/decode."""

from __future__ import annotations

import numpy as np
import pytest

from umbra.payload import DataPayloadCodec


@pytest.fixture()
def codec():
    return DataPayloadCodec(redundancy=9)


def test_round_trip_recovers_data(codec):
    original = b"Hello, Umbra!"
    encoded = codec.encode_file(original, shape=(256, 256))
    recovered = codec.decode_image(encoded)
    assert recovered == original


def test_round_trip_binary_data(codec):
    original = bytes(range(256))
    encoded = codec.encode_file(original, shape=(512, 512))
    recovered = codec.decode_image(encoded)
    assert recovered == original


def test_encode_raises_on_oversized_payload(codec):
    huge = b"\xff" * 100_000
    with pytest.raises(ValueError, match="too big"):
        codec.encode_file(huge, shape=(64, 64))


def test_decode_returns_none_for_noise(codec):
    """Random noise should not produce a valid decode."""
    rng = np.random.default_rng(42)
    noise = rng.random((256, 256, 3), dtype=np.float32)
    result = codec.decode_image(noise)
    # With random noise the magic marker is almost certainly absent
    assert result is None


def test_decode_grayscale_input(codec):
    original = b"gray"
    shape = (128, 128)
    encoded_rgb = codec.encode_file(original, shape=shape)
    # Collapse to grayscale (mean across channels)
    gray = encoded_rgb.mean(axis=2)
    recovered = codec.decode_image(gray)
    assert recovered == original


def test_scramble_permutation_is_deterministic(codec):
    perm1 = codec._get_permutation(1000)
    perm2 = codec._get_permutation(1000)
    np.testing.assert_array_equal(perm1, perm2)
