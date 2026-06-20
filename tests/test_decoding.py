"""Tests for umbra.decoding — NoiseStreamDecoder and gene corrections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from umbra.decoding import DiffusionInpainter, NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder


@dataclass
class _FakeGenes:
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0
    brightness_shift: float = 0.0
    contrast_scale: float = 1.0
    gamma: float = 1.0
    denoise_sigma: float = 0.5


def test_decode_round_trip_with_genes():
    """Encode an image, then decode with gene corrections and check shape/range."""
    rng = np.random.default_rng(42)
    image = rng.random((16, 16, 3), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.1)
    packet = encoder.encode(image, seed=99)

    decoder = NoiseStreamDecoder(denoise_sigma=0.5)
    genes = _FakeGenes(r_gain=1.1, brightness_shift=0.02, gamma=1.2)
    result = decoder.decode(packet, seed=99, genes=genes)

    assert result.shape == image.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_decode_without_genes():
    rng = np.random.default_rng(7)
    image = rng.random((16, 16, 3), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.05)
    packet = encoder.encode(image, seed=42)

    decoder = NoiseStreamDecoder(denoise_sigma=0.3)
    result = decoder.decode(packet, seed=42)
    assert result.shape == image.shape


def test_gene_corrections_clip_extreme_gamma():
    decoder = NoiseStreamDecoder()
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    genes = _FakeGenes(gamma=100.0)  # should be clipped to 3.0
    corrected = decoder.apply_gene_corrections(img, genes)
    assert corrected.min() >= 0.0
    assert corrected.max() <= 1.0


def test_gene_corrections_none_returns_same():
    decoder = NoiseStreamDecoder()
    img = np.full((4, 4, 3), 0.7, dtype=np.float32)
    result = decoder.apply_gene_corrections(img, None)
    np.testing.assert_array_equal(result, img)


def test_diffusion_inpainter_no_latent_returns_decoded():
    inpainter = DiffusionInpainter()
    arr = np.ones((8, 8, 3), dtype=np.float32)
    out = inpainter.inpaint(arr, None)
    np.testing.assert_array_equal(out, arr)


def test_diffusion_inpainter_with_latent_returns_copy():
    inpainter = DiffusionInpainter()
    arr = np.ones((8, 8, 3), dtype=np.float32)
    latent = np.zeros((8, 8, 3), dtype=np.float32)
    out = inpainter.inpaint(arr, latent)
    np.testing.assert_array_equal(out, arr)
    assert out is not arr  # should be a copy


def test_save_image_creates_file(tmp_path):
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out_path = tmp_path / "sub" / "test.png"
    result = NoiseStreamDecoder.save_image(img, out_path)
    assert result.exists()
    assert result.suffix == ".png"


def test_save_image_handles_grayscale(tmp_path):
    img = np.full((8, 8), 0.3, dtype=np.float32)
    out_path = tmp_path / "gray.png"
    result = NoiseStreamDecoder.save_image(img, out_path)
    assert result.exists()
