"""Noise-stream encoder for the Project Umbra test build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class NoisePacket:
    """Container for an encoded image represented as noise."""

    encoded: np.ndarray
    image_shape: Tuple[int, int]
    permutation_seed: int
    sigma: float

    def to_file(self, path: str | Path) -> None:
        """Serialize the packet to disk using NumPy."""
        np.savez_compressed(
            path,
            encoded=self.encoded.astype(np.float32),
            image_shape=np.array(self.image_shape, dtype=np.int64),
            permutation_seed=np.array([self.permutation_seed], dtype=np.int64),
            sigma=np.array([self.sigma], dtype=np.float32),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "NoisePacket":
        with np.load(path) as data:
            encoded = data["encoded"].astype(np.float32)
            height, width = data["image_shape"].astype(int)
            seed = int(data["permutation_seed"][0])
            sigma = float(data["sigma"][0])
        return cls(encoded=encoded, image_shape=(height, width), permutation_seed=seed, sigma=sigma)


class NoiseStreamEncoder:
    """Encode an image into a pseudo-noise stream."""

    def __init__(self, sigma: float = 0.2) -> None:
        self.sigma = sigma

    def to_config(self) -> dict[str, float]:
        """Return a serializable configuration for persistence."""

        return {"sigma": float(self.sigma)}

    @classmethod
    def from_config(cls, config: dict[str, float]) -> "NoiseStreamEncoder":
        """Instantiate the encoder from :meth:`to_config` output."""

        return cls(sigma=float(config.get("sigma", 0.2)))

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load an image as a float32 array in [0, 1]."""
        image = Image.open(path).convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return arr

    def encode(self, image: np.ndarray, seed: int) -> NoisePacket:
        """Encode the image using a permutation driven by ``seed``."""
        if image.ndim != 2:
            raise ValueError("Expected grayscale image array with shape (H, W)")
        height, width = image.shape
        rng = np.random.default_rng(seed)
        flat = image.flatten()
        permutation = rng.permutation(flat.size)
        permuted = flat[permutation]
        noise = rng.normal(0.0, self.sigma, size=permuted.shape)
        encoded = permuted + noise
        return NoisePacket(encoded=encoded, image_shape=(height, width), permutation_seed=seed, sigma=self.sigma)

    def encode_from_path(self, path: str | Path, seed: int) -> NoisePacket:
        image = self.load_image(path)
        return self.encode(image, seed)


__all__ = ["NoisePacket", "NoiseStreamEncoder"]
