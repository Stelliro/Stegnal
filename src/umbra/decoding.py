"""Decoding logic for the Project Umbra toy pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

from .encoding import NoisePacket


class NoiseStreamDecoder:
    """Recover images from :class:`~umbra.encoding.NoisePacket` objects."""

    def __init__(self, denoise_sigma: float | None = 1.0) -> None:
        self.denoise_sigma = denoise_sigma

    def to_config(self) -> dict[str, float | None]:
        """Return a serializable configuration for persistence."""

        return {"denoise_sigma": None if self.denoise_sigma is None else float(self.denoise_sigma)}

    @classmethod
    def from_config(cls, config: dict[str, float | None]) -> "NoiseStreamDecoder":
        """Instantiate the decoder from :meth:`to_config` output."""

        return cls(denoise_sigma=config.get("denoise_sigma"))

    def decode(self, packet: NoisePacket, seed: int) -> np.ndarray:
        """Decode the provided packet using the shared seed."""
        if seed != packet.permutation_seed:
            raise ValueError("Seed mismatch: cannot decode without the original seed")

        height, width = packet.image_shape
        rng = np.random.default_rng(seed)
        flat_size = height * width
        permutation = rng.permutation(flat_size)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(flat_size)

        permuted = packet.encoded
        recovered = permuted[inverse]
        recovered = recovered.reshape(height, width)

        if self.denoise_sigma and self.denoise_sigma > 0:
            recovered = filters.gaussian(recovered, sigma=self.denoise_sigma, preserve_range=True)

        recovered = np.clip(recovered, 0.0, 1.0)
        return recovered.astype(np.float32)

    def decode_to_image(self, packet: NoisePacket, seed: int, path: str | Path) -> None:
        array = self.decode(packet, seed)
        image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L")
        image.save(path)


__all__ = ["NoiseStreamDecoder"]
