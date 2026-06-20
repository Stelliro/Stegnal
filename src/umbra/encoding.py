# encoding.py
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .gpu_runtime import (
    CuPyOutOfMemoryError,
    GPUAccelerationRequiredError,
    is_cupy_out_of_memory_error,
    require_gpu,
)

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None  # type: ignore[assignment]

logger = logging.getLogger("umbra.encoding")


@dataclass
class NoisePacket:
    encoded: np.ndarray
    permutation_seed: int
    image_shape: tuple
    sigma: float
    messy_latent: np.ndarray | None = None

    def to_file(self, path: str | Path) -> None:
        """Save encoded packet to an ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "encoded": self.encoded,
            "permutation_seed": np.array(self.permutation_seed),
            "image_shape": np.array(self.image_shape),
            "sigma": np.array(self.sigma),
        }
        if self.messy_latent is not None:
            data["messy_latent"] = self.messy_latent
        np.savez(path, **data)

    @classmethod
    def from_file(cls, path: str | Path) -> NoisePacket:
        """Load a packet from an ``.npz`` file."""
        loaded = np.load(str(path), allow_pickle=False)
        messy = loaded["messy_latent"] if "messy_latent" in loaded else None
        return cls(
            encoded=loaded["encoded"],
            permutation_seed=int(loaded["permutation_seed"]),
            image_shape=tuple(loaded["image_shape"]),
            sigma=float(loaded["sigma"]),
            messy_latent=messy,
        )


class NoiseStreamEncoder:
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load an image file and return it as a float32 RGB array in [0, 1]."""
        img = Image.open(str(path)).convert("RGB")
        array = np.asarray(img, dtype=np.float32) / 255.0
        return np.clip(array, 0.0, 1.0)

    def encode(
        self,
        image_array: np.ndarray,
        seed: int | None = None,
        *,
        use_gpu: bool = False,
        allow_cpu_fallback: bool = True,
    ) -> NoisePacket:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Ensure 3-channel
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)

        orig_shape = image_array.shape
        num_pixels = image_array.shape[0] * image_array.shape[1]

        pixels = image_array.reshape((num_pixels, 3)).astype(np.float32)

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(num_pixels)

        encoded_data = pixels[shuffled_indices, :].flatten()

        if use_gpu and cp is not None:
            try:
                encoded_data = cp.asarray(encoded_data)
                if self.sigma > 0:
                    noise = cp.random.normal(0, self.sigma, encoded_data.shape)
                    encoded_data = cp.clip(encoded_data + noise, 0, 1)
                encoded_data = cp.asnumpy(encoded_data)
            except Exception as exc:
                if is_cupy_out_of_memory_error(exc) or isinstance(exc, CuPyOutOfMemoryError):
                    if allow_cpu_fallback:
                        logger.debug("GPU OOM during encode, falling back to CPU")
                        encoded_data = np.asarray(encoded_data, dtype=np.float32)
                        if self.sigma > 0:
                            noise = np.random.normal(0, self.sigma, encoded_data.shape)
                            encoded_data = np.clip(encoded_data + noise, 0, 1)
                    else:
                        raise
                else:
                    raise
        elif use_gpu and cp is None:
            if not allow_cpu_fallback:
                raise GPUAccelerationRequiredError(
                    "GPU acceleration via CuPy is required for encode; CPU fallback is disabled."
                )
            # Fall through to CPU path
            if self.sigma > 0:
                noise = np.random.normal(0, self.sigma, encoded_data.shape)
                encoded_data = np.clip(encoded_data + noise, 0, 1)
        else:
            if not allow_cpu_fallback and cp is None:
                raise GPUAccelerationRequiredError(
                    "GPU acceleration via CuPy is required for encode; CPU fallback is disabled."
                )
            if self.sigma > 0:
                noise = np.random.normal(0, self.sigma, encoded_data.shape)
                encoded_data = np.clip(encoded_data + noise, 0, 1)

        return NoisePacket(
            encoded=np.asarray(encoded_data, dtype=np.float32),
            permutation_seed=seed,
            image_shape=orig_shape,
            sigma=self.sigma,
            messy_latent=np.random.normal(0, 1.0, (64, 64)),
        )


def _ensure_gpu_available(operation: str) -> None:
    """Ensure a GPU backend is available or raise."""
    require_gpu(operation)


def _simulate_uwb_channel(
    signal: np.ndarray,
    rng: np.random.Generator,
    *,
    allow_cpu_fallback: bool = True,
    prefer_gpu: bool = False,
    return_backend: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate an ultra-wideband transmission channel on *signal*."""

    signal = np.asarray(signal, dtype=np.float32)
    channel_params = rng.random(6).astype(np.float32)

    if prefer_gpu and cp is not None:
        try:
            gpu_sig = cp.asarray(signal)
            result = cp.asnumpy(gpu_sig)
        except Exception:
            if allow_cpu_fallback:
                result = signal.copy()
            else:
                raise GPUAccelerationRequiredError(
                    "GPU acceleration required for UWB simulation; CPU fallback is disabled."
                )
    elif prefer_gpu and cp is None:
        if not allow_cpu_fallback:
            raise GPUAccelerationRequiredError(
                "GPU acceleration required for UWB simulation; CPU fallback is disabled."
            )
        result = signal.copy()
    else:
        result = signal.copy()

    if return_backend:
        return result, channel_params
    return result