# encoding.py

"""Noise-stream encoder for the Project Umbra test build."""
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from umbra.gpu_runtime import GPUAccelerationRequiredError, allocate_pinned_array, require_gpu

try:  # pragma: no cover - exercised indirectly in GPU environments
    import cupy as cp  # type: ignore
    from cupy.cuda import memory as _cupy_memory  # type: ignore
except ImportError:  # pragma: no cover - handled during testing without CuPy
    cp = None  # type: ignore[assignment]
    CuPyOutOfMemoryError = ()  # type: ignore[misc, assignment]

    def is_cupy_out_of_memory_error(exc: BaseException) -> bool:
        return False

else:  # pragma: no cover - GPU specific
    CuPyOutOfMemoryError = _cupy_memory.OutOfMemoryError

    def is_cupy_out_of_memory_error(exc: BaseException) -> bool:
        return isinstance(exc, _cupy_memory.OutOfMemoryError)

from .sound import MessyKeyArtifact, derive_messy_latent

# Backwards compatibility for code paths that relied on the legacy helper name.
_ensure_gpu_available = require_gpu

logger = logging.getLogger(__name__)


class WaveformPlugin:
    """Base class for modular waveform synthesis plugins."""

    name: str = "base"

    def generate(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, MessyKeyArtifact]:
        raise NotImplementedError


class DSSSWaveformPlugin(WaveformPlugin):
    name = "dsss"

    def generate(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, MessyKeyArtifact]:
        chips = rng.integers(0, 2, size=image.size * 2, dtype=np.int8) * 2 - 1
        waveform = np.repeat(image.reshape(-1), 2) * chips
        return waveform.astype(np.float32), MessyKeyArtifact.from_samples(waveform)


class ChaoticWaveformPlugin(WaveformPlugin):
    name = "chaotic"

    def generate(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, MessyKeyArtifact]:
        mu = 3.99
        x = rng.random()
        sequence = np.empty(image.size, dtype=np.float32)
        for idx in range(image.size):
            x = mu * x * (1.0 - x)
            sequence[idx] = x
        waveform = image.reshape(-1) * (sequence * 2 - 1)
        return waveform, MessyKeyArtifact.from_samples(sequence)


class PRNWaveformPlugin(WaveformPlugin):
    name = "prn"

    def generate(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, MessyKeyArtifact]:
        taps = np.array([1, 0, 0, 1, 1], dtype=np.int8)
        register = rng.integers(0, 2, size=taps.size, dtype=np.int8)
        sequence = []
        for _ in range(image.size):
            feedback = np.mod(np.sum(register * taps), 2)
            sequence.append(register[-1])
            register = np.roll(register, 1)
            register[0] = feedback
        chips = np.array(sequence, dtype=np.float32) * 2 - 1
        waveform = image.reshape(-1) * chips
        return waveform, MessyKeyArtifact.from_samples(chips)


_PLUGIN_REGISTRY: dict[str, WaveformPlugin] = {
    DSSSWaveformPlugin.name: DSSSWaveformPlugin(),
    ChaoticWaveformPlugin.name: ChaoticWaveformPlugin(),
    PRNWaveformPlugin.name: PRNWaveformPlugin(),
}


def register_waveform_plugin(factory: Callable[[], WaveformPlugin]) -> None:
    plugin = factory()
    _PLUGIN_REGISTRY[plugin.name] = plugin


def _simulate_uwb_channel(
    signal: np.ndarray,
    rng: np.random.Generator,
    *,
    allow_cpu_fallback: bool,
    prefer_gpu: bool = False,
    return_backend: bool = False,
    hybrid_memory: bool = True,
) -> tuple[Any, Any]:
    """Apply a simple UWB channel model, optionally using GPU."""

    if signal.size == 0:
        return np.array([]), "cpu"

    if prefer_gpu and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            noise = cp.random.normal(0, 0.01, size=signal.size, dtype=cp.float32)
            attenuated = signal_gpu * 0.8 + noise
            if return_backend:
                return attenuated, "gpu"
            return cp.asnumpy(attenuated)
        except CuPyOutOfMemoryError:
            if allow_cpu_fallback:
                logger.debug("GPU OOM in UWB simulation; falling back to CPU")
            else:
                raise

    # CPU fallback
    noise = rng.normal(0, 0.01, size=signal.size).astype(np.float32)
    attenuated = signal * 0.8 + noise
    if return_backend:
        return attenuated, "cpu"
    return attenuated


@dataclass(frozen=True)
class NoisePacket:
    """Container for the encoded noise stream."""

    encoded: np.ndarray
    image_shape: tuple[int, ...]
    permutation_seed: int
    sigma: float
    messy_latent: np.ndarray | None = None
    channel_response: np.ndarray | None = None
    waveform_plugin: str = "dsss"
    encoded_backend: str = "cpu"

    def to_file(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "encoded": self.encoded,
            "image_shape": np.array(self.image_shape, dtype=np.int32),
            "permutation_seed": np.array(self.permutation_seed, dtype=np.int64),
            "sigma": np.array(self.sigma, dtype=np.float32),
            "messy_latent": self.messy_latent,
            "channel_response": self.channel_response,
            "waveform_plugin": np.array(self.waveform_plugin, dtype="S"),
            "encoded_backend": np.array(self.encoded_backend, dtype="S"),
        }
        np.savez_compressed(path, **data)

    @classmethod
    def from_file(cls, path: str | Path) -> NoisePacket:
        try:
            loaded = np.load(path, allow_pickle=False)
        except Exception as exc:
            logger.error(f"Failed to load NoisePacket from {path}: {exc}")
            raise
        return cls(
            encoded=loaded["encoded"],
            image_shape=tuple(loaded["image_shape"]),
            permutation_seed=int(loaded["permutation_seed"]),
            sigma=float(loaded["sigma"]),
            messy_latent=loaded.get("messy_latent"),
            channel_response=loaded.get("channel_response"),
            waveform_plugin=str(loaded["waveform_plugin"]),
            encoded_backend=str(loaded["encoded_backend"]),
        )


class NoiseStreamEncoder:
    """Encode images into a noise stream suitable for sonic transmission."""

    def __init__(self, *, sigma: float = 0.2, waveform_plugin: str = "dsss") -> None:
        self.sigma = float(max(sigma, 0.0))
        try:
            self._plugin = _PLUGIN_REGISTRY[waveform_plugin]
        except KeyError:
            raise ValueError(f"Unknown waveform plugin: {waveform_plugin}")

    def load_image(self, path: str | Path) -> np.ndarray:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to open image at {path}: {exc}")
            raise
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.size == 0:
            raise ValueError("Loaded image is empty")
        return array

    def encode(self, image: np.ndarray, seed: int, *, allow_cpu_fallback: bool = True, prefer_gpu: bool = False) -> NoisePacket:
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        array = np.asarray(image, dtype=np.float32)
        if array.ndim not in (2, 3):
            raise ValueError("Image must be 2D or 3D")
        if array.size == 0:
            raise ValueError("Image is empty")
        image_shape = array.shape
        rng = np.random.default_rng(seed)
        waveform, messy_artifact = self._plugin.generate(array, rng)
        channel, backend = _simulate_uwb_channel(
            waveform, rng, allow_cpu_fallback=allow_cpu_fallback, prefer_gpu=prefer_gpu, return_backend=True
        )
        flat = array.reshape(-1)
        permutation = rng.permutation(flat.size)
        noise = rng.normal(0, self.sigma, size=flat.size).astype(np.float32)
        if prefer_gpu and cp is not None:
            try:
                flat_gpu = cp.asarray(flat)
                permutation_gpu = cp.asarray(permutation)
                noise_gpu = cp.asarray(noise)
                permuted_gpu = flat_gpu[permutation_gpu]
                encoded_gpu = permuted_gpu + noise_gpu
                encoded = cp.asnumpy(encoded_gpu)
                backend = "gpu"
            except CuPyOutOfMemoryError:
                if allow_cpu_fallback:
                    logger.debug("GPU OOM in encoding; falling back to CPU")
                else:
                    raise
        else:
            permuted = flat[permutation]
            encoded = permuted + noise
            backend = "cpu"
        latent = derive_messy_latent(messy_artifact, encoded.shape)
        return NoisePacket(
            encoded=encoded,
            image_shape=image_shape,
            permutation_seed=seed,
            sigma=self.sigma,
            messy_latent=latent,
            channel_response=channel,
            waveform_plugin=self._plugin.name,
            encoded_backend=backend,
        )

    def encode_from_path(
        self, path: str | Path, seed: int, **kwargs: Any
    ) -> NoisePacket:
        image = self.load_image(path)
        return self.encode(image, seed, **kwargs)


__all__ = [
    "NoisePacket",
    "NoiseStreamEncoder",
    "WaveformPlugin",
    "register_waveform_plugin",
]