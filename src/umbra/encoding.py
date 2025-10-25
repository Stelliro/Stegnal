"""Noise-stream encoder for the Project Umbra test build."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from umbra.gpu_runtime import GPUAccelerationRequiredError, allocate_pinned_array, require_gpu

try:  # pragma: no cover - exercised indirectly in GPU environments
    import cupy as cp  # type: ignore
    from cupy.cuda import memory as _cupy_memory  # type: ignore
except Exception:  # pragma: no cover - handled during testing without CuPy
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
            register[1:] = register[:-1]
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
    """Apply a simple UWB channel model, optionally returning GPU buffers.

    When ``prefer_gpu`` is ``True`` and CuPy is available, the channel simulation
    is executed on the GPU. If ``return_backend`` is also ``True`` the resulting
    waveform and channel response are returned as CuPy arrays so callers can keep
    subsequent work on the accelerator. Otherwise the results are converted back
    to NumPy to avoid unexpected device transfers.
    """

    def _simulate_with_backend(
        backend: Any,
        *,
        keep_backend: bool,
    ) -> tuple[Any, Any]:
        xp = backend
        xp_float32 = getattr(xp, "float32", np.float32)
        if xp is np:
            signal_backend = np.asarray(signal, dtype=np.float32)
        else:
            try:
                signal_backend = xp.asarray(signal, dtype=xp_float32)
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                if (not hybrid_memory or not allow_cpu_fallback) or not is_cupy_out_of_memory_error(exc):
                    raise
                hybrid_signal = allocate_pinned_array(signal.shape, np.float32)
                hybrid_signal[...] = np.asarray(signal, dtype=np.float32)
                signal_backend = hybrid_signal

        taps = 6
        max_delay = max(2, int(signal_backend.size // 8) or 2)
        delays = rng.integers(1, max_delay, size=taps)
        gains_cpu = rng.rayleigh(scale=0.6, size=taps).astype(np.float32)
        if xp is np:
            gains_backend = gains_cpu
        else:
            try:
                gains_backend = xp.asarray(gains_cpu, dtype=xp_float32)
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                if (not hybrid_memory or not allow_cpu_fallback) or not is_cupy_out_of_memory_error(exc):
                    raise
                hybrid_gains = allocate_pinned_array(gains_cpu.shape, np.float32)
                hybrid_gains[...] = gains_cpu
                gains_backend = hybrid_gains

        if xp is np:
            response = np.zeros_like(signal_backend, dtype=np.float32)
        else:
            try:
                response = xp.zeros_like(signal_backend, dtype=xp_float32)
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                if (not hybrid_memory or not allow_cpu_fallback) or not is_cupy_out_of_memory_error(exc):
                    raise
                response = allocate_pinned_array(signal_backend.shape, np.float32)
                response.fill(0)
        for gain, delay in zip(gains_backend, delays):
            response[delay:] += gain * signal_backend[:-delay]

        faded = 0.6 * signal_backend + response
        faded = faded / xp.max(xp.abs(faded) + 1e-6)

        if xp is np or not keep_backend:
            faded_np = (
                faded.astype(np.float32, copy=False)
                if xp is np
                else cp.asnumpy(faded).astype(np.float32, copy=False)
            )
            gains_np = (
                gains_backend.astype(np.float32, copy=False)
                if xp is np
                else cp.asnumpy(gains_backend).astype(np.float32, copy=False)
            )
            return faded_np, gains_np

        return (
            faded.astype(xp_float32, copy=False),
            gains_backend.astype(xp_float32, copy=False),
        )

    if not allow_cpu_fallback and cp is None:
        raise GPUAccelerationRequiredError(
            "GPU acceleration via CuPy is required for UWB channel simulation; CPU fallback is disabled."
        )

    prefer_gpu = (prefer_gpu or not allow_cpu_fallback) and cp is not None

    if prefer_gpu:
        try:
            require_gpu("UWB channel simulation")
        except GPUAccelerationRequiredError:
            if not allow_cpu_fallback:
                raise
        else:
            try:
                return _simulate_with_backend(cp, keep_backend=return_backend)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - exercised via integration tests
                if not allow_cpu_fallback:
                    if isinstance(exc, CuPyOutOfMemoryError) or is_cupy_out_of_memory_error(exc):
                        raise
                    raise GPUAccelerationRequiredError(
                        "GPU acceleration failed while simulating the UWB channel. "
                        "Ensure the CUDA runtime (including nvrtc) is installed or enable CPU fallback."
                    ) from exc

    # CPU fallback or preferred execution.
    return _simulate_with_backend(np, keep_backend=False)


@dataclass
class NoisePacket:
    """Container for an encoded image represented as noise."""

    encoded: np.ndarray
    image_shape: tuple[int, ...]
    permutation_seed: int
    sigma: float
    messy_latent: np.ndarray | None = None
    channel_response: np.ndarray | None = None
    waveform_plugin: str | None = None
    encoded_backend: Any | None = None

    def to_file(self, path: str | Path) -> None:
        """Serialize the packet to disk using NumPy."""
        logger.debug("Serializing noise packet to %s", path)
        payload: dict[str, np.ndarray] = {
            "encoded": self.encoded.astype(np.float32),
            "image_shape": np.array(self.image_shape, dtype=np.int64),
            "permutation_seed": np.array([self.permutation_seed], dtype=np.int64),
            "sigma": np.array([self.sigma], dtype=np.float32),
        }
        if self.messy_latent is not None:
            payload["messy_latent"] = self.messy_latent.astype(np.float32)
        if self.channel_response is not None:
            payload["channel_response"] = self.channel_response.astype(np.float32)
        if self.waveform_plugin is not None:
            payload["waveform_plugin"] = np.array([self.waveform_plugin], dtype=np.str_)

        np.savez_compressed(path, **payload)

    @classmethod
    def from_file(cls, path: str | Path) -> NoisePacket:
        logger.debug("Loading noise packet from %s", path)
        with np.load(path) as data:
            encoded = data["encoded"].astype(np.float32)
            shape = tuple(int(v) for v in data["image_shape"].astype(int))
            seed = int(data["permutation_seed"][0])
            sigma = float(data["sigma"][0])
            messy_latent = data.get("messy_latent")
            channel_response = data.get("channel_response")
            plugin_array = data.get("waveform_plugin")
            plugin_name = str(plugin_array[0]) if plugin_array is not None else None
        return cls(
            encoded=encoded,
            image_shape=shape,
            permutation_seed=seed,
            sigma=sigma,
            messy_latent=messy_latent.astype(np.float32) if messy_latent is not None else None,
            channel_response=(
                channel_response.astype(np.float32) if channel_response is not None else None
            ),
            waveform_plugin=plugin_name,
            encoded_backend=None,
        )


class NoiseStreamEncoder:
    """Encode an image into a pseudo-noise stream."""

    def __init__(self, sigma: float = 0.2, *, waveform: str = "dsss") -> None:
        self.sigma = sigma
        self.waveform = waveform if waveform in _PLUGIN_REGISTRY else "dsss"

    def to_config(self) -> dict[str, float]:
        """Return a serializable configuration for persistence."""

        return {"sigma": float(self.sigma), "waveform": self.waveform}

    @classmethod
    def from_config(cls, config: dict[str, float]) -> NoiseStreamEncoder:
        """Instantiate the encoder from :meth:`to_config` output."""

        return cls(sigma=float(config.get("sigma", 0.2)), waveform=str(config.get("waveform", "dsss")))

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load an image as a float32 array in [0, 1]."""
        image = Image.open(path).convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return arr

    def encode(
        self,
        image: np.ndarray,
        seed: int,
        *,
        allow_cpu_fallback: bool = True,
        use_gpu: bool | None = None,
    ) -> NoisePacket:
        """Encode the image using a permutation driven by ``seed``."""
        if image.ndim not in (2, 3):
            raise ValueError("Expected image array with shape (H, W) or (H, W, C)")
        image_shape: tuple[int, ...] = tuple(int(dim) for dim in image.shape)
        logger.debug(
            "Encoding image with shape %s using sigma %.3f and seed %d",
            image_shape,
            float(self.sigma),
            seed,
        )
        rng = np.random.default_rng(seed)
        plugin = _PLUGIN_REGISTRY[self.waveform]
        plugin_rng = np.random.default_rng(seed ^ 0x5F5A1)
        flat = np.asarray(image, dtype=np.float32).reshape(-1)
        waveform, messy_artifact = plugin.generate(flat, plugin_rng)
        prefer_gpu = cp is not None
        uwb_waveform_backend, channel_backend = _simulate_uwb_channel(
            waveform,
            plugin_rng,
            allow_cpu_fallback=allow_cpu_fallback,
            prefer_gpu=prefer_gpu,
            return_backend=True,
        )

        permutation = rng.permutation(flat.size)
        permuted = flat[permutation]
        noise = rng.normal(0.0, self.sigma, size=permuted.shape)

        channel: np.ndarray | None
        encoded_backend: dict[str, Any] | None
        cp_array_type: tuple[type[Any], ...] = ()
        if cp is not None:
            ndarray_type = getattr(cp, "ndarray", None)
            if ndarray_type is not None:
                cp_array_type = (ndarray_type,)
        using_gpu_backend = bool(cp_array_type) and isinstance(uwb_waveform_backend, cp_array_type)

        if using_gpu_backend:
            uwb_waveform = cp.asnumpy(uwb_waveform_backend).astype(np.float32, copy=False)
            channel = cp.asnumpy(channel_backend).astype(np.float32, copy=False)
            encoded_backend = {
                "uwb_waveform": uwb_waveform_backend,
                "channel_response": channel_backend,
            }
        else:
            uwb_waveform = np.asarray(uwb_waveform_backend, dtype=np.float32)
            channel = (
                None
                if channel_backend is None
                else np.asarray(channel_backend, dtype=np.float32)
            )
            encoded_backend = None

        uwb_waveform = uwb_waveform[: permuted.size]
        use_gpu = cp is not None if use_gpu is None else bool(use_gpu and cp is not None)
        flat_gpu = permutation_gpu = noise_gpu = uwb_gpu = None
        if use_gpu:
            try:
                flat_gpu = cp.asarray(flat)
                permutation_gpu = cp.asarray(permutation)
                noise_gpu = cp.asarray(noise)
                uwb_gpu = cp.asarray(uwb_waveform)
            except CuPyOutOfMemoryError:  # type: ignore[misc]
                if allow_cpu_fallback:
                    logger.debug(
                        "CuPy ran out of memory while allocating GPU buffers; falling back to CPU.",
                        exc_info=True,
                    )
                    use_gpu = False
                else:
                    raise
            except Exception as exc:
                if allow_cpu_fallback and is_cupy_out_of_memory_error(exc):
                    logger.debug(
                        "CuPy ran out of memory while allocating GPU buffers; falling back to CPU.",
                        exc_info=True,
                    )
                    use_gpu = False
                else:
                    raise
        if (
            use_gpu
            and flat_gpu is not None
            and permutation_gpu is not None
            and noise_gpu is not None
            and uwb_gpu is not None
        ):
            permuted_gpu = flat_gpu[permutation_gpu]
            encoded_gpu = permuted_gpu + noise_gpu
            if self.sigma >= 0.3:
                encoded_gpu = encoded_gpu + 0.01 * uwb_gpu
            encoded = cp.asnumpy(encoded_gpu)
        else:
            encoded = permuted + noise
            if self.sigma >= 0.3:
                encoded = encoded + 0.01 * uwb_waveform
        latent = derive_messy_latent(messy_artifact, encoded.shape)
        return NoisePacket(
            encoded=encoded,
            image_shape=image_shape,
            permutation_seed=seed,
            sigma=self.sigma,
            messy_latent=latent,
            channel_response=channel,
            waveform_plugin=plugin.name,
            encoded_backend=encoded_backend,
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
