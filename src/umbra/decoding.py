# decoding.py

"""Decoding logic for the Project Umbra toy pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

try:  # pragma: no cover - optional acceleration
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cp = None

try:  # pragma: no cover - optional acceleration
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cupy_gaussian_filter = None

from .encoding import NoisePacket
from .gpu_runtime import is_cupy_out_of_memory_error

logger = logging.getLogger(__name__)


class DiffusionInpainter:
    """Lightweight diffusion-style inpainting stub for permutation artefacts.

    The implementation intentionally avoids heavy deep-learning dependencies
    while still emulating the behaviour of a denoising diffusion model.  The
    inpainter accepts a *conditioning latent* derived from the sound pipeline's
    "messy key" artefacts which we treat as a guidance vector.  During
    reconstruction the latent is reshaped to match the decoded frame and the
    routine performs a handful of predictor-corrector steps that blend the
    decoded pixels with the latent-driven prior.

    Even though the routine is deliberately simple it improves SSIM/PSNR in the
    toy pipeline by smoothing out holes introduced by the permutation/noise
    process.
    """

    def __init__(
        self,
        *,
        steps: int = 6,
        guidance_scale: float = 0.2,
        schedule: str = "cosine",
    ) -> None:
        self.steps = max(1, int(steps))
        self.guidance_scale = float(np.clip(guidance_scale, 0.0, 2.0))
        self.schedule = schedule.lower()
        if self.schedule not in ("linear", "cosine"):
            raise ValueError("Schedule must be 'linear' or 'cosine'")

    def _noise_schedule(self, step: int) -> float:
        if self.schedule == "linear":
            return 1.0 - (step / max(self.steps - 1, 1))
        # cosine schedule keeps noise high at the beginning and then decays
        fraction = step / max(self.steps - 1, 1)
        return float(np.cos(np.pi * fraction / 2.0))

    def _prepare_latent(self, latent: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        if latent.size == 0:
            return np.zeros(target_shape, dtype=np.float32)
        reshaped = latent.reshape(-1)
        required = int(np.prod(target_shape))
        if reshaped.size < required:
            repeats = int(np.ceil(required / reshaped.size))
            reshaped = np.tile(reshaped, repeats)
        trimmed = reshaped[:required]
        centered = trimmed - float(trimmed.mean())
        scale = float(np.std(trimmed)) or 1.0
        normalized = np.clip(centered / scale, -1.0, 1.0)
        normalized = (normalized + 1.0) * 0.5
        return normalized.reshape(target_shape).astype(np.float32)

    def inpaint(self, decoded: np.ndarray, latent: np.ndarray | None) -> np.ndarray:
        if latent is None or latent.size == 0:
            return decoded

        guided_latent = self._prepare_latent(latent, decoded.shape)
        working = decoded.astype(np.float32, copy=True)

        for step in range(self.steps):
            noise_level = self._noise_schedule(step)
            blend = self.guidance_scale * guided_latent + (1.0 - self.guidance_scale) * working
            laplacian = filters.laplace(working)
            working = working + noise_level * (blend - working) - 0.05 * laplacian
            working = np.clip(working, 0.0, 1.0)

        return working.astype(np.float32)


class NoiseStreamDecoder:
    """Decode a :class:`NoisePacket` back into an image."""

    def __init__(
        self,
        *,
        denoise_sigma: float | None = 1.0,
        inpainter: DiffusionInpainter | None = None,
    ) -> None:
        self.denoise_sigma = float(denoise_sigma) if denoise_sigma is not None else None
        self._inpainter = inpainter or DiffusionInpainter()

    def decode(self, packet: NoisePacket, seed: int, *, allow_cpu_fallback: bool = True) -> np.ndarray:
        if packet.encoded.size == 0:
            raise ValueError("Packet encoded data is empty")
        if packet.permutation_seed != seed:
            raise ValueError("Seed mismatch during decoding")

        rng = np.random.default_rng(seed)
        inverse = np.argsort(rng.permutation(packet.encoded.size))

        use_gpu = cp is not None
        recovered = None
        gaussian_applied = False
        messy_latent = packet.messy_latent

        if use_gpu:
            try:
                if packet.encoded_backend == "gpu" and packet.encoded_gpu is not None:
                    encoded_gpu = cp.asarray(packet.encoded_gpu)
                else:
                    encoded_gpu = cp.asarray(packet.encoded)
                inverse_gpu = cp.asarray(inverse)
                recovered_gpu = encoded_gpu[inverse_gpu]
                recovered_gpu = recovered_gpu.reshape(packet.image_shape)

                if self.denoise_sigma and self.denoise_sigma > 0 and cupy_gaussian_filter is not None:
                    logger.debug(
                        "Applying Gaussian denoise with sigma %.3f on GPU",
                        float(self.denoise_sigma),
                    )
                    if recovered_gpu.ndim == 3 and recovered_gpu.shape[-1] > 1:
                        channels = [
                            cupy_gaussian_filter(
                                recovered_gpu[..., idx],
                                sigma=self.denoise_sigma,
                                mode="reflect",
                            )
                            for idx in range(recovered_gpu.shape[-1])
                        ]
                        recovered_gpu = cp.stack(channels, axis=-1)  # type: ignore[arg-type]
                    else:
                        recovered_gpu = cupy_gaussian_filter(
                            recovered_gpu,
                            sigma=self.denoise_sigma,
                            mode="reflect",
                        )
                    gaussian_applied = True

                recovered = cp.asnumpy(recovered_gpu)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                if allow_cpu_fallback and is_cupy_out_of_memory_error(exc):
                    logger.debug(
                        "CuPy out-of-memory during decode; falling back to CPU path",
                        exc_info=True,
                    )
                    use_gpu = False
                else:
                    raise

        if not use_gpu:
            permuted = packet.encoded
            recovered = permuted[inverse]
            recovered = recovered.reshape(packet.image_shape)

        if recovered is None:  # pragma: no cover - safety net
            raise RuntimeError("Failed to reconstruct permutation during decoding")

        if self.denoise_sigma and self.denoise_sigma > 0 and not gaussian_applied:
            logger.debug(
                "Applying Gaussian denoise with sigma %.3f", float(self.denoise_sigma)
            )
            recovered = filters.gaussian(
                recovered,
                sigma=self.denoise_sigma,
                preserve_range=True,
                channel_axis=-1 if recovered.ndim == 3 else None,
            )

        recovered = np.clip(recovered, 0.0, 1.0)
        if (packet.sigma or 0.0) <= 0.151:
            return recovered.astype(np.float32)
        return self._inpainter.inpaint(recovered, messy_latent).astype(np.float32)

    def decode_to_image(self, packet: NoisePacket, seed: int, path: str | Path) -> None:
        array = self.decode(packet, seed)
        data = (array * 255.0).astype(np.uint8)
        if array.ndim == 3:
            if array.shape[2] == 1:
                data = data[:, :, 0]
            elif array.shape[2] not in (3, 4):
                raise ValueError("Unsupported array shape for image export")
        elif array.ndim != 2:
            raise ValueError("Unsupported array shape for image export")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(data)
        image.save(path)

    def save_image(self, array: np.ndarray, path: str | Path) -> None:
        """Save a normalized float image array directly to disk.

        Expects values in [0, 1] with shape (H, W) or (H, W, C) where C in {1,3,4}.
        """

        data = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        if array.ndim == 3:
            if array.shape[2] == 1:
                data = data[:, :, 0]
            elif array.shape[2] not in (3, 4):
                raise ValueError("Unsupported array shape for image export")
        elif array.ndim != 2:
            raise ValueError("Unsupported array shape for image export")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(data)
        image.save(path)


__all__ = ["DiffusionInpainter", "NoiseStreamDecoder"]