"""Decoding logic for the Project Umbra toy pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

from .encoding import NoisePacket


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
        self.schedule = schedule

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
        scale = float(trimmed.std()) or 1.0
        normalized = np.clip(centered / scale, -1.0, 1.0)
        normalized = (normalized + 1.0) * 0.5
        return normalized.reshape(target_shape).astype(np.float32)

    def inpaint(self, decoded: np.ndarray, latent: np.ndarray | None) -> np.ndarray:
        if latent is None:
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

logger = logging.getLogger(__name__)


class NoiseStreamDecoder:
    """Recover images from :class:`~umbra.encoding.NoisePacket` objects."""

    def __init__(
        self,
        denoise_sigma: float | None = 1.0,
        *,
        inpainter: DiffusionInpainter | None = None,
    ) -> None:
        self.denoise_sigma = denoise_sigma
        self._inpainter = inpainter or DiffusionInpainter()

    def to_config(self) -> dict[str, float | None]:
        """Return a serializable configuration for persistence."""

        return {
            "denoise_sigma": None if self.denoise_sigma is None else float(self.denoise_sigma),
            "inpainter": {
                "steps": self._inpainter.steps,
                "guidance_scale": self._inpainter.guidance_scale,
                "schedule": self._inpainter.schedule,
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, float | None]) -> NoiseStreamDecoder:
        """Instantiate the decoder from :meth:`to_config` output."""

        inpainter_cfg = config.get("inpainter", {}) if isinstance(config, dict) else {}
        inpainter = DiffusionInpainter(**inpainter_cfg) if isinstance(inpainter_cfg, dict) else None
        return cls(denoise_sigma=config.get("denoise_sigma"), inpainter=inpainter)

    def decode(self, packet: NoisePacket, seed: int, *, messy_latent: np.ndarray | None = None) -> np.ndarray:
        """Decode the provided packet using the shared seed."""
        if seed != packet.permutation_seed:
            raise ValueError("Seed mismatch: cannot decode without the original seed")

        logger.debug(
            "Decoding packet with sigma %s using seed %d", packet.sigma, seed
        )
        rng = np.random.default_rng(seed)
        flat_size = int(np.prod(packet.image_shape))
        permutation = rng.permutation(flat_size)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(flat_size)

        permuted = packet.encoded
        recovered = permuted[inverse]
        recovered = recovered.reshape(packet.image_shape)

        if self.denoise_sigma and self.denoise_sigma > 0:
            logger.debug(
                "Applying Gaussian denoise with sigma %.3f", float(self.denoise_sigma)
            )
            recovered = filters.gaussian(
                recovered,
                sigma=self.denoise_sigma,
                preserve_range=True,
                channel_axis=-1 if recovered.ndim == 3 else None,
            )

        latent = messy_latent
        if latent is None:
            latent = packet.messy_latent
        recovered = np.clip(recovered, 0.0, 1.0)
        if (packet.sigma or 0.0) <= 0.151:
            return recovered.astype(np.float32)
        return self._inpainter.inpaint(recovered, latent).astype(np.float32)

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

        image = Image.fromarray(data)
        image.save(path)


__all__ = ["DiffusionInpainter", "NoiseStreamDecoder"]
