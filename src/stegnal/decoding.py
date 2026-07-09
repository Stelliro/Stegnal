# decoding.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter  # type: ignore
except ImportError:
    cp = None  # type: ignore[assignment]
    cupy_gaussian_filter = None  # type: ignore[assignment]

from .gpu_runtime import (
    CuPyOutOfMemoryError,
    GPUAccelerationRequiredError,
    is_cupy_out_of_memory_error,
)

logger = logging.getLogger(__name__)

class DiffusionInpainter:
    def __init__(self, steps: int = 6, guidance_scale: float = 0.2, schedule: str = "cosine") -> None:
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.schedule = schedule

    def inpaint(self, decoded: np.ndarray, latent: np.ndarray | None) -> np.ndarray:
        if latent is None:
            return decoded
        return decoded.copy()

class NoiseStreamDecoder:
    def __init__(self, denoise_sigma: float | None = 1.0, inpainter: DiffusionInpainter | None = None) -> None:
        self.denoise_sigma = denoise_sigma
        self._inpainter = inpainter or DiffusionInpainter()

    def apply_gene_corrections(self, image: np.ndarray, genes: object) -> np.ndarray:
        """Applies the evolved color/contrast genes to the reconstructed image."""
        if genes is None:
            return image
        
        # 1. RGB Gains
        if image.shape[-1] == 3:
            gains = np.array([genes.r_gain, genes.g_gain, genes.b_gain])
            image = image * gains.reshape(1, 1, 3)
            
        # 2. Brightness / Contrast
        image = (image + genes.brightness_shift) * genes.contrast_scale
        
        # 3. Gamma Correction
        safe_gamma = np.clip(genes.gamma, 0.1, 3.0)
        image = np.clip(image, 0, 1) ** (1.0 / safe_gamma)
        
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def decode(
        self,
        packet: object,
        seed: int,
        genes: object = None,
        *,
        use_gpu: bool = False,
        allow_cpu_fallback: bool = True,
    ) -> np.ndarray:
        num_pixels = packet.image_shape[0] * packet.image_shape[1]

        # 1. Handle corrupted/partial packets
        if packet.encoded.size < num_pixels * 3:
            missing = (num_pixels * 3) - packet.encoded.size
            packet.encoded = np.pad(packet.encoded, (0, missing), 'constant')

        # The inverse permutation is derived from the shared seed (cheap, CPU).
        rng = np.random.default_rng(seed)
        inverse_permutation = np.argsort(rng.permutation(num_pixels))

        sigma = genes.denoise_sigma if genes else self.denoise_sigma

        # 2. Un-permute + denoise on the requested device, with graceful fallback.
        if use_gpu and cp is None and not allow_cpu_fallback:
            raise GPUAccelerationRequiredError(
                "GPU acceleration via CuPy is required for decode; CPU fallback is disabled."
            )

        if use_gpu and cp is not None:
            try:
                recovered = self._decode_array_gpu(
                    packet.encoded, inverse_permutation, packet.image_shape, sigma
                )
            except Exception as exc:
                fatal_oom = is_cupy_out_of_memory_error(exc) or isinstance(exc, CuPyOutOfMemoryError)
                if not allow_cpu_fallback:
                    raise
                logger.debug(
                    "GPU decode failed (%s%s); falling back to CPU",
                    "OOM: " if fatal_oom else "", exc,
                )
                recovered = self._decode_array_cpu(
                    packet.encoded, inverse_permutation, packet.image_shape, sigma
                )
        else:
            recovered = self._decode_array_cpu(
                packet.encoded, inverse_permutation, packet.image_shape, sigma
            )

        # 3. Apply Color/Contrast Genes (device-independent, cheap)
        if genes:
            recovered = self.apply_gene_corrections(recovered, genes)

        return np.clip(recovered, 0.0, 1.0).astype(np.float32)

    def _decode_array_cpu(self, encoded, inverse_permutation, image_shape, sigma) -> np.ndarray:
        """CPU un-permute + Gaussian denoise. Returns the recovered image array."""
        num_pixels = image_shape[0] * image_shape[1]
        encoded_pixels = np.asarray(encoded).reshape((num_pixels, 3))
        recovered = encoded_pixels[inverse_permutation, :].reshape(image_shape)
        if sigma and sigma > 0:
            recovered = filters.gaussian(
                recovered, sigma=sigma, preserve_range=True, channel_axis=-1
            )
        return recovered

    def _decode_array_gpu(self, encoded, inverse_permutation, image_shape, sigma) -> np.ndarray:
        """GPU un-permute + Gaussian denoise via CuPy; returns a host (numpy) array."""
        num_pixels = image_shape[0] * image_shape[1]
        g_encoded = cp.asarray(encoded).reshape((num_pixels, 3))
        g_perm = cp.asarray(inverse_permutation)
        recovered = g_encoded[g_perm, :].reshape(image_shape)
        if sigma and sigma > 0:
            # Blur spatially per channel only -> zero sigma on the channel axis.
            recovered = cupy_gaussian_filter(
                recovered, sigma=(float(sigma), float(sigma), 0.0)
            )
        return cp.asnumpy(recovered)

    @staticmethod
    def save_image(image: np.ndarray, path: str | Path) -> Path:
        """Save a reconstructed image to *path* as PNG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        png = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
        png.save(path)
        return path