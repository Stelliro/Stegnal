# adversarial.py

"""Adversarial co-evolution between a predictive generator and decoder.

The generator learns editable parameters to predict how an image would look after
passing through the stochastic encode/decode process, using only the original
image at inference time. During training it uses the actual reconstructions as
supervision. The decoder evolves its denoise parameter to counter the generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional acceleration
    from numba import njit
except ImportError:  # pragma: no cover - fallback
    def njit(*_args, **_kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

logger = logging.getLogger(__name__)


@dataclass
class GeneratorParams:
    blur_sigma: float = 0.8
    contrast: float = 1.05
    brightness: float = 0.0


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = float(max(0.05, sigma))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= float(kernel.sum() + 1e-6)  # Avoid div by zero
    return kernel


@njit(cache=True)
def _blur_channel(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:  # pragma: no cover - compiled
    radius = kernel.shape[0] // 2
    rows, cols = data.shape
    tmp = np.zeros_like(data)
    out = np.zeros_like(data)
    for r in range(rows):
        for c in range(cols):
            acc = 0.0
            for k in range(kernel.shape[0]):
                col = c + k - radius
                if col < 0:
                    col = 0
                elif col >= cols:
                    col = cols - 1
                acc += kernel[k] * data[r, col]
            tmp[r, c] = acc
    for r in range(rows):
        for c in range(cols):
            acc = 0.0
            for k in range(kernel.shape[0]):
                row = r + k - radius
                if row < 0:
                    row = 0
                elif row >= rows:
                    row = rows - 1
                acc += kernel[k] * tmp[row, c]
            out[r, c] = acc
    return out


def _separable_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return np.asarray(image, dtype=np.float32)
    kernel = _gaussian_kernel1d(sigma).astype(np.float32)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        out = np.empty_like(arr)
        for channel in range(arr.shape[2]):
            out[..., channel] = _blur_channel(arr[..., channel], kernel)
        return out
    return _blur_channel(arr, kernel)


def apply_generator(original: np.ndarray, params: GeneratorParams) -> np.ndarray:
    """Apply editable transforms to predict a reconstruction from original.

    The transform sequence is: blur -> contrast -> brightness -> clip.
    """

    image = np.asarray(original, dtype=np.float32)
    blurred = _separable_gaussian_blur(image, params.blur_sigma)
    adjusted = blurred * float(params.contrast) + float(params.brightness)
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


@dataclass
class CoevolutionState:
    generator: GeneratorParams
    decoder_sigma: float
    best_score: float
    step_count: int


class AdversarialManager:
    """Coordinate co-evolution of generator and decoder hyperparameters."""

    def __init__(self, *, initial_generator: GeneratorParams | None = None, initial_decoder_sigma: float = 1.0) -> None:
        self.rng = np.random.default_rng()
        if initial_generator is None:
            initial_generator = GeneratorParams()
        self.state = CoevolutionState(
            generator=initial_generator,
            decoder_sigma=float(np.clip(initial_decoder_sigma, 0.05, 2.5)),
            best_score=0.0,
            step_count=0,
        )
        logger.info(
            "Initialised adversarial manager with decoder sigma %.3f",
            self.state.decoder_sigma,
        )

    def _score(self, original: np.ndarray, target: np.ndarray, params: GeneratorParams) -> float:
        pred = apply_generator(original, params)
        overlap = float(np.mean(np.clip(pred * target, 0.0, 1.0)))
        ssim_proxy = float(np.mean(1.0 - np.abs(pred - target)))
        return 0.6 * overlap + 0.4 * ssim_proxy

    def step(self, original: np.ndarray, target: np.ndarray) -> tuple[GeneratorParams, float, float]:
        """Perform one co-evolution step.

        The generator proposes parameter jitters; the decoder responds by nudging
        its sigma based on target similarity. Returns (best_params, best_score, decoder_sigma).
        """

        if original.shape != target.shape:
            raise ValueError("Original and target must have the same shape")

        current = self.state.generator
        base_score = self._score(original, target, current)
        best_params = current
        best_score = base_score
        logger.debug(
            "Adversarial step %d base score %.4f", self.state.step_count, base_score
        )

        # Propose a few jittered candidates around the current generator parameters
        for _ in range(6):
            candidate = GeneratorParams(
                blur_sigma=float(np.clip(current.blur_sigma + self.rng.normal(0.0, 0.12), 0.0, 3.0)),
                contrast=float(np.clip(current.contrast + self.rng.normal(0.0, 0.05), 0.5, 2.0)),
                brightness=float(np.clip(current.brightness + self.rng.normal(0.0, 0.03), -0.25, 0.25)),
            )
            score = self._score(original, target, candidate)
            if score > best_score:
                best_score = score
                best_params = candidate

        # Update generator towards the best local candidate with inertia
        inertia = 0.7
        self.state.generator = GeneratorParams(
            blur_sigma=float(inertia * current.blur_sigma + (1.0 - inertia) * best_params.blur_sigma),
            contrast=float(inertia * current.contrast + (1.0 - inertia) * best_params.contrast),
            brightness=float(inertia * current.brightness + (1.0 - inertia) * best_params.brightness),
        )

        # Decoder responds: if target looks over-blurred, reduce sigma; if noisy, increase
        target_var = float(np.var(target))
        adjustment = np.interp(target_var, [0.005, 0.05], [0.05, -0.05])
        self.state.decoder_sigma = float(np.clip(self.state.decoder_sigma + adjustment, 0.05, 2.5))

        self.state.best_score = max(self.state.best_score, best_score)
        self.state.step_count += 1
        logger.info(
            "Adversarial iteration %d best score %.4f decoder sigma %.3f",
            self.state.step_count,
            best_score,
            self.state.decoder_sigma,
        )
        return self.state.generator, float(best_score), float(self.state.decoder_sigma)

    def inject_burst_noise(self, image: np.ndarray, *, severity: float = 0.2) -> np.ndarray:
        """Simulate adversarial jamming by injecting burst noise."""

        arr = np.array(image, dtype=np.float32)  # always copy to avoid mutating input
        bursts = max(1, int(0.05 * arr.size))
        flat = arr.reshape(-1)
        indices = self.rng.integers(0, flat.size, size=bursts)
        noise = self.rng.normal(0.0, severity, size=bursts)
        flat[indices] = np.clip(flat[indices] + noise, 0.0, 1.0)
        return flat.reshape(arr.shape).astype(np.float32)


__all__ = [
    "GeneratorParams",
    "AdversarialManager",
    "apply_generator",
]