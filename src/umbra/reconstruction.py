"""Noise-to-image reconstruction helpers for experimental workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .decoding import NoiseStreamDecoder
    from .encoding import NoisePacket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedShape:
    """Description of a synthetic geometric primitive used in a collage."""

    color: tuple[float, float, float]
    shape: str
    center: tuple[int, int]
    rotation: float
    size: int


@dataclass(frozen=True)
class ReconstructionResult:
    """Outcome from a noise reconstruction experiment."""

    base_image: np.ndarray
    variations: np.ndarray
    ensemble_prediction: np.ndarray
    audio_reconstruction: np.ndarray
    hybrid_prediction: np.ndarray
    coverage: np.ndarray
    waveform: np.ndarray
    sample_rate: int
    shapes: tuple[GeneratedShape, ...]


def _draw_filled_circle(
    canvas: np.ndarray, center: tuple[int, int], radius: int, color: np.ndarray
) -> None:
    rows, cols = canvas.shape[:2]
    y_indices, x_indices = np.ogrid[:rows, :cols]
    cy, cx = center
    mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius**2
    canvas[mask] = np.maximum(canvas[mask], color)


def _draw_filled_polygon(
    canvas: np.ndarray, vertices: np.ndarray, color: np.ndarray
) -> None:
    rows, cols = canvas.shape[:2]
    poly = np.asarray(vertices, dtype=np.float32)
    if poly.size == 0:
        return

    min_y = max(int(np.floor(poly[:, 0].min())), 0)
    max_y = min(int(np.ceil(poly[:, 0].max())), rows - 1)
    min_x = max(int(np.floor(poly[:, 1].min())), 0)
    max_x = min(int(np.ceil(poly[:, 1].max())), cols - 1)

    if min_y > max_y or min_x > max_x:
        return

    y_coords = np.arange(min_y, max_y + 1)
    x_coords = np.arange(min_x, max_x + 1)
    yy = y_coords[:, None].astype(np.float32) + 0.5
    xx = x_coords[None, :].astype(np.float32) + 0.5

    inside = np.zeros((y_coords.size, x_coords.size), dtype=bool)
    y_vertices = poly[:, 0]
    x_vertices = poly[:, 1]
    count = len(poly)

    for idx in range(count):
        nxt = (idx + 1) % count
        y0, y1 = y_vertices[idx], y_vertices[nxt]
        x0, x1 = x_vertices[idx], x_vertices[nxt]

        if np.isclose(y0, y1):
            continue

        intersects = (y0 > yy) != (y1 > yy)
        x_intersect = (x1 - x0) * (yy - y0) / (y1 - y0) + x0
        inside ^= intersects & (xx < x_intersect)

    subregion = canvas[min_y : max_y + 1, min_x : max_x + 1]
    subregion[inside] = np.maximum(subregion[inside], color)
    canvas[min_y : max_y + 1, min_x : max_x + 1] = subregion


def _rotate_offsets(points: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    return points @ rotation.T


def generate_shape_collage(
    seed: int,
    *,
    resolution: tuple[int, int] = (192, 192),
    shape_count: int | None = None,
) -> tuple[np.ndarray, tuple[GeneratedShape, ...]]:
    """Create a colour image composed of multiple geometric primitives.

    Each collage contains between three and fifteen shapes (inclusive) unless
    ``shape_count`` is provided explicitly.
    """

    rows, cols = resolution
    canvas = np.zeros((rows, cols, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    count = int(shape_count or rng.integers(3, 16))
    count = int(np.clip(count, 3, 15))

    min_extent = max(min(rows, cols) // 9, 12)
    max_extent = max(min(rows, cols) // 3, min_extent + 6)
    padding = int(np.ceil(max_extent * 0.75))

    shapes: list[GeneratedShape] = []
    shape_types = ("circle", "square", "triangle", "diamond")

    for _ in range(count):
        extent = int(rng.integers(min_extent, max_extent + 1))
        cy = int(rng.integers(padding, rows - padding)) if rows > 2 * padding else rows // 2
        cx = int(rng.integers(padding, cols - padding)) if cols > 2 * padding else cols // 2
        center = (cy, cx)
        shape = str(rng.choice(shape_types))
        rotation = float(rng.uniform(0, 2 * np.pi)) if shape != "circle" else 0.0
        color = rng.uniform(0.25, 1.0, size=3).astype(np.float32)

        if shape == "circle":
            _draw_filled_circle(canvas, center, extent, color)
        else:
            if shape == "square":
                half = float(extent)
                base = np.array(
                    [
                        [-half, -half],
                        [half, -half],
                        [half, half],
                        [-half, half],
                    ],
                    dtype=np.float32,
                )
            elif shape == "triangle":
                height = float(extent)
                base = np.array(
                    [
                        [0.0, -height],
                        [height, height],
                        [-height, height],
                    ],
                    dtype=np.float32,
                )
            else:  # diamond
                radius = float(extent)
                base = np.array(
                    [
                        [0.0, -radius],
                        [radius, 0.0],
                        [0.0, radius],
                        [-radius, 0.0],
                    ],
                    dtype=np.float32,
                )

            rotated = _rotate_offsets(base, rotation)
            vertices = [(center[0] + pt[1], center[1] + pt[0]) for pt in rotated]
            _draw_filled_polygon(canvas, np.asarray(vertices, dtype=np.float32), color)

        shapes.append(
            GeneratedShape(
                color=(float(color[0]), float(color[1]), float(color[2])),
                shape=shape,
                center=center,
                rotation=np.degrees(rotation),
                size=extent,
            )
        )

    collage = np.clip(canvas, 0.0, 1.0)
    logger.debug("Generated collage with %d shapes", len(shapes))
    return collage, tuple(shapes)


def create_variations(
    image: np.ndarray,
    *,
    variation_count: int = 6,
    noise_sigma: float = 0.3,
    dropout_probability: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create multiple noisy glimpses of ``image`` by masking and corrupting pixels."""

    if variation_count <= 0:
        raise ValueError("variation_count must be positive")

    base = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    generator = rng or np.random.default_rng()
    variations: list[np.ndarray] = []

    for _ in range(int(variation_count)):
        dropout_mask = generator.random(base.shape[:2], dtype=np.float32) > dropout_probability
        dropout_mask = dropout_mask[..., None]
        noise = generator.normal(0.0, noise_sigma, size=base.shape).astype(np.float32)
        jittered = np.clip(base + noise, 0.0, 1.0)
        filler = generator.random(base.shape, dtype=np.float32) * 0.35
        variant = np.where(dropout_mask, jittered, filler)
        variations.append(np.clip(variant, 0.0, 1.0))

    stacked = np.stack(variations, axis=0)
    logger.debug("Created %d noisy variations", stacked.shape[0])
    return stacked


def predict_missing_pixels(variations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the underlying image and coverage map from ``variations``."""

    stack = np.asarray(variations, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError("Expected variations with shape (n, h, w, 3)")
    coverage = np.mean(stack > 0.05, axis=0)
    if coverage.ndim == 3:
        coverage = coverage.mean(axis=2)
    ensemble = np.median(stack, axis=0)
    ensemble = np.clip(ensemble, 0.0, 1.0)
    return ensemble.astype(np.float32), coverage.astype(np.float32)


def tiled_reconstruction(
    decoder: NoiseStreamDecoder,
    packet: NoisePacket,
    seed: int,
    *,
    tile_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Decode ``packet`` using tiles to limit peak memory usage."""

    decoded_full = decoder.decode(packet, seed)
    single_channel = decoded_full.ndim == 2 or (
        decoded_full.ndim == 3 and decoded_full.shape[2] == 1
    )
    if decoded_full.ndim == 2:
        decoded_full = decoded_full[:, :, None]
    height, width = decoded_full.shape[:2]
    tile_h, tile_w = tile_size
    assembled = np.zeros_like(decoded_full)
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            tile = decoded_full[y : y + tile_h, x : x + tile_w]
            assembled[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    if single_channel:
        return assembled[:, :, 0]
    return assembled


def image_to_waveform(
    image: np.ndarray,
    *,
    sample_rate: int = 48_000,
) -> np.ndarray:
    """Encode ``image`` into a mono waveform using spectral weighting."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image for conversion to waveform")

    weights = np.array([0.5, 0.35, 0.15], dtype=np.float32)
    intensities = array[..., :3] @ weights
    intensities = intensities.reshape(-1)
    if intensities.size == 0:
        raise ValueError("Image contains no pixels")

    intensities -= intensities.min()
    max_val = float(intensities.max())
    if max_val > 0:
        intensities /= max_val

    num_bins = sample_rate // 2 + 1
    xp = np.linspace(0, intensities.size - 1, num_bins)
    spectrum = np.interp(xp, np.arange(intensities.size), intensities)
    waveform = np.fft.irfft(spectrum, n=sample_rate)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform /= peak

    waveform = waveform.astype(np.float32)
    logger.debug("Encoded image to waveform with %d samples", waveform.size)
    return waveform


def reconstruct_from_waveform(
    waveform: np.ndarray,
    *,
    resolution: tuple[int, int],
    sample_rate: int,
) -> np.ndarray:
    """Approximate an RGB image from a mono waveform."""

    rows, cols = resolution
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1:
        wave = wave.reshape(-1)
    if wave.size == 0:
        raise ValueError("Waveform must contain samples")

    spectrum = np.abs(np.fft.rfft(wave, n=sample_rate))
    if spectrum.size == 0:
        raise ValueError("Unable to derive spectrum from waveform")

    spectrum = spectrum.astype(np.float32)
    max_val = float(spectrum.max())
    if max_val > 0:
        spectrum /= max_val

    total_pixels = rows * cols
    needed = total_pixels * 3
    if spectrum.size < needed:
        repeats = int(np.ceil(needed / spectrum.size))
        spectrum = np.tile(spectrum, repeats)[:needed]
    else:
        spectrum = spectrum[:needed]

    image = spectrum.reshape(3, rows, cols).transpose(1, 2, 0)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def blend_predictions(
    ensemble: np.ndarray, audio: np.ndarray, coverage: np.ndarray
) -> np.ndarray:
    """Combine ensemble and audio predictions based on coverage confidence."""

    ens = np.asarray(ensemble, dtype=np.float32)
    aud = np.asarray(audio, dtype=np.float32)
    cov = np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0)

    if ens.shape != aud.shape:
        raise ValueError("Ensemble and audio predictions must share a shape")
    if cov.ndim == 3 and cov.shape[2] == ens.shape[2]:
        cov = np.mean(cov, axis=2)
    if cov.shape != ens.shape[:2]:
        raise ValueError("Coverage map must match prediction height/width")

    weights = cov[..., None]
    blended = ens * weights + aud * (1.0 - weights)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode ``waveform`` into 16-bit PCM WAV bytes."""

    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1:
        wave = wave.reshape(-1)
    if wave.size == 0:
        raise ValueError("Waveform must contain samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    scaled = np.clip(wave, -1.0, 1.0)
    scaled = (scaled * 32767.0).astype(np.int16)
    import wave as _wave

    buffer = BytesIO()
    with _wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(scaled.tobytes())

    return buffer.getvalue()


def run_reconstruction_cycle(
    seed: int,
    *,
    resolution: tuple[int, int] = (192, 192),
    variation_count: int = 6,
    noise_sigma: float = 0.3,
    dropout_probability: float = 0.35,
    sample_rate: int = 48_000,
) -> ReconstructionResult:
    """Generate a collage, corrupt it, and attempt to reconstruct missing pixels."""

    collage, shapes = generate_shape_collage(seed, resolution=resolution)
    rng = np.random.default_rng(seed + 1)
    variations = create_variations(
        collage,
        variation_count=variation_count,
        noise_sigma=noise_sigma,
        dropout_probability=dropout_probability,
        rng=rng,
    )
    ensemble, coverage = predict_missing_pixels(variations)
    waveform = image_to_waveform(collage, sample_rate=sample_rate)
    audio_image = reconstruct_from_waveform(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
    )
    hybrid = blend_predictions(ensemble, audio_image, coverage)

    logger.info(
        "Completed reconstruction cycle with %d variations and sample_rate=%d",
        variations.shape[0],
        sample_rate,
    )

    return ReconstructionResult(
        base_image=collage.astype(np.float32),
        variations=variations.astype(np.float32),
        ensemble_prediction=ensemble.astype(np.float32),
        audio_reconstruction=audio_image.astype(np.float32),
        hybrid_prediction=hybrid.astype(np.float32),
        coverage=coverage.astype(np.float32),
        waveform=waveform.astype(np.float32),
        sample_rate=int(sample_rate),
        shapes=shapes,
    )


__all__ = [
    "GeneratedShape",
    "ReconstructionResult",
    "blend_predictions",
    "create_variations",
    "generate_shape_collage",
    "image_to_waveform",
    "predict_missing_pixels",
    "tiled_reconstruction",
    "reconstruct_from_waveform",
    "run_reconstruction_cycle",
    "waveform_to_wav_bytes",
]

