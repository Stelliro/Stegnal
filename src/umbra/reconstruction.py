# reconstruction.py

"""Noise-to-image reconstruction helpers for experimental workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from .gpu_runtime import GPUAccelerationRequiredError, cp, require_gpu

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Hard cap for fax-style encodings to avoid exhausting host memory when gene
# mutations attempt to explore extremely long transmissions. Ten minutes keeps
# experiments flexible while bounding allocations to roughly 600 * sample_rate
# samples (≈115 MB at 48 kHz float32).
_MAX_FAX_DURATION_SECONDS = 600.0
_MARKER_BASE_FREQUENCY = 1_600.0
_MARKER_STEP_FREQUENCY = 220.0
_MARKER_RATIO_BAND = 180.0
_SEGMENT_RATIO_MIN = 0.05
_GPU_MIN_FFT_SAMPLES = 65_536


def suggest_sample_rate(image: np.ndarray) -> int:
    """Return a stable audio sample rate for ``image`` based on its size."""

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions for sample rate suggestion")
    height = int(max(array.shape[0], 1))
    width = int(max(array.shape[1], 1))
    area = max(height * width, 1)
    rate = int(round(16_000 + 28.0 * np.sqrt(float(area))))
    return max(rate, 16_000)


def suggest_transmission_profile(image: np.ndarray) -> tuple[int, float]:
    """Return fax-style transmission parameters tailored to ``image``."""

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions for transmission profile")
    height = int(max(array.shape[0], 1))
    segments = max(1, int(np.ceil(height / 96.0)))
    marker_duration = float(max(0.01, 0.03 + 0.0015 * segments))
    return segments, marker_duration


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

    min_y = max(int(np.floor(poly[:, 1].min())), 0)
    max_y = min(int(np.ceil(poly[:, 1].max())), rows - 1)
    min_x = max(int(np.floor(poly[:, 0].min())), 0)
    max_x = min(int(np.ceil(poly[:, 0].max())), cols - 1)

    if min_y > max_y or min_x > max_x:
        return

    y_coords = np.arange(min_y, max_y + 1)
    x_coords = np.arange(min_x, max_x + 1)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Inside polygon check (ray casting)
    inside = np.zeros((len(y_coords), len(x_coords)), dtype=bool)
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        x1, y1 = poly[i]
        x2, y2 = poly[j]
        cond1 = ((yy >= y1) != (yy >= y2))
        cond2 = (xx < x1 + ((yy - y1) / (y2 - y1 + 1e-6)) * (x2 - x1))
        inside = np.logical_xor(inside, cond1 & cond2)

    canvas[yy[inside], xx[inside]] = np.maximum(canvas[yy[inside], xx[inside]], color)


def _rotate_offsets(offsets: np.ndarray, angle: float) -> np.ndarray:
    rad = np.deg2rad(angle)
    cos, sin = np.cos(rad), np.sin(rad)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return np.dot(offsets, rotation)


def generate_shape_collage(
    seed: int, *, resolution: tuple[int, int] = (192, 192), shape_count: int = 3
) -> tuple[np.ndarray, tuple[GeneratedShape, ...]]:
    """Generate a collage of random shapes on a canvas."""

    rng = np.random.default_rng(seed)
    canvas = np.zeros(resolution + (3,), dtype=np.float32)
    shapes = []

    prototypes = {
        "circle": lambda size: np.array([[0, 0]]),  # Placeholder, uses circle func
        "square": lambda size: np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size,
        "triangle": lambda size: np.array([[0, -0.577], [-0.5, 0.289], [0.5, 0.289]]) * size,
    }

    for _ in range(shape_count):
        shape_type = rng.choice(list(prototypes.keys()))
        color = rng.uniform(0.2, 0.8, size=3)
        size = int(rng.uniform(20, min(resolution) / 2))
        center = tuple(rng.integers(size // 2, dim - size // 2) for dim in resolution)
        rotation = float(rng.uniform(0, 360))

        if shape_type == "circle":
            _draw_filled_circle(canvas, center, size // 2, color)
        else:
            offsets = prototypes[shape_type](size)
            rotated = _rotate_offsets(offsets, rotation)
            vertices = rotated + np.array(center)
            _draw_filled_polygon(canvas, vertices, color)

        shapes.append(GeneratedShape(color=tuple(color), shape=shape_type, center=center, rotation=rotation, size=size))

    return np.clip(canvas, 0.0, 1.0), tuple(shapes)


def create_variations(
    image: np.ndarray,
    *,
    variation_count: int = 6,
    noise_sigma: float = 0.3,
    dropout_probability: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create corrupted variations of ``image`` for ensemble prediction."""

    if rng is None:
        rng = np.random.default_rng()

    array = np.asarray(image, dtype=np.float32)
    variations = np.empty((variation_count,) + array.shape, dtype=np.float32)

    for idx in range(variation_count):
        noise = rng.normal(0.0, noise_sigma, size=array.shape).astype(np.float32)
        noisy = array + noise
        mask = rng.random(size=array.shape) < dropout_probability
        noisy[mask] = 0.0
        variations[idx] = np.clip(noisy, 0.0, 1.0)

    return variations


def predict_missing_pixels(variations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict an ensemble image and coverage map from ``variations``."""

    if variations.ndim != 4:
        raise ValueError("Variations must be 4D (N, H, W, C)")

    valid_mask = variations > 0.0
    count = np.sum(valid_mask, axis=0)
    ensemble = np.sum(variations, axis=0) / np.maximum(count, 1)
    coverage = count / variations.shape[0]

    return ensemble.astype(np.float32), coverage.astype(np.float32)


def image_to_waveform(image: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert ``image`` to a fax-style waveform with markers."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image")

    height, width = array.shape[:2]
    flat = array.mean(axis=2).reshape(-1)

    total_samples = flat.size
    duration = height / sample_rate if height > 0 else 0.0
    t = np.linspace(0, duration, total_samples, endpoint=False)
    carrier = np.sin(2 * np.pi * 1200 * t)
    if carrier.size != flat.size:
        carrier = np.resize(carrier, flat.size)
    waveform = carrier * flat

    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def segment_image_rows(image: np.ndarray, num_segments: int) -> list[slice]:
    """Divide ``image`` rows into ``num_segments`` for tiled reconstruction."""

    height = image.shape[0]
    segment_size = height // num_segments
    remainder = height % num_segments
    segments = []
    start = 0
    for i in range(num_segments):
        end = start + segment_size + (1 if i < remainder else 0)
        segments.append(slice(start, end))
        start = end
    return segments


def tiled_reconstruction(waveform: np.ndarray, resolution: tuple[int, int], sample_rate: int) -> np.ndarray:
    """Reconstruct large waveforms in tiles to avoid OOM."""

    if waveform.size < _GPU_MIN_FFT_SAMPLES or cp is None:
        return reconstruct_from_waveform(waveform, resolution, sample_rate)  # Fallback to full

    require_gpu("tiled FFT reconstruction")

    num_tiles = max(1, int(np.ceil(waveform.size / _GPU_MIN_FFT_SAMPLES)))
    tile_size = waveform.size // num_tiles
    reconstructed = np.zeros(resolution + (3,), dtype=np.float32)

    for i in range(num_tiles):
        start = i * tile_size
        end = start + tile_size if i < num_tiles - 1 else waveform.size
        tile_wave = waveform[start:end]
        tile_res = (resolution[0] // num_tiles, resolution[1])
        tile_recon = reconstruct_from_waveform(tile_wave, tile_res, sample_rate)
        row_start = i * tile_res[0]
        row_end = row_start + tile_recon.shape[0]
        reconstructed[row_start:row_end] = tile_recon

    return reconstructed


def reconstruct_from_waveform(
    waveform: np.ndarray,
    resolution: tuple[int, int],
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> np.ndarray:
    """Reconstruct an image from ``waveform`` using heuristic decoding."""

    if waveform.size == 0:
        return np.zeros(resolution + (3,), dtype=np.float32)

    freqs = np.fft.rfftfreq(waveform.size, 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(waveform))

    marker_idx = np.argmin(np.abs(freqs - _MARKER_BASE_FREQUENCY))
    required = int(np.prod(resolution))
    image_data = spectrum[marker_idx : marker_idx + required]
    if image_data.size < required:
        repeats = int(np.ceil(required / max(image_data.size, 1)))
        image_data = np.tile(image_data, repeats)[:required]
    else:
        image_data = image_data[:required]

    reconstructed = image_data.reshape(resolution)
    reconstructed /= np.max(reconstructed) + 1e-6

    return np.repeat(reconstructed[..., np.newaxis], 3, axis=-1).astype(np.float32)


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
    "GPUAccelerationRequiredError",
    "GeneratedShape",
    "ReconstructionResult",
    "blend_predictions",
    "create_variations",
    "generate_shape_collage",
    "image_to_waveform",
    "predict_missing_pixels",
    "segment_image_rows",
    "tiled_reconstruction",
    "reconstruct_from_waveform",
    "run_reconstruction_cycle",
    "waveform_to_wav_bytes",
    "suggest_sample_rate",
    "suggest_transmission_profile",
]