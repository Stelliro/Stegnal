# reconstruction.py

"""Noise-to-image reconstruction helpers for experimental workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import hilbert as scipy_hilbert

# --- GPU ACCELERATION IMPORTS ---
try:
    import cupy as cp
    from cupyx.scipy.signal import hilbert as cupy_hilbert
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_hilbert = None
    CUPY_AVAILABLE = False
# --- END GPU IMPORTS ---

from .gpu_runtime import (
    GPUAccelerationRequiredError,
    require_gpu,
)

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
        "circle": lambda size: np.array([[0, 0]]),
        "square": lambda size: np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size,
        "triangle": lambda size: np.array([[0, -0.577], [-0.5, 0.289], [0.5, 0.289]]) * size,
    }

    for _ in range(shape_count):
        shape_type = rng.choice(list(prototypes.keys()))
        color = rng.uniform(0.2, 0.8, size=3)
        size_upper = max(2.0, min(resolution) / 2)
        size_lower = min(20.0, size_upper - 1.0)
        size = int(rng.uniform(max(1, size_lower), size_upper))
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
    if coverage.ndim == 3:
        coverage = np.mean(coverage, axis=-1)

    return ensemble.astype(np.float32), coverage.astype(np.float32)


def image_to_waveform(
    image: np.ndarray,
    sample_rate: int,
    *,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> np.ndarray:
    """Convert ``image`` to a fax-style waveform using Amplitude Modulation."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image")

    flat = array.mean(axis=2).reshape(-1)
    num_samples = min(sample_rate, flat.size)

    duration = num_samples / float(sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)

    carrier = np.sin(2 * np.pi * 1200 * t)
    base_waveform = np.clip(carrier * flat[:num_samples], -1.0, 1.0).astype(np.float32)

    if segments <= 1:
        # Pad to exactly sample_rate samples
        if base_waveform.size < sample_rate:
            base_waveform = np.pad(base_waveform, (0, sample_rate - base_waveform.size))
        else:
            base_waveform = base_waveform[:sample_rate]
        return base_waveform

    # Cap total duration
    marker_samples = int(round(marker_duration * sample_rate))
    total_samples = (sample_rate + marker_samples) * segments
    max_samples = int(_MAX_FAX_DURATION_SECONDS * sample_rate)
    if total_samples > max_samples:
        segments = max(1, max_samples // (sample_rate + marker_samples))
        total_samples = (sample_rate + marker_samples) * segments

    parts = []
    rows_per_seg = max(1, array.shape[0] // segments)
    for seg_idx in range(segments):
        row_start = seg_idx * rows_per_seg
        row_end = min(row_start + rows_per_seg, array.shape[0])
        seg_flat = array[row_start:row_end].mean(axis=2).reshape(-1)
        seg_samples = min(sample_rate, seg_flat.size)
        if seg_samples == 0:
            seg_samples = sample_rate
            seg_flat = np.zeros(seg_samples, dtype=np.float32)
        dur = seg_samples / float(sample_rate)
        seg_t = np.linspace(0, dur, seg_samples, endpoint=False, dtype=np.float32)
        seg_carrier = np.sin(2 * np.pi * 1200 * seg_t)
        seg_wave = np.clip(seg_carrier * seg_flat[:seg_samples], -1.0, 1.0).astype(np.float32)
        # Pad segment to exactly sample_rate samples
        if seg_wave.size < sample_rate:
            seg_wave = np.pad(seg_wave, (0, sample_rate - seg_wave.size))
        else:
            seg_wave = seg_wave[:sample_rate]
        # Add marker tone
        marker_t = np.linspace(0, marker_duration, marker_samples, endpoint=False, dtype=np.float32)
        freq = _MARKER_BASE_FREQUENCY + _MARKER_STEP_FREQUENCY * seg_idx
        marker = np.sin(2 * np.pi * freq * marker_t).astype(np.float32) * 0.5
        parts.append(np.concatenate([seg_wave, marker]))

    return np.concatenate(parts).astype(np.float32)


def segment_image_rows(image: np.ndarray, num_segments: int) -> list[tuple[int, int, float]]:
    """Divide ``image`` rows into ``num_segments`` using contrast-aware boundaries."""
    height = image.shape[0]
    if num_segments <= 1:
        gray = np.mean(image, axis=(-1,)) if image.ndim == 3 else image
        contrast = float(np.std(gray))
        return [(0, height, contrast)]

    # Compute per-row contrast scores
    gray = np.mean(image, axis=2) if image.ndim == 3 else image.astype(np.float32)
    row_contrast = np.array([float(np.std(gray[r])) for r in range(height)])

    # Accumulate contrast and split at roughly equal cumulative contrast
    cumulative = np.cumsum(row_contrast)
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0

    splits = [0]
    for seg_idx in range(1, num_segments):
        target = (seg_idx / num_segments) * total
        idx = int(np.searchsorted(cumulative, target))
        idx = max(splits[-1] + 1, min(idx, height - (num_segments - seg_idx)))
        splits.append(idx)
    splits.append(height)

    segments = []
    for i in range(num_segments):
        start, end = splits[i], splits[i + 1]
        contrast = float(np.std(gray[start:end]))
        segments.append((start, end, contrast))
    return segments


def tiled_reconstruction(waveform: np.ndarray, resolution: tuple[int, int], sample_rate: int) -> np.ndarray:
    """Reconstruct large waveforms in tiles to avoid OOM."""
    if waveform.size < _GPU_MIN_FFT_SAMPLES or cp is None:
        return reconstruct_from_waveform(waveform, resolution, sample_rate)
    
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
    resolution: tuple[int, int] | None = None,
    sample_rate: int | None = None,
    *,
    segments: int | None = None,
    marker_duration: float = 0.05,
    advanced_logging: bool = False,
    return_segments: bool = False,
    allow_cpu_fallback: bool = True,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Reconstruct an image from ``waveform`` using AM demodulation, with GPU acceleration."""
    if resolution is None:
        resolution = (64, 64)
    if sample_rate is None:
        sample_rate = 48000

    if waveform.size == 0:
        blank = np.zeros(resolution + (3,), dtype=np.float32)
        if return_segments:
            return blank, segments or 1
        return blank

    detected_segments = segments or 1

    # Attempt segment detection if segments is None
    if segments is None and waveform.size > sample_rate:
        marker_samples = int(round(marker_duration * sample_rate))
        expected_seg_len = sample_rate + marker_samples
        if expected_seg_len > 0:
            detected_segments = max(1, round(waveform.size / expected_seg_len))
    elif segments is not None:
        detected_segments = segments

    num_pixels_expected = np.prod(resolution)

    if advanced_logging:
        logger.info(
            "Reconstructing from waveform: samples=%d resolution=%s sr=%d segments=%d",
            waveform.size, resolution, sample_rate, detected_segments,
        )

    # Per-segment demodulation for multi-segment waveforms
    if detected_segments > 1 and waveform.size > sample_rate:
        marker_samples = int(round(marker_duration * sample_rate))
        segment_total = sample_rate + marker_samples
        rows_per_seg = max(1, resolution[0] // detected_segments)

        result = np.zeros(resolution + (3,), dtype=np.float32)
        for seg_idx in range(detected_segments):
            seg_start = seg_idx * segment_total
            seg_end = min(seg_start + sample_rate, waveform.size)
            if seg_start >= waveform.size:
                break
            seg_wave = waveform[seg_start:seg_end]

            analytic = scipy_hilbert(seg_wave)
            envelope = np.abs(analytic).astype(np.float32)

            row_start = seg_idx * rows_per_seg
            row_end = min(row_start + rows_per_seg, resolution[0])
            if seg_idx == detected_segments - 1:
                row_end = resolution[0]
            seg_pixels = (row_end - row_start) * resolution[1]

            if envelope.size < seg_pixels:
                demod = np.pad(envelope, (0, seg_pixels - envelope.size))
            else:
                demod = envelope[:seg_pixels]

            seg_image = demod.reshape(row_end - row_start, resolution[1])
            max_val = np.max(seg_image)
            if max_val > 1e-6:
                seg_image /= max_val

            result[row_start:row_end] = np.repeat(seg_image[..., np.newaxis], 3, axis=-1)

        if return_segments:
            return result.astype(np.float32), detected_segments
        return result.astype(np.float32)

    if CUPY_AVAILABLE:
        try:
            logger.debug("Attempting GPU-accelerated Hilbert transform.")
            waveform_gpu = cp.asarray(waveform)
            analytic_signal_gpu = cupy_hilbert(waveform_gpu)
            amplitude_envelope_gpu = cp.abs(analytic_signal_gpu)
            
            if amplitude_envelope_gpu.size < num_pixels_expected:
                pad_width = (0, num_pixels_expected - amplitude_envelope_gpu.size)
                recovered_data_gpu = cp.pad(amplitude_envelope_gpu, pad_width, 'constant')
            else:
                recovered_data_gpu = amplitude_envelope_gpu[:num_pixels_expected]
            
            reconstructed_gpu = recovered_data_gpu.reshape(resolution)
            
            max_val_gpu = cp.max(reconstructed_gpu)
            if max_val_gpu > 1e-6:
                reconstructed_gpu /= max_val_gpu
            
            reconstructed = cp.asnumpy(reconstructed_gpu)
            logger.debug("GPU Hilbert transform successful.")
            
            result = np.repeat(reconstructed[..., np.newaxis], 3, axis=-1).astype(np.float32)
            if return_segments:
                return result, detected_segments
            return result

        except Exception as e:
            logger.warning(f"GPU Hilbert transform failed, falling back to CPU. Error: {e}")
    
    logger.debug("Using CPU-based Hilbert transform.")
    analytic_signal = scipy_hilbert(waveform)
    amplitude_envelope = np.abs(analytic_signal)
    
    if amplitude_envelope.size < num_pixels_expected:
        recovered_data = np.pad(
            amplitude_envelope, 
            (0, num_pixels_expected - amplitude_envelope.size), 
            'constant'
        )
    else:
        recovered_data = amplitude_envelope[:num_pixels_expected]

    reconstructed = recovered_data.reshape(resolution)

    max_val = np.max(reconstructed)
    if max_val > 1e-6:
        reconstructed /= max_val

    result = np.repeat(reconstructed[..., np.newaxis], 3, axis=-1).astype(np.float32)
    if return_segments:
        return result, detected_segments
    return result


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
        buffer = BytesIO()
        import wave as _wave
        with _wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(b'')
        return buffer.getvalue()

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


def _encode_stripe_waveform(
    stripe: np.ndarray,
    sample_count: int,
    *,
    allow_cpu_fallback: bool = True,
) -> np.ndarray:
    """Encode a stripe of pixel data into a waveform of ``sample_count`` samples."""
    array = np.asarray(stripe, dtype=np.float32).reshape(-1)

    if cp is not None:
        try:
            gpu_arr = cp.asarray(array)
            resampled = cp.asnumpy(gpu_arr)
            # Resample to target length
            if resampled.size >= sample_count:
                result = resampled[:sample_count]
            else:
                result = np.pad(resampled, (0, sample_count - resampled.size))
            return result.astype(np.float32)
        except Exception:
            if not allow_cpu_fallback:
                raise GPUAccelerationRequiredError(
                    "GPU acceleration required for stripe encoding; CPU fallback disabled."
                )

    if cp is None and not allow_cpu_fallback:
        raise GPUAccelerationRequiredError(
            "GPU acceleration required for stripe encoding; CPU fallback disabled."
        )

    if array.size >= sample_count:
        return array[:sample_count].astype(np.float32)
    return np.pad(array, (0, sample_count - array.size)).astype(np.float32)


def _fft_magnitude(
    signal: np.ndarray,
    n: int,
    *,
    advanced_logging: bool = False,
    allow_cpu_fallback: bool = True,
) -> np.ndarray:
    """Return the magnitude of the first ``n//2 + 1`` FFT bins."""
    signal = np.asarray(signal, dtype=np.float32)

    if cp is not None and signal.size >= _GPU_MIN_FFT_SAMPLES:
        try:
            gpu_sig = cp.asarray(signal, dtype=cp.float32)
            fft_result = cp.fft.rfft(gpu_sig, n=n)
            mag = cp.abs(fft_result)
            return cp.asnumpy(mag).astype(np.float32)
        except Exception:
            if not allow_cpu_fallback:
                raise GPUAccelerationRequiredError(
                    "GPU acceleration required for FFT; CPU fallback disabled."
                )

    if cp is None and not allow_cpu_fallback:
        raise GPUAccelerationRequiredError(
            "GPU acceleration required for FFT; CPU fallback disabled."
        )

    fft_result = np.fft.rfft(signal, n=n)
    return np.abs(fft_result).astype(np.float32)


def _as_backend(array: np.ndarray, xp) -> np.ndarray:
    """Convert ``array`` to the backend (*xp*) array type, retrying on OOM."""
    for attempt in range(2):
        try:
            return xp.asarray(array)
        except Exception:
            if attempt == 0:
                continue
            return np.asarray(array, dtype=np.float32)


__all__ = [
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