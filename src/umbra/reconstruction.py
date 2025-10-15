"""Noise-to-image reconstruction helpers for experimental workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np

try:  # pragma: no cover - optional GPU acceleration
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - GPU optional dependency
    cp = None

if TYPE_CHECKING:
    from .decoding import NoiseStreamDecoder
    from .encoding import NoisePacket

logger = logging.getLogger(__name__)


# Hard cap for fax-style encodings to avoid exhausting host memory when gene
# mutations attempt to explore extremely long transmissions. Ten minutes keeps
# experiments flexible while bounding allocations to roughly 600 * sample_rate
# samples (≈115 MB at 48 kHz float32).
_MAX_FAX_DURATION_SECONDS = 600.0


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


def _as_backend(array: Any, xp: Any) -> Any:
    """Return ``array`` as an ``xp`` ndarray with float32 dtype."""

    if xp is cp:
        return cp.asarray(array, dtype=cp.float32)
    return np.asarray(array, dtype=np.float32)


def _to_numpy(array: Any) -> np.ndarray:
    """Convert ``array`` to a NumPy float32 array."""

    if cp is not None and isinstance(array, cp.ndarray):  # pragma: no cover - runtime guard
        return cp.asnumpy(array.astype(cp.float32, copy=False))
    return np.asarray(array, dtype=np.float32)


def _encode_stripe_waveform(
    stripe: np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    """Encode an image stripe into ``sample_count`` audio samples."""

    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)

    xp = cp if cp is not None else np

    weights = _as_backend([0.5, 0.35, 0.15], xp)
    intensities = _as_backend(stripe, xp)[..., :3] @ weights
    intensities = intensities.reshape(-1)
    if intensities.size == 0:
        return np.zeros(sample_count, dtype=np.float32)

    intensities -= intensities.min()
    max_val = float(xp.max(intensities))
    if max_val > 0:
        intensities /= max_val

    bins = sample_count // 2 + 1
    xp_lin = xp.linspace(0.0, max(float(intensities.size - 1), 0.0), bins, dtype=xp.float32)
    xp_idx = xp.arange(intensities.size, dtype=xp.float32)
    spectrum = xp.interp(xp_lin, xp_idx, intensities)
    waveform = xp.fft.irfft(spectrum, n=sample_count)
    if waveform.size < sample_count:
        waveform = xp.pad(waveform, (0, sample_count - waveform.size))
    peak = float(xp.max(xp.abs(waveform)))
    if peak > 0:
        waveform /= peak
    return _to_numpy(waveform)


def _marker_tone(
    *,
    sample_rate: int,
    marker_samples: int,
    index: int,
    base_frequency: float = 1_600.0,
    step: float = 220.0,
) -> np.ndarray:
    """Return a short sinusoidal marker identifying ``index``."""

    if marker_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    frequency = max(base_frequency + index * step, 80.0)
    t = np.linspace(0.0, marker_samples / sample_rate, marker_samples, endpoint=False)
    envelope = np.linspace(0.2, 1.0, marker_samples, dtype=np.float32)
    tone = np.sin(2 * np.pi * frequency * t)
    tone = tone.astype(np.float32) * envelope
    return tone


def image_to_waveform(
    image: np.ndarray,
    *,
    sample_rate: int = 48_000,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> np.ndarray:
    """Encode ``image`` into a mono waveform using spectral weighting.

    When ``segments`` is greater than one a fax-style transmission is produced
    where each stripe of the image is emitted as its own block separated by a
    short audible marker tone. The markers help the decoder realign segments
    when reconstructing the image from an extended audio clip.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image for conversion to waveform")

    safe_segments = max(1, int(segments))
    marker_seconds = float(max(0.0, marker_duration))
    marker_samples = max(int(round(marker_seconds * sample_rate)), 0)
    payload_samples = max(sample_rate, 1)
    segment_length = marker_samples + payload_samples

    max_total_samples = int(max(_MAX_FAX_DURATION_SECONDS * sample_rate, sample_rate))
    if safe_segments > 1:
        max_segments = max(1, max_total_samples // max(segment_length, 1))
        if safe_segments > max_segments:
            logger.warning(
                "Truncating fax transmission from %d to %d segments to respect the %.1f s cap",
                safe_segments,
                max_segments,
                _MAX_FAX_DURATION_SECONDS,
            )
            safe_segments = max_segments

    if safe_segments == 1:
        waveform = _encode_stripe_waveform(array, sample_count=sample_rate)
        if waveform.size == 0:
            raise ValueError("Image contains no pixels")
        logger.debug("Encoded image to waveform with %d samples", waveform.size)
        return waveform.astype(np.float32, copy=False)

    rows = array.shape[0]
    stripe_height = int(np.ceil(rows / safe_segments))
    segments_wave: list[np.ndarray] = []
    total_samples = 0
    for idx in range(safe_segments):
        remaining = max_total_samples - total_samples
        if remaining <= 0:
            break
        start_row = idx * stripe_height
        end_row = min(rows, start_row + stripe_height)
        if start_row >= rows:
            stripe = array[-1:]
        else:
            stripe = array[start_row:end_row]
        stripe_wave = _encode_stripe_waveform(stripe, sample_count=max(payload_samples, 1))
        marker = _marker_tone(
            sample_rate=sample_rate,
            marker_samples=marker_samples,
            index=idx,
        )
        segment_wave = np.concatenate(
            [marker.astype(np.float32), stripe_wave.astype(np.float32)]
        )
        if segment_wave.size < segment_length:
            segment_wave = np.pad(segment_wave, (0, segment_length - segment_wave.size))
        elif segment_wave.size > segment_length:
            segment_wave = segment_wave[:segment_length]
        if segment_wave.size > remaining:
            segment_wave = segment_wave[:remaining]
        if segment_wave.size == 0:
            break
        segments_wave.append(segment_wave.astype(np.float32))
        total_samples += segment_wave.size

    if not segments_wave:
        return np.zeros(0, dtype=np.float32)

    waveform = np.concatenate(segments_wave).astype(np.float32)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform /= peak

    logger.debug(
        "Encoded image to waveform with %d samples across %d segments",
        waveform.size,
        safe_segments,
    )
    return waveform.astype(np.float32)


def _infer_segment_count(
    total_samples: int, sample_rate: int, marker_duration: float
) -> int:
    """Heuristically estimate the number of fax segments in ``waveform``."""

    if total_samples <= 0:
        raise ValueError("Waveform must contain samples")
    payload_samples = max(int(sample_rate), 1)
    marker_samples = max(int(round(marker_duration * sample_rate)), 0)
    segment_length = payload_samples + marker_samples
    if segment_length <= 0:
        return 1

    # A lower bound assuming the final segment may be truncated when the
    # transmission is stopped early. ``max`` defends against extremely short
    # clips that still need to be treated as a single segment.
    minimum_segments = max(int(np.ceil(total_samples / (segment_length + 1))), 1)
    approx_segments = int(total_samples // segment_length)
    remainder = total_samples - approx_segments * segment_length
    if remainder > marker_samples // 2:
        approx_segments += 1

    estimated = max(approx_segments, minimum_segments)
    # Constrain the estimate so that the implied payload does not exceed the
    # configured ten minute cap. This mirrors the guard in ``image_to_waveform``
    # and prevents pathological sample counts from returning runaway values.
    max_segments = max(
        1,
        int(np.ceil(total_samples / max(payload_samples, 1))),
    )
    return max(1, min(estimated, max_segments))


def reconstruct_from_waveform(
    waveform: np.ndarray,
    *,
    resolution: tuple[int, int],
    sample_rate: int,
    segments: int | None = 1,
    marker_duration: float = 0.05,
    advanced_logging: bool = False,
    return_segments: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Approximate an RGB image from a mono waveform.

    When ``segments`` is ``None`` the decoder attempts to estimate how many
    fax-style stripes were transmitted by examining the waveform length. The
    ``return_segments`` flag can be used to retrieve the detected segment count
    alongside the reconstructed image for bookkeeping purposes. Enabling
    ``advanced_logging`` surfaces additional debug information that can help
    diagnose decoding issues without polluting logs for standard workflows.
    """

    rows, cols = resolution
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1:
        wave = wave.reshape(-1)
    if wave.size == 0:
        raise ValueError("Waveform must contain samples")

    if advanced_logging:
        logger.debug(
            "Starting waveform reconstruction: samples=%d resolution=%s sample_rate=%d segments=%s marker_duration=%.5f",
            wave.size,
            resolution,
            sample_rate,
            "auto" if segments is None else int(segments),
            marker_duration,
        )

    if segments is None:
        safe_segments = _infer_segment_count(wave.size, sample_rate, marker_duration)
        if advanced_logging:
            logger.debug(
                "Inferred %d segments from waveform length", safe_segments
            )
    else:
        safe_segments = max(1, int(segments))
    marker_seconds = float(max(0.0, marker_duration))
    marker_samples = max(int(round(marker_seconds * sample_rate)), 0)
    payload_samples = max(sample_rate, 1)
    segment_length = marker_samples + payload_samples

    if advanced_logging:
        logger.debug(
            "Segment configuration: marker_samples=%d payload_samples=%d segment_length=%d",
            marker_samples,
            payload_samples,
            segment_length,
        )

    if safe_segments == 1:
        usable = wave
        if usable.size < sample_rate:
            usable = np.pad(usable, (0, sample_rate - usable.size))
        else:
            usable = usable[:sample_rate]
        spectrum = np.abs(np.fft.rfft(usable, n=sample_rate))
        if spectrum.size == 0:
            raise ValueError("Unable to derive spectrum from waveform")
        if advanced_logging:
            logger.debug("Processed single-segment waveform for reconstruction")
    else:
        available_segments = max(1, wave.size // segment_length)
        available_segments = min(available_segments, safe_segments)
        if available_segments <= 0:
            raise ValueError("Waveform is shorter than one segment")

        if advanced_logging:
            logger.debug(
                "Processing %d/%d available segments", available_segments, safe_segments
            )

        stripes: list[np.ndarray] = []
        for idx in range(available_segments):
            start = idx * segment_length
            end = start + segment_length
            segment_wave = wave[start:end]
            if segment_wave.size < segment_length:
                segment_wave = np.pad(segment_wave, (0, segment_length - segment_wave.size))
            payload = segment_wave[marker_samples : marker_samples + payload_samples]
            if payload.size < payload_samples:
                payload = np.pad(payload, (0, payload_samples - payload.size))
            else:
                payload = payload[:payload_samples]
            spectrum = np.abs(np.fft.rfft(payload, n=payload_samples))
            stripes.append(spectrum.astype(np.float32))

            if advanced_logging:
                logger.debug(
                    "Segment %d: start=%d end=%d payload_size=%d",
                    idx + 1,
                    start,
                    min(end, wave.size),
                    payload.size,
                )

        total_pixels = rows * cols
        needed = total_pixels * 3
        combined = np.zeros(needed, dtype=np.float32)
        stripe_height = int(np.ceil(rows / available_segments))
        cursor = 0
        for idx, spectrum in enumerate(stripes):
            start_row = idx * stripe_height
            end_row = min(rows, start_row + stripe_height)
            if start_row >= rows:
                break
            stripe_rows = max(end_row - start_row, 1)
            stripe_pixels = stripe_rows * cols * 3
            xp = np.linspace(0, max(spectrum.size - 1, 0), stripe_pixels)
            source_idx = np.arange(spectrum.size, dtype=np.float32)
            stripe_values = np.interp(
                xp,
                source_idx if source_idx.size else np.array([0.0], dtype=np.float32),
                spectrum,
            )
            end_cursor = min(cursor + stripe_pixels, needed)
            combined[cursor:end_cursor] = stripe_values[: end_cursor - cursor]
            cursor = end_cursor
            if cursor >= needed:
                break
        if cursor < needed and cursor > 0:
            combined[cursor:] = combined[cursor - 1]
        spectrum = combined
        safe_segments = available_segments

        if advanced_logging:
            logger.debug(
                "Combined %d stripes into %d values", len(stripes), spectrum.size
            )

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
    reconstructed = np.clip(np.nan_to_num(image, nan=0.0), 0.0, 1.0).astype(np.float32)
    if return_segments:
        if advanced_logging:
            logger.debug("Reconstruction complete; returning segment count %d", safe_segments)
        return reconstructed, int(safe_segments)
    if advanced_logging:
        logger.debug("Reconstruction complete without segment count")
    return reconstructed


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
    "suggest_sample_rate",
    "suggest_transmission_profile",
]

