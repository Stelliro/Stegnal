"""Synthetic sound-driven image generation utilities."""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import wave
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MessyKeyArtifact:
    """Container describing the noisy latent used for diffusion guidance."""

    hash: str
    latent: np.ndarray

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> MessyKeyArtifact:
        buffer = np.asarray(samples, dtype=np.float32).reshape(-1)
        digest = hashlib.sha1(buffer.tobytes()).hexdigest()
        return cls(hash=digest, latent=buffer)


def derive_messy_latent(artifact: MessyKeyArtifact, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a :class:`MessyKeyArtifact` to ``shape`` for diffusion guidance."""

    if artifact.latent.size == 0:
        return np.zeros(shape, dtype=np.float32)
    repeats = int(np.ceil(np.prod(shape) / artifact.latent.size))
    tiled = np.tile(artifact.latent, repeats)
    return tiled[: int(np.prod(shape))].reshape(shape).astype(np.float32)


def messy_key_hash_from_overlap(overlap_map: np.ndarray) -> str:
    """Create a reproducible messy-key hash from an overlap activation map."""

    array = np.asarray(overlap_map, dtype=np.float32)
    normalized = (array - float(array.min())) / (float(np.ptp(array)) + 1e-6)
    digest = hashlib.sha1(normalized.tobytes()).hexdigest()
    return digest


@dataclass(frozen=True)
class SyntheticSound:
    """Representation of a randomly generated sound clip."""

    seed: int
    sample_rate: int
    samples: np.ndarray
    band_volumes: dict[str, float]


@dataclass(frozen=True)
class ShapeSpec:
    """Description of a coloured geometric shape encoded in the sound image."""

    color: str
    shape: str
    volume: float
    center: tuple[int, int]
    rotation: float
    size: int


@dataclass(frozen=True)
class ShapeGuess:
    """Prediction made by the shape-guessing helper."""

    color: str
    guess: str
    confidence: float
    volume: float


def _seed_from_samples(samples: np.ndarray) -> int:
    """Derive a deterministic seed from ``samples`` for waveform synthesis."""

    digest = hashlib.sha1(np.asarray(samples, dtype=np.float32).tobytes()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _normalized_band_volumes(spectrum: np.ndarray) -> dict[str, float]:
    """Split ``spectrum`` into three bands and normalise their magnitudes."""

    if spectrum.ndim != 1:
        raise ValueError("Expected a one-dimensional spectrum array")

    band_edges = np.linspace(0, spectrum.size, 4, dtype=int)
    bands = []
    for idx in range(3):
        start, end = band_edges[idx], band_edges[idx + 1]
        band = spectrum[start:end]
        if band.size == 0:
            magnitude = 0.0
        else:
            magnitude = float(np.mean(np.abs(band)))
        bands.append(magnitude)

    max_val = max(bands)
    if max_val <= 0.0:
        norm = [1.0, 1.0, 1.0]
    else:
        norm = [val / max_val for val in bands]

    return {"red": norm[0], "green": norm[1], "blue": norm[2]}


def _draw_circle(canvas: np.ndarray, center: tuple[int, int], radius: int, channel: int, intensity: float) -> None:
    rows, cols = canvas.shape[:2]
    y_indices, x_indices = np.ogrid[:rows, :cols]
    cy, cx = center
    mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius ** 2
    canvas[..., channel][mask] = np.maximum(canvas[..., channel][mask], intensity)


def _rotate_offsets(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``points`` (x, y) offsets by ``angle`` radians."""

    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    return points @ rotation.T


def _draw_polygon(
    canvas: np.ndarray,
    vertices: Sequence[tuple[float, float]],
    channel: int,
    intensity: float,
) -> None:
    """Rasterise a filled polygon defined by ``vertices`` onto ``canvas``."""

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

    subregion = canvas[min_y : max_y + 1, min_x : max_x + 1, channel]
    subregion[inside] = np.maximum(subregion[inside], intensity)
    canvas[min_y : max_y + 1, min_x : max_x + 1, channel] = subregion


def _synthesise_sound_image(
    rng: np.random.Generator,
    volumes: dict[str, float],
    image_size: tuple[int, int],
    *,
    shape_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ShapeSpec]]:
    """Create the colour/grayscale pair representing ``volumes``."""

    color_canvas = np.zeros((*image_size, 3), dtype=np.float32)
    rows, cols = image_size
    min_extent = max(int(min(rows, cols) * 0.05), 8)
    max_extent = max(int(min(rows, cols) * 0.4), min_extent + 6)

    shapes: list[ShapeSpec] = []
    shape_types = ("circle", "square", "triangle")
    channels = {"red": 0, "green": 1, "blue": 2}
    colour_order = ["red", "green", "blue"]

    total_shapes = int(shape_count) if shape_count is not None else len(colour_order)
    total_shapes = max(3, total_shapes)
    weight_array = np.array([
        max(float(volumes.get(name, 0.0)), 1e-3) for name in colour_order
    ])
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0.0:
        weight_array = np.ones_like(weight_array, dtype=np.float32)
        weight_sum = float(weight_array.sum())
    probabilities = (weight_array / weight_sum).astype(np.float32)

    sequence: list[str] = []
    for base_colour in colour_order:
        sequence.append(base_colour)
        if len(sequence) >= total_shapes:
            break
    while len(sequence) < total_shapes:
        chosen = str(rng.choice(colour_order, p=probabilities))
        sequence.append(chosen)

    for color_name in sequence:
        channel = channels[color_name]
        base_extent = float(rng.uniform(min_extent, max_extent))
        cy = int(rng.integers(0, rows))
        cx = int(rng.integers(0, cols))
        center = (cy, cx)
        shape = rng.choice(shape_types)
        rotation = float(rng.uniform(0, 2 * np.pi)) if shape != "circle" else 0.0
        base_volume = float(np.clip(volumes.get(color_name, 0.0), 0.05, 1.0))
        variation = float(rng.uniform(0.85, 1.15))
        scaled = float(np.clip(base_volume * variation, 0.05, 1.0))
        intensity = scaled

        if shape == "circle":
            radius = int(max(2, round(base_extent * rng.uniform(0.4, 0.9))))
            _draw_circle(color_canvas, center, radius, channel, intensity)
            size_value = max(2, radius * 2)
        else:
            if shape == "square":
                half = float(base_extent * rng.uniform(0.4, 0.9))
                base = np.array(
                    [
                        [-half, -half],
                        [half, -half],
                        [half, half],
                        [-half, half],
                    ],
                    dtype=np.float32,
                )
                size_value = int(max(2, round(half * 2)))
            else:
                height = float(base_extent * rng.uniform(0.5, 1.1))
                width = float(base_extent * rng.uniform(0.4, 1.0))
                base = np.array(
                    [
                        [0.0, -height],
                        [width, height],
                        [-width, height],
                    ],
                    dtype=np.float32,
                )
                size_value = int(max(2, round(max(height, width))))

            rotated = _rotate_offsets(base, rotation)
            vertices = [(center[0] + pt[1], center[1] + pt[0]) for pt in rotated]
            _draw_polygon(color_canvas, vertices, channel, intensity)

        shapes.append(
            ShapeSpec(
                color=color_name,
                shape=shape,
                volume=intensity,
                center=center,
                rotation=np.degrees(rotation),
                size=int(size_value),
            )
        )

    color_canvas = np.clip(color_canvas, 0.0, 1.0)
    grayscale = np.clip(
        0.299 * color_canvas[..., 0]
        + 0.587 * color_canvas[..., 1]
        + 0.114 * color_canvas[..., 2],
        0.0,
        1.0,
    ).astype(np.float32)

    return color_canvas, grayscale, shapes


def generate_sound_art(
    seed: int,
    *,
    image_size: tuple[int, int] = (192, 192),
    sample_rate: int = 48_000,
    shape_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, SyntheticSound, list[ShapeSpec]]:
    """Create a colour image and grayscale reference from a synthetic sound clip."""

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(sample_rate).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples))
    volumes = _normalized_band_volumes(spectrum)
    logger.info(
        "Generated sound spectrum for seed=%d sample_rate=%d with bands %s",
        seed,
        sample_rate,
        {key: round(val, 3) for key, val in volumes.items()},
    )

    color_canvas, grayscale, shapes = _synthesise_sound_image(
        rng, volumes, image_size, shape_count=shape_count
    )

    sound = SyntheticSound(
        seed=seed,
        sample_rate=sample_rate,
        samples=samples,
        band_volumes=volumes,
    )

    logger.debug("Generated %d shapes for seed=%d", len(shapes), seed)

    return color_canvas, grayscale, sound, shapes


def generate_sound_art_from_waveform(
    samples: np.ndarray,
    sample_rate: int,
    *,
    image_size: tuple[int, int] = (192, 192),
    seed: int | None = None,
    shape_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, SyntheticSound, list[ShapeSpec]]:
    """Create a colour/grayscale pair from an uploaded waveform."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    wave = np.asarray(samples, dtype=np.float32)
    if wave.ndim > 1:
        wave = np.mean(wave, axis=1)
    if wave.size == 0:
        raise ValueError("Audio clip contains no samples")

    center = float(np.max(np.abs(wave)))
    if center > 0:
        wave = wave / center

    spectrum = np.abs(np.fft.rfft(wave))
    volumes = _normalized_band_volumes(spectrum)

    if seed is None:
        seed = _seed_from_samples(wave)

    rng = np.random.default_rng(int(seed))
    color_canvas, grayscale, shapes = _synthesise_sound_image(
        rng, volumes, image_size, shape_count=shape_count
    )

    sound = SyntheticSound(
        seed=int(seed),
        sample_rate=int(sample_rate),
        samples=wave.astype(np.float32),
        band_volumes=volumes,
    )

    logger.info(
        "Generated sound spectrum from waveform sample_rate=%d with bands %s",
        sample_rate,
        {key: round(val, 3) for key, val in volumes.items()},
    )

    return color_canvas, grayscale, sound, shapes


def generate_sound_art_gallery(
    sounds: Sequence[SyntheticSound],
    *,
    resolution: tuple[int, int] = (192, 192),
) -> list[np.ndarray]:
    """Create FFT-guided canvases for a collection of sounds."""

    gallery: list[np.ndarray] = []
    for sound in sounds:
        spectrum = np.fft.rfft(sound.samples)
        priors = _normalized_band_volumes(np.abs(spectrum))
        rng = np.random.default_rng(int(sound.seed))
        image, _, _ = _synthesise_sound_image(rng, priors, resolution)
        gallery.append(image)
    return gallery


def guess_shapes(image: np.ndarray, threshold: float = 0.2) -> list[ShapeGuess]:
    """Attempt to recover geometric primitives from ``image`` on a per-channel basis."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a colour image with three channels")

    results: list[ShapeGuess] = []
    prototypes = {"square": 1.0, "circle": np.pi / 4.0, "triangle": 0.5}
    channels = {"red": 0, "green": 1, "blue": 2}

    for color_name, channel in channels.items():
        layer = np.clip(image[..., channel], 0.0, 1.0)
        if layer.max() <= 0.0:
            continue
        mask = layer > (threshold * layer.max())
        if not np.any(mask):
            continue

        indices = np.argwhere(mask)
        ymin, xmin = indices.min(axis=0)
        ymax, xmax = indices.max(axis=0)
        height = max(int(ymax - ymin + 1), 1)
        width = max(int(xmax - xmin + 1), 1)
        bbox_area = float(height * width)
        filled_area = float(mask.sum())
        ratio = filled_area / bbox_area if bbox_area > 0 else 0.0

        guess = min(prototypes.items(), key=lambda item: abs(ratio - item[1]))[0]
        diff = abs(ratio - prototypes[guess])
        confidence = float(max(0.0, 1.0 - diff / 0.5))
        volume = float(layer[mask].mean())

        results.append(ShapeGuess(color=color_name, guess=guess, confidence=confidence, volume=volume))

    logger.debug("Shape guesses: %s", results)

    return results


def load_waveform_from_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode PCM WAV ``data`` into normalised mono samples."""

    buffer = io.BytesIO(data)
    with contextlib.closing(wave.open(buffer, "rb")) as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        if frame_count == 0:
            raise ValueError("WAV file is empty")

        raw = wav_file.readframes(frame_count)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bit")

    samples = np.frombuffer(raw, dtype=dtype_map[sample_width])
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)

    if sample_width == 1:
        samples -= 128.0

    max_val = float(np.max(np.abs(samples)))
    if max_val > 0:
        samples /= max_val

    return samples.astype(np.float32), int(sample_rate)


__all__ = [
    "SyntheticSound",
    "ShapeSpec",
    "ShapeGuess",
    "generate_sound_art",
    "generate_sound_art_from_waveform",
    "guess_shapes",
    "load_waveform_from_wav",
    "MessyKeyArtifact",
    "derive_messy_latent",
    "messy_key_hash_from_overlap",
    "generate_sound_art_gallery",
]
