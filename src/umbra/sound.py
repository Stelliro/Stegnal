# sound.py

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
        if buffer.size == 0:
            return cls(hash="", latent=np.array([]))
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
    if array.size == 0:
        return ""
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

    if samples.size == 0:
        return 0
    digest = hashlib.sha1(np.asarray(samples, dtype=np.float32).tobytes()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _normalized_band_volumes(spectrum: np.ndarray) -> dict[str, float]:
    """Split ``spectrum`` into three bands and normalise their magnitudes."""

    if spectrum.ndim != 1:
        raise ValueError("Expected a one-dimensional spectrum array")
    if spectrum.size == 0:
        return {"red": 1.0, "green": 1.0, "blue": 1.0}

    band_edges = np.linspace(0, spectrum.size, 4, dtype=int)
    bands = []
    for idx in range(3):
        start, end = band_edges[idx], band_edges[idx + 1]
        band = spectrum[start:end]
        magnitude = float(np.mean(np.abs(band))) if band.size > 0 else 0.0
        bands.append(magnitude)

    max_val = max(bands) or 1.0
    norm = [val / max_val for val in bands]

    return {"red": norm[0], "green": norm[1], "blue": norm[2]}


def _draw_circle(canvas: np.ndarray, center: tuple[int, int], radius: int, intensity: float) -> None:
    rows, cols = canvas.shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    canvas[mask] = intensity


def _draw_square(canvas: np.ndarray, center: tuple[int, int], size: int, intensity: float, rotation: float) -> None:
    half = size / 2
    offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rotated = _rotate_offsets(offsets, rotation)
    vertices = rotated + np.array(center)
    _draw_filled_polygon(canvas, vertices, intensity)


def _draw_triangle(canvas: np.ndarray, center: tuple[int, int], size: int, intensity: float, rotation: float) -> None:
    height = size * np.sqrt(3) / 2
    offsets = np.array([[0, -height / 3], [-size / 2, height * 2 / 3], [size / 2, height * 2 / 3]])
    rotated = _rotate_offsets(offsets, rotation)
    vertices = rotated + np.array(center)
    _draw_filled_polygon(canvas, vertices, intensity)


def _draw_filled_polygon(canvas: np.ndarray, vertices: np.ndarray, intensity: float) -> None:
    rows, cols = canvas.shape
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

    canvas[yy[inside], xx[inside]] = intensity


def _rotate_offsets(offsets: np.ndarray, angle: float) -> np.ndarray:
    rad = np.deg2rad(angle)
    cos, sin = np.cos(rad), np.sin(rad)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return np.dot(offsets, rotation)


def _synthesise_sound_image(
    rng: np.random.Generator, priors: dict[str, float], resolution: tuple[int, int]
) -> tuple[np.ndarray, tuple[ShapeSpec, ...], np.ndarray]:
    """Generate an image from sound priors with shapes."""

    canvas = np.zeros(resolution + (3,), dtype=np.float32)
    specs = []
    prototypes = {"circle": _draw_circle, "square": _draw_square, "triangle": _draw_triangle}
    colors = {"red": (1,0,0), "green": (0,1,0), "blue": (0,0,1)}

    for color_name, vol in priors.items():
        if vol <= 0:
            continue
        shape_type = rng.choice(list(prototypes.keys()))
        size = int(vol * min(resolution) / 2)
        center = tuple(rng.integers(size // 2, dim - size // 2) for dim in resolution)
        rotation = float(rng.uniform(0, 360))
        channel = list(colors[color_name]).index(1)
        if shape_type == "circle":
            prototypes[shape_type](canvas[..., channel], center, size, vol)
        else:
            prototypes[shape_type](canvas[..., channel], center, size, vol, rotation)
        specs.append(ShapeSpec(color=color_name, shape=shape_type, volume=vol, center=center, rotation=rotation, size=size))

    return np.clip(canvas, 0.0, 1.0), tuple(specs), np.mean(canvas, axis=2)


def generate_sound_art(
    sound: SyntheticSound | None = None,
    resolution: tuple[int, int] = (128, 128),
    *,
    seed: int | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[ShapeSpec, ...], np.ndarray] | tuple[np.ndarray, np.ndarray, SyntheticSound, tuple[ShapeSpec, ...]]:
    """Generate an image inspired by sound.

    Two calling conventions are supported:

    * ``generate_sound_art(sound)`` → ``(color, shapes, gray)``  (legacy)
    * ``generate_sound_art(seed=N, image_size=HW)`` → ``(color, gray, sound, shapes)``
    """

    if seed is not None:
        res = image_size or resolution
        rng = np.random.default_rng(seed)
        sr = 22050
        samples = rng.standard_normal(sr).astype(np.float32)
        synth = SyntheticSound(
            seed=seed,
            sample_rate=sr,
            samples=samples,
            band_volumes=_normalized_band_volumes(np.abs(np.fft.rfft(samples))),
        )
        color, shapes, gray = _synthesise_sound_image(rng, synth.band_volumes, res)
        return color, gray, synth, shapes

    if sound is None:
        return np.zeros(resolution + (3,)), (), np.zeros(resolution)

    if sound.samples.size == 0:
        return np.zeros(resolution + (3,)), (), np.zeros(resolution)

    spectrum = np.fft.rfft(sound.samples)
    priors = _normalized_band_volumes(np.abs(spectrum))
    rng = np.random.default_rng(sound.seed)
    return _synthesise_sound_image(rng, priors, resolution)


def generate_sound_art_from_waveform(waveform: np.ndarray, sample_rate: int, resolution: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, tuple[ShapeSpec, ...], np.ndarray]:
    """Generate an image from a raw ``waveform``."""

    if waveform.size == 0:
        return np.zeros(resolution + (3,)), (), np.zeros(resolution)

    seed = _seed_from_samples(waveform)
    sound = SyntheticSound(seed=seed, sample_rate=sample_rate, samples=waveform, band_volumes={})
    return generate_sound_art(sound, resolution)


def generate_sound_art_gallery(sounds: Sequence[SyntheticSound], resolution: tuple[int, int] = (128, 128)) -> list[np.ndarray]:
    """Generate FFT-guided canvases for a collection of sounds."""

    gallery: list[np.ndarray] = []
    for sound in sounds:
        if sound.samples.size == 0:
            gallery.append(np.zeros(resolution + (3,)))
            continue
        spectrum = np.fft.rfft(sound.samples)
        priors = _normalized_band_volumes(np.abs(spectrum))
        rng = np.random.default_rng(sound.seed)
        image, _, _ = _synthesise_sound_image(rng, priors, resolution)
        gallery.append(image)
    return gallery


def guess_shapes(image: np.ndarray, threshold: float = 0.2) -> list[ShapeGuess]:
    """Attempt to recover geometric primitives from ``image`` on a per-channel basis."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a colour image with three channels")
    if image.size == 0:
        return []

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
        volume = float(layer[mask].mean()) if np.any(mask) else 0.0

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

    max_val = float(np.max(np.abs(samples))) or 1.0
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