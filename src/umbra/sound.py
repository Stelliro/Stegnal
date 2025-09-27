"""Synthetic sound-driven image generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticSound:
    """Representation of a randomly generated sound clip."""

    seed: int
    sample_rate: int
    samples: np.ndarray
    band_volumes: Dict[str, float]


@dataclass(frozen=True)
class ShapeSpec:
    """Description of a coloured geometric shape encoded in the sound image."""

    color: str
    shape: str
    volume: float


@dataclass(frozen=True)
class ShapeGuess:
    """Prediction made by the shape-guessing helper."""

    color: str
    guess: str
    confidence: float
    volume: float


def _normalized_band_volumes(spectrum: np.ndarray) -> Dict[str, float]:
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


def _draw_circle(canvas: np.ndarray, center: Tuple[int, int], radius: int, channel: int, intensity: float) -> None:
    rows, cols = canvas.shape[:2]
    y_indices, x_indices = np.ogrid[:rows, :cols]
    cy, cx = center
    mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius ** 2
    canvas[..., channel][mask] = np.maximum(canvas[..., channel][mask], intensity)


def _draw_square(canvas: np.ndarray, center: Tuple[int, int], size: int, channel: int, intensity: float) -> None:
    rows, cols = canvas.shape[:2]
    half = size // 2
    top = max(center[0] - half, 0)
    bottom = min(center[0] + half, rows)
    left = max(center[1] - half, 0)
    right = min(center[1] + half, cols)
    canvas[top:bottom, left:right, channel] = np.maximum(
        canvas[top:bottom, left:right, channel], intensity
    )


def _triangle_mask(rows: int, cols: int, vertices: Sequence[Tuple[int, int]]) -> np.ndarray:
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    (y0, x0), (y1, x1), (y2, x2) = vertices
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return np.zeros((rows, cols), dtype=bool)
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    return (a >= 0) & (b >= 0) & (c >= 0)


def _draw_triangle(
    canvas: np.ndarray,
    center: Tuple[int, int],
    size: int,
    channel: int,
    intensity: float,
) -> None:
    rows, cols = canvas.shape[:2]
    cy, cx = center
    top = max(cy - size, 0)
    bottom = min(cy + size, rows - 1)
    left = max(cx - size, 0)
    right = min(cx + size, cols - 1)

    vertices = ((top, cx), (bottom, left), (bottom, right))
    mask = _triangle_mask(rows, cols, vertices)
    canvas[..., channel][mask] = np.maximum(canvas[..., channel][mask], intensity)


def generate_sound_art(
    seed: int,
    *,
    image_size: Tuple[int, int] = (192, 192),
    sample_rate: int = 48_000,
) -> Tuple[np.ndarray, np.ndarray, SyntheticSound, List[ShapeSpec]]:
    """Create a colour image and grayscale reference from a synthetic sound clip."""

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(sample_rate).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples))
    volumes = _normalized_band_volumes(spectrum)

    color_canvas = np.zeros((*image_size, 3), dtype=np.float32)
    rows, cols = image_size
    min_extent = max(min(rows, cols) // 6, 12)
    max_extent = max(min(rows, cols) // 3, min_extent + 4)

    shapes: List[ShapeSpec] = []
    shape_types = ("circle", "square", "triangle")
    channels = {"red": 0, "green": 1, "blue": 2}

    for color_name in ("red", "green", "blue"):
        channel = channels[color_name]
        extent = int(rng.integers(min_extent, max_extent + 1))
        cy = int(rng.integers(extent, rows - extent)) if rows > 2 * extent else rows // 2
        cx = int(rng.integers(extent, cols - extent)) if cols > 2 * extent else cols // 2
        center = (cy, cx)
        shape = rng.choice(shape_types)
        intensity = float(np.clip(volumes[color_name], 0.05, 1.0))

        if shape == "circle":
            _draw_circle(color_canvas, center, extent, channel, intensity)
        elif shape == "square":
            _draw_square(color_canvas, center, extent * 2, channel, intensity)
        else:
            _draw_triangle(color_canvas, center, extent, channel, intensity)

        shapes.append(ShapeSpec(color=color_name, shape=shape, volume=intensity))

    color_canvas = np.clip(color_canvas, 0.0, 1.0)
    grayscale = np.clip(
        0.299 * color_canvas[..., 0]
        + 0.587 * color_canvas[..., 1]
        + 0.114 * color_canvas[..., 2],
        0.0,
        1.0,
    ).astype(np.float32)

    sound = SyntheticSound(
        seed=seed,
        sample_rate=sample_rate,
        samples=samples,
        band_volumes=volumes,
    )

    return color_canvas, grayscale, sound, shapes


def guess_shapes(image: np.ndarray, threshold: float = 0.2) -> List[ShapeGuess]:
    """Attempt to recover geometric primitives from ``image`` on a per-channel basis."""

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected a colour image with three channels")

    results: List[ShapeGuess] = []
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

    return results


__all__ = [
    "SyntheticSound",
    "ShapeSpec",
    "ShapeGuess",
    "generate_sound_art",
    "guess_shapes",
]
