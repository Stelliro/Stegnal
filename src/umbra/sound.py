"""Synthetic sound-driven image generation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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


def generate_sound_art(
    seed: int,
    *,
    image_size: tuple[int, int] = (192, 192),
    sample_rate: int = 48_000,
) -> tuple[np.ndarray, np.ndarray, SyntheticSound, list[ShapeSpec]]:
    """Create a colour image and grayscale reference from a synthetic sound clip."""

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(sample_rate).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples))
    volumes = _normalized_band_volumes(spectrum)

    color_canvas = np.zeros((*image_size, 3), dtype=np.float32)
    rows, cols = image_size
    min_extent = max(min(rows, cols) // 5, 18)
    max_extent = max(min(rows, cols) // 4, min_extent + 6)
    padding = int(np.ceil(max_extent * 1.25))

    shapes: list[ShapeSpec] = []
    shape_types = ("circle", "square", "triangle")
    channels = {"red": 0, "green": 1, "blue": 2}

    for color_name in ("red", "green", "blue"):
        channel = channels[color_name]
        extent = int(rng.integers(min_extent, max_extent + 1))
        cy = int(rng.integers(padding, rows - padding)) if rows > 2 * padding else rows // 2
        cx = int(rng.integers(padding, cols - padding)) if cols > 2 * padding else cols // 2
        center = (cy, cx)
        shape = rng.choice(shape_types)
        rotation = float(rng.uniform(0, 2 * np.pi)) if shape != "circle" else 0.0
        intensity = float(np.clip(volumes[color_name], 0.05, 1.0))

        if shape == "circle":
            _draw_circle(color_canvas, center, extent, channel, intensity)
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
            else:
                height = float(extent)
                width = float(extent)
                base = np.array(
                    [
                        [0.0, -height],
                        [width, height],
                        [-width, height],
                    ],
                    dtype=np.float32,
                )

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
                size=extent,
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

    sound = SyntheticSound(
        seed=seed,
        sample_rate=sample_rate,
        samples=samples,
        band_volumes=volumes,
    )

    return color_canvas, grayscale, sound, shapes


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

    return results


__all__ = [
    "SyntheticSound",
    "ShapeSpec",
    "ShapeGuess",
    "generate_sound_art",
    "guess_shapes",
]
