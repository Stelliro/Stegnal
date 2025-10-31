# codec.py

"""Deterministic helpers for converting between images, text payloads, and WAV waveforms."""

from __future__ import annotations

import logging
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .reconstruction import (
    image_to_waveform,
    reconstruct_from_waveform,
    suggest_sample_rate,
    suggest_transmission_profile,
    waveform_to_wav_bytes,
)
from .sound import load_waveform_from_wav

logger = logging.getLogger(__name__)


def _ensure_rgb_image(image: np.ndarray | Image.Image) -> np.ndarray:
    """Normalise ``image`` to a ``float32`` RGB array in ``[0, 1]``."""

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        array /= 255.0
        return np.clip(array, 0.0, 1.0)

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")
    if array.max() > 1.0 or array.min() < 0.0:
        max_val = float(np.max(array)) or 1.0
        array = np.clip(array / max_val, 0.0, 1.0)
    return array[..., :3].astype(np.float32)


def _generate_placeholder_image(resolution: tuple[int, int]) -> np.ndarray:
    """Return a deterministic gradient placeholder for ``resolution``."""

    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")

    y_gradient = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    x_gradient = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    base = (0.55 * x_gradient + 0.45 * y_gradient) % 1.0
    base = base.astype(np.float32)
    accent = np.sqrt(np.clip(base, 0.0, 1.0)).astype(np.float32)
    highlight = np.clip(1.0 - base, 0.0, 1.0).astype(np.float32)
    return np.stack((base, accent, highlight), axis=2).astype(np.float32)


def _waveform_preview_image(waveform: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Create a visually informative preview derived from ``waveform``."""

    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return np.zeros((rows, cols, 3), dtype=np.float32)

    # Downsample to fit cols
    step = max(1, waveform.size // cols)
    downsampled = waveform[::step][:cols]
    normalized = np.abs(downsampled)
    normalized /= np.max(normalized) + 1e-6

    # Repeat vertically with falloff
    grid = np.repeat(normalized[None, :], rows, axis=0)
    falloff = np.linspace(1.0, 0.2, rows)[:, None]
    grid *= falloff

    return np.repeat(grid[..., None], 3, axis=2).astype(np.float32)


def _pack_text_into_image(text: str, resolution: tuple[int, int]) -> np.ndarray:
    """Embed ``text`` into a grayscale image of ``resolution``."""

    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")

    compressed = zlib.compress(text.encode("utf-8"))
    packed = np.frombuffer(compressed, dtype=np.uint8)
    if packed.size > rows * cols:
        raise ValueError(f"Compressed text ({packed.size} bytes) exceeds image capacity ({rows * cols} pixels)")

    padded = np.pad(packed, (0, rows * cols - packed.size), mode="constant")
    return padded.reshape(rows, cols).astype(np.float32) / 255.0


def _unpack_text_from_image(image: np.ndarray) -> str:
    """Extract embedded text from a grayscale ``image``."""

    array = np.asarray(image, dtype=np.float32).reshape(-1)
    unpacked = (array * 255.0).astype(np.uint8)
    try:
        decompressed = zlib.decompress(unpacked.tobytes())
    except zlib.error as exc:
        logger.warning(f"Failed to decompress text from image: {exc}")
        return ""
    return decompressed.decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class TextEncodingMetadata:
    """Metadata embedded during text-to-image encoding."""

    resolution: tuple[int, int]
    payload_length: int
    sample_rate: int = 48000
    segments: int = 1
    marker_duration: float = 0.05
    raw_text: str | None = None

    def with_waveform(
        self,
        *,
        sample_rate: int | None = None,
        segments: int | None = None,
        marker_duration: float | None = None,
    ) -> TextEncodingMetadata:
        return TextEncodingMetadata(
            resolution=self.resolution,
            payload_length=self.payload_length,
            sample_rate=int(sample_rate) if sample_rate is not None else self.sample_rate,
            segments=int(segments) if segments is not None else self.segments,
            marker_duration=float(marker_duration) if marker_duration is not None else self.marker_duration,
            raw_text=self.raw_text,
        )


def encode_text_to_image(
    text: str,
    *,
    resolution: tuple[int, int] | None = None,
    placeholder: bool = True,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[np.ndarray, TextEncodingMetadata]:
    """Encode ``text`` into an RGB image suitable for sonic transmission."""

    if not text:
        text = "Hello, Umbra!"

    if resolution is None:
        side = int(math.ceil(math.sqrt(len(text.encode("utf-8")) * 1.5)))
        resolution = (side, side)

    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")

    grayscale = _pack_text_into_image(text, resolution)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)

    if placeholder:
        placeholder_image = _generate_placeholder_image(resolution)
        rgb = np.maximum(rgb, placeholder_image * 0.3)

    metadata = TextEncodingMetadata(
        resolution=resolution,
        payload_length=len(text),
        sample_rate=suggest_sample_rate(rgb) if sample_rate is None else int(sample_rate),
        segments=suggest_transmission_profile(rgb)[0] if segments is None else int(segments),
        marker_duration=suggest_transmission_profile(rgb)[1] if marker_duration is None else float(marker_duration),
        raw_text=text,
    )

    return rgb.astype(np.float32), metadata


def decode_image_to_text(
    image: np.ndarray,
    *,
    resolution: tuple[int, int] | None = None,
) -> tuple[str, TextEncodingMetadata]:
    """Extract the embedded text from ``image``."""

    array = _ensure_rgb_image(image)
    if resolution is not None:
        array = resize(array, resolution + (3,), mode="reflect", anti_aliasing=True)
    grayscale = np.mean(array, axis=2)
    text = _unpack_text_from_image(grayscale)

    metadata = TextEncodingMetadata(
        resolution=tuple(array.shape[:2]),
        payload_length=len(text),
    )

    return text, metadata


def encode_text_to_waveform(
    text: str,
    *,
    resolution: tuple[int, int] | None = None,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[np.ndarray, TextEncodingMetadata]:
    """Encode ``text`` into a waveform suitable for sonic transmission."""

    image, metadata = encode_text_to_image(
        text,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    waveform = image_to_waveform(
        image,
        sample_rate=metadata.sample_rate,
    )
    return waveform.astype(np.float32), metadata


def encode_text_to_wav_bytes(
    text: str,
    *,
    resolution: tuple[int, int] | None = None,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[bytes, TextEncodingMetadata]:
    """Encode ``text`` into WAV bytes."""

    waveform, metadata = encode_text_to_waveform(
        text,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    wav_bytes = waveform_to_wav_bytes(waveform, metadata.sample_rate)
    return wav_bytes, metadata


def encode_image_to_waveform(
    image: np.ndarray | Image.Image,
    *,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> np.ndarray:
    """Encode ``image`` into a waveform suitable for sonic transmission."""

    array = _ensure_rgb_image(image)
    if sample_rate is None:
        sample_rate = suggest_sample_rate(array)
    if segments is None or marker_duration is None:
        seg, dur = suggest_transmission_profile(array)
        segments = segments or seg
        marker_duration = marker_duration or dur

    waveform = image_to_waveform(
        array,
        sample_rate=sample_rate,
    )
    return waveform.astype(np.float32)


def encode_image_to_wav_bytes(
    image: np.ndarray | Image.Image,
    *,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> bytes:
    """Encode ``image`` into WAV bytes."""

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    return waveform_to_wav_bytes(waveform, sample_rate or suggest_sample_rate(image))


@dataclass(frozen=True)
class DecodedWavMetadata:
    """Metadata extracted during WAV-to-image decoding."""

    sample_rate: int
    segments: int
    marker_duration: float
    raw_text: str | None = None


def decode_waveform_to_image(
    waveform: np.ndarray,
    resolution: tuple[int, int],
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
    return_metadata: bool = False,
    advanced_logging: bool = False,
) -> tuple[np.ndarray, DecodedWavMetadata]:
    """Decode ``waveform`` into an RGB image using heuristic reconstruction."""

    if waveform.size == 0:
        return np.zeros(resolution + (3,), dtype=np.float32), DecodedWavMetadata(sample_rate, segments, marker_duration)

    image = reconstruct_from_waveform(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )

    metadata = DecodedWavMetadata(
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )

    if return_metadata:
        return image.astype(np.float32), metadata
    return image.astype(np.float32), metadata


def decode_wav_bytes_to_image(
    data: bytes,
    resolution: tuple[int, int],
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
    return_metadata: bool = False,
    advanced_logging: bool = False,
) -> tuple[np.ndarray, DecodedWavMetadata]:
    """Decode WAV ``data`` into an RGB image."""

    if not data:
        return np.zeros(resolution + (3,), dtype=np.float32), DecodedWavMetadata(48000, 1, 0.05)

    waveform, loaded_rate = load_waveform_from_wav(data)
    sample_rate = sample_rate or loaded_rate

    if segments is None or marker_duration is None:
        seg, dur = suggest_transmission_profile(np.zeros(resolution))
        segments = segments or seg
        marker_duration = marker_duration or dur

    image, metadata = decode_waveform_to_image(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
        return_metadata=True,
        advanced_logging=advanced_logging,
    )

    if return_metadata:
        return image, metadata
    return image, metadata


def decode_waveform_to_text(
    waveform: np.ndarray,
    resolution: tuple[int, int],
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> tuple[str, DecodedWavMetadata]:
    """Decode ``waveform`` into text."""

    image, metadata = decode_waveform_to_image(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
        return_metadata=True,
    )
    try:
        text = _unpack_text_from_image(np.mean(image, axis=2))
    except zlib.error as exc:
        logger.warning(f"Failed to unpack text from waveform image: {exc}")
        text = ""
    return text, metadata


def decode_wav_bytes_to_text(
    data: bytes,
    resolution: tuple[int, int],
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[str, DecodedWavMetadata]:
    """Decode WAV ``data`` into text."""

    waveform, loaded_rate = load_waveform_from_wav(data)
    sample_rate = sample_rate or loaded_rate

    if segments is None or marker_duration is None:
        seg, dur = suggest_transmission_profile(np.zeros(resolution))
        segments = segments or seg
        marker_duration = marker_duration or dur

    return decode_waveform_to_text(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )


def save_image_as_png(data: np.ndarray, path: str | Path) -> Path:
    """Persist ``data`` as a PNG image at ``path`` and return the path."""

    array = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
    png = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    png.save(destination)
    return destination


def save_waveform_as_wav(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    path: str | Path,
) -> Path:
    """Write ``waveform`` to ``path`` as 16-bit PCM WAV and return the path."""

    wav_bytes = waveform_to_wav_bytes(np.asarray(waveform, dtype=np.float32), sample_rate)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(wav_bytes)
    return destination


__all__ = [
    "DecodedWavMetadata",
    "decode_wav_bytes_to_image",
    "decode_waveform_to_image",
    "encode_image_to_waveform",
    "encode_image_to_wav_bytes",
    "encode_text_to_image",
    "encode_text_to_waveform",
    "encode_text_to_wav_bytes",
    "decode_image_to_text",
    "decode_waveform_to_text",
    "decode_wav_bytes_to_text",
    "TextEncodingMetadata",
    "save_image_as_png",
    "save_waveform_as_wav",
]