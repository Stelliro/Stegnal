"""Deterministic helpers for converting between images and WAV waveforms."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from dataclasses import dataclass

from .reconstruction import (
    image_to_waveform,
    reconstruct_from_waveform,
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
        max_val = float(np.max(array))
        if max_val == 0:
            max_val = 1.0
        array = np.clip(array / max_val, 0.0, 1.0)
    return array[..., :3].astype(np.float32)


@dataclass(frozen=True)
class DecodedWavMetadata:
    """Metadata describing a reconstructed image extracted from WAV bytes."""

    sample_rate: int
    segments: int
    marker_duration: float


def encode_image_to_waveform(
    image: np.ndarray | Image.Image,
    *,
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> np.ndarray:
    """Return a mono waveform encoding of ``image`` at ``sample_rate`` samples."""

    rgb = _ensure_rgb_image(image)
    waveform = image_to_waveform(
        rgb,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    logger.debug(
        "Encoded image to waveform with resolution %s at %d Hz",
        rgb.shape[:2],
        sample_rate,
    )
    return waveform.astype(np.float32)


def encode_image_to_wav_bytes(
    image: np.ndarray | Image.Image,
    *,
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> bytes:
    """Encode ``image`` into deterministic WAV bytes."""

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    return waveform_to_wav_bytes(waveform, sample_rate)


def decode_waveform_to_image(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    resolution: tuple[int, int],
    segments: int | None = 1,
    marker_duration: float = 0.05,
    advanced_logging: bool = False,
) -> np.ndarray:
    """Decode ``waveform`` back into an RGB image."""

    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")
    if advanced_logging:
        logger.debug(
            "Decoding waveform with advanced logging: samples=%d resolution=%s sample_rate=%d segments=%s marker_duration=%.5f",
            np.asarray(waveform).size,
            resolution,
            sample_rate,
            "auto" if segments is None else int(segments),
            marker_duration,
        )
    image = reconstruct_from_waveform(
        waveform,
        resolution=(int(rows), int(cols)),
        sample_rate=int(sample_rate),
        segments=segments,
        marker_duration=float(marker_duration),
        advanced_logging=advanced_logging,
    )
    return image.astype(np.float32)


def decode_wav_bytes_to_image(
    data: bytes,
    *,
    resolution: tuple[int, int],
    sample_rate: int | None = None,
    segments: int | None = 1,
    marker_duration: float = 0.05,
    return_metadata: bool = False,
    advanced_logging: bool = False,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, DecodedWavMetadata]:
    """Decode WAV ``data`` into an image.

    Parameters
    ----------
    data:
        The PCM WAV byte stream to decode.
    resolution:
        Target resolution for the reconstructed image.
    sample_rate:
        Optional override for the waveform sample rate. When omitted the rate is
        extracted from the WAV header.
    segments:
        Number of fax-style segments embedded in the waveform. When ``None``
        the decoder will attempt to infer this value using the waveform length.
    marker_duration:
        Duration, in seconds, of each marker tone that separates a segment.
    return_metadata:
        When ``True`` the second element of the return tuple is a
        :class:`DecodedWavMetadata` instance containing the detected sample rate
        together with the segments and marker configuration used during
        reconstruction. The default behaviour preserves the legacy tuple of
        ``(image, sample_rate)``.
    advanced_logging:
        When ``True`` additional debug logging is emitted while reconstructing
        the image, aiding investigations into problematic waveforms.
    """

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Expected WAV bytes for decoding")

    waveform, detected_rate = load_waveform_from_wav(bytes(data))
    target_rate = int(sample_rate or detected_rate)
    if advanced_logging:
        logger.debug(
            "Loaded WAV bytes: detected_rate=%d override=%s resolution=%s segments=%s marker_duration=%.5f",
            detected_rate,
            sample_rate,
            resolution,
            "auto" if segments is None else int(segments),
            marker_duration,
        )
    reconstructed, used_segments = reconstruct_from_waveform(
        waveform,
        resolution=(int(resolution[0]), int(resolution[1])),
        sample_rate=target_rate,
        segments=segments,
        marker_duration=float(marker_duration),
        advanced_logging=advanced_logging,
        return_segments=True,
    )
    logger.debug(
        "Decoded WAV bytes to image at %d Hz with resolution %s",
        target_rate,
        resolution,
    )
    if return_metadata:
        metadata = DecodedWavMetadata(
            sample_rate=target_rate,
            segments=int(used_segments),
            marker_duration=float(marker_duration),
        )
        return reconstructed.astype(np.float32), metadata
    return reconstructed.astype(np.float32), target_rate


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
    "save_image_as_png",
    "save_waveform_as_wav",
]

