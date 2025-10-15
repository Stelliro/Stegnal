"""Deterministic helpers for converting between images and WAV waveforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

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

    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return _generate_placeholder_image(resolution)

    samples = max(rows * cols, 1)
    spectrum = np.abs(np.fft.fft(wave, n=samples))
    spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if spectrum.size < samples:
        spectrum = np.pad(spectrum, (0, samples - spectrum.size))
    elif spectrum.size > samples:
        spectrum = spectrum[:samples]

    preview = spectrum.reshape(rows, cols)
    max_val = float(np.max(preview, initial=0.0))
    min_val = float(np.min(preview, initial=0.0))
    if not np.isfinite(max_val) or not np.isfinite(min_val) or max_val <= min_val:
        return _generate_placeholder_image(resolution)

    base = ((preview - min_val) / max(max_val - min_val, np.finfo(np.float32).eps)).astype(
        np.float32
    )
    accent = np.sqrt(base).astype(np.float32)
    highlight = np.clip(1.0 - base, 0.0, 1.0).astype(np.float32)
    return np.stack((base, accent, highlight), axis=2).astype(np.float32)


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


def _reconstruct_with_strategies(
    waveform: np.ndarray,
    *,
    resolution: tuple[int, int],
    sample_rate: int,
    marker_duration: float,
    segments: int | None,
    advanced_logging: bool,
) -> tuple[np.ndarray, int]:
    """Attempt waveform reconstruction using multiple segment strategies."""

    attempts: list[tuple[str, int | None]] = []
    if segments is not None:
        attempts.append(("requested", int(segments)))
    attempts.append(("auto", None))
    attempts.append(("single", 1))

    tried: set[int | str] = set()
    failures: list[str] = []

    for label, segment_hint in attempts:
        key: int | str = "auto" if segment_hint is None else int(segment_hint)
        if key in tried:
            continue
        tried.add(key)

        if advanced_logging:
            logger.debug(
                "Attempting waveform reconstruction with %s segments", key
            )

        try:
            result = reconstruct_from_waveform(
                waveform,
                resolution=resolution,
                sample_rate=sample_rate,
                segments=segment_hint,
                marker_duration=marker_duration,
                advanced_logging=advanced_logging,
                return_segments=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            message = f"{label} ({key}) failed: {exc}"
            failures.append(message)
            if advanced_logging:
                logger.debug(message)
            continue

        image: np.ndarray
        used_segments: int
        if isinstance(result, tuple):
            image, used_segments = result
        else:  # pragma: no cover - legacy behaviour guard
            image = result
            used_segments = 1 if segment_hint in (None, 0) else int(segment_hint)

        return np.asarray(image, dtype=np.float32), int(max(used_segments, 1))

    raise RuntimeError("; ".join(failures) if failures else "reconstruction failed")


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

    rows, cols = int(resolution[0]), int(resolution[1])
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
    try:
        image, _used_segments = _reconstruct_with_strategies(
            waveform,
            resolution=(rows, cols),
            sample_rate=int(sample_rate),
            segments=segments,
            marker_duration=float(marker_duration),
            advanced_logging=advanced_logging,
        )
        return image.astype(np.float32)
    except Exception as exc:  # pragma: no cover - defensive audio fallback
        logger.warning(
            "Failed to reconstruct image from waveform; returning preview image: %s",
            exc,
        )
        return _waveform_preview_image(waveform, (rows, cols))


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

    rows, cols = int(resolution[0]), int(resolution[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")

    fallback_waveform: np.ndarray | None = None
    fallback_image = _generate_placeholder_image((rows, cols))
    fallback_segments = 1
    if isinstance(segments, int) and segments > 0:
        fallback_segments = int(segments)

    detected_rate: int | None = None
    target_rate: int | None = None
    used_segments: int | None = None

    try:
        waveform, detected_rate = load_waveform_from_wav(bytes(data))
        fallback_waveform = waveform
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
        reconstructed, used_segments = _reconstruct_with_strategies(
            waveform,
            resolution=(rows, cols),
            sample_rate=target_rate,
            segments=segments,
            marker_duration=float(marker_duration),
            advanced_logging=advanced_logging,
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
    except Exception as exc:  # pragma: no cover - defensive audio fallback
        logger.warning(
            "Failed to decode WAV bytes into image; returning preview image: %s",
            exc,
        )
        fallback_rate_candidates = (
            sample_rate,
            target_rate,
            detected_rate,
        )
        fallback_rate = 16_000
        for candidate in fallback_rate_candidates:
            try:
                rate = int(candidate)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if rate > 0:
                fallback_rate = rate
                break

        if fallback_waveform is not None:
            fallback_image = _waveform_preview_image(fallback_waveform, (rows, cols))

        if return_metadata:
            metadata = DecodedWavMetadata(
                sample_rate=fallback_rate,
                segments=int(used_segments) if used_segments else fallback_segments,
                marker_duration=float(marker_duration),
            )
            return fallback_image, metadata
        return fallback_image, fallback_rate


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

