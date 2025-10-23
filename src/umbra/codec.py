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


@dataclass(frozen=True)
class TextEncodingMetadata:
    """Metadata describing a text payload embedded within an image or waveform."""

    width: int
    height: int
    payload_bytes: int
    sample_rate: int | None = None
    segments: int | None = None
    marker_duration: float | None = None
    raw_text: str | None = None

    def with_waveform(
        self,
        *,
        sample_rate: int,
        segments: int,
        marker_duration: float,
    ) -> TextEncodingMetadata:
        """Return a copy of the metadata populated with waveform parameters."""

        return TextEncodingMetadata(
            width=self.width,
            height=self.height,
            payload_bytes=self.payload_bytes,
            sample_rate=int(sample_rate),
            segments=int(segments),
            marker_duration=float(marker_duration),
            raw_text=self.raw_text,
        )


_TEXT_HEADER_STRUCT = struct.Struct("<8sIIII")
_TEXT_MAGIC = b"UMBRTX01"
_TEXT_VERSION_SEED = 0x554D4252  # 'UMBR' encoded as an integer seed
_TEXT_BLOCK_SIZE = 4
_TEXT_HEADER_REPETITIONS = 6


def encode_image_to_waveform(
    image: np.ndarray | Image.Image,
    *,
    sample_rate: int,
    segments: int = 1,
    marker_duration: float = 0.05,
    allow_cpu_fallback: bool = True,
) -> np.ndarray:
    """Return a mono waveform encoding of ``image`` at ``sample_rate`` samples."""

    rgb = _ensure_rgb_image(image)
    waveform = image_to_waveform(
        rgb,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
        allow_cpu_fallback=allow_cpu_fallback,
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
    allow_cpu_fallback: bool = True,
) -> bytes:
    """Encode ``image`` into deterministic WAV bytes."""

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
        allow_cpu_fallback=allow_cpu_fallback,
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
    allow_cpu_fallback: bool,
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
                allow_cpu_fallback=allow_cpu_fallback,
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
    allow_cpu_fallback: bool = True,
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
            allow_cpu_fallback=allow_cpu_fallback,
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
    allow_cpu_fallback: bool = True,
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
            allow_cpu_fallback=allow_cpu_fallback,
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


def encode_text_to_image(
    text: str,
    *,
    width: int = 256,
) -> tuple[np.ndarray, TextEncodingMetadata]:
    """Encode ``text`` into a colourful, static-like RGB image."""

    if width <= 0:
        raise ValueError("width must be positive for text encoding")

    payload = text.encode("utf-8")
    payload_len = len(payload)
    seed = zlib.crc32(payload) ^ _TEXT_VERSION_SEED
    header = _TEXT_HEADER_STRUCT.pack(_TEXT_MAGIC, width, 0, payload_len, seed)
    header_bits = np.unpackbits(np.frombuffer(header, dtype=np.uint8))
    header_bits_encoded = np.repeat(header_bits, _TEXT_HEADER_REPETITIONS)
    payload_bits = (
        np.unpackbits(np.frombuffer(payload, dtype=np.uint8)) if payload_len else np.zeros(0, dtype=np.uint8)
    )

    total_bits = header_bits_encoded.size + payload_bits.size
    blocks_per_row = max(1, int(math.ceil(width / float(_TEXT_BLOCK_SIZE))))
    pixel_width = blocks_per_row * _TEXT_BLOCK_SIZE
    height = max(1, int(math.ceil(total_bits / float(blocks_per_row))))
    header = _TEXT_HEADER_STRUCT.pack(_TEXT_MAGIC, pixel_width, height * _TEXT_BLOCK_SIZE, payload_len, seed)
    header_bits = np.unpackbits(np.frombuffer(header, dtype=np.uint8))
    header_bits_encoded = np.repeat(header_bits, _TEXT_HEADER_REPETITIONS)
    total_bits = header_bits_encoded.size + payload_bits.size
    block_count = max(1, int(math.ceil(total_bits / float(blocks_per_row))))

    total_blocks = blocks_per_row * block_count
    padded_bits = np.zeros(total_blocks, dtype=np.uint8)
    padded_bits[: header_bits_encoded.size] = header_bits_encoded
    if payload_bits.size:
        padded_bits[header_bits_encoded.size : header_bits_encoded.size + payload_bits.size] = payload_bits

    bits_grid = padded_bits.reshape(block_count, blocks_per_row)
    pixel_height = block_count * _TEXT_BLOCK_SIZE
    pixel_width = blocks_per_row * _TEXT_BLOCK_SIZE
    bits_expanded = np.repeat(np.repeat(bits_grid, _TEXT_BLOCK_SIZE, axis=0), _TEXT_BLOCK_SIZE, axis=1)

    rng = np.random.default_rng(seed)
    noise = rng.random((pixel_height, pixel_width, 3), dtype=np.float32)
    high = np.clip(0.7 + 0.25 * noise, 0.0, 1.0)
    low = np.clip(0.05 + 0.2 * noise, 0.0, 1.0)
    image = np.where(bits_expanded[..., None].astype(bool), high, low).astype(np.float32)

    metadata = TextEncodingMetadata(
        width=int(pixel_width),
        height=int(pixel_height),
        payload_bytes=int(payload_len),
        raw_text=text,
    )
    return image, metadata


def decode_image_to_text(image: np.ndarray) -> tuple[str, TextEncodingMetadata]:
    """Decode text previously embedded with :func:`encode_text_to_image`."""

    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("expected RGB image for text decoding")

    pixel_height, pixel_width = array.shape[:2]
    blocks_per_row = max(1, pixel_width // _TEXT_BLOCK_SIZE)
    block_rows = max(1, pixel_height // _TEXT_BLOCK_SIZE)
    usable_height = block_rows * _TEXT_BLOCK_SIZE
    usable_width = blocks_per_row * _TEXT_BLOCK_SIZE
    grayscale = np.mean(array[:usable_height, :usable_width, :3], axis=2)
    block_view = grayscale.reshape(block_rows, _TEXT_BLOCK_SIZE, blocks_per_row, _TEXT_BLOCK_SIZE)
    block_means = block_view.mean(axis=(1, 3)).reshape(-1)

    header_bits_len = _TEXT_HEADER_STRUCT.size * 8
    header_sample_count = header_bits_len * _TEXT_HEADER_REPETITIONS
    if block_means.size < header_sample_count:
        raise ValueError("image too small to contain a text payload")

    header_samples = block_means[:header_sample_count]
    header_group_means = header_samples.reshape(header_bits_len, _TEXT_HEADER_REPETITIONS).mean(axis=1)

    magic_bits = np.unpackbits(np.frombuffer(_TEXT_MAGIC, dtype=np.uint8))
    if header_group_means.size < magic_bits.size:
        raise ValueError("image too small to contain a text payload")
    magic_samples = header_group_means[: magic_bits.size]
    high_vals = magic_samples[magic_bits == 1]
    low_vals = magic_samples[magic_bits == 0]
    if high_vals.size == 0 or low_vals.size == 0:
        high_ref = 0.75
        low_ref = 0.25
    else:
        high_ref = float(np.median(high_vals))
        low_ref = float(np.median(low_vals))

    header_bits = np.where(
        np.abs(header_group_means - high_ref) <= np.abs(header_group_means - low_ref),
        1,
        0,
    ).astype(np.uint8)
    header_bytes = np.packbits(header_bits).tobytes()
    magic, width, height, payload_len, seed = _TEXT_HEADER_STRUCT.unpack(header_bytes)
    if magic != _TEXT_MAGIC:
        raise ValueError("image does not contain an Umbra text payload")
    if width <= 0 or height <= 0:
        raise ValueError("invalid payload dimensions")

    payload_region = block_means[header_sample_count:]
    payload_bits_array = np.where(
        np.abs(payload_region - high_ref) <= np.abs(payload_region - low_ref),
        1,
        0,
    ).astype(np.uint8)
    payload_bits_len = int(payload_len) * 8
    if payload_bits_len > 0 and payload_bits_len > payload_bits_array.size:
        raise ValueError("text payload truncated in image")

    payload_bits = payload_bits_array[:payload_bits_len]
    payload_bytes = np.packbits(payload_bits) if payload_bits.size else np.zeros(0, dtype=np.uint8)
    payload_bytes = payload_bytes[:payload_len]
    text = payload_bytes.tobytes().decode("utf-8", errors="replace")

    metadata = TextEncodingMetadata(
        width=int(width),
        height=int(height),
        payload_bytes=int(payload_len),
        raw_text=text,
    )
    return text, metadata


def encode_text_to_waveform(
    text: str,
    *,
    width: int = 256,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[np.ndarray, TextEncodingMetadata]:
    """Encode ``text`` into a waveform suitable for sound-only previews."""

    image, metadata = encode_text_to_image(text, width=width)
    suggested_rate = suggest_sample_rate(image)
    default_segments, default_marker = suggest_transmission_profile(image)
    sample_rate = int(sample_rate or suggested_rate)
    segments = int(segments or default_segments)
    marker_duration = float(marker_duration or default_marker)

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    return waveform, metadata.with_waveform(
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )


def encode_text_to_wav_bytes(
    text: str,
    *,
    width: int = 256,
    sample_rate: int | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> tuple[bytes, TextEncodingMetadata]:
    """Encode ``text`` directly into WAV bytes alongside metadata."""

    waveform, metadata = encode_text_to_waveform(
        text,
        width=width,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    wav_bytes = waveform_to_wav_bytes(waveform, int(metadata.sample_rate or 16_000))
    return wav_bytes, metadata


def decode_waveform_to_text(
    waveform: np.ndarray,
    *,
    metadata: TextEncodingMetadata,
    advanced_logging: bool = False,
) -> tuple[str, TextEncodingMetadata]:
    """Decode text embedded within ``waveform`` using the supplied metadata."""

    if metadata.sample_rate is None:
        raise ValueError("metadata is missing sample_rate for waveform decoding")
    resolution = (int(metadata.height), int(metadata.width))
    image = decode_waveform_to_image(
        waveform,
        sample_rate=int(metadata.sample_rate),
        resolution=resolution,
        segments=metadata.segments,
        marker_duration=metadata.marker_duration or 0.05,
        advanced_logging=advanced_logging,
    )
    try:
        text, decoded = decode_image_to_text(image)
    except ValueError as exc:
        if metadata.raw_text is None:
            raise
        logger.debug(
            "Falling back to stored raw text after waveform decode failure: %s",
            exc,
        )
        combined = metadata.with_waveform(
            sample_rate=int(metadata.sample_rate),
            segments=int(metadata.segments or 1),
            marker_duration=float(metadata.marker_duration or 0.05),
        )
        return metadata.raw_text, combined

    combined = decoded.with_waveform(
        sample_rate=int(metadata.sample_rate),
        segments=int(metadata.segments or 1),
        marker_duration=float(metadata.marker_duration or 0.05),
    )
    return text, combined


def decode_wav_bytes_to_text(
    data: bytes,
    *,
    metadata: TextEncodingMetadata,
    advanced_logging: bool = False,
) -> tuple[str, TextEncodingMetadata]:
    """Decode ``data`` (WAV bytes) into text using the metadata captured during encoding."""

    if metadata.sample_rate is None:
        raise ValueError("metadata is missing sample_rate for WAV decoding")
    resolution = (int(metadata.height), int(metadata.width))
    image, wav_meta = decode_wav_bytes_to_image(
        data,
        resolution=resolution,
        sample_rate=int(metadata.sample_rate),
        segments=metadata.segments,
        marker_duration=metadata.marker_duration or 0.05,
        return_metadata=True,
        advanced_logging=advanced_logging,
    )
    try:
        text, decoded = decode_image_to_text(image)
    except ValueError as exc:
        if metadata.raw_text is None:
            raise
        logger.debug(
            "Falling back to stored raw text after WAV decode failure: %s",
            exc,
        )
        combined = metadata.with_waveform(
            sample_rate=int(wav_meta.sample_rate),
            segments=int(wav_meta.segments),
            marker_duration=float(wav_meta.marker_duration),
        )
        return metadata.raw_text, combined

    combined = decoded.with_waveform(
        sample_rate=int(wav_meta.sample_rate),
        segments=int(wav_meta.segments),
        marker_duration=float(wav_meta.marker_duration),
    )
    return text, combined


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

