import numpy as np
import pytest

import umbra.codec as codec_module
from umbra.codec import (
    DecodedWavMetadata,
    _ensure_rgb_image,
    decode_image_to_text,
    decode_wav_bytes_to_image,
    decode_waveform_to_image,
    encode_image_to_wav_bytes,
    encode_image_to_waveform,
    encode_text_to_image,
    encode_text_to_wav_bytes,
    encode_text_to_waveform,
)
from umbra.metrics import compute_metrics


def _random_image(size: int = 8) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.random((size, size, 3), dtype=np.float32)


def test_image_wav_round_trip_recovers_shape() -> None:
    image = _random_image()
    sample_rate = 4096

    waveform = encode_image_to_waveform(image, sample_rate=sample_rate)
    assert waveform.ndim == 1
    assert waveform.size == sample_rate

    wav_bytes = encode_image_to_wav_bytes(image, sample_rate=sample_rate)
    decoded, detected_rate = decode_wav_bytes_to_image(
        wav_bytes, resolution=image.shape[:2]
    )

    assert detected_rate == sample_rate
    assert decoded.shape == image.shape

    metrics = compute_metrics(image, decoded)
    assert metrics.psnr > 0.0
    assert -1.0 <= metrics.ssim <= 1.0


def test_encode_image_accepts_grayscale() -> None:
    gray = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    waveform = encode_image_to_waveform(gray, sample_rate=1024)
    assert waveform.size == 1024


def test_segmented_waveform_round_trip() -> None:
    image = _random_image(size=12)
    sample_rate = 2048
    segments = 3
    marker = 0.02

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker,
    )
    assert waveform.ndim == 1
    marker_samples = int(round(marker * sample_rate))
    expected_segment = sample_rate + marker_samples
    assert waveform.size == expected_segment * segments

    decoded = decode_wav_bytes_to_image(
        encode_image_to_wav_bytes(
            image,
            sample_rate=sample_rate,
            segments=segments,
            marker_duration=marker,
        ),
        resolution=image.shape[:2],
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker,
    )[0]

    metrics = compute_metrics(image, decoded)
    assert metrics.psnr > 0.0
    assert -1.0 <= metrics.ssim <= 1.0


def test_waveform_duration_is_capped() -> None:
    image = _random_image(size=6)
    sample_rate = 8_000
    absurd_segments = 500_000

    waveform = encode_image_to_waveform(
        image,
        sample_rate=sample_rate,
        segments=absurd_segments,
        marker_duration=0.5,
    )

    assert waveform.ndim == 1
    # Ten minute cap from reconstruction module
    max_samples = int(600.0 * sample_rate)
    assert 0 < waveform.size <= max_samples


def test_ensure_rgb_accepts_single_channel_dimension() -> None:
    single_channel = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3, 1)
    rgb = _ensure_rgb_image(single_channel)
    assert rgb.shape == (3, 3, 3)
    assert rgb.dtype == np.float32
    assert np.all((0.0 <= rgb) & (rgb <= 1.0))


def test_decode_requires_bytes() -> None:
    with pytest.raises(TypeError):
        decode_wav_bytes_to_image("not-bytes", resolution=(4, 4))  # type: ignore[arg-type]


def test_decode_waveform_metadata_inference() -> None:
    image = _random_image(size=10)
    sample_rate = 8192
    segments = 4
    marker = 0.03

    wav_bytes = encode_image_to_wav_bytes(
        image,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker,
    )

    decoded, metadata = decode_wav_bytes_to_image(
        wav_bytes,
        resolution=image.shape[:2],
        return_metadata=True,
        marker_duration=marker,
        segments=None,
    )

    assert isinstance(metadata, DecodedWavMetadata)
    assert metadata.sample_rate == sample_rate
    assert metadata.segments == segments
    assert pytest.approx(metadata.marker_duration, rel=1e-6) == marker

    metrics = compute_metrics(image, decoded)
    assert metrics.psnr > 0.0


def test_decode_wav_bytes_supports_advanced_logging() -> None:
    image = _random_image(size=6)
    sample_rate = 4096

    wav_bytes = encode_image_to_wav_bytes(image, sample_rate=sample_rate)
    decoded, metadata = decode_wav_bytes_to_image(
        wav_bytes,
        resolution=image.shape[:2],
        sample_rate=sample_rate,
        return_metadata=True,
        advanced_logging=True,
    )

    assert decoded.shape == image.shape
    assert metadata.sample_rate == sample_rate


def test_decode_waveform_to_image_returns_preview_on_failure() -> None:
    waveform = np.zeros(0, dtype=np.float32)

    decoded = decode_waveform_to_image(
        waveform,
        sample_rate=1024,
        resolution=(4, 4),
    )

    assert decoded.shape == (4, 4, 3)
    assert decoded.dtype == np.float32
    assert decoded.min() >= 0.0
    assert decoded.max() <= 1.0
    assert not np.allclose(decoded, decoded[0, 0, 0])


def test_decode_waveform_to_image_attempts_alternative_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int | None] = []

    def fake_reconstruct(
        waveform: np.ndarray,
        *,
        resolution: tuple[int, int],
        sample_rate: int,
        segments: int | None,
        marker_duration: float,
        advanced_logging: bool,
        return_segments: bool,
        allow_cpu_fallback: bool,
    ) -> tuple[np.ndarray, int]:
        calls.append(segments)
        rows, cols = resolution
        if segments == 2:
            raise ValueError("requested segments invalid")
        image = np.full((rows, cols, 3), 0.25, dtype=np.float32)
        detected = 5 if segments is None else int(segments)
        return image, detected

    monkeypatch.setattr(codec_module, "reconstruct_from_waveform", fake_reconstruct)

    waveform = np.ones(16, dtype=np.float32)
    decoded = decode_waveform_to_image(
        waveform,
        sample_rate=16,
        resolution=(2, 2),
        segments=2,
        marker_duration=0.01,
        advanced_logging=True,
    )

    assert calls == [2, None]
    assert decoded.shape == (2, 2, 3)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, 0.25)


def test_decode_wav_bytes_to_image_returns_preview_on_failure() -> None:
    bogus = b"not a wav"

    decoded, detected_rate = decode_wav_bytes_to_image(
        bogus,
        resolution=(4, 4),
        sample_rate=22_050,
    )

    assert detected_rate == 22_050
    assert decoded.shape == (4, 4, 3)
    assert decoded.dtype == np.float32
    assert decoded.min() >= 0.0
    assert decoded.max() <= 1.0
    assert not np.allclose(decoded, decoded[0, 0, 0])


def test_decode_wav_bytes_to_image_attempts_segment_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = np.ones(32, dtype=np.float32)

    def fake_loader(data: bytes) -> tuple[np.ndarray, int]:
        return waveform, 22_050

    calls: list[int | None] = []

    def fake_reconstruct(
        waveform: np.ndarray,
        *,
        resolution: tuple[int, int],
        sample_rate: int,
        segments: int | None,
        marker_duration: float,
        advanced_logging: bool,
        return_segments: bool,
        allow_cpu_fallback: bool,
    ) -> tuple[np.ndarray, int]:
        calls.append(segments)
        rows, cols = resolution
        if segments == 3:
            raise RuntimeError("primary attempt failed")
        image = np.full((rows, cols, 3), 0.5, dtype=np.float32)
        detected = 7 if segments is None else int(segments)
        return image, detected

    monkeypatch.setattr(codec_module, "load_waveform_from_wav", fake_loader)
    monkeypatch.setattr(codec_module, "reconstruct_from_waveform", fake_reconstruct)

    decoded, metadata = decode_wav_bytes_to_image(
        b"fake wav",
        resolution=(3, 3),
        segments=3,
        return_metadata=True,
    )

    assert calls == [3, None]
    assert decoded.shape == (3, 3, 3)
    assert np.allclose(decoded, 0.5)
    assert metadata.sample_rate == 22_050
    assert metadata.segments == 7


def test_decode_wav_bytes_to_image_metadata_on_failure() -> None:
    bogus = b"still not a wav"
    marker = 0.12

    decoded, metadata = decode_wav_bytes_to_image(
        bogus,
        resolution=(2, 2),
        segments=None,
        return_metadata=True,
        marker_duration=marker,
    )

    assert decoded.shape == (2, 2, 3)
    assert decoded.dtype == np.float32
    assert decoded.min() >= 0.0
    assert decoded.max() <= 1.0
    assert not np.allclose(decoded, decoded[0, 0, 0])
    assert metadata.sample_rate == 16_000
    assert metadata.segments == 1
    assert metadata.marker_duration == pytest.approx(marker, rel=1e-6)


def test_decode_waveform_to_image_returns_preview_when_reconstruction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(16, dtype=np.float32)

    def fake_reconstruct(
        waveform: np.ndarray,
        *,
        resolution: tuple[int, int],
        sample_rate: int,
        segments: int | None,
        marker_duration: float,
        advanced_logging: bool,
        return_segments: bool,
    ) -> tuple[np.ndarray, int]:
        raise RuntimeError("boom")

    monkeypatch.setattr(codec_module, "_reconstruct_with_strategies", fake_reconstruct)

    preview = decode_waveform_to_image(
        waveform,
        sample_rate=8,
        resolution=(4, 4),
        advanced_logging=True,
    )

    assert preview.shape == (4, 4, 3)
    assert preview.dtype == np.float32
    assert preview.min() >= 0.0
    assert preview.max() <= 1.0
    assert not np.allclose(preview, preview[0, 0, 0])


def test_text_image_round_trip() -> None:
    payload = "The quick brown fox jumps over the lazy dog." * 5
    image, metadata = encode_text_to_image(payload, width=64)

    assert metadata.payload_bytes == len(payload.encode("utf-8"))
    assert image.dtype == np.float32
    assert image.shape[2] == 3

    decoded, decoded_meta = decode_image_to_text(image)
    assert decoded.rstrip("\0") == payload
    assert decoded_meta.payload_bytes == metadata.payload_bytes


def test_text_waveform_round_trip() -> None:
    payload = "Bee movie script line " * 200
    waveform, metadata = encode_text_to_waveform(payload, width=128)

    assert waveform.ndim == 1
    assert metadata.sample_rate is not None
    assert metadata.segments is not None

    text, decoded_meta = codec_module.decode_waveform_to_text(
        waveform,
        metadata=metadata,
    )
    assert text.rstrip("\0") == payload
    assert decoded_meta.payload_bytes == metadata.payload_bytes


def test_text_wav_bytes_round_trip() -> None:
    payload = "Signal data" * 100
    wav_bytes, metadata = encode_text_to_wav_bytes(payload, width=96)

    text, decoded_meta = codec_module.decode_wav_bytes_to_text(wav_bytes, metadata=metadata)
    assert text.rstrip("\0") == payload
    assert decoded_meta.sample_rate == metadata.sample_rate
