import numpy as np
import pytest

from umbra.codec import (
    _ensure_rgb_image,
    decode_wav_bytes_to_image,
    encode_image_to_waveform,
    encode_image_to_wav_bytes,
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


def test_ensure_rgb_accepts_single_channel_dimension() -> None:
    single_channel = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3, 1)
    rgb = _ensure_rgb_image(single_channel)
    assert rgb.shape == (3, 3, 3)
    assert rgb.dtype == np.float32
    assert np.all((0.0 <= rgb) & (rgb <= 1.0))


def test_decode_requires_bytes() -> None:
    with pytest.raises(TypeError):
        decode_wav_bytes_to_image("not-bytes", resolution=(4, 4))  # type: ignore[arg-type]

