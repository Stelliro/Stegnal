"""Tests for the noise reconstruction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from umbra import reconstruction
from umbra.reconstruction import (
    create_variations,
    generate_shape_collage,
    image_to_waveform,
    reconstruct_from_waveform,
    run_reconstruction_cycle,
    waveform_to_wav_bytes,
)


def test_generate_shape_collage_produces_multiple_shapes() -> None:
    collage, shapes = generate_shape_collage(1234, resolution=(96, 128))
    assert collage.shape == (96, 128, 3)
    assert np.all(collage >= 0.0) and np.all(collage <= 1.0)
    assert 3 <= len(shapes) <= 15


def test_create_variations_preserves_bounds() -> None:
    base, _ = generate_shape_collage(42, resolution=(64, 64), shape_count=7)
    rng = np.random.default_rng(1)
    variations = create_variations(
        base,
        variation_count=5,
        noise_sigma=0.4,
        dropout_probability=0.25,
        rng=rng,
    )
    assert variations.shape == (5, 64, 64, 3)
    assert np.all(variations >= 0.0) and np.all(variations <= 1.0)


def test_run_reconstruction_cycle_returns_audio_and_hybrid() -> None:
    result = run_reconstruction_cycle(
        77,
        resolution=(48, 48),
        variation_count=4,
        noise_sigma=0.25,
        dropout_probability=0.3,
        sample_rate=24_000,
    )
    assert result.base_image.shape == (48, 48, 3)
    assert result.ensemble_prediction.shape == (48, 48, 3)
    assert result.hybrid_prediction.shape == (48, 48, 3)
    assert result.coverage.shape == (48, 48)
    assert result.variations.shape[0] == 4
    assert np.all(result.coverage >= 0.0) and np.all(result.coverage <= 1.0)

    wav_bytes = waveform_to_wav_bytes(result.waveform, result.sample_rate)
    assert wav_bytes[:4] == b"RIFF"


def test_waveform_segment_detection() -> None:
    collage, _ = generate_shape_collage(99, resolution=(40, 36))
    waveform = image_to_waveform(
        collage,
        sample_rate=16_000,
        segments=6,
        marker_duration=0.02,
    )

    recovered, detected = reconstruct_from_waveform(
        waveform,
        resolution=collage.shape[:2],
        sample_rate=16_000,
        segments=None,
        marker_duration=0.02,
        return_segments=True,
    )

    assert detected == 6
    assert recovered.shape == collage.shape


def test_segment_image_rows_respects_contrast() -> None:
    bright = np.ones((160, 64, 3), dtype=np.float32)
    dark = np.zeros((32, 64, 3), dtype=np.float32)
    gradient = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :, None]
    transition = np.repeat(gradient, 40, axis=0)
    transition = np.repeat(transition, 3, axis=2)
    image = np.vstack([bright, transition, dark])

    segments = reconstruction.segment_image_rows(image, 5)
    assert len(segments) == 5
    heights = [end - start for start, end, _ in segments]
    assert sum(heights) == image.shape[0]
    assert max(heights) != min(heights)


def test_reconstruct_from_waveform_supports_advanced_logging() -> None:
    collage, _ = generate_shape_collage(101, resolution=(32, 24))
    waveform = image_to_waveform(collage, sample_rate=12_000, segments=2)

    recovered = reconstruct_from_waveform(
        waveform,
        resolution=collage.shape[:2],
        sample_rate=12_000,
        segments=2,
        advanced_logging=True,
    )

    assert recovered.shape == collage.shape


def test_generation_round_trip_with_mach_data() -> None:
    rows, cols = 48, 60
    gradient_y = np.linspace(0.05, 0.95, rows, dtype=np.float32)
    gradient_x = np.linspace(0.1, 0.9, cols, dtype=np.float32)
    mach_base = gradient_y[:, None]
    horizontal = np.repeat(mach_base, cols, axis=1)
    vertical = np.repeat(gradient_x[None, :], rows, axis=0)
    mach_pattern = np.stack(
        (
            horizontal,
            1.0 - horizontal * 0.8,
            np.clip(vertical ** 0.5, 0.0, 1.0),
        ),
        axis=-1,
    )

    waveform = image_to_waveform(
        mach_pattern,
        sample_rate=18_000,
        segments=3,
        marker_duration=0.025,
    )

    recovered, detected = reconstruct_from_waveform(
        waveform,
        resolution=mach_pattern.shape[:2],
        sample_rate=18_000,
        segments=3,
        marker_duration=0.025,
        advanced_logging=True,
        return_segments=True,
    )

    assert detected == 3
    assert recovered.shape == mach_pattern.shape
    difference = np.mean(np.abs(recovered - mach_pattern))
    assert difference < 0.26


def test_waveform_encoding_falls_back_to_numpy(monkeypatch) -> None:
    stripe = np.ones((4, 4, 3), dtype=np.float32)
    original_cp = reconstruction.cp

    class BrokenCp:
        float32 = np.float32
        ndarray = np.ndarray

        @staticmethod
        def asarray(array: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
            raise RuntimeError("GPU backend unavailable")

        @staticmethod
        def asnumpy(array: np.ndarray) -> np.ndarray:
            return np.asarray(array, dtype=np.float32)

    monkeypatch.setattr(reconstruction, "cp", BrokenCp)
    try:
        waveform = reconstruction._encode_stripe_waveform(
            stripe,
            sample_count=16,
            allow_cpu_fallback=True,
        )
    finally:
        monkeypatch.setattr(reconstruction, "cp", original_cp)

    assert waveform.shape == (16,)
    assert np.isfinite(waveform).all()


def test_fft_magnitude_prefers_gpu(monkeypatch) -> None:
    class TrackingCp:
        float32 = np.float32
        ndarray = np.ndarray
        used = False

        class fft:
            @staticmethod
            def rfft(array: np.ndarray, n: int) -> np.ndarray:
                TrackingCp.used = True
                return np.fft.rfft(np.asarray(array, dtype=np.float32), n=n)

        @staticmethod
        def asarray(array: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
            TrackingCp.used = True
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def abs(array: np.ndarray) -> np.ndarray:
            TrackingCp.used = True
            return np.abs(array)

        @staticmethod
        def asnumpy(array: np.ndarray) -> np.ndarray:
            TrackingCp.used = True
            return np.asarray(array, dtype=np.float32)

    monkeypatch.setattr(reconstruction, "cp", TrackingCp)
    monkeypatch.setattr(reconstruction, "_GPU_MIN_FFT_SAMPLES", 1)

    result = reconstruction._fft_magnitude(
        np.ones(8, dtype=np.float32),
        8,
        advanced_logging=False,
        allow_cpu_fallback=True,
    )

    assert TrackingCp.used is True
    assert result.shape == (5,)


def test_fft_magnitude_falls_back_to_numpy(monkeypatch) -> None:
    class BrokenCp:
        float32 = np.float32
        ndarray = np.ndarray

        @staticmethod
        def asarray(array: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
            raise RuntimeError("GPU backend unavailable")

        @staticmethod
        def asnumpy(array: np.ndarray) -> np.ndarray:
            return np.asarray(array, dtype=np.float32)

    monkeypatch.setattr(reconstruction, "cp", BrokenCp)
    monkeypatch.setattr(reconstruction, "_GPU_MIN_FFT_SAMPLES", 1)

    result = reconstruction._fft_magnitude(
        np.ones(8, dtype=np.float32),
        8,
        advanced_logging=False,
        allow_cpu_fallback=True,
    )

    assert result.shape == (5,)


def test_fft_magnitude_requires_gpu_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(reconstruction, "cp", None)
    monkeypatch.setattr(reconstruction, "_GPU_MIN_FFT_SAMPLES", 1)

    with pytest.raises(reconstruction.GPUAccelerationRequiredError):
        reconstruction._fft_magnitude(
            np.ones(4, dtype=np.float32),
            4,
            advanced_logging=False,
            allow_cpu_fallback=False,
        )


def test_waveform_encoding_requires_gpu_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(reconstruction, "cp", None)

    with pytest.raises(reconstruction.GPUAccelerationRequiredError):
        reconstruction._encode_stripe_waveform(
            np.ones((2, 2, 3), dtype=np.float32),
            sample_count=4,
            allow_cpu_fallback=False,
        )


def test_as_backend_hybrid_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    xp = reconstruction.cp
    if xp is None:
        pytest.skip("CuPy unavailable")

    calls = {"asarray": 0}

    def _failing_asarray(array, dtype=None):  # type: ignore[override]
        calls["asarray"] += 1
        if calls["asarray"] == 1:
            raise xp.cuda.memory.OutOfMemoryError("OOM")
        return np.asarray(array, dtype=dtype)

    monkeypatch.setattr(xp, "asarray", _failing_asarray, raising=False)

    backend = reconstruction._as_backend(np.ones(8, dtype=np.float32), xp)
    assert backend.shape == (8,)

