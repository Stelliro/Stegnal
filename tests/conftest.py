"""Test fixtures for Project Umbra."""

from __future__ import annotations

import numpy as np
import pytest

import umbra.reconstruction as reconstruction


class _CuPyStub:
    """Minimal CuPy stand-in that delegates operations to NumPy."""

    _umbra_skip_nvrtc_check = True
    float32 = np.float32
    ndarray = np.ndarray

    class fft:
        @staticmethod
        def rfft(array: np.ndarray, n: int) -> np.ndarray:
            return np.fft.rfft(np.asarray(array, dtype=np.float32), n=n)

        @staticmethod
        def irfft(array: np.ndarray, n: int) -> np.ndarray:
            return np.fft.irfft(np.asarray(array, dtype=np.float32), n=n)

    @staticmethod
    def asarray(array: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def asnumpy(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def abs(array: np.ndarray) -> np.ndarray:
        return np.abs(array)

    @staticmethod
    def max(array: np.ndarray) -> float:
        return float(np.max(array))

    @staticmethod
    def zeros_like(array: np.ndarray, dtype=np.float32):  # type: ignore[override]
        return np.zeros_like(array, dtype=dtype)

    @staticmethod
    def pad(array: np.ndarray, pad_width, mode: str = "constant") -> np.ndarray:  # type: ignore[override]
        return np.pad(array, pad_width, mode=mode)

    @staticmethod
    def linspace(start, stop, num, dtype=np.float32):  # type: ignore[override]
        return np.linspace(start, stop, num, dtype=dtype)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=np.float32):  # type: ignore[override]
        return np.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def interp(x, xp, fp):  # type: ignore[override]
        return np.interp(x, xp, fp)


@pytest.fixture(autouse=True)
def _install_cupy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GPU-requiring code paths see a CuPy-compatible backend during tests."""

    import umbra.gpu_runtime as gpu_runtime

    monkeypatch.setattr(gpu_runtime, "cp", _CuPyStub, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_CHECKED", True, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_ERROR", None, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_PATH_CACHED", True, raising=False)

    monkeypatch.setattr(reconstruction, "cp", _CuPyStub, raising=False)

    import umbra.encoding as encoding

    monkeypatch.setattr(encoding, "cp", _CuPyStub, raising=False)
