"""Test fixtures for Project Umbra."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from skimage import filters

import umbra.reconstruction as reconstruction


class _CuPyStub:
    """Minimal CuPy stand-in that delegates operations to NumPy."""

    _umbra_skip_nvrtc_check = True
    float32 = np.float32
    ndarray = np.ndarray

    @staticmethod
    def zeros(shape, dtype=np.float32):  # type: ignore[override]
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=np.float32):  # type: ignore[override]
        return np.empty(shape, dtype=dtype)

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

    @staticmethod
    def stack(arrays, axis=0):  # type: ignore[override]
        return np.stack(arrays, axis=axis)

    class random:
        @staticmethod
        def default_rng(seed: int | None = None):
            rng = np.random.default_rng(seed)

            class _Generator:
                def normal(
                    self,
                    loc: float = 0.0,
                    scale: float = 1.0,
                    size=None,
                    dtype=np.float32,
                ):
                    return np.asarray(rng.normal(loc, scale, size), dtype=dtype)

                def standard_normal(
                    self,
                    size=None,
                    dtype=np.float32,
                ):
                    return np.asarray(rng.standard_normal(size), dtype=dtype)

                def permutation(self, n: int):
                    return np.asarray(rng.permutation(n))

            return _Generator()

    class cuda:
        class memory:
            class OutOfMemoryError(MemoryError):
                pass

            class UnownedMemory:
                def __init__(self, ptr: int, size: int, owner: object) -> None:
                    self.ptr = ptr
                    self.size = size
                    self.owner = owner

            class MemoryPointer:
                def __init__(self, mem: _CuPyStub.cuda.memory.UnownedMemory, offset: int) -> None:
                    self.mem = mem
                    self.ptr = mem.ptr + offset

        @staticmethod
        def alloc_pinned_memory(size: int):  # type: ignore[override]
            buffer = np.zeros(int(size), dtype=np.uint8)

            class _Pinned:
                def __init__(self, data: np.ndarray) -> None:
                    self._data = data
                    self.ptr = int(data.ctypes.data)

            return _Pinned(buffer)

        class runtime:
            @staticmethod
            def runtimeGetVersion() -> int:
                return 11040

            @staticmethod
            def driverGetVersion() -> int:
                return 11040

            @staticmethod
            def getDeviceCount() -> int:
                return 1

            @staticmethod
            def getDeviceProperties(index: int):
                return {
                    "name": "Stub GPU",
                    "totalGlobalMem": 8 * 1024 * 1024 * 1024,
                }

        class cudnn:
            @staticmethod
            def getVersion() -> int:
                return 8900


@pytest.fixture(autouse=True)
def _install_cupy_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GPU-requiring code paths see a CuPy-compatible backend during tests."""

    import umbra.gpu_runtime as gpu_runtime

    monkeypatch.setattr(gpu_runtime, "cp", _CuPyStub, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_CHECKED", True, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_ERROR", None, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_PATH_CACHED", True, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_DETECTED_VERSION", None, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_DETECTED_LIBRARY", None, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_VERSION_MATCHED", False, raising=False)

    monkeypatch.setattr(reconstruction, "cp", _CuPyStub, raising=False)

    import umbra.encoding as encoding

    monkeypatch.setattr(encoding, "cp", _CuPyStub, raising=False)

    cupyx_module = types.ModuleType("cupyx")
    scipy_module = types.ModuleType("cupyx.scipy")
    ndimage_module = types.ModuleType("cupyx.scipy.ndimage")

    def _gaussian_filter_stub(array, sigma, mode="reflect"):
        channel_axis = -1 if array.ndim == 3 and array.shape[2] in (1, 3, 4) else None
        filtered = filters.gaussian(
            np.asarray(array, dtype=np.float32),
            sigma=sigma,
            preserve_range=True,
            channel_axis=channel_axis,
        )
        return np.asarray(filtered, dtype=np.float32)

    ndimage_module.gaussian_filter = _gaussian_filter_stub  # type: ignore[attr-defined]
    cupyx_module.scipy = types.SimpleNamespace(ndimage=ndimage_module)
    scipy_module.ndimage = ndimage_module

    monkeypatch.setitem(sys.modules, "cupyx", cupyx_module)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", ndimage_module)
