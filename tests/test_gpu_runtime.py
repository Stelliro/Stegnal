"""Unit tests for the GPU runtime helpers."""

from __future__ import annotations

import sys
import types

import umbra.gpu_runtime as gpu_runtime


def test_recommend_command_uses_detected_distribution(monkeypatch):
    """The helper should surface the detected CuPy wheel in its recommendation."""

    fake_metadata = types.SimpleNamespace(
        packages_distributions=lambda: {"cupy": ["cupy-cuda13x"]}
    )
    monkeypatch.setattr(gpu_runtime, "importlib_metadata", fake_metadata, raising=False)
    monkeypatch.setattr(
        gpu_runtime,
        "cp",
        types.SimpleNamespace(__name__="cupy"),
        raising=False,
    )

    assert gpu_runtime.recommend_cupy_install_command() == 'pip install -U "cupy-cuda13x"'


def test_recommend_command_without_cupy(monkeypatch):
    """Missing CuPy should fall back to a generic CUDA 12 wheel suggestion."""

    monkeypatch.setattr(gpu_runtime, "cp", None, raising=False)
    assert gpu_runtime.recommend_cupy_install_command() == 'pip install -U "cupy-cuda12x"'


def test_describe_required_runtime_uses_cuda_version(monkeypatch):
    """Runtime version detection should surface a precise NVRTC requirement."""

    runtime = types.SimpleNamespace(runtimeGetVersion=lambda: 11020)
    fake_cp = types.SimpleNamespace(__name__="cupy", cuda=types.SimpleNamespace(runtime=runtime))

    monkeypatch.setattr(gpu_runtime, "cp", fake_cp, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_REQUIRED_VERSION", None, raising=False)

    expected = "CUDA Toolkit 11.2 (NVRTC libnvrtc.so.11.2)"
    if sys.platform == "win32":  # pragma: no cover - platform-specific expectation
        expected = "CUDA Toolkit 11.2 (NVRTC nvrtc64_112_0.dll)"

    assert gpu_runtime.describe_required_cuda_runtime() == expected


def test_describe_required_runtime_falls_back_to_distribution(monkeypatch):
    """When runtime detection fails, fall back to the CuPy wheel metadata."""

    class RaisingRuntime:
        @staticmethod
        def runtimeGetVersion():  # pragma: no cover - executed via tests
            raise RuntimeError("missing CUDA runtime")

    fake_cuda = types.SimpleNamespace(runtime=RaisingRuntime())
    fake_cp = types.SimpleNamespace(__name__="cupy", cuda=fake_cuda)
    fake_metadata = types.SimpleNamespace(
        packages_distributions=lambda: {"cupy": ["cupy-cuda11x"]}
    )

    monkeypatch.setattr(gpu_runtime, "cp", fake_cp, raising=False)
    monkeypatch.setattr(gpu_runtime, "importlib_metadata", fake_metadata, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_REQUIRED_VERSION", None, raising=False)

    expected = "CUDA Toolkit 11.x (NVRTC libnvrtc.so.11)"
    if sys.platform == "win32":  # pragma: no cover - platform-specific expectation
        expected = "CUDA Toolkit 11.x (NVRTC nvrtc64_11*_0.dll)"

    assert gpu_runtime.describe_required_cuda_runtime() == expected
