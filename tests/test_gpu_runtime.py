"""Unit tests for the GPU runtime helpers."""

from __future__ import annotations

import importlib.util as importlib_util
import sys
import types
from pathlib import Path

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

    # Simulate a truly CuPy-free environment: no importable module AND no
    # installed distribution (otherwise the helper correctly echoes whatever
    # cupy-cudaXXx wheel happens to be installed in the current environment).
    fake_metadata = types.SimpleNamespace(packages_distributions=lambda: {})
    monkeypatch.setattr(gpu_runtime, "importlib_metadata", fake_metadata, raising=False)
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


def test_detected_runtime_reports_mismatch(monkeypatch, tmp_path):
    """Detected NVRTC versions should be exposed for debugging mismatches."""

    fake_dir = Path(tmp_path)
    if sys.platform == "win32":
        library = fake_dir / "nvrtc64_130_0.dll"
    elif sys.platform == "darwin":  # pragma: no cover - macOS not in CI
        library = fake_dir / "libnvrtc.13.0.dylib"
    else:
        library = fake_dir / "libnvrtc.so.13.0"

    library.write_bytes(b"")

    monkeypatch.setattr(
        gpu_runtime,
        "_iter_candidate_directories",
        lambda: iter([fake_dir]),
        raising=False,
    )
    monkeypatch.setattr(gpu_runtime, "_NVRTC_REQUIRED_VERSION", (11, 8), raising=False)

    located = gpu_runtime._find_nvrtc_library((11, 8))
    assert located == library

    description = gpu_runtime.describe_detected_cuda_runtime()
    if sys.platform == "win32":  # pragma: no cover - platform-specific expectation
        expected = "CUDA Toolkit 13.x (NVRTC nvrtc64_130_0.dll)"
    else:
        expected = "CUDA Toolkit 13.x (NVRTC libnvrtc.so.13.0)"
    assert description == expected

    assert gpu_runtime.nvrtc_version_matches_requirement() is False


def test_iter_candidate_directories_respects_hint(monkeypatch, tmp_path):
    """Custom hint environment variables should be considered during discovery."""

    hint_file = tmp_path / "nvrtc64_118_0.dll"
    hint_file.write_bytes(b"")

    monkeypatch.setenv("UMBRA_NVRTC_PATH_HINTS", str(hint_file))
    for env_var in ("CUPY_CUDA_PATH", "CUDA_PATH", "CUDA_HOME", "PATH"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(importlib_util, "find_spec", lambda name: None, raising=False)

    directories = list(gpu_runtime._iter_candidate_directories())
    assert hint_file.parent in directories
