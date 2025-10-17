"""Unit tests for the GPU runtime helpers."""

from __future__ import annotations

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
