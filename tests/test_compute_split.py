"""Tests for CPU/GPU split processing: GPU decode path + hybrid evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from umbra import decoding
from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import (
    EvolutionManager,
    _env_compute_mode,
    _env_gpu_fraction,
)

HAS_CUPY = decoding.cp is not None
gpu_only = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU not available")


def _packet():
    rng = np.random.default_rng(7)
    img = rng.random((64, 64, 3), dtype=np.float64).astype(np.float32)
    return NoiseStreamEncoder(sigma=0.1).encode(img, seed=11)


# --- GPU decode path ------------------------------------------------------

@gpu_only
def test_gpu_decode_matches_cpu_without_denoise():
    """With denoise off the GPU path is a pure gather -> must be identical."""
    packet = _packet()
    dec = NoiseStreamDecoder(denoise_sigma=0.0)
    cpu = dec.decode(packet, seed=11, use_gpu=False)
    gpu = dec.decode(packet, seed=11, use_gpu=True)
    assert np.array_equal(cpu, gpu)


@gpu_only
def test_gpu_decode_close_with_denoise():
    """With denoise on, GPU vs CPU differ only by the gaussian implementation."""
    packet = _packet()
    dec = NoiseStreamDecoder(denoise_sigma=0.9)
    cpu = dec.decode(packet, seed=11, use_gpu=False)
    gpu = dec.decode(packet, seed=11, use_gpu=True)
    assert gpu.shape == cpu.shape
    assert float(np.abs(cpu - gpu).mean()) < 1e-2


def test_decode_use_gpu_falls_back_when_cupy_missing(monkeypatch):
    """use_gpu=True must transparently fall back to CPU when CuPy is absent."""
    monkeypatch.setattr(decoding, "cp", None, raising=False)
    packet = _packet()
    dec = NoiseStreamDecoder(denoise_sigma=0.5)
    out = dec.decode(packet, seed=11, use_gpu=True)  # should not raise
    assert out.shape == packet.image_shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


# --- hybrid / split evaluation -------------------------------------------

@pytest.mark.parametrize("mode", ["cpu", "gpu", "hybrid"])
def test_run_generation_all_modes_build_full_population(mode):
    """Every compute mode produces a complete generation (GPU falls back if absent)."""
    ref = np.random.default_rng(0).random((48, 48, 3)).astype(np.float32)
    mgr = EvolutionManager(
        original=ref,
        encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(denoise_sigma=0.8),
        population_size=10,
        base_seed=123,
        enable_waveform=False,
        compute_mode=mode,
        gpu_fraction=0.5,
    )
    rec = mgr.run_generation()
    assert len(mgr.generations) == 1
    assert rec.best_candidate is not None
    assert np.isfinite(rec.mean_reward)


def test_hybrid_matches_cpu_when_no_gpu(monkeypatch):
    """Without a GPU, hybrid mode must equal pure-CPU results exactly."""
    monkeypatch.setattr("umbra.evolution._cupy_available", lambda: False)

    def build(mode):
        return EvolutionManager(
            original=np.random.default_rng(1).random((40, 40, 3)).astype(np.float32),
            encoder=NoiseStreamEncoder(sigma=0.0),  # sigma=0 -> deterministic encode
            decoder=NoiseStreamDecoder(denoise_sigma=0.7),
            population_size=8, base_seed=99, enable_waveform=False, compute_mode=mode,
        )

    cpu_rec = build("cpu").run_generation()
    hyb_rec = build("hybrid").run_generation()
    assert cpu_rec.best_candidate.reward == pytest.approx(hyb_rec.best_candidate.reward)


def test_update_settings_changes_compute_mode():
    ref = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    mgr = EvolutionManager(
        original=ref, encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(), population_size=4, compute_mode="cpu",
    )
    mgr.update_settings(compute_mode="hybrid", gpu_fraction=0.25)
    assert mgr._compute_mode == "hybrid"
    assert mgr._gpu_fraction == pytest.approx(0.25)


def test_env_helpers(monkeypatch):
    monkeypatch.setenv("UMBRA_COMPUTE_MODE", "HYBRID")
    monkeypatch.setenv("UMBRA_GPU_FRACTION", "0.3")
    assert _env_compute_mode() == "hybrid"
    assert _env_gpu_fraction() == pytest.approx(0.3)
    monkeypatch.setenv("UMBRA_COMPUTE_MODE", "nonsense")
    assert _env_compute_mode() == "cpu"
