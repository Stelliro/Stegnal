from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np

from umbra.evolution import EvolutionManager


def test_ensure_manager_preserves_infinite_flag(monkeypatch) -> None:
    stub_state: dict[str, object] = {
        "run_infinite": False,
        "evolution_trees": {},
        "shared_seed": 123,
        "active_sound_seed": 99,
        "current_sound_sample_rate": 48_000,
        "current_sound_resolution": 128,
        "active_parent_seeds": [],
    }
    class _StubFrame:
        def __init__(self, data: object) -> None:
            self._data = list(data) if data is not None else []
            self.empty = len(self._data) == 0

        def replace(self, *_args, **_kwargs):
            return self

        def dropna(self, *_args, **_kwargs):
            return self

        def to_csv(self, *_args, **_kwargs) -> str:
            return ""

    def _data_frame(data: object) -> _StubFrame:
        return _StubFrame(data)

    fake_pandas = SimpleNamespace(DataFrame=_data_frame)

    fake_streamlit = SimpleNamespace(
        session_state=stub_state,
        experimental_rerun=lambda: None,
        rerun=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    ui = importlib.import_module("umbra.ui")
    try:
        from umbra.decoding import NoiseStreamDecoder
        from umbra.encoding import NoiseStreamEncoder

        original = np.zeros((8, 8), dtype=np.float32)
        encoder = NoiseStreamEncoder(sigma=0.2)
        decoder = NoiseStreamDecoder(denoise_sigma=1.0)

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=123,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is False

        stub_state["run_infinite"] = True
        stub_state["shared_seed"] = 456
        stub_state["active_sound_seed"] = 456

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=456,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is True
    finally:
        sys.modules.pop("umbra.ui", None)


def test_session_export_payload_contains_provenance(monkeypatch) -> None:
    from umbra.decoding import NoiseStreamDecoder
    from umbra.encoding import NoiseStreamEncoder
    from umbra.sound import generate_sound_art

    stub_state: dict[str, object] = {}

    class _StubFrame:
        def __init__(self, data: object) -> None:
            self._data = list(data) if data is not None else []
            self.empty = len(self._data) == 0

        def replace(self, *_args, **_kwargs):
            return self

        def dropna(self, *_args, **_kwargs):
            return self

        def to_csv(self, *_args, **_kwargs) -> str:
            return ""

    def _data_frame(data: object) -> _StubFrame:
        return _StubFrame(data)

    fake_pandas = SimpleNamespace(DataFrame=_data_frame)
    fake_streamlit = SimpleNamespace(
        session_state=stub_state,
        experimental_rerun=lambda: None,
        rerun=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    from umbra.ui import _session_export_payload

    original = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.2)
    decoder = NoiseStreamDecoder(denoise_sigma=0.9)
    manager = EvolutionManager(
        original=original,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=2024,
        autosave_interval=2,
    )

    generation = manager.run_generation()
    best_candidate = generation.best_candidate

    state: dict[str, object] = {
        "hardware_backend": "CPU (NumPy)",
        "difficulty_progress": 0.2,
        "max_overlap_seen": float(best_candidate.overlap_score),
        "sound_reseed_count": 1,
        "difficulty_improvement": 0.1,
        "difficulty_volatility": 0.05,
        "performance_history": [],
        "latest_generation_difficulty": float(generation.difficulty_level),
        "latest_generation_difficulty_raw": float(generation.difficulty_raw),
    }

    _, _, sound_clip, _ = generate_sound_art(
        seed=manager.base_seed,
        image_size=(32, 32),
        sample_rate=44_100,
    )

    metrics = best_candidate.metrics
    payload = _session_export_payload(
        state=state,
        manager=manager,
        metrics=metrics,
        sound_metrics=metrics,
        ai_sound_alignment=metrics,
        ai_overlap_score=best_candidate.overlap_score,
        sound_overlap_score=best_candidate.overlap_score,
        sound_clip=sound_clip,
        base_encoder_sigma=0.2,
        base_decoder_sigma=1.0,
        encoder_sigma=0.2,
        denoise_sigma=0.9,
        current_sample_rate=sound_clip.sample_rate,
        current_resolution=32,
        sample_rate_range=(20_000, 48_000),
        resolution_range=(32, 64),
        seed=manager.base_seed,
        sound_seed=sound_clip.seed,
        target_dwell=5,
        best_candidate_summary={
            "generation": generation.index,
            "seed": best_candidate.seed,
            "psnr": best_candidate.metrics.psnr,
            "ssim": best_candidate.metrics.ssim,
            "overlap": best_candidate.overlap_score,
        },
    )

    assert "provenance" in payload
    assert "random_seeds" in payload["provenance"]
    metrics_block = payload["metrics"]
    assert metrics_block["global_pooled"]["desc"] == "pooled/global comparator"
    assert metrics_block["per_candidate_strict"]["desc"] == "gallery/best-candidate strict comparator"
    difficulty_block = payload["difficulty"]
    for key in ("raw", "normalized", "target"):
        assert key in difficulty_block


def test_handle_auto_pause_retains_infinite(monkeypatch) -> None:
    from umbra import ui

    messages: list[str] = []

    sidebar_stub = SimpleNamespace(info=lambda message: messages.append(str(message)))
    monkeypatch.setattr(ui, "st", SimpleNamespace(sidebar=sidebar_stub), raising=False)

    state: dict[str, object] = {"run_infinite": True, "pending_generations": 5}

    ui._handle_auto_pause(state, difficulty_progress=0.95, pause_threshold=0.9)

    assert state["run_infinite"] is True
    assert state["auto_pause_acknowledged"] is True
    assert messages, "Expected an informational message when auto pause triggers in infinite mode"


def test_handle_auto_pause_pauses_finite(monkeypatch) -> None:
    from umbra import ui

    messages: list[str] = []

    sidebar_stub = SimpleNamespace(info=lambda message: messages.append(str(message)))
    monkeypatch.setattr(ui, "st", SimpleNamespace(sidebar=sidebar_stub), raising=False)

    state: dict[str, object] = {"run_infinite": False, "pending_generations": 7}

    ui._handle_auto_pause(state, difficulty_progress=0.95, pause_threshold=0.9)

    assert state["run_infinite"] is False
    assert state["pending_generations"] == 0
    assert state["auto_pause_acknowledged"] is True
    assert messages, "Expected an informational message when auto pause pauses finite runs"

    ui._handle_auto_pause(state, difficulty_progress=0.2, pause_threshold=0.9)

    assert state["auto_pause_acknowledged"] is False
