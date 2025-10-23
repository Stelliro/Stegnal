import itertools

import pytest

import umbra.ui as ui
from umbra.metrics import (
    ReconstructionMetrics,
    audio_fidelity_score,
    composite_score,
    readability_score,
    team_cohesion_score,
)
from umbra.ui import UmbraAppState, _generate_unique_model_path, _normalize_pinterest_url


def test_compute_composite_score_increases_with_metrics() -> None:
    baseline = composite_score(50.0, 30.0, 0.5)
    improved_overlap = composite_score(80.0, 30.0, 0.5)
    improved_psnr = composite_score(50.0, 55.0, 0.5)
    improved_ssim = composite_score(50.0, 30.0, 0.9)

    assert improved_overlap > baseline
    assert improved_psnr > baseline
    assert improved_ssim > baseline


def test_readability_score_tracks_metrics() -> None:
    baseline = readability_score(50.0, 30.0, 0.5)
    improved_overlap = readability_score(80.0, 30.0, 0.5)
    improved_psnr = readability_score(50.0, 55.0, 0.5)
    improved_ssim = readability_score(50.0, 30.0, 0.9)

    assert improved_overlap > baseline
    assert improved_psnr > baseline
    assert improved_ssim > baseline


def test_app_state_records_generations() -> None:
    state = UmbraAppState()
    metrics = ReconstructionMetrics(psnr=42.0, ssim=0.92)

    sound_metrics = ReconstructionMetrics(psnr=28.0, ssim=0.66)
    reference_partial = 0.42
    alignment_partial = 0.35
    sound_score = audio_fidelity_score(
        70.0, metrics.psnr, metrics.ssim, partial_credit=reference_partial
    )
    sound_readability = readability_score(70.0, metrics.psnr, metrics.ssim)
    alignment_score = audio_fidelity_score(
        64.0, sound_metrics.psnr, sound_metrics.ssim, partial_credit=alignment_partial
    )
    team_value = team_cohesion_score(
        76.0,
        metrics.psnr,
        metrics.ssim,
        sound_reference_overlap=70.0,
        sound_reference_psnr=metrics.psnr,
        sound_reference_ssim=metrics.ssim,
        sound_alignment_overlap=64.0,
        sound_alignment_psnr=sound_metrics.psnr,
        sound_alignment_ssim=sound_metrics.ssim,
        sound_reference_partial=reference_partial,
        sound_alignment_partial=alignment_partial,
        readability=sound_readability,
    )
    entry = state.record_generation(
        5,
        metrics,
        76.0,
        sound_metrics=sound_metrics,
        sound_overlap=64.0,
        sound_reference_metrics=metrics,
        sound_reference_overlap=70.0,
        sound_reference_partial=reference_partial,
        sound_alignment_partial=alignment_partial,
        sound_score=sound_score,
        sound_readability_score=sound_readability,
        sound_alignment_score=alignment_score,
        team_score=team_value,
        ai_score_value=composite_score(76.0, metrics.psnr, metrics.ssim),
        frame_time_ms=123.4,
        execution_backend="gpu",
    )
    assert entry["generation"] == 5
    assert entry["overlap"] == 76.0
    assert entry["psnr"] == 42.0
    assert entry["ssim"] == 0.92
    assert entry["ai_score"] > 0
    assert entry["sound_psnr"] == pytest.approx(metrics.psnr)
    assert entry["sound_ssim"] == pytest.approx(metrics.ssim)
    assert entry["sound_overlap"] == pytest.approx(70.0)
    assert entry["sound_score"] == pytest.approx(sound_score)
    assert entry["sound_readability_score"] == pytest.approx(sound_readability)
    assert entry["sound_alignment_score"] == pytest.approx(alignment_score)
    assert entry["sound_reference_partial"] == pytest.approx(reference_partial * 100.0)
    assert entry["sound_alignment_partial"] == pytest.approx(alignment_partial * 100.0)
    assert entry["team_score"] == pytest.approx(team_value)
    assert entry["composite_score"] == pytest.approx(team_value)
    assert entry["frame_time_ms"] == pytest.approx(123.4)
    assert entry["frame_fps"] == pytest.approx(1000.0 / 123.4)
    assert entry["execution_backend"] == "gpu"
    assert state.sound_scores[-1] == entry["sound_score"]
    assert state.readability_scores[-1] == entry["sound_readability_score"]

    # Fill more than the history limit to ensure trimming works.
    for index in range(1000):
        state.record_generation(index, metrics, 60.0 + 0.01 * index)

    assert len(state.history) <= state.history.maxlen  # type: ignore[operator]
    assert len(state.composite_scores) <= state.composite_scores.maxlen  # type: ignore[operator]
    assert len(state.sound_scores) <= state.sound_scores.maxlen  # type: ignore[operator]
    assert len(state.readability_scores) <= state.readability_scores.maxlen  # type: ignore[operator]


def test_normalize_pinterest_url_strips_tracking() -> None:
    messy = "https://i.pinimg.com/originals/d5/3b/01/d53b014d86a6b6761bf649a0ed813c2b.png?foo=1#fragment"
    clean = _normalize_pinterest_url(messy)
    assert clean == "https://i.pinimg.com/originals/d5/3b/01/d53b014d86a6b6761bf649a0ed813c2b.png"


def test_generate_unique_model_path_avoids_collisions(tmp_path, monkeypatch) -> None:
    sequence = itertools.cycle("abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setattr(ui.secrets, "choice", lambda _alphabet: next(sequence))

    first_path = _generate_unique_model_path(tmp_path, generation_index=12)
    first_path.touch()
    second_path = _generate_unique_model_path(tmp_path, generation_index=12)

    assert first_path != second_path
    assert first_path.parent == tmp_path
    assert second_path.parent == tmp_path
    assert first_path.suffix == ".json"
    assert second_path.suffix == ".json"
    assert first_path.name.endswith("_12.json")
    assert second_path.name.endswith("_12.json")
