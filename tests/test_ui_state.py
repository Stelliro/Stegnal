from umbra.metrics import ReconstructionMetrics
from umbra.ui import (
    UmbraAppState,
    _compute_composite_score,
    _compute_readability_score,
    _normalize_pinterest_url,
)


def test_compute_composite_score_increases_with_metrics() -> None:
    baseline = _compute_composite_score(50.0, 30.0, 0.5)
    improved_overlap = _compute_composite_score(80.0, 30.0, 0.5)
    improved_psnr = _compute_composite_score(50.0, 55.0, 0.5)
    improved_ssim = _compute_composite_score(50.0, 30.0, 0.9)

    assert improved_overlap > baseline
    assert improved_psnr > baseline
    assert improved_ssim > baseline


def test_readability_score_tracks_metrics() -> None:
    baseline = _compute_readability_score(50.0, 30.0, 0.5)
    improved_overlap = _compute_readability_score(80.0, 30.0, 0.5)
    improved_psnr = _compute_readability_score(50.0, 55.0, 0.5)
    improved_ssim = _compute_readability_score(50.0, 30.0, 0.9)

    assert improved_overlap > baseline
    assert improved_psnr > baseline
    assert improved_ssim > baseline


def test_app_state_records_generations() -> None:
    state = UmbraAppState()
    metrics = ReconstructionMetrics(psnr=42.0, ssim=0.92)

    sound_metrics = ReconstructionMetrics(psnr=28.0, ssim=0.66)
    entry = state.record_generation(
        5,
        metrics,
        76.0,
        sound_metrics=sound_metrics,
        sound_overlap=64.0,
        sound_reference_metrics=metrics,
        sound_reference_overlap=70.0,
    )
    assert entry["generation"] == 5
    assert entry["overlap"] == 76.0
    assert entry["psnr"] == 42.0
    assert entry["ssim"] == 0.92
    assert entry["ai_score"] > 0
    assert entry["composite_score"] == entry["sound_score"]
    assert entry["sound_psnr"] == sound_metrics.psnr
    assert entry["sound_ssim"] == sound_metrics.ssim
    assert entry["sound_overlap"] == 64.0
    assert entry["sound_score"] > 0
    assert entry["sound_readability_score"] > 0
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
