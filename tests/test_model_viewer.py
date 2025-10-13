from __future__ import annotations

import json

from umbra.tools.model_viewer import compute_total_score, load_model_stats


def test_compute_total_score_with_metrics() -> None:
    stats = {
        "metrics": {
            "ai_vs_reference": {"psnr": 14.0, "ssim": 0.82},
            "sound_vs_ai": {"psnr": 13.2, "ssim": 0.78},
            "overlap": {
                "ai_vs_reference": 67.0,
                "sound_vs_ai": 66.0,
                "sound_vs_reference": 65.0,
            },
            "global_pooled": {"psnr": 13.5},
        },
        "manager": {
            "total_score": 420.0,
            "latest_total_score": 58.0,
            "best_candidate": {"overlap": 68.5},
        },
        "lifetime_reward": 220.0,
    }

    breakdown = compute_total_score(stats)

    assert breakdown.total is not None
    assert 0.0 <= breakdown.total <= 100.0
    assert "AI PSNR" in breakdown.components
    assert breakdown.components["AI PSNR"] > 0.0


def test_compute_total_score_without_components() -> None:
    breakdown = compute_total_score({})

    assert breakdown.total is None
    assert breakdown.components == {}


def test_load_model_stats_prefers_named_files(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    fallback = tmp_path / "metrics.json"
    summary.write_text(json.dumps({"key": "summary"}), encoding="utf-8")
    fallback.write_text(json.dumps({"key": "metrics"}), encoding="utf-8")

    data, source = load_model_stats(tmp_path)

    assert data == {"key": "summary"}
    assert source == summary


def test_load_model_stats_from_file(tmp_path) -> None:
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps({"key": 123}), encoding="utf-8")

    data, source = load_model_stats(stats_file)

    assert data == {"key": 123}
    assert source == stats_file
