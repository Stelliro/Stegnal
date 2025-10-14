import math

from umbra.progress import (
    prepare_metrics_chart,
    prepare_trend_chart,
    sanitize_progress_rows,
)


def test_sanitize_progress_rows_handles_non_finite_values() -> None:
    rows = [
        {"Generation": 0, "Best SSIM": 0.9, "Best PSNR": math.inf},
        {"Generation": 1, "Best SSIM": 0.85, "Best PSNR": 21.0},
    ]

    sanitized, discarded = sanitize_progress_rows(rows)

    assert discarded is True
    assert [row["Generation"] for row in sanitized] == [0.0, 1.0]
    assert "Best PSNR" not in sanitized[0]
    assert sanitized[1]["Best PSNR"] == 21.0


def test_prepare_trend_chart_requires_finite_values() -> None:
    rows = [
        {"Generation": 0, "Best SSIM": 0.9, "Best PSNR": math.inf},
        {"Generation": 1, "Best SSIM": 0.85, "Best PSNR": 21.0},
    ]
    sanitized, discarded = sanitize_progress_rows(rows)

    spec, message = prepare_trend_chart(sanitized, had_non_finite=discarded)

    assert spec is not None
    assert message is None


def test_prepare_trend_chart_returns_spec_when_data_varies() -> None:
    rows = [
        {"Generation": 0, "Best SSIM": 0.8, "Best PSNR": 20.0},
        {"Generation": 1, "Best SSIM": 0.9, "Best PSNR": 21.5},
    ]
    sanitized, discarded = sanitize_progress_rows(rows)
    spec, message = prepare_trend_chart(sanitized, had_non_finite=discarded)

    assert message is None
    assert spec is not None
    modifiers = spec["usermeta"]["embedOptions"]["tooltip"]["modifiers"]
    names = [item["name"] for item in modifiers]
    assert names == ["offset", "preventOverflow", "hide", "flip"]
    assert spec["encoding"]["x"]["scale"]["domain"] == [0.0, 1.0]
    assert spec["encoding"]["y"]["scale"]["domain"] == [0.8, 21.5]


def test_prepare_trend_chart_handles_none_values() -> None:
    rows = [
        {"Generation": 0.0, "Best SSIM": None},
        {"Generation": 1.0, "Best SSIM": 0.5},
    ]

    spec, message = prepare_trend_chart(rows, had_non_finite=True)

    assert spec is None
    assert message is not None


def test_sanitize_progress_rows_preserves_additional_columns() -> None:
    rows = [
        {
            "Generation": 0,
            "reward_total": 1.0,
            "difficulty_raw": 0.8,
            "checkpoint_tag": "plateau_kick",
        },
        {
            "Generation": 1,
            "reward_total": 1.2,
            "difficulty_raw": 0.82,
        },
    ]

    sanitized, discarded = sanitize_progress_rows(rows)

    assert discarded is True
    assert sanitized[0]["reward_total"] == 1.0
    assert sanitized[0]["difficulty_raw"] == 0.8
    assert "checkpoint_tag" not in sanitized[0]


def test_prepare_metrics_chart_uses_root_dataset() -> None:
    history = [
        {"ai_overlap": 10.0, "ai_ssim": 0.1, "sound_overlap": 11.0},
        {"ai_overlap": 12.0, "ai_ssim": 0.2, "sound_overlap": 13.0},
    ]

    spec = prepare_metrics_chart(history, markers=[1])

    assert spec is not None
    assert spec.get("data", {}).get("values")
    assert "layer" in spec
    base_layer = spec["layer"][0]
    marker_layer = spec["layer"][1]
    assert "data" not in base_layer
    assert marker_layer.get("data", {}).get("values") == [{"Step": 1.0}]
    encoding = base_layer["encoding"]
    assert encoding["y"]["field"] == "ScaledValue"
    assert encoding["y"]["title"].lower().startswith("normalised")
    assert encoding["y"]["scale"]["domain"] == [0.0, 1.0]
    tooltip_titles = {item.get("title", item.get("field")) for item in encoding["tooltip"]}
    assert "Score" in tooltip_titles
    assert "Normalised score" in tooltip_titles


def test_prepare_metrics_chart_scales_per_metric() -> None:
    history = [
        {
            "ai_overlap": 80.0,
            "ai_ssim": 0.50,
            "ai_psnr": 30.0,
            "ai_score": 55.0,
            "composite_score": 60.0,
            "sound_overlap": 82.0,
        },
        {
            "ai_overlap": 81.0,
            "ai_ssim": 0.51,
            "ai_psnr": 31.0,
            "ai_score": 56.0,
            "composite_score": 61.0,
            "sound_overlap": 82.1,
        },
    ]

    spec = prepare_metrics_chart(history)

    assert spec is not None
    values = spec.get("data", {}).get("values", [])
    assert values, "expected chart values"
    metrics = {value["Metric"] for value in values}
    assert "AI SSIM" in metrics
    assert "Sound↔AI composite score" in metrics
    assert "AI baseline score" in metrics
    ssim_values = [value for value in values if value["Metric"] == "AI SSIM"]
    overlap_values = [value for value in values if value["Metric"] == "AI overlap (%)"]
    composite_values = [
        value for value in values if value["Metric"] == "Sound↔AI composite score"
    ]
    baseline_values = [value for value in values if value["Metric"] == "AI baseline score"]
    assert {entry["ScaledValue"] for entry in ssim_values} == {0.0, 1.0}
    assert {entry["ScaledValue"] for entry in overlap_values} == {0.0, 1.0}
    assert {entry["ScaledValue"] for entry in composite_values} == {0.0, 1.0}
    assert {entry["ScaledValue"] for entry in baseline_values} == {0.0, 1.0}
