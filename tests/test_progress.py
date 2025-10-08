import math

from umbra.progress import prepare_trend_chart, sanitize_progress_rows


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
    assert names[0] == "offset"
    assert "preventOverflow" in names
    assert "hide" not in names
