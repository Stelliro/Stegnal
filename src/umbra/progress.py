"""Utilities for preparing evolution progress analytics."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


def sanitize_progress_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, float]], bool]:
    """Return sanitized rows and whether any values were discarded."""

    raw_rows = list(rows)
    sanitized: list[dict[str, float]] = []
    discarded = False

    for row in raw_rows:
        try:
            generation = float(row.get("Generation"))
        except (TypeError, ValueError):
            discarded = True
            continue

        clean_row: dict[str, float] = {"Generation": generation}
        metrics_found = False
        for key, value in row.items():
            if key == "Generation":
                continue
            if key.startswith("_"):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                discarded = True
                continue
            if math.isfinite(numeric):
                clean_row[key] = numeric
                metrics_found = True
            else:
                discarded = True

        if metrics_found:
            sanitized.append(clean_row)

    sanitized.sort(key=lambda item: item["Generation"])

    logger.debug(
        "Sanitised %d rows -> %d usable rows (discarded=%s)",
        len(raw_rows),
        len(sanitized),
        discarded,
    )
    return sanitized, discarded


def _metric_names(rows: Sequence[Mapping[str, float]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(key for key in row.keys() if key != "Generation")
    return sorted(names)


def prepare_trend_chart(
    rows: Sequence[Mapping[str, float]],
    *,
    had_non_finite: bool = False,
) -> tuple[dict[str, object] | None, str | None]:
    """Create a Vega-Lite spec from sanitized rows or a user-facing message."""

    def _empty_message() -> str:
        return (
            "Trend chart hidden until generations contain finite metric values."
            if had_non_finite
            else "Trend chart will appear once generations contain finite metric values."
        )

    if not rows:
        return None, _empty_message()

    metrics = _metric_names(rows)
    if not metrics:
        return None, "Trend chart requires metric columns to display."

    values: list[dict[str, object]] = []
    for row in rows:
        generation = row["Generation"]
        if generation is None:
            continue
        try:
            generation_value = float(generation)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(generation_value) or generation_value < 0:
            logger.debug("Skipping negative generation value: %s", generation)
            continue
        for metric in metrics:
            if metric in row:
                entry: dict[str, object] = {
                    "Generation": generation_value,
                    "Metric": metric,
                    "Value": row[metric],
                }
                run_id = row.get("_run_id")
                if run_id is not None:
                    entry["Run"] = str(run_id)
                run_generation = row.get("_run_generation")
                if run_generation is not None:
                    try:
                        entry["Run generation"] = float(run_generation)
                    except (TypeError, ValueError):
                        pass
                run_seed = row.get("_run_seed")
                if run_seed is not None:
                    try:
                        entry["Run seed"] = float(run_seed)
                    except (TypeError, ValueError):
                        pass
                values.append(entry)

    filtered_values = []
    for entry in values:
        generation = entry["Generation"]
        value = entry["Value"]
        if value is None or not math.isfinite(value):
            logger.debug(
                "Skipping non-finite metric value for generation %s metric %s",
                generation,
                entry["Metric"],
            )
            continue
        if not math.isfinite(generation):
            logger.debug("Skipping non-finite generation value: %s", generation)
            continue
        filtered_values.append(entry)

    if not filtered_values:
        return None, _empty_message()

    metrics_with_data = sorted({entry["Metric"] for entry in filtered_values})
    if not metrics_with_data:
        return None, "Trend chart requires metric columns to display."

    raw_generations = [entry["Generation"] for entry in filtered_values]
    unique_generations = len(set(raw_generations))

    metric_variations = [
        len({entry["Value"] for entry in filtered_values if entry["Metric"] == metric}) > 1
        for metric in metrics_with_data
    ]

    if unique_generations <= 1 or not any(metric_variations):
        return None, "Trend chart will appear once multiple non-identical generations are available."

    score_values = [entry["Value"] for entry in filtered_values]
    x_min_raw = min(raw_generations)
    x_max_raw = max(raw_generations)
    use_log_scale = x_max_raw >= 10

    generation_values = list(raw_generations)
    if use_log_scale and generation_values:
        if x_min_raw <= 0:
            shift = 1.0 - x_min_raw
            for entry in filtered_values:
                entry["Generation"] = float(entry["Generation"]) + shift
            generation_values = [entry["Generation"] for entry in filtered_values]
        x_min = min(generation_values)
        x_max = max(generation_values)
    else:
        x_min = x_min_raw
        x_max = x_max_raw
        use_log_scale = False

    x_domain = [x_min, x_max]
    y_min = min(score_values)
    y_max = max(score_values)

    if y_min == y_max:
        return None, "Trend chart will appear once multiple non-identical generations are available."

    tooltip_fields = [
        {
            "field": "Generation",
            "type": "quantitative",
            "title": "Cumulative generation",
        },
        {"field": "Metric", "type": "nominal", "title": "Metric"},
        {"field": "Value", "type": "quantitative", "title": "Score"},
    ]

    if any("Run" in entry for entry in filtered_values):
        tooltip_fields.append({"field": "Run", "type": "nominal", "title": "Run"})
    if any("Run generation" in entry for entry in filtered_values):
        tooltip_fields.append(
            {"field": "Run generation", "type": "quantitative", "title": "Run generation"}
        )
    if any("Run seed" in entry for entry in filtered_values):
        tooltip_fields.append(
            {"field": "Run seed", "type": "quantitative", "title": "Run seed"}
        )

    log_min = math.floor(math.log10(x_min)) if x_min > 0 else 0
    log_max = math.ceil(math.log10(x_max)) if x_max > 0 else 0
    tick_values = [10 ** power for power in range(log_min, log_max + 1)]
    tick_values = [value for value in tick_values if x_min <= value <= x_max]
    if use_log_scale and not tick_values:
        tick_values = [x_min, x_max]

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"values": filtered_values},
        "autosize": {"type": "fit", "contains": "padding"},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {
                "field": "Generation",
                "type": "quantitative",
                "title": "Cumulative generation",
            },
            "y": {"field": "Value", "type": "quantitative", "title": "Score"},
            "color": {"field": "Metric", "type": "nominal", "title": "Metric"},
            "tooltip": tooltip_fields,
        },
        "config": {"legend": {"orient": "bottom", "title": ""}},
        "usermeta": {
            "embedOptions": {
                "tooltip": {
                    "modifiers": [
                        {"name": "offset", "options": {"mainAxis": 8, "crossAxis": 0}},
                        {"name": "preventOverflow", "options": {"padding": 16}},
                        {"name": "hide"},
                        {"name": "flip"},
                    ]
                }
            }
        },
    }

    if use_log_scale:
        x_scale: dict[str, object] = {"domain": x_domain, "type": "log", "base": 10}
        spec["encoding"]["x"]["scale"] = x_scale
        if tick_values:
            spec["encoding"]["x"]["axis"] = {"values": tick_values, "format": "s"}
    else:
        spec["encoding"]["x"]["scale"] = {"domain": x_domain}
    spec["encoding"]["y"]["scale"] = {"domain": [y_min, y_max]}

    logger.debug(
        "Prepared trend chart spec with %d records and metrics %s",
        len(filtered_values),
        metrics_with_data,
    )
    return spec, None


def prepare_metrics_chart(
    history: Sequence[Mapping[str, float]],
    *,
    markers: Sequence[float | int] | None = None,
    window: int | None = None,
    auto_follow: bool = True,
) -> dict[str, object] | None:
    """Return a Vega-Lite spec visualising the performance history."""

    if len(history) < 2:
        return None

    if window is not None and window <= 0:
        window = None

    start_index = 0
    if window is not None and len(history) > window and auto_follow:
        start_index = len(history) - window
    sliced_history = history[start_index:]

    metric_labels = {
        "ai_overlap": "AI overlap (%)",
        "ai_ssim": "AI SSIM",
        "ai_psnr": "AI PSNR (dB)",
        "sound_overlap": "AI↔Sound overlap (%)",
    }
    if any("composite_score" in entry for entry in sliced_history):
        metric_labels["composite_score"] = "Sound↔AI composite score"
    if any("ai_score" in entry for entry in sliced_history):
        metric_labels["ai_score"] = "AI baseline score"
    if any("sound_reference_overlap" in entry for entry in sliced_history):
        metric_labels["sound_reference_overlap"] = "Sound↔Reference overlap (%)"

    values: list[dict[str, float | str]] = []
    for offset, entry in enumerate(sliced_history, start=start_index + 1):
        for key, label in metric_labels.items():
            if key not in entry:
                continue
            try:
                numeric = float(entry[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            values.append({"Step": float(offset), "Metric": label, "Value": numeric})

    if not values:
        return None

    unique_steps = {value["Step"] for value in values}
    if len(unique_steps) <= 1:
        return None

    metric_variations: dict[str, set[float]] = {}
    for entry in values:
        label = str(entry["Metric"])
        metric_variations.setdefault(label, set()).add(float(entry["Value"]))
    varying_metrics = {label for label, samples in metric_variations.items() if len(samples) > 1}
    if not varying_metrics:
        return None

    values = [value for value in values if str(value["Metric"]) in varying_metrics]
    if not values:
        return None

    step_values = [value["Step"] for value in values]

    metric_stats: dict[str, tuple[float, float]] = {}
    for entry in values:
        label = str(entry["Metric"])
        value = float(entry["Value"])
        if label in metric_stats:
            current_min, current_max = metric_stats[label]
            metric_stats[label] = (min(current_min, value), max(current_max, value))
        else:
            metric_stats[label] = (value, value)

    for entry in values:
        label = str(entry["Metric"])
        value = float(entry["Value"])
        metric_min, metric_max = metric_stats[label]
        span = metric_max - metric_min
        if span <= 0:
            scaled = 0.5
        else:
            scaled = (value - metric_min) / span
        entry["ScaledValue"] = scaled

    x_domain = [min(step_values), max(step_values)]
    y_domain = [0.0, 1.0]

    schema = "https://vega.github.io/schema/vega-lite/v6.json"
    base_layer: dict[str, object] = {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {
                "field": "Step",
                "type": "quantitative",
                "title": "Observation",
                "scale": {"domain": x_domain},
            },
            "y": {
                "field": "ScaledValue",
                "type": "quantitative",
                "title": "Normalised score",
                "scale": {"domain": y_domain, "nice": False},
            },
            "color": {"field": "Metric", "type": "nominal", "title": "Metric"},
            "tooltip": [
                {"field": "Step", "type": "quantitative"},
                {"field": "Metric", "type": "nominal"},
                {
                    "field": "Value",
                    "type": "quantitative",
                    "title": "Score",
                },
                {
                    "field": "ScaledValue",
                    "type": "quantitative",
                    "title": "Normalised score",
                },
            ],
        },
    }

    config = {"legend": {"orient": "bottom", "title": ""}}

    marker_values: list[dict[str, float]] = []
    if markers:
        seen_steps: set[float] = set()
        x_min, x_max = x_domain
        for raw_marker in markers:
            try:
                step = float(raw_marker)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(step):
                continue
            if step < x_min or step > x_max:
                continue
            if step in seen_steps:
                continue
            seen_steps.add(step)
            marker_values.append({"Step": step})

    if marker_values:
        marker_layer = {
            "data": {"values": marker_values},
            "mark": {
                "type": "rule",
                "color": "#ff6f61",
                "strokeWidth": 1.5,
                "strokeDash": [6, 4],
            },
            "encoding": {
                "x": {
                    "field": "Step",
                    "type": "quantitative",
                    "title": "Observation",
                },
                "tooltip": [
                    {
                        "field": "Step",
                        "type": "quantitative",
                        "title": "Sound target",
                    }
                ],
            },
        }
        spec = {
            "$schema": schema,
            "config": config,
            "data": {"values": values},
            "layer": [base_layer, marker_layer],
            "resolve": {"scale": {"y": "shared"}},
        }
    else:
        spec = {
            "$schema": schema,
            "config": config,
            "data": {"values": values},
            **base_layer,
        }

    return spec


__all__ = ["sanitize_progress_rows", "prepare_trend_chart", "prepare_metrics_chart"]