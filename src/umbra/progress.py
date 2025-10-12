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

    values: list[dict[str, float]] = []
    for row in rows:
        generation = row["Generation"]
        for metric in metrics:
            if metric in row:
                values.append(
                    {
                        "Generation": generation,
                        "Metric": metric,
                        "Value": row[metric],
                    }
                )

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

    generation_values = [entry["Generation"] for entry in filtered_values]
    unique_generations = len(set(generation_values))

    metric_variations = [
        len({entry["Value"] for entry in filtered_values if entry["Metric"] == metric}) > 1
        for metric in metrics_with_data
    ]

    if unique_generations <= 1 or not any(metric_variations):
        return None, "Trend chart will appear once multiple non-identical generations are available."

    score_values = [entry["Value"] for entry in filtered_values]
    x_domain = [min(generation_values), max(generation_values)]
    y_min = min(score_values)
    y_max = max(score_values)

    if y_min == y_max:
        return None, "Trend chart will appear once multiple non-identical generations are available."

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"values": filtered_values},
        "autosize": {"type": "fit", "contains": "padding"},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "Generation", "type": "quantitative", "title": "Generation"},
            "y": {"field": "Value", "type": "quantitative", "title": "Score"},
            "color": {"field": "Metric", "type": "nominal", "title": "Metric"},
            "tooltip": [
                {"field": "Generation", "type": "quantitative"},
                {"field": "Metric", "type": "nominal"},
                {"field": "Value", "type": "quantitative"},
            ],
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
    markers: Sequence[int] | None = None,
    window: int | None = None,
    auto_follow: bool = True,
) -> dict[str, object] | None:
    """Return a Vega-Lite spec visualising the performance history.

    Args:
        history: Sanitised performance samples.
        markers: Optional observation indices that highlight sound target
            transitions.
        window: Desired number of observations to keep visible when auto-
            following. ``None`` disables windowing.
        auto_follow: Whether the default chart view should stick to the most
            recent observations.
    """

    if len(history) < 2:
        return None

    metric_labels = {
        "ai_overlap": "AI overlap (%)",
        "ai_ssim": "AI SSIM",
        "ai_psnr": "AI PSNR (dB)",
        "sound_overlap": "Sound overlap (%)",
        "difficulty_progress": "Adaptive difficulty (%)",
        "difficulty_target": "Difficulty target (%)",
        "reward_signal": "Reward signal (%)",
        "reward_points": "Lifetime reward (pts)",
    }

    values: list[dict[str, float | str]] = []
    for index, entry in enumerate(history, start=1):
        step = float(entry.get("step", index))
        for key, label in metric_labels.items():
            if key not in entry:
                continue
            try:
                numeric = float(entry[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            values.append({"Step": step, "Metric": label, "Value": numeric})

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

    step_values = [value["Step"] for value in values]
    score_values = [value["Value"] for value in values]

    step_min = min(step_values)
    step_max = max(step_values)
    y_domain = [min(score_values), max(score_values)]
    if y_domain[0] == y_domain[1]:
        return None

    x_scale: dict[str, object] = {"nice": False}
    if auto_follow and window:
        window = max(int(window), 1)
        domain_start = max(step_max - window + 1, step_min)
        x_scale["domain"] = [domain_start, step_max]
    else:
        x_scale["domain"] = [step_min, step_max]

    spec: dict[str, object] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"values": values},
        "encoding": {
            "x": {
                "field": "Step",
                "type": "quantitative",
                "title": "Observation",
                "scale": x_scale,
            },
            "y": {
                "field": "Value",
                "type": "quantitative",
                "title": "Score",
                "scale": {"domain": y_domain},
            },
            "color": {"field": "Metric", "type": "nominal", "title": "Metric"},
            "tooltip": [
                {"field": "Step", "type": "quantitative"},
                {"field": "Metric", "type": "nominal"},
                {"field": "Value", "type": "quantitative"},
            ],
        },
        "layer": [{"mark": {"type": "line", "point": True}}],
        "config": {
            "legend": {"orient": "bottom", "title": ""},
            "point": {"filled": True, "size": 30},
        },
        "params": [
            {
                "name": "history_view",
                "select": {
                    "type": "interval",
                    "encodings": ["x"],
                    "bind": "scales",
                    "translate": "[mousedown[event.shiftKey], mousemove[event.shiftKey], mouseup]",
                    "zoom": "wheel![event.shiftKey]",
                },
            }
        ],
    }

    marker_values: list[dict[str, float | str]] = []
    if markers:
        seen: set[int] = set()
        ordered_markers: list[int] = []
        for marker in markers:
            if marker in seen:
                continue
            seen.add(marker)
            ordered_markers.append(marker)
        marker_values = [
            {
                "Step": float(marker),
                "Label": "Sound target reseeded",
            }
            for marker in ordered_markers
            if step_min <= float(marker) <= step_max
        ]

    if marker_values:
        spec["layer"].append(
            {
                "data": {"values": marker_values},
                "mark": {
                    "type": "rule",
                    "color": "#6b7280",
                    "strokeDash": [4, 4],
                    "size": 1,
                },
                "encoding": {
                    "x": {"field": "Step", "type": "quantitative"},
                    "tooltip": [
                        {"field": "Label", "type": "nominal"},
                        {"field": "Step", "type": "quantitative"},
                    ],
                },
            }
        )
        spec["layer"].append(
            {
                "data": {"values": marker_values},
                "mark": {
                    "type": "point",
                    "shape": "triangle-down",
                    "color": "#6b7280",
                    "filled": True,
                    "size": 80,
                },
                "encoding": {
                    "x": {"field": "Step", "type": "quantitative"},
                    "y": {"value": y_domain[1]},
                    "tooltip": [
                        {"field": "Label", "type": "nominal"},
                        {"field": "Step", "type": "quantitative"},
                    ],
                },
            }
        )

    return spec


__all__ = ["sanitize_progress_rows", "prepare_trend_chart", "prepare_metrics_chart"]

