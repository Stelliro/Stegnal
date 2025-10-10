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


__all__ = ["sanitize_progress_rows", "prepare_trend_chart"]

