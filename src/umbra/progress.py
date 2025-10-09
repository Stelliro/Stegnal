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

    if not rows:
        message = (
            "Trend chart hidden until generations contain finite metric values."
            if had_non_finite
            else "Trend chart will appear once generations contain finite metric values."
        )
        return None, message

    metrics = _metric_names(rows)
    if not metrics:
        return None, "Trend chart requires metric columns to display."

    unique_generations = len({row["Generation"] for row in rows})
    metric_variations = []
    for metric in metrics:
        values = {row[metric] for row in rows if metric in row}
        metric_variations.append(len(values) > 1)

    has_variation = unique_generations > 1 and any(metric_variations)

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

    if not values:
        return None, "Trend chart requires metric columns to display."

    for entry in values:
        if not math.isfinite(entry["Generation"]) or not math.isfinite(entry["Value"]):
            logger.warning(
                "Discarding trend chart due to non-finite values after sanitization"
            )
            return None, "Trend chart hidden until generations contain finite metric values."

    if not has_variation:
        return None, "Trend chart will appear once multiple non-identical generations are available."

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"values": values},
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

    logger.debug(
        "Prepared trend chart spec with %d records and metrics %s",
        len(values),
        metrics,
    )
    return spec, None


__all__ = ["sanitize_progress_rows", "prepare_trend_chart"]

