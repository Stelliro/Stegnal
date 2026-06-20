# chart_export.py

"""Utilities for exporting Vega-Lite charts to PNG images."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import vl_convert as vlc
    vl_convert = vlc
except ImportError:  # pragma: no cover - optional dependency
    vlc = None
    vl_convert = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on optional symbol availability
    from vl_convert import VegaLite as _VegaLite
except ImportError:  # pragma: no cover - fallback for older binaries
    _VegaLite = None

logger = logging.getLogger(__name__)


def _render_png(spec: Mapping[str, Any], *, scale: float | None = None) -> bytes:
    """Return PNG bytes for ``spec`` using the fastest available backend."""

    if vlc is None:
        raise ImportError("vl-convert-python is required for chart export")

    options: dict[str, Any] = {}
    if scale is not None:
        options["scale"] = scale

    if _VegaLite is not None:
        converter = _VegaLite()
        return converter.convert(dict(spec), format="png", **options)

    return vlc.vegalite_to_png(spec, **options)


def export_chart_png(
    spec: Mapping[str, Any],
    output_path: str | Path,
    *,
    scale: float | None = None,
) -> Path:
    """Serialise a Vega-Lite ``spec`` to ``output_path`` as a PNG image."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = dict(spec)
    sanitized.pop("params", None)

    try:
        png_bytes = _render_png(sanitized, scale=scale)
    except Exception as exc:
        logger.error(f"Failed to render chart to PNG: {exc}")
        raise

    path.write_bytes(png_bytes)
    return path


__all__ = ["export_chart_png"]