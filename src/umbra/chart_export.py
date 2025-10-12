"""Utilities for exporting Vega-Lite charts to PNG images."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from vl_convert import vl_convert

try:  # pragma: no cover - depends on optional symbol availability
    from vl_convert import VegaLite as _VegaLite
except ImportError:  # pragma: no cover - fallback for older binaries
    _VegaLite = None


def _sanitise_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied version of ``spec`` safe for static export."""

    cleaned: dict[str, Any] = deepcopy(dict(spec))
    cleaned.pop("params", None)
    return cleaned


def _render_png(spec: Mapping[str, Any], *, scale: float | None = None) -> bytes:
    """Return PNG bytes for ``spec`` using the fastest available backend."""

    options: dict[str, Any] = {}
    if scale is not None:
        options["scale"] = scale

    cleaned = _sanitise_spec(spec)
    if _VegaLite is not None:
        converter = _VegaLite()
        return converter.convert(cleaned, format="png", **options)

    return vl_convert.vegalite_to_png(cleaned, **options)


def export_chart_png(
    spec: Mapping[str, Any],
    output_path: str | Path,
    *,
    scale: float | None = None,
) -> Path:
    """Serialise a Vega-Lite ``spec`` to ``output_path`` as a PNG image."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = _render_png(spec, scale=scale)
    path.write_bytes(png_bytes)
    return path


__all__ = ["export_chart_png"]

