"""Filesystem helpers for organising Umbra evolution runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Collection of important directories for a single run."""

    root: Path
    charts: Path


_DEFAULT_RUNS_ROOT = Path("runs")


def runs_root(base: str | Path | None = None) -> Path:
    """Return the directory that stores all run artefacts."""

    root = Path(base) if base is not None else _DEFAULT_RUNS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_run_paths(run_id: str, *, base: str | Path | None = None) -> RunPaths:
    """Ensure directories for ``run_id`` exist and return their paths."""

    if not run_id:
        raise ValueError("run_id must be provided")

    runs_dir = runs_root(base=base)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(root=run_dir, charts=charts_dir)


def chart_file(run_id: str, filename: str, *, base: str | Path | None = None) -> Path:
    """Return the path to ``filename`` inside ``run_id``'s charts directory."""

    if not filename:
        raise ValueError("filename must be provided")
    paths = ensure_run_paths(run_id, base=base)
    return paths.charts / filename


__all__ = ["RunPaths", "runs_root", "ensure_run_paths", "chart_file"]

