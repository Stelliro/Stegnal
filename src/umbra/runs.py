# runs.py

"""Helpers for managing persistent evolution run directories."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from uuid import uuid4

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - defensive
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "runs"


def _resolve_base(directory: str | Path | None = None) -> Path:
    base = Path(directory) if directory is not None else _DEFAULT_RUN_ROOT
    return base


def new_run(base_dir: str | Path | None = None) -> tuple[str, Path]:
    """Create a fresh run directory and return its identifier and path."""

    base = _resolve_base(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    for _ in range(32):
        run_id = uuid4().hex
        run_dir = base / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        logger.debug("Created run directory %s", run_dir)
        return run_id, run_dir

    raise RuntimeError("Failed to allocate a unique run directory")


def get_run_paths(run_id: str, base_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Return the directory and history path for ``run_id``."""

    base = _resolve_base(base_dir)
    run_dir = base / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.parquet"
    return run_dir, history_path


def _normalise_records(
    records: Iterable[Mapping[str, object]] | Mapping[str, object]
) -> list[dict[str, object]]:
    if isinstance(records, Mapping):
        return [dict(records)]
    return [dict(row) for row in records]


class _HistoryRecords(list[dict[str, object]]):
    """Minimal DataFrame-like container when pandas is unavailable."""

    @property
    def empty(self) -> bool:
        return not self

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            return [row.get(key) for row in self]
        return super().__getitem__(key)


def _load_history_json(path: Path) -> _HistoryRecords:
    if not path.exists():
        return _HistoryRecords()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return _HistoryRecords(data)
        return _HistoryRecords([data])
    except Exception as exc:
        logger.debug("Failed to load JSON history from %s: %s", path, exc)
        return _HistoryRecords()


def _write_history_json(path: Path, data: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_history(
    run_id: str,
    records: Iterable[Mapping[str, object]] | Mapping[str, object],
    *,
    replace: bool = False,
    base_dir: str | Path | None = None,
) -> Path:
    """Persist ``records`` to the history file for ``run_id``."""

    rows = _normalise_records(records)
    if not rows:
        return Path()

    _, history_path = get_run_paths(run_id, base_dir=base_dir)

    if pd is None:
        existing = [] if replace else list(_load_history_json(history_path))
        data = existing + rows if not replace else rows
        if replace and not data and history_path.exists():
            try:
                history_path.unlink()
            except OSError:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to remove existing history file %s", history_path, exc_info=True
                )
            return history_path
        _write_history_json(history_path, data)
        logger.debug(
            "Persisted %d history row(s) to %s (JSON fallback)", len(data), history_path
        )
        return history_path

    df = pd.DataFrame(rows)
    if df.empty:
        if replace and history_path.exists():
            try:
                history_path.unlink()
            except OSError:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to remove existing history file %s", history_path, exc_info=True
                )
        return history_path

    if not replace and history_path.exists():
        try:
            existing = pd.read_parquet(history_path)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to read existing history at %s", history_path, exc_info=True)
            existing = pd.DataFrame()
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(history_path, index=False)
    logger.debug("Persisted %d history row(s) to %s", len(df), history_path)
    return history_path


def load_history(
    run_id: str,
    *,
    base_dir: str | Path | None = None,
) -> object:
    """Load the history dataframe for ``run_id``.

    Returns an empty dataframe when no history is present.
    """

    _, history_path = get_run_paths(run_id, base_dir=base_dir)
    if not history_path.exists():
        return _HistoryRecords() if pd is None else pd.DataFrame()
    if pd is None:
        return _load_history_json(history_path)
    try:
        return pd.read_parquet(history_path)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to load history from %s", history_path, exc_info=True)
        return pd.DataFrame()


__all__ = ["append_history", "get_run_paths", "load_history", "new_run"]