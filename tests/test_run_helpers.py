"""Tests for umbra.run_helpers — path sanitization and run directory setup."""

from __future__ import annotations

import pytest

from umbra.run_helpers import RunPaths, chart_file, ensure_run_paths, runs_root


def test_runs_root_creates_directory(tmp_path):
    root = runs_root(base=tmp_path / "custom_runs")
    assert root.exists()


def test_ensure_run_paths_creates_directories(tmp_path):
    paths = ensure_run_paths("my-run", base=tmp_path)
    assert paths.root.exists()
    assert paths.charts.exists()
    assert paths.root.name == "my-run"


def test_ensure_run_paths_sanitizes_special_characters(tmp_path):
    paths = ensure_run_paths("run/with:bad<chars>", base=tmp_path)
    assert paths.root.exists()
    # No raw special characters in path name
    assert "/" not in paths.root.name
    assert ":" not in paths.root.name


def test_ensure_run_paths_rejects_empty_id(tmp_path):
    with pytest.raises(ValueError, match="run_id must be provided"):
        ensure_run_paths("", base=tmp_path)


def test_chart_file_returns_valid_path(tmp_path):
    path = chart_file("run1", "trend.png", base=tmp_path)
    assert path.parent.exists()
    assert "trend" in path.name


def test_chart_file_rejects_empty_filename(tmp_path):
    with pytest.raises(ValueError, match="filename must be provided"):
        chart_file("run1", "", base=tmp_path)


def test_chart_file_sanitizes_filename(tmp_path):
    path = chart_file("run1", "my file?.png", base=tmp_path)
    assert "?" not in path.name
