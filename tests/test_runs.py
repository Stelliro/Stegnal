"""Tests for umbra.runs — run directory management and history persistence."""

from __future__ import annotations

import json

import pytest

from umbra.runs import append_history, get_run_paths, load_history, new_run


def test_new_run_creates_directory(tmp_path):
    run_id, run_dir = new_run(base_dir=tmp_path)
    assert run_dir.exists()
    assert run_id in str(run_dir)


def test_new_run_ids_are_unique(tmp_path):
    ids = set()
    for _ in range(10):
        rid, _ = new_run(base_dir=tmp_path)
        ids.add(rid)
    assert len(ids) == 10


def test_get_run_paths_creates_directory(tmp_path):
    run_dir, history_path = get_run_paths("test_run", base_dir=tmp_path)
    assert run_dir.exists()
    assert history_path.name == "history.parquet"


def test_append_and_load_history_json_fallback(tmp_path, monkeypatch):
    """Test history persistence with the JSON fallback (no pandas)."""
    import umbra.runs as runs_mod
    monkeypatch.setattr(runs_mod, "pd", None)

    run_id = "json_test"
    append_history(run_id, {"gen": 1, "score": 0.5}, base_dir=tmp_path)
    append_history(run_id, {"gen": 2, "score": 0.7}, base_dir=tmp_path)

    history = load_history(run_id, base_dir=tmp_path)
    assert len(history) == 2
    assert history["gen"] == [1, 2]
    assert history["score"] == [0.5, 0.7]


def test_append_history_replace_mode(tmp_path, monkeypatch):
    import umbra.runs as runs_mod
    monkeypatch.setattr(runs_mod, "pd", None)

    run_id = "replace_test"
    append_history(run_id, {"gen": 1}, base_dir=tmp_path)
    append_history(run_id, {"gen": 99}, replace=True, base_dir=tmp_path)

    history = load_history(run_id, base_dir=tmp_path)
    assert len(history) == 1
    assert history["gen"] == [99]


def test_load_history_returns_empty_when_missing(tmp_path, monkeypatch):
    import umbra.runs as runs_mod
    monkeypatch.setattr(runs_mod, "pd", None)

    history = load_history("nonexistent", base_dir=tmp_path)
    assert history.empty


def test_append_history_with_dict_input(tmp_path, monkeypatch):
    import umbra.runs as runs_mod
    monkeypatch.setattr(runs_mod, "pd", None)

    run_id = "single_dict"
    append_history(run_id, {"x": 42}, base_dir=tmp_path)
    history = load_history(run_id, base_dir=tmp_path)
    assert history["x"] == [42]
