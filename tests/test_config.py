"""Tests for umbra.config — load/save with validation and atomic writes."""

from __future__ import annotations

import json
import os

import pytest

from umbra.config import DEFAULT_CONFIG, load_config, save_config


@pytest.fixture()
def tmp_config(tmp_path):
    """Return a path to a temporary config file."""
    return str(tmp_path / "test_settings.json")


def test_load_returns_defaults_when_file_missing(tmp_config):
    cfg = load_config(path=tmp_config)
    assert cfg == DEFAULT_CONFIG


def test_save_then_load_round_trip(tmp_config):
    data = {**DEFAULT_CONFIG, "master_volume": 0.8, "difficulty": 0.3}
    save_config(data, path=tmp_config)
    loaded = load_config(path=tmp_config)
    assert loaded["master_volume"] == pytest.approx(0.8)
    assert loaded["difficulty"] == pytest.approx(0.3)


def test_load_clamps_out_of_range_values(tmp_config):
    bad = {"master_volume": 5.0, "difficulty": -1.0}
    with open(tmp_config, "w") as f:
        json.dump(bad, f)
    cfg = load_config(path=tmp_config)
    assert cfg["master_volume"] == 1.0
    assert cfg["difficulty"] == 0.0


def test_load_returns_defaults_on_corrupt_json(tmp_config):
    with open(tmp_config, "w") as f:
        f.write("{broken json!!!")
    cfg = load_config(path=tmp_config)
    assert cfg == DEFAULT_CONFIG


def test_load_returns_defaults_when_file_is_not_object(tmp_config):
    with open(tmp_config, "w") as f:
        json.dump([1, 2, 3], f)
    cfg = load_config(path=tmp_config)
    assert cfg == DEFAULT_CONFIG


def test_save_atomic_does_not_corrupt_on_existing(tmp_config):
    """Saving twice should leave the file with the latest data."""
    save_config({"master_volume": 0.1}, path=tmp_config)
    save_config({"master_volume": 0.9}, path=tmp_config)
    cfg = load_config(path=tmp_config)
    assert cfg["master_volume"] == pytest.approx(0.9)


def test_load_merges_partial_config_with_defaults(tmp_config):
    with open(tmp_config, "w") as f:
        json.dump({"acoustic_mode": True}, f)
    cfg = load_config(path=tmp_config)
    assert cfg["acoustic_mode"] is True
    # all other fields are defaults
    assert cfg["master_volume"] == DEFAULT_CONFIG["master_volume"]
