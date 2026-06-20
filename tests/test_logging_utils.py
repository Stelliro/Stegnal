"""Tests for umbra.logging_utils — structured logging and provenance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from umbra.logging_utils import JsonFormatter, configure_logging


def test_json_formatter_emits_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"


def test_json_formatter_includes_custom_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="", lineno=0,
        msg="test", args=(), exc_info=None,
    )
    record.run_id = "abc123"  # type: ignore[attr-defined]
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["run_id"] == "abc123"


def test_configure_logging_creates_directory(tmp_path):
    log_dir = tmp_path / "logs"
    result = configure_logging(log_dir=log_dir)
    assert result == log_dir
    assert log_dir.exists()


def test_configure_logging_idempotent(tmp_path):
    """Calling configure_logging twice should not raise."""
    log_dir = tmp_path / "logs"
    configure_logging(log_dir=log_dir)
    configure_logging(log_dir=log_dir)
    assert log_dir.exists()
