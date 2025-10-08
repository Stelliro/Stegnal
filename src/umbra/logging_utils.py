"""Centralised logging configuration helpers for Project Umbra."""

from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path

DEFAULT_LOG_DIR = Path.home() / ".umbra" / "logs"
"""Default directory that stores the application's log files."""

_LOGGING_CONFIGURED = False


def _handler_config(filename: Path) -> dict[str, object]:
    """Return a rotating file handler configuration dictionary."""

    return {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "DEBUG",
        "filename": str(filename),
        "maxBytes": 1_048_576,  # 1 MiB per log file
        "backupCount": 5,
        "encoding": "utf-8",
        "formatter": "detailed",
    }


def configure_logging(
    log_dir: str | Path | None = None,
    *,
    console_level: int = logging.INFO,
) -> Path:
    """Configure structured logging with dedicated files per subsystem.

    The configuration is applied once per process. Subsequent calls simply
    return the directory without mutating global logging state.
    """

    global _LOGGING_CONFIGURED

    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    if _LOGGING_CONFIGURED:
        return directory

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
            "console": {
                "format": "%(levelname)s - %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.getLevelName(console_level),
                "formatter": "console",
            },
            "ui_file": _handler_config(directory / "ui.log"),
            "evolution_file": _handler_config(directory / "evolution.log"),
            "audio_file": _handler_config(directory / "audio.log"),
            "pipeline_file": _handler_config(directory / "pipeline.log"),
            "metrics_file": _handler_config(directory / "metrics.log"),
        },
        "loggers": {
            "umbra": {
                "handlers": ["console"],
                "level": logging.getLevelName(console_level),
                "propagate": False,
            },
            "umbra.ui": {
                "handlers": ["console", "ui_file"],
                "level": "DEBUG",
                "propagate": False,
            },
            "umbra.evolution": {
                "handlers": ["console", "evolution_file"],
                "level": "INFO",
                "propagate": False,
            },
            "umbra.sound": {
                "handlers": ["console", "audio_file"],
                "level": "INFO",
                "propagate": False,
            },
            "umbra.pipeline": {
                "handlers": ["console", "pipeline_file"],
                "level": "INFO",
                "propagate": False,
            },
            "umbra.metrics": {
                "handlers": ["console", "metrics_file"],
                "level": "INFO",
                "propagate": False,
            },
            "umbra.adversarial": {
                "handlers": ["console", "evolution_file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": logging.getLevelName(console_level),
        },
    }

    dictConfig(config)
    _LOGGING_CONFIGURED = True

    logging.getLogger(__name__).debug("Logging configured; output directory: %s", directory)
    return directory


__all__ = ["configure_logging", "DEFAULT_LOG_DIR"]

