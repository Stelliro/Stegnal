"""Centralised logging configuration helpers for Project Umbra."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from logging.config import dictConfig
from pathlib import Path
from typing import Any

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


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON payloads for metric aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


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
            "json": {
                "()": JsonFormatter,
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
            "pipeline_file": {**_handler_config(directory / "pipeline.log"), "formatter": "json"},
            "metrics_file": {**_handler_config(directory / "metrics.log"), "formatter": "json"},
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


def _repo_root() -> Path | None:
    """Return the repository root if the current file lives inside a git repo."""

    current = Path(__file__).resolve()
    for ancestor in (current,) + tuple(current.parents):
        if (ancestor / ".git").exists():
            return ancestor
    return None


def _safe_run(command: list[str], cwd: Path | None = None) -> str | None:
    """Execute ``command`` and return stripped stdout or ``None`` on failure."""

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    output = result.stdout.strip()
    return output or None


def collect_provenance(config_hash: str | None = None) -> dict[str, object]:
    """Gather runtime provenance information for reproducibility breadcrumbs.

    The helper is intentionally defensive: failures to query optional binaries
    or GPU metadata return ``"UNKNOWN"`` or ``None`` instead of raising.
    """

    repo = _repo_root()
    git_hash = "UNKNOWN"
    if repo is not None:
        result = _safe_run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
        if result:
            git_hash = result

    binary_versions: dict[str, object] = {
        "python": platform.python_version(),
        "cupy": "UNKNOWN",
        "cuda": "UNKNOWN",
        "cudnn": "UNKNOWN",
        "nvidia_driver": "UNKNOWN",
    }

    device_info: dict[str, object] = {
        "gpu_name": "UNKNOWN",
        "gpu_count": 0,
        "compute_capability": "UNKNOWN",
    }

    try:
        import cupy as cp  # type: ignore

        binary_versions["cupy"] = getattr(cp, "__version__", "UNKNOWN")

        try:
            cuda_version = int(cp.cuda.runtime.runtimeGetVersion())
            major, minor = divmod(cuda_version, 1000)
            minor, patch = divmod(minor, 10)
            binary_versions["cuda"] = f"{major}.{minor}.{patch}"
        except Exception:
            pass

        try:
            cudnn_version = int(cp.cuda.cudnn.getVersion())
            binary_versions["cudnn"] = str(cudnn_version)
        except Exception:
            pass

        try:
            driver_version = int(cp.cuda.runtime.driverGetVersion())
            driver_major, driver_minor = divmod(driver_version, 100)
            binary_versions["nvidia_driver"] = f"{driver_major}.{driver_minor}"
        except Exception:
            pass

        try:
            gpu_count = int(cp.cuda.runtime.getDeviceCount())
            device_info["gpu_count"] = gpu_count
            if gpu_count > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                name = props.get("name") if isinstance(props, dict) else None
                if isinstance(name, bytes):
                    name = name.decode(errors="ignore")
                device_info["gpu_name"] = name or "UNKNOWN"
                major = props.get("major") if isinstance(props, dict) else None
                minor = props.get("minor") if isinstance(props, dict) else None
                if major is not None and minor is not None:
                    device_info["compute_capability"] = f"{major}.{minor}"
        except Exception:
            pass
    except Exception:
        # CuPy or CUDA is not available; leave UNKNOWN defaults in place.
        pass

    provenance: dict[str, Any] = {
        "git_hash": git_hash,
        "binary_versions": binary_versions,
        "device": device_info,
        "config_hash": config_hash,
    }
    return provenance


__all__ = ["configure_logging", "collect_provenance", "DEFAULT_LOG_DIR", "JsonFormatter"]