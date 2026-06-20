import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

CONFIG_FILE = "umbra_settings.json"

DEFAULT_CONFIG = {
    "audio_out_index": 0,
    "audio_in_index": 0,
    "master_volume": 0.5,
    "acoustic_mode": False,
    "auto_difficulty": False,
    "synth_link": False,
    "difficulty": 0.5
}

_NUMERIC_BOUNDS = {
    "master_volume": (0.0, 1.0),
    "difficulty": (0.0, 1.0),
}


def _validate_config(cfg: dict) -> dict:
    """Clamp numeric values to their valid ranges."""
    for key, (lo, hi) in _NUMERIC_BOUNDS.items():
        if key in cfg and isinstance(cfg[key], (int, float)):
            cfg[key] = max(lo, min(hi, float(cfg[key])))
    return cfg


def load_config(path: str | None = None) -> dict:
    filepath = path or CONFIG_FILE
    if not os.path.exists(filepath):
        return dict(DEFAULT_CONFIG)
    try:
        with open(filepath, encoding="utf-8") as f:
            user = json.load(f)
        if not isinstance(user, dict):
            logger.warning("Config file %s does not contain a JSON object; using defaults", filepath)
            return dict(DEFAULT_CONFIG)
        merged = {**DEFAULT_CONFIG, **user}
        return _validate_config(merged)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load config from %s: %s; using defaults", filepath, exc)
        return dict(DEFAULT_CONFIG)


def save_config(data: dict, path: str | None = None) -> None:
    filepath = path or CONFIG_FILE
    try:
        dir_name = os.path.dirname(os.path.abspath(filepath))
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, filepath)
        except BaseException:
            os.unlink(tmp_path)
            raise
    except OSError as exc:
        logger.error("Failed to save config to %s: %s", filepath, exc)