"""Utilities for managing optional CuPy GPU acceleration."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

try:  # pragma: no cover - Python 3.10+ ships with importlib.metadata
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover - fallback for very old interpreters
    import importlib_metadata  # type: ignore

try:  # pragma: no cover - optional dependency that may be absent at runtime
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - when CuPy itself is unavailable
    cp = None  # type: ignore

logger = logging.getLogger(__name__)

_NVRTC_CHECKED = False
_NVRTC_AVAILABLE = False
_NVRTC_ERROR: Exception | None = None
_NVRTC_PATH_CACHED = False


def _detect_cupy_distribution_name() -> str | None:
    """Return the installed distribution name that provides :mod:`cupy`."""

    if cp is None:
        return None

    try:
        packages = importlib_metadata.packages_distributions()
    except Exception:  # pragma: no cover - importlib metadata may be missing
        packages = {}

    for package_name in (getattr(cp, "__name__", "cupy"), "cupy"):
        distributions = packages.get(package_name)
        if distributions:
            return distributions[0]

    try:
        distribution = importlib_metadata.distribution("cupy")
        return distribution.metadata.get("Name")
    except Exception:  # pragma: no cover - distribution metadata may be absent
        return None


def recommend_cupy_install_command() -> str | None:
    """Return a pip command that installs the detected CuPy wheel."""

    if cp is None:
        # Without CuPy we cannot infer the wheel; suggest the CUDA 12 build.
        return 'pip install -U "cupy-cuda12x"'

    distribution_name = _detect_cupy_distribution_name()
    if not distribution_name:
        return 'pip install -U "cupy-cuda12x"'

    normalized = distribution_name.lower()
    if normalized.startswith("cupy"):
        return f'pip install -U "{distribution_name}"'

    return 'pip install -U "cupy-cuda12x"'


def _iter_candidate_directories() -> Iterable[Path]:
    """Yield directories that are likely to contain the NVRTC runtime."""

    search_roots: list[Path] = []

    for env_var in ("CUPY_CUDA_PATH", "CUDA_PATH", "CUDA_HOME"):
        candidate = os.environ.get(env_var)
        if candidate:
            root = Path(candidate)
            search_roots.extend(
                path
                for path in (
                    root,
                    root / "bin",
                    root / "lib64",
                    root / "lib",
                )
            )

    if sys.platform == "win32":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if cuda_root.exists():
            for version_dir in sorted(cuda_root.glob("v*"), reverse=True):
                search_roots.extend(
                    path
                    for path in (
                        version_dir,
                        version_dir / "bin",
                        version_dir / "lib",
                    )
                )
    else:
        search_roots.extend(
            Path(path)
            for path in ("/usr/local/cuda", "/usr/local/cuda/lib64", "/usr/local/cuda/lib")
        )

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        search_roots.append(Path(entry))

    seen: set[Path] = set()
    for root in search_roots:
        try:
            resolved = root.resolve()
        except Exception:  # pragma: no cover - permission issues
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            yield resolved


def _find_nvrtc_library() -> Path | None:
    """Locate the NVRTC shared library across common installation paths."""

    if sys.platform == "win32":
        pattern = "nvrtc64*.dll"
    elif sys.platform == "darwin":  # pragma: no cover - macOS not in CI
        pattern = "libnvrtc*.dylib"
    else:
        pattern = "libnvrtc.so*"

    for directory in _iter_candidate_directories():
        try:
            matches = sorted(directory.glob(pattern), reverse=True)
        except Exception:  # pragma: no cover - permission issues
            continue
        for candidate in matches:
            if candidate.is_file():
                return candidate
    return None


def _configure_nvrtc_path() -> None:
    """Populate ``CUPY_NVRTC_PATH`` with a discovered runtime when possible."""

    global _NVRTC_PATH_CACHED

    if cp is None or _NVRTC_PATH_CACHED:
        return

    if os.environ.get("CUPY_NVRTC_PATH"):
        _NVRTC_PATH_CACHED = True
        return

    library = _find_nvrtc_library()
    if library is None:
        return

    os.environ["CUPY_NVRTC_PATH"] = str(library)
    _NVRTC_PATH_CACHED = True
    logger.debug("Configured CUPY_NVRTC_PATH to %s", library)


def ensure_nvrtc_configured() -> bool:
    """Return ``True`` when NVRTC is accessible for the active CuPy build."""

    global _NVRTC_CHECKED, _NVRTC_AVAILABLE, _NVRTC_ERROR

    if cp is None:
        _NVRTC_CHECKED = True
        _NVRTC_AVAILABLE = False
        return False

    if getattr(cp, "_umbra_skip_nvrtc_check", False):  # pragma: no cover - test hook
        _NVRTC_CHECKED = True
        _NVRTC_AVAILABLE = True
        _NVRTC_ERROR = None
        return True

    if _NVRTC_CHECKED:
        return _NVRTC_AVAILABLE

    _configure_nvrtc_path()

    try:
        if hasattr(cp, "cuda"):
            from cupy_backends.cuda.libs import nvrtc  # type: ignore

            nvrtc.getVersion()
        _NVRTC_AVAILABLE = True
        _NVRTC_ERROR = None
    except Exception as exc:  # pragma: no cover - depends on runtime setup
        _NVRTC_AVAILABLE = False
        _NVRTC_ERROR = exc
        logger.debug("NVRTC validation failed", exc_info=True)

    _NVRTC_CHECKED = True
    return _NVRTC_AVAILABLE


def describe_last_error() -> str | None:
    """Return a human-readable description of the last NVRTC failure."""

    if _NVRTC_ERROR is None:
        return None
    return f"{_NVRTC_ERROR}"

