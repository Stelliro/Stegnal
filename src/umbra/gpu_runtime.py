"""Utilities for managing optional CuPy GPU acceleration."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

try:  # pragma: no cover - Python 3.10+ ships with importlib.metadata
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover - fallback for very old interpreters
    import importlib_metadata  # type: ignore

try:  # pragma: no cover - optional dependency that may be absent at runtime
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - when CuPy itself is unavailable
    cp = None  # type: ignore


class GPUAccelerationRequiredError(RuntimeError):
    """Raised when GPU execution is required but no accelerator is available."""

logger = logging.getLogger(__name__)

_HYBRID_PINNED_BUFFERS: list[object] = []

_NVRTC_CHECKED = False
_NVRTC_AVAILABLE = False
_NVRTC_ERROR: Exception | None = None
_NVRTC_PATH_CACHED = False
_NVRTC_REQUIRED_VERSION: tuple[int, int | None] | None = None
_NVRTC_DETECTED_VERSION: tuple[int, int | None] | None = None
_NVRTC_DETECTED_LIBRARY: Path | None = None
_NVRTC_VERSION_MATCHED = False

if cp is not None:
    try:  # pragma: no cover - attribute may be missing on older CuPy builds
        _OOM_CLASSES: tuple[type[BaseException], ...] = (
            cp.cuda.memory.OutOfMemoryError,  # type: ignore[attr-defined]
        )
    except Exception:  # pragma: no cover - fallback when CuPy API changes
        _OOM_CLASSES = (MemoryError,)
else:  # pragma: no cover - runtime without CuPy
    _OOM_CLASSES = (MemoryError,)


try:
    CuPyOutOfMemoryError = _OOM_CLASSES[0]
except Exception:  # pragma: no cover - defensive fallback

    class CuPyOutOfMemoryError(MemoryError):
        pass


def is_cupy_out_of_memory_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a CuPy out-of-memory condition."""

    return isinstance(exc, _OOM_CLASSES)


def _retain_hybrid_buffer(buffer: object) -> None:
    """Keep a strong reference to pinned buffers to avoid premature GC."""

    _HYBRID_PINNED_BUFFERS.append(buffer)


def allocate_pinned_array(shape: Iterable[int], dtype: np.dtype | type = np.float32):
    """Allocate a CuPy array backed by pinned host memory."""

    if cp is None:
        raise GPUAccelerationRequiredError(
            "GPU acceleration via CuPy is required for hybrid pinned allocations."
        )

    dtype_np = np.dtype(dtype)
    size = int(np.prod(tuple(int(dim) for dim in shape))) * int(dtype_np.itemsize)

    try:
        pinned = cp.cuda.alloc_pinned_memory(size)  # type: ignore[attr-defined]
        _retain_hybrid_buffer(pinned)
        mem = cp.cuda.memory.UnownedMemory(int(pinned.ptr), size, pinned)  # type: ignore[attr-defined]
        memptr = cp.cuda.memory.MemoryPointer(mem, 0)  # type: ignore[attr-defined]
        try:
            array = cp.ndarray(shape, dtype=dtype_np, memptr=memptr)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for unusual CuPy builds
            array = cp.empty(shape, dtype=dtype_np)  # pragma: no cover
    except Exception:  # pragma: no cover - fallback when pinned allocation fails
        array = cp.empty(shape, dtype=dtype_np)
    else:
        # Ensure the pinned owner is retained even if CuPy discards it.
        try:
            setattr(array, "_umbra_pinned_owner", pinned)
        except Exception:  # pragma: no cover - ndarray may disallow attributes
            _retain_hybrid_buffer((array, pinned))

    return array


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

    hints = os.environ.get("UMBRA_NVRTC_PATH_HINTS")
    if hints:
        for raw_hint in hints.split(os.pathsep):
            if not raw_hint:
                continue
            hint_path = Path(raw_hint)
            if hint_path.suffix:
                hint_path = hint_path.parent
            search_roots.append(hint_path)

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

    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
    except Exception:  # pragma: no cover - torch may not be importable
        torch_spec = None

    if torch_spec and torch_spec.origin:
        torch_root = Path(torch_spec.origin).resolve().parent
        search_roots.extend(
            path for path in (torch_root, torch_root / "lib") if path not in search_roots
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


def _matches_required_version(candidate: Path, required: tuple[int, int | None]) -> bool:
    """Return ``True`` if *candidate* matches the requested NVRTC version."""

    major, minor = required
    name = candidate.name.lower()

    if sys.platform == "win32":
        base_token = f"nvrtc64_{major}"
        if base_token not in name:
            return False
        if minor is None:
            return True
        version_digits = major * 10 + minor
        return f"nvrtc64_{version_digits}" in name

    if sys.platform == "darwin":  # pragma: no cover - macOS not in CI
        if "libnvrtc" not in name:
            return False
        if minor is None:
            return f".{major}" in name
    return f".{major}.{minor}" in name


def _parse_nvrtc_version(path: Path) -> tuple[int, int | None] | None:
    """Best-effort extraction of the CUDA version from an NVRTC library name."""

    name = path.name.lower()

    if sys.platform == "win32":
        match = re.search(r"nvrtc64_(\d+)", name)
        if not match:
            return None
        digits = int(match.group(1))
        major = digits // 10
        minor_val = digits % 10
        minor = minor_val if minor_val != 0 else None
        return major, minor

    match = re.search(r"libnvrtc(?:\.so|\.dylib)\.(\d+)(?:\.(\d+))?", name)
    if not match:
        return None

    major = int(match.group(1))
    minor = match.group(2)
    minor_version: int | None = int(minor) if minor is not None else None
    if minor_version == 0:
        minor_version = None
    return major, minor_version

    # Linux / other POSIX platforms
    if "libnvrtc" not in name:
        return False
    if minor is None:
        return f".{major}" in name
    return f".{major}.{minor}" in name or f".{major}{minor}" in name


def _find_nvrtc_library(required: tuple[int, int | None] | None = None) -> Path | None:
    """Locate the NVRTC shared library across common installation paths."""

    global _NVRTC_DETECTED_LIBRARY, _NVRTC_DETECTED_VERSION, _NVRTC_VERSION_MATCHED

    _NVRTC_DETECTED_LIBRARY = None
    _NVRTC_DETECTED_VERSION = None
    _NVRTC_VERSION_MATCHED = False

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

        preferred: list[tuple[Path, tuple[int, int | None] | None]] = []
        fallback: list[tuple[Path, tuple[int, int | None] | None]] = []

        for candidate in matches:
            if not candidate.is_file():
                continue

            version = _parse_nvrtc_version(candidate)
            if required is not None and _matches_required_version(candidate, required):
                preferred.append((candidate, version))
            else:
                fallback.append((candidate, version))

        if preferred:
            library, version = preferred[0]
            _NVRTC_DETECTED_LIBRARY = library
            _NVRTC_DETECTED_VERSION = version
            _NVRTC_VERSION_MATCHED = True
            return library
        if fallback:
            library, version = fallback[0]
            _NVRTC_DETECTED_LIBRARY = library
            _NVRTC_DETECTED_VERSION = version
            _NVRTC_VERSION_MATCHED = required is None
            return library
    return None


def _infer_cuda_version() -> tuple[int, int | None] | None:
    """Best-effort detection of the CUDA major/minor expected by CuPy."""

    global _NVRTC_REQUIRED_VERSION

    if cp is None:
        return None

    try:
        version = cp.cuda.runtime.runtimeGetVersion()
    except Exception:
        version = None

    if isinstance(version, int) and version > 0:
        major = version // 1000
        minor_val = (version % 1000) // 10
        minor = minor_val if minor_val != 0 else None
        _NVRTC_REQUIRED_VERSION = (major, minor)
        return _NVRTC_REQUIRED_VERSION

    distribution_name = _detect_cupy_distribution_name()
    if not distribution_name:
        return None

    normalized = distribution_name.lower()
    for major in range(20, 8, -1):
        token = f"cuda{major}"
        if token not in normalized:
            continue

        suffix = normalized.split(token, 1)[1]
        match = re.match(r"(\d+)", suffix)
        if match:
            minor_val = int(match.group(1))
            minor_version: int | None = minor_val if minor_val != 0 else None
        else:
            minor_version = None

        _NVRTC_REQUIRED_VERSION = (major, minor_version)
        return _NVRTC_REQUIRED_VERSION

    return None


def _configure_nvrtc_path() -> None:
    """Populate ``CUPY_NVRTC_PATH`` with a discovered runtime when possible."""

    global _NVRTC_PATH_CACHED

    if cp is None or _NVRTC_PATH_CACHED:
        return

    if os.environ.get("CUPY_NVRTC_PATH"):
        _NVRTC_PATH_CACHED = True
        return

    required = _infer_cuda_version()
    library = _find_nvrtc_library(required)
    if library is None:
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


def describe_required_cuda_runtime() -> str | None:
    """Return guidance for the CUDA toolkit version CuPy expects."""

    if cp is None:
        return None

    if _NVRTC_REQUIRED_VERSION is None or _NVRTC_REQUIRED_VERSION[1] is None:
        version = _infer_cuda_version()
    else:
        version = _NVRTC_REQUIRED_VERSION

    if version is None:
        return None

    major, minor = version

    if sys.platform == "win32":
        if minor is not None:
            version_digits = major * 10 + minor
            filename = f"nvrtc64_{version_digits}_0.dll"
        else:
            filename = f"nvrtc64_{major}*_0.dll"
    elif sys.platform == "darwin":  # pragma: no cover - macOS not in CI
        filename = f"libnvrtc.{major}{f'.{minor}' if minor is not None else ''}.dylib"
    else:
        if minor is not None:
            filename = f"libnvrtc.so.{major}.{minor}"
        else:
            filename = f"libnvrtc.so.{major}"

    if minor is not None and minor != 0:
        return f"CUDA Toolkit {major}.{minor} (NVRTC {filename})"

    return f"CUDA Toolkit {major}.x (NVRTC {filename})"


def describe_detected_cuda_runtime() -> str | None:
    """Return a description of the NVRTC runtime discovered on the system."""

    library = _NVRTC_DETECTED_LIBRARY
    version = _NVRTC_DETECTED_VERSION

    if library is None and version is None:
        return None

    filename = library.name if library is not None else "NVRTC"

    if version is None:
        return f"NVRTC library {filename}"

    major, minor = version
    if minor is not None:
        version_text = f"CUDA Toolkit {major}.{minor}"
    else:
        version_text = f"CUDA Toolkit {major}.x"

    return f"{version_text} (NVRTC {filename})"


def nvrtc_version_matches_requirement() -> bool | None:
    """Return ``True`` if the detected NVRTC version matches CuPy's needs."""

    if _NVRTC_REQUIRED_VERSION is None or _NVRTC_DETECTED_VERSION is None:
        return None

    return _NVRTC_VERSION_MATCHED


def require_gpu(operation: str) -> None:
    """Ensure a GPU backend is available for *operation* or raise an error."""

    if cp is None:
        raise GPUAccelerationRequiredError(
            f"GPU acceleration via CuPy is required for {operation}; CPU fallback is disabled."
        )

    if getattr(cp, "_umbra_skip_nvrtc_check", False):  # pragma: no cover - exercised in tests
        return

    if ensure_nvrtc_configured():
        return

    detail = describe_last_error()
    requirement = describe_required_cuda_runtime()
    detected = describe_detected_cuda_runtime()
    matches_requirement = nvrtc_version_matches_requirement()
    hint = "CuPy is installed but failed to load the CUDA NVRTC runtime."
    if requirement:
        hint = f"{hint} The installed wheel expects {requirement}."
    if detected:
        if matches_requirement is False:
            hint = f"{hint} Detected {detected}, which does not satisfy the requirement."
        else:
            hint = f"{hint} Detected {detected}."
    hint = f"{hint} Install the matching CUDA toolkit or allow CPU fallback."
    install_hint = recommend_cupy_install_command()
    if install_hint:
        hint = f"{hint} Try reinstalling CuPy with `{install_hint}`."
    if detail:
        hint = f"{hint} (Detail: {detail})"
    raise GPUAccelerationRequiredError(hint)


__all__ = [
    "cp",
    "GPUAccelerationRequiredError",
    "CuPyOutOfMemoryError",
    "allocate_pinned_array",
    "is_cupy_out_of_memory_error",
    "require_gpu",
    "ensure_nvrtc_configured",
    "recommend_cupy_install_command",
    "describe_last_error",
    "describe_required_cuda_runtime",
    "describe_detected_cuda_runtime",
    "nvrtc_version_matches_requirement",
]

