# gpu_runtime.py

"""Utilities for managing optional CuPy GPU acceleration."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - Python 3.10+ ships with importlib.metadata
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for very old interpreters
    import importlib_metadata  # type: ignore

try:  # pragma: no cover - optional dependency that may be absent at runtime
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - when CuPy itself is unavailable
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
    except AttributeError:  # pragma: no cover - fallback when CuPy API changes
        _OOM_CLASSES = (MemoryError,)
else:  # pragma: no cover - runtime without CuPy
    _OOM_CLASSES = (MemoryError,)


try:
    CuPyOutOfMemoryError = _OOM_CLASSES[0]
except IndexError:  # pragma: no cover - defensive fallback

    class CuPyOutOfMemoryError(MemoryError):
        pass


def is_cupy_out_of_memory_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a CuPy out-of-memory condition."""

    return isinstance(exc, _OOM_CLASSES)


def _retain_hybrid_buffer(buffer: object) -> None:
    """Keep a strong reference to pinned buffers to avoid premature GC."""

    _HYBRID_PINNED_BUFFERS.append(buffer)


def allocate_pinned_array(shape: Iterable[int], dtype: np.dtype | type = np.float32) -> Any:
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
        array = cp.ndarray(shape, dtype=dtype_np, memptr=memptr)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback when pinned allocation fails
        array = cp.empty(shape, dtype=dtype_np)
    else:
        # Ensure the pinned owner is retained even if CuPy discards it.
        try:
            setattr(array, "_umbra_pinned_owner", pinned)
        except AttributeError:  # pragma: no cover - ndarray may disallow attributes
            pass
    return array


def ensure_nvrtc_configured() -> bool:
    """Return ``True`` if the NVRTC runtime is available and properly configured."""

    global _NVRTC_CHECKED, _NVRTC_AVAILABLE, _NVRTC_ERROR
    if _NVRTC_CHECKED:
        return _NVRTC_AVAILABLE

    if cp is None:
        _NVRTC_ERROR = ImportError("CuPy is not installed")
        _NVRTC_CHECKED = True
        return False

    try:
        from cupy.cuda import compiler  # type: ignore
        compiler.compile_using_nvrtc("void test() {}")
        _NVRTC_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - runtime dependent
        _NVRTC_ERROR = exc
        _NVRTC_AVAILABLE = False

    _NVRTC_CHECKED = True
    return _NVRTC_AVAILABLE


def _infer_cuda_version() -> tuple[int, int | None] | None:
    """Attempt to infer the required CUDA toolkit version from CuPy metadata."""

    try:
        version_str = importlib_metadata.version("cupy")
    except importlib_metadata.PackageNotFoundError:
        return None

    match = re.search(r"cuda(\d+)([a-z]?)", version_str)
    if match:
        major = int(match.group(1)) // 10
        minor_str = match.group(2)
        minor = ord(minor_str) - ord('a') if minor_str else None
        return major, minor
    return None


def recommend_cupy_install_command() -> str | None:
    """Return a suggested pip install command for the appropriate CuPy wheel."""

    # First try to detect the installed CuPy distribution name
    try:
        dists = importlib_metadata.packages_distributions()
        if "cupy" in dists:
            wheel_name = dists["cupy"][0]
            return f'pip install -U "{wheel_name}"'
    except Exception:
        pass

    if cp is None:
        return 'pip install -U "cupy-cuda12x"'

    required = _NVRTC_REQUIRED_VERSION
    if required is None:
        required = _infer_cuda_version()

    if required is None:
        return 'pip install -U "cupy-cuda12x"'

    major, minor = required
    if minor is not None:
        return f'pip install -U "cupy-cuda{major}{minor}"'
    return f'pip install -U "cupy-cuda{major}x"'


def describe_last_error() -> str | None:
    """Return a description of the most recent NVRTC configuration error."""

    if _NVRTC_ERROR is None:
        return None
    return str(_NVRTC_ERROR)


def _iter_candidate_directories():
    """Yield directories that may contain NVRTC libraries."""
    # Custom hint paths
    hints = os.environ.get("UMBRA_NVRTC_PATH_HINTS", "")
    if hints:
        for hint in hints.split(os.pathsep):
            hint_path = Path(hint.strip())
            if hint_path.is_file():
                yield hint_path.parent
            elif hint_path.is_dir():
                yield hint_path

    # CUDA_PATH / CUPY_CUDA_PATH / CUDA_HOME
    for env_var in ("CUPY_CUDA_PATH", "CUDA_PATH", "CUDA_HOME"):
        val = os.environ.get(env_var, "")
        if val:
            p = Path(val)
            if p.exists():
                yield p

    # PATH directories
    system_path = os.environ.get("PATH", "")
    if system_path:
        for entry in system_path.split(os.pathsep):
            entry = entry.strip()
            if entry:
                p = Path(entry)
                if p.is_dir():
                    yield p

    # CuPy's own directory
    try:
        import importlib.util
        spec = importlib.util.find_spec("cupy")
        if spec and spec.origin:
            yield Path(spec.origin).parent
    except Exception:
        pass


def _find_nvrtc_library(version: tuple[int, int | None] | None = None) -> Path | None:
    """Locate the NVRTC shared library on the system."""

    global _NVRTC_PATH_CACHED, _NVRTC_DETECTED_LIBRARY, _NVRTC_DETECTED_VERSION, _NVRTC_VERSION_MATCHED

    # Reset cache when called with explicit version
    if version is not None:
        _NVRTC_PATH_CACHED = False

    if _NVRTC_PATH_CACHED:
        return _NVRTC_DETECTED_LIBRARY

    required = version or _NVRTC_REQUIRED_VERSION
    if required is None:
        required = _infer_cuda_version()

    # Build candidate patterns
    if sys.platform == "win32":
        pattern = "nvrtc64_*_0.dll"
    elif sys.platform == "darwin":
        pattern = "libnvrtc.*"
    else:
        pattern = "libnvrtc.so.*"

    for base in _iter_candidate_directories():
        candidates = list(base.glob(pattern))
        if candidates:
            lib = candidates[0]
            _NVRTC_DETECTED_LIBRARY = lib
            # Parse version from filename
            _detect_version_from_library(lib)
            if required is not None and _NVRTC_DETECTED_VERSION is not None:
                _NVRTC_VERSION_MATCHED = (
                    _NVRTC_DETECTED_VERSION[0] == required[0]
                    and (required[1] is None or _NVRTC_DETECTED_VERSION[1] == required[1])
                )
            else:
                _NVRTC_VERSION_MATCHED = True
            _NVRTC_PATH_CACHED = True
            return _NVRTC_DETECTED_LIBRARY

    _NVRTC_PATH_CACHED = True
    return None


def _detect_version_from_library(lib: Path) -> None:
    """Parse version information from an NVRTC library filename."""
    global _NVRTC_DETECTED_VERSION
    name = lib.name
    # Windows: nvrtc64_130_0.dll → major=13
    m = re.search(r"nvrtc64_(\d+)(\d?)_0\.dll", name)
    if m:
        digits = m.group(1)
        if len(digits) >= 2:
            major = int(digits[:-1])
        else:
            major = int(digits)
        _NVRTC_DETECTED_VERSION = (major, None)
        return
    # Linux: libnvrtc.so.13.0 → major=13
    m = re.search(r"libnvrtc\.so\.(\d+)\.(\d+)", name)
    if m:
        _NVRTC_DETECTED_VERSION = (int(m.group(1)), int(m.group(2)))
        return
    # macOS: libnvrtc.13.0.dylib
    m = re.search(r"libnvrtc\.(\d+)\.(\d+)\.dylib", name)
    if m:
        _NVRTC_DETECTED_VERSION = (int(m.group(1)), int(m.group(2)))
        return


def describe_required_cuda_runtime() -> str | None:
    """Return a description of the CUDA runtime expected by the CuPy wheel."""

    global _NVRTC_REQUIRED_VERSION

    # Try runtimeGetVersion() API first
    if _NVRTC_REQUIRED_VERSION is None and cp is not None:
        try:
            raw = cp.cuda.runtime.runtimeGetVersion()
            major = raw // 1000
            minor = (raw % 1000) // 10
            _NVRTC_REQUIRED_VERSION = (major, minor)
        except Exception:
            pass

    # Fall back to distribution metadata
    if _NVRTC_REQUIRED_VERSION is None:
        try:
            dists = importlib_metadata.packages_distributions()
            if "cupy" in dists:
                import re as _re
                wheel_name = dists["cupy"][0]
                m = _re.search(r"cuda(\d+)(x?)", wheel_name)
                if m:
                    major = int(m.group(1))
                    if m.group(2) == "x":
                        _NVRTC_REQUIRED_VERSION = (major, None)
                    else:
                        _NVRTC_REQUIRED_VERSION = (major, None)
        except Exception:
            pass

    if _NVRTC_REQUIRED_VERSION is None:
        _NVRTC_REQUIRED_VERSION = _infer_cuda_version()

    version = _NVRTC_REQUIRED_VERSION
    if version is None:
        return None

    major, minor = version

    if sys.platform == "win32":
        if minor is not None:
            filename = f"nvrtc64_{major}{minor}_0.dll"
        else:
            filename = f"nvrtc64_{major}*_0.dll"
    elif sys.platform == "darwin":
        filename = f"libnvrtc.{major}{f'.{minor}' if minor is not None else ''}.dylib"
    else:
        if minor is not None:
            filename = f"libnvrtc.so.{major}.{minor}"
        else:
            filename = f"libnvrtc.so.{major}"

    if minor is not None:
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

    _find_nvrtc_library()  # Ensure detection has run
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