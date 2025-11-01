"""Utilities for managing optional CuPy GPU acceleration."""

from __future__ import annotations

import importlib.util as importlib_util
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
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
_GPU_MEMORY_POOL: Any | None = None
_PINNED_MEMORY_POOL: Any | None = None
_GPU_MEMORY_LIMIT_BYTES: int = 0

_NVRTC_CHECKED = False
_NVRTC_AVAILABLE = False
_NVRTC_ERROR: Exception | None = None
_NVRTC_PATH_CACHED = False
_NVRTC_REQUIRED_VERSION: tuple[int, int | None] | None = None
_NVRTC_DETECTED_VERSION: tuple[int, int | None] | None = None
_NVRTC_DETECTED_LIBRARY: Path | None = None
_NVRTC_VERSION_MATCHED = False

if cp is not None:  # pragma: no branch - exercised through tests via monkeypatch
    try:
        _OOM_CLASSES: tuple[type[BaseException], ...] = (
            cp.cuda.memory.OutOfMemoryError,  # type: ignore[attr-defined]
        )
    except AttributeError:  # pragma: no cover - legacy CuPy builds
        _OOM_CLASSES = (MemoryError,)
else:  # pragma: no cover - runtime without CuPy
    _OOM_CLASSES = (MemoryError,)


try:
    CuPyOutOfMemoryError = _OOM_CLASSES[0]
except IndexError:  # pragma: no cover - defensive fallback

    class CuPyOutOfMemoryError(MemoryError):
        """Fallback error mirroring CuPy's out-of-memory exception."""


def is_cupy_out_of_memory_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* represents a CuPy out-of-memory condition."""

    if isinstance(exc, _OOM_CLASSES):
        return True
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    return module.startswith("cupy.cuda") and "OutOfMemory" in name


def _retain_hybrid_buffer(buffer: object) -> None:
    """Keep a strong reference to pinned buffers to avoid premature GC."""

    _HYBRID_PINNED_BUFFERS.append(buffer)


def _memory_target_bytes(total_bytes: int, fraction: float | None) -> int:
    """Compute a desired GPU memory budget based on configuration."""

    explicit = os.getenv("UMBRA_GPU_MEMORY_TARGET_BYTES")
    if explicit:
        try:
            target = int(explicit)
        except ValueError:
            target = 0
    else:
        target = 0

    if target <= 0 and fraction is not None:
        target = int(total_bytes * max(0.0, min(fraction, 1.0)))

    if target <= 0:
        default = 12 * 1024**3  # 12 GiB default target for desktop GPUs
        target = default if total_bytes >= default else int(total_bytes * 0.9)

    return max(0, min(int(total_bytes), int(target)))


def configure_device_memory_pool(
    target_bytes: int | None = None,
    *,
    fraction: float | None = None,
) -> int:
    """Configure CuPy's memory pools to aggressively reserve GPU VRAM.

    The function returns the configured budget in bytes so callers can log the
    expected utilisation.  When CuPy is unavailable the function returns ``0``.
    """

    if cp is None:
        return 0

    device = cp.cuda.Device()  # type: ignore[attr-defined]
    free_mem, total_mem = device.mem_info  # type: ignore[attr-defined]

    budget = int(target_bytes) if target_bytes is not None else 0
    if budget <= 0:
        env_fraction = os.getenv("UMBRA_GPU_MEMORY_TARGET_FRACTION")
        if env_fraction is not None:
            try:
                fraction = float(env_fraction)
            except ValueError:
                fraction = None
        budget = _memory_target_bytes(int(total_mem), fraction)

    if budget <= 0:
        return 0

    budget = min(int(total_mem), max(int(free_mem), budget))

    pool = cp.cuda.MemoryPool()  # type: ignore[attr-defined]
    pool.set_limit(budget)
    cp.cuda.set_allocator(pool.malloc)  # type: ignore[attr-defined]

    pinned_pool = cp.cuda.PinnedMemoryPool()  # type: ignore[attr-defined]
    pinned_pool.set_limit(budget)
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)  # type: ignore[attr-defined]

    global _GPU_MEMORY_POOL, _PINNED_MEMORY_POOL, _GPU_MEMORY_LIMIT_BYTES
    _GPU_MEMORY_POOL = pool
    _PINNED_MEMORY_POOL = pinned_pool
    _GPU_MEMORY_LIMIT_BYTES = budget

    # Trigger lazy initialisation so the pools are ready for the first kernel.
    if budget > 0:
        pool.malloc(1)
        pinned_pool.malloc(1)

    return budget


def allocate_pinned_array(
    shape: Iterable[int],
    dtype: np.dtype | type = np.float32,
) -> Any:
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
        unowned = cp.cuda.memory.UnownedMemory(  # type: ignore[attr-defined]
            int(pinned.ptr), size, pinned
        )
        memptr = cp.cuda.memory.MemoryPointer(unowned, 0)  # type: ignore[attr-defined]
        array = cp.ndarray(shape, dtype=dtype_np, memptr=memptr)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback when pinned allocation fails
        array = cp.empty(shape, dtype=dtype_np)  # type: ignore[attr-defined]
    else:
        try:
            setattr(array, "_umbra_pinned_owner", pinned)
        except AttributeError:  # pragma: no cover - ndarray may disallow attributes
            pass
    return array


def _parse_distribution_version(name: str) -> tuple[int, int | None] | None:
    """Extract a CUDA major/minor pair from a CuPy wheel distribution name."""

    match = re.search(r"cupy-cuda(\d+)([a-z]*)", name)
    if not match:
        return None

    digits = match.group(1)
    suffix = match.group(2)
    if not digits:
        return None

    major = int(digits)
    minor: int | None = None
    if len(digits) > 2:
        major = int(digits[:-1])
        minor = int(digits[-1])
    elif len(digits) == 2 and suffix and suffix != "x":
        # Handles forms like cupy-cuda11a where the suffix encodes the minor.
        minor = ord(suffix[0]) - ord("a")
    elif len(digits) == 2 and suffix == "":
        minor = None

    if suffix == "x":
        minor = None

    if minor == 0:
        minor = None
    return major, minor


def _infer_cuda_version_from_runtime() -> tuple[int, int | None] | None:
    """Attempt to infer the CUDA runtime version from CuPy's runtime API."""

    if cp is None:
        return None

    try:
        runtime = cp.cuda.runtime  # type: ignore[attr-defined]
    except AttributeError:
        return None

    try:
        version = runtime.runtimeGetVersion()  # type: ignore[attr-defined]
    except Exception:
        return None

    try:
        version_int = int(version)
    except Exception:  # pragma: no cover - defensive
        return None

    major = version_int // 1000
    minor = (version_int % 1000) // 10
    if minor == 0:
        minor = None
    return major, minor


def _infer_cuda_version_from_distribution() -> tuple[int, int | None] | None:
    """Infer the CUDA toolkit requirement from the installed CuPy distribution."""

    try:
        packages = importlib_metadata.packages_distributions()
    except Exception:  # pragma: no cover - metadata query failure
        return None

    module_name = getattr(cp, "__name__", "cupy") if cp is not None else "cupy"
    distributions = packages.get(module_name)
    if not distributions:
        return None

    for dist in distributions:
        parsed = _parse_distribution_version(dist)
        if parsed is not None:
            return parsed
    return None


def _determine_required_version() -> tuple[int, int | None] | None:
    """Return and cache the CUDA version required by the current CuPy wheel."""

    global _NVRTC_REQUIRED_VERSION
    if _NVRTC_REQUIRED_VERSION is not None:
        return _NVRTC_REQUIRED_VERSION

    runtime_version = _infer_cuda_version_from_runtime()
    if runtime_version is not None:
        _NVRTC_REQUIRED_VERSION = runtime_version
        return runtime_version

    distribution_version = _infer_cuda_version_from_distribution()
    if distribution_version is not None:
        _NVRTC_REQUIRED_VERSION = distribution_version
        return distribution_version

    return None


def recommend_cupy_install_command() -> str | None:
    """Return a suggested pip command for installing a compatible CuPy wheel."""

    distribution: str | None = None

    try:
        packages = importlib_metadata.packages_distributions()
    except Exception:  # pragma: no cover - metadata query failure
        packages = {}

    if cp is not None:
        module_name = getattr(cp, "__name__", "cupy")
        distributions = packages.get(module_name)
        if distributions:
            distribution = distributions[0]

    if distribution is None:
        required = _determine_required_version()
        if required is not None:
            major, minor = required
            if minor is None:
                distribution = f"cupy-cuda{major}x"
            else:
                distribution = f"cupy-cuda{major}{minor}"
        else:
            distribution = "cupy-cuda12x"

    if distribution is None:
        return None

    return f'pip install -U "{distribution}"'


def describe_last_error() -> str | None:
    """Return a description of the most recent NVRTC configuration error."""

    if _NVRTC_ERROR is None:
        return None
    return str(_NVRTC_ERROR)


def _iter_candidate_directories() -> Iterator[Path]:
    """Yield directories that may contain NVRTC runtime libraries."""

    seen: set[Path] = set()
    queue: list[Path] = []

    def register(path: Path) -> None:
        original = path
        path = path.expanduser()
        if path.is_file():
            path = path.parent
        try:
            resolved = path.resolve(strict=False)
        except Exception:  # pragma: no cover - unusual filesystems
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        queue.append(resolved)
        logger.debug("Registered NVRTC search directory: %s", original)

    def register_env(var: str) -> None:
        value = os.environ.get(var)
        if not value:
            return
        for entry in value.split(os.pathsep):
            if entry:
                register(Path(entry))

    register_env("UMBRA_NVRTC_PATH_HINTS")
    register_env("CUPY_NVRTC_PATH")
    register_env("CUPY_CUDA_PATH")
    register_env("CUDA_PATH")
    register_env("CUDA_HOME")

    spec = importlib_util.find_spec("cupy_backends.cuda.libs")
    if spec and spec.origin:
        register(Path(spec.origin).parent)

    path_env = os.environ.get("PATH")
    if path_env:
        for entry in path_env.split(os.pathsep):
            if entry and ("cuda" in entry.lower() or "nvidia" in entry.lower()):
                register(Path(entry))

    defaults: list[Path] = []
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:/Program Files")
        defaults.extend(
            [
                Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA",
            ]
        )
    else:
        defaults.extend(
            [
                Path("/usr/local/cuda"),
                Path("/usr/lib/cuda"),
                Path("/opt/cuda"),
            ]
        )

    for base in defaults:
        register(base)
        register(base / "lib")
        register(base / "lib64")
        register(base / "bin")

    yield from queue


def _parse_nvrtc_library_name(filename: str) -> tuple[int, int | None] | None:
    """Parse a CUDA version tuple from an NVRTC library filename."""

    lower = filename.lower()
    if lower.startswith("nvrtc64_") and lower.endswith("_0.dll"):
        digits = lower[len("nvrtc64_") : -len("_0.dll")]
        if digits.isdigit() and len(digits) >= 2:
            major = int(digits[:-1])
            minor = int(digits[-1])
            if minor == 0:
                minor = None
            return major, minor
        return None

    if lower.startswith("libnvrtc.so."):
        match = re.match(r"libnvrtc\.so\.(\d+)(?:\.(\d+))?", lower)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) is not None else None
            if minor == 0:
                minor = None
            return major, minor
        return None

    if lower.startswith("libnvrtc.") and lower.endswith(".dylib"):
        match = re.match(r"libnvrtc\.(\d+)(?:\.(\d+))?\.dylib", lower)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2)) if match.group(2) is not None else None
            if minor == 0:
                minor = None
            return major, minor
        return None

    return None


def _versions_match(
    found: tuple[int, int | None] | None,
    required: tuple[int, int | None] | None,
) -> bool:
    if found is None or required is None:
        return False
    found_major, found_minor = found
    req_major, req_minor = required
    if found_major != req_major:
        return False
    if req_minor is None or found_minor is None:
        return True
    return req_minor == found_minor


def _score_candidate(
    version: tuple[int, int | None],
    required: tuple[int, int | None] | None,
) -> tuple[int, int, int]:
    match_score = 0 if _versions_match(version, required) else 1
    major_rank = -version[0]
    minor_rank = -1 if version[1] is None else -version[1]
    return (match_score, major_rank, minor_rank)


def _find_nvrtc_library(
    required: tuple[int, int | None] | None = None,
) -> Path | None:
    """Locate the NVRTC shared library on the system."""

    global _NVRTC_PATH_CACHED, _NVRTC_DETECTED_LIBRARY, _NVRTC_DETECTED_VERSION, _NVRTC_VERSION_MATCHED

    if required is None:
        required = _determine_required_version()

    best_path: Path | None = None
    best_version: tuple[int, int | None] | None = None
    best_score: tuple[int, int, int] | None = None

    for directory in _iter_candidate_directories():
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            version = _parse_nvrtc_library_name(entry.name)
            if version is None:
                continue
            score = _score_candidate(version, required)
            if best_score is None or score < best_score:
                best_path = entry
                best_version = version
                best_score = score
                if score[0] == 0:
                    break
        if best_score is not None and best_score[0] == 0:
            break

    _NVRTC_PATH_CACHED = True
    _NVRTC_DETECTED_LIBRARY = best_path
    _NVRTC_DETECTED_VERSION = best_version
    _NVRTC_VERSION_MATCHED = _versions_match(best_version, required)

    return best_path


def describe_required_cuda_runtime() -> str | None:
    """Return a description of the CUDA runtime expected by the CuPy wheel."""

    version = _determine_required_version()
    if version is None:
        return None

    major, minor = version

    if sys.platform == "win32":
        if minor is None:
            filename = f"nvrtc64_{major}*_0.dll"
        else:
            filename = f"nvrtc64_{major}{minor}_0.dll"
    elif sys.platform == "darwin":  # pragma: no cover - macOS not in CI
        if minor is None:
            filename = f"libnvrtc.{major}.dylib"
        else:
            filename = f"libnvrtc.{major}.{minor}.dylib"
    else:
        if minor is None:
            filename = f"libnvrtc.so.{major}"
        else:
            filename = f"libnvrtc.so.{major}.{minor}"

    if minor is None:
        return f"CUDA Toolkit {major}.x (NVRTC {filename})"
    return f"CUDA Toolkit {major}.{minor} (NVRTC {filename})"


def describe_detected_cuda_runtime() -> str | None:
    """Return a description of the NVRTC runtime discovered on the system."""

    library = _find_nvrtc_library()
    version = _NVRTC_DETECTED_VERSION

    if library is None and version is None:
        return None

    filename = library.name if library is not None else "NVRTC"

    if version is None:
        return f"NVRTC library {filename}"

    major, minor = version
    if minor is None:
        version_text = f"CUDA Toolkit {major}.x"
    else:
        version_text = f"CUDA Toolkit {major}.{minor}"
    return f"{version_text} (NVRTC {filename})"


def nvrtc_version_matches_requirement() -> bool | None:
    """Return ``True`` if the detected NVRTC version matches CuPy's needs."""

    if _NVRTC_DETECTED_VERSION is None or _NVRTC_REQUIRED_VERSION is None:
        return None
    return _NVRTC_VERSION_MATCHED


def ensure_nvrtc_configured() -> bool:
    """Return ``True`` if the NVRTC runtime is available and properly configured."""

    global _NVRTC_CHECKED, _NVRTC_AVAILABLE, _NVRTC_ERROR

    if _NVRTC_CHECKED:
        return _NVRTC_AVAILABLE

    if cp is None:
        _NVRTC_ERROR = ImportError("CuPy is not installed")
        _NVRTC_AVAILABLE = False
        _NVRTC_CHECKED = True
        return False

    try:
        from cupy_backends.cuda.libs import nvrtc as nvrtc_lib  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime
        _NVRTC_ERROR = exc
        _NVRTC_AVAILABLE = False
    else:
        try:
            version_info = nvrtc_lib.getVersion()  # type: ignore[attr-defined]
            if isinstance(version_info, tuple):
                major = int(version_info[0])
                minor = int(version_info[1]) if len(version_info) > 1 else None
            else:
                value = int(version_info)
                major = value // 1000
                minor = (value % 1000) // 10
            if minor == 0:
                minor = None
            if major:
                global _NVRTC_DETECTED_VERSION
                if _NVRTC_DETECTED_VERSION is None:
                    _NVRTC_DETECTED_VERSION = (major, minor)
            _NVRTC_ERROR = None
            _NVRTC_AVAILABLE = True
            _find_nvrtc_library(_NVRTC_DETECTED_VERSION or (major, minor))
        except Exception as exc:  # pragma: no cover - depends on runtime
            _NVRTC_ERROR = exc
            _NVRTC_AVAILABLE = False

    _NVRTC_CHECKED = True
    return _NVRTC_AVAILABLE


def require_gpu(operation: str) -> None:
    """Ensure a GPU backend is available for *operation* or raise an error."""

    if cp is None:
        raise GPUAccelerationRequiredError(
            f"GPU acceleration via CuPy is required for {operation}; CPU fallback is disabled."
        )

    if getattr(cp, "_umbra_skip_nvrtc_check", False):  # pragma: no cover - set in tests
        return

    if ensure_nvrtc_configured():
        return

    detail = describe_last_error()
    requirement = describe_required_cuda_runtime()
    detected = describe_detected_cuda_runtime()
    matches = nvrtc_version_matches_requirement()

    hint = "CuPy is installed but failed to load the CUDA NVRTC runtime."
    if requirement:
        hint = f"{hint} The installed wheel expects {requirement}."
    if detected:
        if matches is False:
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
    "cp",
    "is_cupy_out_of_memory_error",
    "require_gpu",
    "ensure_nvrtc_configured",
    "recommend_cupy_install_command",
    "describe_last_error",
    "describe_required_cuda_runtime",
    "describe_detected_cuda_runtime",
    "nvrtc_version_matches_requirement",
]
