import logging

import numpy as np
import pytest

import umbra.encoding as encoding
from umbra.encoding import NoiseStreamEncoder


class _FakeCuPyOOMError(RuntimeError):
    """Test helper standing in for CuPy's OOM error."""


def test_encode_gpu_oom_fallback(monkeypatch, caplog):
    """GPU OOM should fall back to CPU when allowed."""

    class FakeCuPy:
        def asarray(self, *_args, **_kwargs):
            raise _FakeCuPyOOMError("OOM")

    monkeypatch.setattr(encoding, "cp", FakeCuPy())
    monkeypatch.setattr(encoding, "CuPyOutOfMemoryError", _FakeCuPyOOMError)
    monkeypatch.setattr(
        encoding,
        "is_cupy_out_of_memory_error",
        lambda exc: isinstance(exc, _FakeCuPyOOMError),
    )

    caplog.set_level(logging.DEBUG, logger="umbra.encoding")
    encoder = NoiseStreamEncoder(sigma=0.05)
    image = np.full((4, 4), 0.5, dtype=np.float32)

    packet = encoder.encode(image, seed=42, use_gpu=True, allow_cpu_fallback=True)

    assert isinstance(packet.encoded, np.ndarray)
    assert any("falling back to CPU" in record.message for record in caplog.records)


def test_encode_gpu_oom_without_fallback(monkeypatch):
    class FakeCuPy:
        def asarray(self, *_args, **_kwargs):
            raise _FakeCuPyOOMError("OOM")

    monkeypatch.setattr(encoding, "cp", FakeCuPy())
    monkeypatch.setattr(encoding, "CuPyOutOfMemoryError", _FakeCuPyOOMError)
    monkeypatch.setattr(
        encoding,
        "is_cupy_out_of_memory_error",
        lambda exc: isinstance(exc, _FakeCuPyOOMError),
    )

    encoder = NoiseStreamEncoder(sigma=0.05)
    image = np.full((2, 2), 0.2, dtype=np.float32)

    with pytest.raises(_FakeCuPyOOMError):
        encoder.encode(image, seed=7, use_gpu=True, allow_cpu_fallback=False)
