from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np


def test_ensure_manager_preserves_infinite_flag(monkeypatch) -> None:
    stub_state: dict[str, object] = {"run_infinite": False}
    class _StubFrame:
        def __init__(self, data: object) -> None:
            self._data = list(data) if data is not None else []
            self.empty = len(self._data) == 0

        def replace(self, *_args, **_kwargs):
            return self

        def dropna(self, *_args, **_kwargs):
            return self

        def to_csv(self, *_args, **_kwargs) -> str:
            return ""

    def _data_frame(data: object) -> _StubFrame:
        return _StubFrame(data)

    fake_pandas = SimpleNamespace(DataFrame=_data_frame)

    fake_streamlit = SimpleNamespace(
        session_state=stub_state,
        experimental_rerun=lambda: None,
        rerun=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    ui = importlib.import_module("umbra.ui")
    try:
        from umbra.decoding import NoiseStreamDecoder
        from umbra.encoding import NoiseStreamEncoder

        original = np.zeros((8, 8), dtype=np.float32)
        encoder = NoiseStreamEncoder(sigma=0.2)
        decoder = NoiseStreamDecoder(denoise_sigma=1.0)

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=123,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is False

        stub_state["run_infinite"] = True

        ui._ensure_manager(
            original,
            encoder,
            decoder,
            population_size=2,
            seed=456,
            autosave_interval=2,
        )

        assert stub_state["run_infinite"] is True
    finally:
        sys.modules.pop("umbra.ui", None)
