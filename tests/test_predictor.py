import numpy as np

import umbra.predictor as predictor


def test_predictor_falls_back_without_torch(monkeypatch) -> None:
    monkeypatch.setattr(predictor, "torch", None)

    waveform = np.random.default_rng(0).random(2048, dtype=np.float32)
    image = predictor.predict_image_from_waveform(
        waveform,
        sample_rate=2048,
        resolution=(16, 16),
        model=None,
    )

    assert image.shape == (16, 16, 3)
    assert image.dtype == np.float32


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def view(self, *_shape) -> "_FakeTensor":
        self._array = self._array.reshape(*_shape)
        return self

    def to(self, *_args, **_kwargs) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeNoGrad:
    def __enter__(self) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, *_exc) -> None:  # pragma: no cover - trivial
        return None


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def as_tensor(array: np.ndarray) -> _FakeTensor:
        return _FakeTensor(array)

    @staticmethod
    def no_grad() -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeModel:
    def __call__(self, *_args, **kwargs) -> np.ndarray:
        rows, cols = kwargs.get("resolution", (4, 4))
        return np.full((1, 3, rows, cols), 0.5, dtype=np.float32)


def test_predictor_uses_model_when_torch_available(monkeypatch) -> None:
    monkeypatch.setattr(predictor, "torch", _FakeTorch())

    waveform = np.random.default_rng(1).random(1024, dtype=np.float32)
    image = predictor.predict_image_from_waveform(
        waveform,
        sample_rate=1024,
        resolution=(4, 4),
        model=_FakeModel(),
    )

    assert image.shape == (4, 4, 3)
    assert np.allclose(image, 0.5, atol=1e-6)

