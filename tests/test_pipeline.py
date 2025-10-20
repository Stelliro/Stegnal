import numpy as np
import pytest

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoisePacket, NoiseStreamEncoder
from umbra.metrics import compute_metrics


def create_test_image(size: int = 64) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    circle = ((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2) < 0.2
    gradient = (grid_x + grid_y) / 2
    image = np.clip(gradient + circle.astype(np.float32) * 0.5, 0.0, 1.0)
    return image.astype(np.float32)


def test_encode_decode_round_trip(tmp_path):
    encoder = NoiseStreamEncoder(sigma=0.15)
    decoder = NoiseStreamDecoder(denoise_sigma=0.8)

    image = create_test_image()
    packet = encoder.encode(image, seed=42)
    packet_path = tmp_path / "packet.npz"
    packet.to_file(packet_path)

    loaded = NoisePacket.from_file(packet_path)
    assert loaded.permutation_seed == packet.permutation_seed
    assert loaded.image_shape == packet.image_shape
    assert np.allclose(loaded.encoded, packet.encoded)

    decoded = decoder.decode(loaded, seed=42)
    metrics = compute_metrics(image, decoded)

    assert metrics.psnr > 18
    assert metrics.ssim > 0.55


def test_encode_requires_gpu_when_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.1)

    import umbra.encoding as encoding

    monkeypatch.setattr(encoding, "cp", None, raising=False)

    from umbra.reconstruction import GPUAccelerationRequiredError

    with pytest.raises(GPUAccelerationRequiredError):
        encoder.encode(image, seed=7, allow_cpu_fallback=False)


def test_simulate_uwb_channel_gpu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import umbra.encoding as encoding

    class _FailingCuPyStub:
        _umbra_skip_nvrtc_check = True
        float32 = np.float32

        @staticmethod
        def asarray(*args, **kwargs):  # type: ignore[override]
            raise RuntimeError("nvrtc missing")

    monkeypatch.setattr(encoding, "cp", _FailingCuPyStub, raising=False)

    from umbra.reconstruction import GPUAccelerationRequiredError

    with pytest.raises(GPUAccelerationRequiredError):
        encoding._simulate_uwb_channel(
            np.ones(16, dtype=np.float32),
            np.random.default_rng(0),
            allow_cpu_fallback=False,
            prefer_gpu=True,
        )


def test_ensure_gpu_available_missing_nvrtc(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import umbra.encoding as encoding

    stub = types.SimpleNamespace(_umbra_skip_nvrtc_check=False, cuda=types.SimpleNamespace())
    monkeypatch.setattr(encoding, "cp", stub, raising=False)

    import umbra.gpu_runtime as gpu_runtime

    monkeypatch.setattr(gpu_runtime, "cp", stub, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_CHECKED", False, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_AVAILABLE", False, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_ERROR", None, raising=False)
    monkeypatch.setattr(gpu_runtime, "_NVRTC_PATH_CACHED", False, raising=False)

    nvrtc_module = types.ModuleType("cupy_backends.cuda.libs.nvrtc")

    def _raise_missing():
        raise RuntimeError("missing NVRTC runtime")

    nvrtc_module.getVersion = _raise_missing  # type: ignore[attr-defined]

    libs_module = types.ModuleType("cupy_backends.cuda.libs")
    libs_module.nvrtc = nvrtc_module  # type: ignore[attr-defined]

    cuda_module = types.ModuleType("cupy_backends.cuda")
    cuda_module.libs = libs_module  # type: ignore[attr-defined]

    backends_module = types.ModuleType("cupy_backends")
    backends_module.cuda = cuda_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cupy_backends", backends_module)
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda.libs", libs_module)
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda.libs.nvrtc", nvrtc_module)

    from umbra.reconstruction import GPUAccelerationRequiredError

    with pytest.raises(GPUAccelerationRequiredError):
        encoding._ensure_gpu_available("validation")
