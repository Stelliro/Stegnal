import numpy as np

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
