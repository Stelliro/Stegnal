from pathlib import Path

import numpy as np
from PIL import Image

from umbra.pipeline import run_pipeline


def _make_image(path: Path, size: int = 64) -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    circle = ((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2) < 0.2
    gradient = (grid_x + grid_y) / 2
    image = np.clip(gradient + circle.astype(np.float32) * 0.5, 0.0, 1.0)
    arr = (image * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_run_pipeline_saves_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    packet_path = tmp_path / "packet.npz"
    recon_path = tmp_path / "recon.png"
    _make_image(image_path)

    result = run_pipeline(
        image_path=image_path,
        seed=123,
        sigma=0.2,
        packet_path=packet_path,
        reconstruction_path=recon_path,
        denoise_sigma=0.9,
    )

    assert packet_path.exists()
    assert recon_path.exists()
    assert result.packet_path == packet_path
    assert result.reconstruction_path == recon_path
    assert 0.0 <= result.metrics.ssim <= 1.0
    assert result.metrics.psnr > 5.0

