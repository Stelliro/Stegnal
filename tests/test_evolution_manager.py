import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager


def test_evolution_manager_runs_multiple_generations() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((32, 32), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.2)
    decoder = NoiseStreamDecoder(denoise_sigma=0.9)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=3,
        base_seed=123,
        autosave_interval=2,
    )

    for _ in range(3):
        manager.run_generation()

    assert len(manager.generations) == 3
    best_metrics = [record.best_candidate.metrics for record in manager.generations]
    assert all(np.isfinite(metric.psnr) for metric in best_metrics)
    assert all(0.0 <= metric.ssim <= 1.0 for metric in best_metrics)
