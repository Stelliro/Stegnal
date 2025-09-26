"""Tests for the evolutionary search helpers."""

from __future__ import annotations

import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager


def test_evolution_generation_and_persistence(tmp_path) -> None:
    image = np.full((8, 8), 0.5, dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.05)
    decoder = NoiseStreamDecoder(denoise_sigma=None)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=3,
        base_seed=42,
        autosave_interval=1,
    )

    record = manager.run_generation()
    assert len(record.candidates) == 3

    save_path = manager.save(tmp_path)
    assert save_path.exists()

    restored = EvolutionManager.load(tmp_path)
    assert len(restored.generations) == len(manager.generations)
    assert restored.image_signature == manager.image_signature
    assert restored.population_size == manager.population_size
    assert restored.autosave_interval == manager.autosave_interval


def test_update_settings_preserves_history() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    manager = EvolutionManager(
        original=image,
        encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(denoise_sigma=None),
        population_size=1,
        base_seed=7,
        autosave_interval=2,
    )

    manager.run_generation()
    manager.update_settings(population_size=5)

    assert len(manager.generations) == 1
    assert manager.population_size == 5
