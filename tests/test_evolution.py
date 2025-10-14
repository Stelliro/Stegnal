"""Tests for the evolutionary search helpers."""

from __future__ import annotations

import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager, _chaotic_seed_mix


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


def test_chaotic_seed_mix_varies_with_noise() -> None:
    base = [1, 2, 3]
    first = _chaotic_seed_mix(base, noise=123, logistic=0.5)
    second = _chaotic_seed_mix(base, noise=123, logistic=0.5)
    different = _chaotic_seed_mix(base, noise=456, logistic=0.5)

    assert first == second
    assert first != different


def test_spawn_child_seed_uses_chaotic_mix(monkeypatch) -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    manager = EvolutionManager(
        original=image,
        encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(denoise_sigma=None),
        population_size=1,
        base_seed=5,
        autosave_interval=1,
    )

    anchors = [11, 19, 23, 31]

    class DummyRNG:
        def __init__(self) -> None:
            self._integer_values = iter([111, 222])

        def choice(self, values, size, replace):
            return np.array(list(values)[:size], dtype=np.int64)

        def random(self) -> float:
            return 0.25

        def integers(self, low, high):  # noqa: D401 - signature mirrors numpy Generator
            return next(self._integer_values)

    monkeypatch.setattr(manager, "rng", DummyRNG())

    logistic = 3.999 * 0.25 * (1.0 - 0.25)
    selected = np.array(anchors[:3], dtype=np.int64)
    combined = 0
    for idx, parent_seed in enumerate(selected):
        shift = (idx * 17) % 31
        combined ^= (int(parent_seed) << shift) & 0x7FFFFFFF
    walsh = int(np.bitwise_xor.reduce(selected ^ np.roll(selected, 1))) & 0x7FFFFFFF
    chaotic = _chaotic_seed_mix(selected.tolist(), 111, logistic)
    logistic_component = int(abs(logistic) * 0x7FFFFFFF) & 0x7FFFFFFF
    mutation = 222
    expected = (combined ^ walsh ^ chaotic ^ logistic_component ^ mutation) & 0x7FFFFFFF

    child = manager._spawn_child_seed(anchors)
    assert child == expected
