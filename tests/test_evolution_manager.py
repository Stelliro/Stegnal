import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager
from umbra.neural import NeuralRewardModel


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
    assert manager.lifetime_reward > 0.0
    assert all(record.reward_summary >= 0.0 for record in manager.generations)


def test_parent_lineage_retains_elites_and_children() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((16, 16), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.25)
    decoder = NoiseStreamDecoder(denoise_sigma=0.8)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=4,
        base_seed=321,
        autosave_interval=2,
    )

    first_generation = manager.run_generation()
    assert manager.parent_lineage
    parent_seeds = {candidate.seed for candidate in first_generation.candidates}

    second_generation = manager.run_generation(parent_selection=list(parent_seeds))
    second_seeds = {candidate.seed for candidate in second_generation.candidates}

    # ensure at least one parent seed persisted and new offspring were introduced
    assert parent_seeds.intersection(second_seeds)
    assert len(second_seeds) > len(parent_seeds.intersection(second_seeds))
    lineage_map = {entry.seed: entry for entry in manager.parent_lineage}
    lineage_entry = lineage_map[second_generation.best_candidate.seed]
    assert lineage_entry.appearances >= 1
    assert lineage_entry.cumulative_reward >= 0.0


def test_neural_reward_model_learns_signal() -> None:
    model = NeuralRewardModel(input_dim=5, hidden_layers=(8, 4), learning_rate=0.05, max_epochs=10)
    rng = np.random.default_rng(5)
    features = rng.random((32, 5), dtype=np.float32)
    # Reward emphasises first and third feature components
    rewards = 0.7 * features[:, 0] + 0.3 * features[:, 2]
    model.update(features, rewards)
    prediction = model.predict(features[0])
    assert np.isfinite(prediction)
    assert prediction != 0.0


def test_plateau_ramp_reduces_sigma_and_improves_overlap() -> None:
    image = np.full((12, 12), 0.5, dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.6)
    decoder = NoiseStreamDecoder(denoise_sigma=1.0)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=777,
        autosave_interval=2,
    )

    initial_sigma = float(manager.encoder.sigma)
    overlaps: list[float] = []
    for _ in range(12):
        generation = manager.run_generation()
        overlaps.append(generation.best_candidate.overlap_score)

    assert float(manager.encoder.sigma) < initial_sigma
    assert max(overlaps) > overlaps[0]
    assert manager.mutation_boost >= 0
