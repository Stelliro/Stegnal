import importlib

import numpy as np

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionLimitReached, EvolutionManager
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
        enable_waveform=False,
    )

    for _ in range(3):
        manager.run_generation()

    assert len(manager.generations) == 3
    best_metrics = [record.best_candidate.metrics for record in manager.generations]
    assert all(np.isfinite(metric.psnr) for metric in best_metrics)
    assert all(0.0 <= metric.ssim <= 1.0 for metric in best_metrics)
    assert manager.lifetime_reward > 0.0
    assert all(record.reward_summary >= 0.0 for record in manager.generations)


def test_legacy_evolution_improves_and_reports_real_metrics() -> None:
    """The UI's evolve_generation path must (a) reconstruct sanely through the
    sim channel and (b) actually raise the score across generations."""
    import random

    random.seed(0)
    np.random.seed(0)

    # A smooth, structured RGB image the channel can plausibly recover.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 64.0
    image = np.stack([xx, yy, (xx + yy) * 0.5], axis=-1).astype(np.float32)

    manager = EvolutionManager(
        image, simulation_mode=True, population_size=20, compute_mode="cpu",
    )

    rewards = []
    for _ in range(12):
        manager.evolve_generation(difficulty=0.1)
        rewards.append(manager.population[0].reward)

    best = manager.population[0]
    # Unit-bug regression guard: before the int16/float fix this was ~6 dB.
    assert best.metrics.psnr > 12.0
    assert 0.0 <= best.metrics.ssim <= 1.0
    # Reward is the structure-gated composite (0-100), and the search improved it.
    assert best.reward > 0.0
    assert max(rewards) > rewards[0]


def test_gene_unlock_toggle_controls_color_genes() -> None:
    """Purist (default) keeps colors neutral; unlocked perturbs them."""
    import random

    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)

    random.seed(1)
    purist = EvolutionManager(image, simulation_mode=True, population_size=8)
    assert purist.unlock_genes is False
    g = purist._random_gene()
    assert (g.contrast_scale, g.gamma, g.brightness_shift) == (1.0, 1.0, 0.0)
    assert (g.r_gain, g.g_gain, g.b_gain) == (1.0, 1.0, 1.0)

    random.seed(1)
    unlocked = EvolutionManager(image, simulation_mode=True, population_size=8,
                                unlock_genes=True)
    assert unlocked.unlock_genes is True
    genes = [unlocked._random_gene() for _ in range(8)]
    # at least one gene differs from neutral across the sample
    assert any(abs(g.gamma - 1.0) > 1e-6 or abs(g.contrast_scale - 1.0) > 1e-6
               for g in genes)

    unlocked.update_settings(unlock_genes=False)
    assert unlocked.unlock_genes is False


def test_recommend_difficulty_gates_on_ssim() -> None:
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    mgr = EvolutionManager(image, simulation_mode=True, population_size=4,
                           difficulty_min_ssim=0.5, difficulty_step=0.02)
    # below the SSIM threshold difficulty must NOT rise (it eases off)
    assert mgr.recommend_difficulty(0.3, best_ssim=0.4) < 0.3
    # at/above the threshold it ratchets up — harder is the reward for doing well
    assert mgr.recommend_difficulty(0.3, best_ssim=0.6) > 0.3
    # bounds are respected
    assert mgr.recommend_difficulty(0.999, 0.9) <= 1.0
    assert mgr.recommend_difficulty(0.01, 0.0) >= 0.01


def test_reward_is_competence_led_and_escalates() -> None:
    """Reward = 100*competence + quality, so difficulty actually climbs off the
    floor and the score reflects the hardest channel cleared (not an easy one)."""
    import random

    random.seed(0)
    np.random.seed(0)
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32) / 48.0
    image = np.stack([xx, yy, (xx + yy) * 0.5], axis=-1).astype(np.float32)
    mgr = EvolutionManager(image, simulation_mode=True, population_size=16,
                           difficulty_min_ssim=0.4)
    for _ in range(20):
        mgr.evolve_generation(0.1)

    best = mgr.population[0]
    comp = best.heritage.competence
    # difficulty escalated well off the 0.01 floor (the old bug left it stuck)
    assert comp > 0.05
    # reward is competence-led: reward == 100*competence + quality, quality in [0,1]
    assert 0.0 <= (best.reward - 100.0 * comp) <= 1.0 + 1e-6
    # the champion captured the hardest difficulty cleared
    assert mgr.champion is not None
    assert mgr.champion["difficulty_cleared"] >= comp - 1e-9


def test_heritage_child_inherits_stronger_competence() -> None:
    from umbra.evolution import Heritage

    a = Heritage(competence=0.3, depth=2)
    b = Heritage(competence=0.5, depth=1)
    child = a.child(b, generation=5, parent_seeds=[11, 22])
    assert child.competence == 0.5      # the stronger lineage's competence
    assert child.depth == 3             # max(2, 1) + 1
    assert child.parents == [11, 22]
    assert child.birth_generation == 5


def test_per_lineage_competence_ratchets_up() -> None:
    import random

    random.seed(0)
    np.random.seed(0)
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32) / 48.0
    image = np.stack([xx, yy, (xx + yy) * 0.5], axis=-1).astype(np.float32)
    mgr = EvolutionManager(image, simulation_mode=True, population_size=20,
                           difficulty_min_ssim=0.4)

    max_competence = []
    for _ in range(15):
        mgr.evolve_generation(0.1)
        # every candidate carries a heritage record
        assert all(c.heritage is not None for c in mgr.population)
        max_competence.append(max(c.heritage.competence for c in mgr.population))

    # As lineages clear the SSIM bar at harder channels, competence ratchets up.
    assert max_competence[-1] > max_competence[0] + 0.01
    # the learned controller accumulated experience
    assert mgr.difficulty_controller.samples > 0


def test_generation_limit_enforced() -> None:
    rng = np.random.default_rng(1234)
    image = rng.random((16, 16), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.2)
    decoder = NoiseStreamDecoder(denoise_sigma=0.9)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=9876,
        autosave_interval=1,
        enable_waveform=False,
        max_generations=2,
    )

    first = manager.run_generation()
    second = manager.run_generation()
    assert first.index == 0
    assert second.index == 1
    try:
        manager.run_generation()
    except EvolutionLimitReached as exc:
        assert exc.limit == 2
        assert exc.attempted_generation == 2
    else:  # pragma: no cover - defensive
        raise AssertionError("expected EvolutionLimitReached to be raised")


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
        enable_waveform=False,
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
    predictions = model.predict(features)
    assert all(np.isfinite(predictions))
    # After training, predictions should show variance (model learned something)
    assert predictions.std() > 1e-6


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
        enable_waveform=False,
    )

    initial_sigma = float(manager.encoder.sigma)
    overlaps: list[float] = []
    for _ in range(12):
        generation = manager.run_generation()
        overlaps.append(generation.best_candidate.overlap_score)

    assert float(manager.encoder.sigma) < initial_sigma
    assert max(overlaps) > overlaps[0]
    assert manager.mutation_boost >= 0


def test_perfect_overlap_is_reachable_with_zero_noise() -> None:
    image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    encoder = NoiseStreamEncoder(sigma=0.0)
    decoder = NoiseStreamDecoder(denoise_sigma=0.0)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=1,
        base_seed=2024,
        autosave_interval=1,
        enable_waveform=False,
    )

    generation = manager.run_generation()
    best = generation.best_candidate
    assert best.overlap_score > 40.0
    assert best.metrics.psnr > 4.0


def test_difficulty_respects_overlap_improvement() -> None:
    image = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    encoder = NoiseStreamEncoder(sigma=0.9)
    decoder = NoiseStreamDecoder(denoise_sigma=1.2)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=1,
        base_seed=31415,
        autosave_interval=1,
    )

    first = manager.run_generation()
    manager.encoder.sigma = 0.0
    manager.decoder.denoise_sigma = 0.0
    second = manager.run_generation()

    assert second.best_candidate.overlap_score >= first.best_candidate.overlap_score
    assert second.difficulty_level >= first.difficulty_level


def test_hyper_mode_profile_adapts(monkeypatch) -> None:
    monkeypatch.setenv("UMBRA_HYPER_MODE", "1")
    import umbra.evolution as evolution

    evolution = importlib.reload(evolution)

    rng = np.random.default_rng(11)
    image = rng.random((16, 16), dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.3)
    decoder = NoiseStreamDecoder(denoise_sigma=0.8)
    manager = evolution.EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=404,
        autosave_interval=1,
    )

    profile = manager.hyper_profile
    assert profile.enabled
    assert manager.population_size == profile.batch_size
    assert profile.batch_size >= 5

    generation = manager.run_generation()
    updated = manager.hyper_profile
    assert updated.last_update == generation.index
    assert manager.population_size == updated.batch_size
    assert updated.batch_size >= profile.batch_size
    assert updated.dwell_generations > 0
    assert (
        updated.batch_size * updated.dwell_generations
        >= profile.batch_size * profile.dwell_generations
    )

    monkeypatch.delenv("UMBRA_HYPER_MODE", raising=False)
    importlib.reload(evolution)


def test_decoder_falls_back_after_cupy_oom(monkeypatch) -> None:
    rng = np.random.default_rng(2024)
    image = rng.random((12, 12), dtype=np.float32)
    seed = 99

    encoder = NoiseStreamEncoder(sigma=0.35)
    decoder = NoiseStreamDecoder(denoise_sigma=0.8)

    packet = encoder.encode(image, seed)
    baseline = decoder.decode(packet, seed)

    class FakeOutOfMemoryError(RuntimeError):
        pass

    FakeOutOfMemoryError.__module__ = "cupy.cuda.memory"

    class FakeCuPy:
        float32 = np.float32

        def asarray(self, array, dtype=None):  # type: ignore[no-untyped-def]
            raise FakeOutOfMemoryError("synthetic OOM")

        @staticmethod
        def asnumpy(array):  # type: ignore[no-untyped-def]
            return np.asarray(array)

    monkeypatch.setattr("umbra.decoding.cp", FakeCuPy())
    monkeypatch.setattr("umbra.decoding.cupy_gaussian_filter", None)

    result = decoder.decode(packet, seed, allow_cpu_fallback=True)
    np.testing.assert_allclose(result, baseline)
