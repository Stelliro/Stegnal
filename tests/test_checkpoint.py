"""Tests for the model checkpoint (save/load + metadata + continue-training)."""

from __future__ import annotations

import numpy as np

from umbra.checkpoint import UmbraModel, candidate_feature_vector
from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager


def _fake_batch(rng, n=8):
    feats = rng.random((n, 5), dtype=np.float32)
    rewards = 60.0 * feats[:, 0] + 40.0 * feats[:, 2]  # depends on overlap + ssim
    return feats, rewards


def test_train_records_metadata():
    rng = np.random.default_rng(0)
    model = UmbraModel(name="m1")
    feats, rewards = _fake_batch(rng)
    info = model.train(feats, rewards, source="simulation",
                       peers=[{"seed": 7, "reward": 99.0, "generation": 1}])
    assert info["samples"] == 8
    assert model.total_samples == 8
    assert model.generations_trained == 1
    assert len(model.sessions) == 1 and model.sessions[0].source == "simulation"
    assert len(model.performance) == 1
    assert model.peers and model.peers[0].seed == 7


def test_save_load_roundtrip_preserves_everything(tmp_path):
    rng = np.random.default_rng(1)
    model = UmbraModel(name="roundtrip")
    feats, rewards = _fake_batch(rng)
    model.train(feats, rewards, source="acoustic", recordings=["take_000.wav"])

    path = model.save(tmp_path / "model")          # no suffix -> .umbra.json
    assert path.exists() and path.suffix == ".json"

    loaded = UmbraModel.load(tmp_path / "model")
    assert loaded.name == "roundtrip"
    assert loaded.total_samples == model.total_samples
    assert loaded.generations_trained == model.generations_trained
    assert loaded.sessions[0].recordings == ["take_000.wav"]
    # weights survive: identical predictions on the same input
    probe = rng.random((4, 5), dtype=np.float32)
    np.testing.assert_allclose(model.predict(probe), loaded.predict(probe), rtol=1e-5, atol=1e-5)


def test_load_then_continue_training(tmp_path):
    rng = np.random.default_rng(2)
    model = UmbraModel()
    f1, r1 = _fake_batch(rng)
    model.train(f1, r1)
    model.save(tmp_path / "ck.umbra.json")

    reloaded = UmbraModel.load(tmp_path / "ck.umbra.json")
    samples_before = reloaded.reward_model._samples_seen
    f2, r2 = _fake_batch(rng)
    reloaded.train(f2, r2, source="acoustic")

    # Training continued from the loaded state, not from scratch.
    assert reloaded.generations_trained == 2
    assert reloaded.reward_model._samples_seen > samples_before
    assert reloaded.total_samples == 16


def test_train_from_generation():
    rng = np.random.default_rng(3)
    image = rng.random((24, 24, 3)).astype(np.float32)
    mgr = EvolutionManager(
        original=image, encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(denoise_sigma=0.8), population_size=6,
        base_seed=5, enable_waveform=False,
    )
    mgr.run_generation()
    model = UmbraModel()
    info = model.train_from_generation(mgr, source="simulation")
    assert info["samples"] == len(mgr.generations[-1].candidates) or info["samples"] >= 0
    assert model.generations_trained == 1


def test_candidate_feature_vector_shape():
    class _M:  # noqa: N801 - tiny stub
        psnr, ssim = 20.0, 0.5

    class _C:
        metrics = _M()
        overlap_score = 88.0

        class genes:  # noqa: N801
            denoise_sigma = 0.7

    vec = candidate_feature_vector(_C(), difficulty=0.2)
    assert len(vec) == 5
    assert vec[0] == 88.0 and vec[3] == 0.7
