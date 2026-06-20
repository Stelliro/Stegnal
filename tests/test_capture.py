"""Tests for acoustic batch capture (with a stubbed audio engine)."""

from __future__ import annotations

import numpy as np

from umbra.capture import (
    average_recordings,
    load_recording,
    record_batch,
    score_recordings,
)
from umbra.encoding import NoiseStreamEncoder


class StubAudioEngine:
    """Mimics AudioEngine.transmit_and_record over a near-ideal channel.

    Returns the transmitted signal normalized to [-1, 1] plus a little noise, so
    each 'recording' is a slightly different realisation of the same signal.
    """

    def __init__(self, noise=0.01, fail_indices=()):
        self.noise = noise
        self.fail_indices = set(fail_indices)
        self._calls = 0

    def transmit_and_record(self, wav, sr, idx_out, idx_in, use_sync_pulse=True):
        i = self._calls
        self._calls += 1
        if i in self.fail_indices:
            return None
        norm = np.asarray(wav, dtype=np.float32) / 32767.0
        rng = np.random.default_rng(100 + i)
        return np.clip(norm + rng.normal(0, self.noise, norm.shape), -1.0, 1.0)


def _encoded(seed=11, size=(32, 32, 3)):
    img = np.random.default_rng(0).random(size, dtype=np.float32)
    return NoiseStreamEncoder(sigma=0.05).encode(img, seed=seed), img


def test_record_batch_writes_files_and_manifest(tmp_path):
    packet, _ = _encoded()
    manifest = record_batch(StubAudioEngine(), packet.encoded, out_dir=tmp_path,
                            repeats=10, label="take")
    assert manifest["successful"] == 10
    assert len(manifest["takes"]) == 10
    wavs = sorted(tmp_path.glob("take_*.wav"))
    assert len(wavs) == 10
    assert (tmp_path / "manifest.json").exists()
    # each take records signal stats
    assert all(t["peak"] > 0 for t in manifest["takes"])


def test_record_batch_handles_failed_captures(tmp_path):
    packet, _ = _encoded()
    engine = StubAudioEngine(fail_indices={1, 3})
    manifest = record_batch(engine, packet.encoded, out_dir=tmp_path, repeats=5)
    assert manifest["successful"] == 3
    assert sum(1 for t in manifest["takes"] if not t["ok"]) == 2


def test_average_recordings_reduces_noise(tmp_path):
    packet, _ = _encoded()
    record_batch(StubAudioEngine(noise=0.05), packet.encoded, out_dir=tmp_path,
                 repeats=8, label="take")
    paths = sorted(tmp_path.glob("take_*.wav"))
    shape = (32, 32, 3)
    avg = average_recordings(paths, target_shape=shape)
    singles = np.stack([load_recording(p, shape) for p in paths])
    # the average is closer to the take-mean than a single noisy take is
    mean = singles.mean(axis=0)
    assert np.abs(avg - mean).mean() < np.abs(singles[0] - mean).mean()


def test_score_recordings_returns_features_and_rewards(tmp_path):
    packet, image = _encoded(seed=21)
    record_batch(StubAudioEngine(noise=0.0), packet.encoded, out_dir=tmp_path,
                 repeats=4, label="take")
    paths = sorted(tmp_path.glob("take_*.wav"))
    feats, rewards = score_recordings(paths, image, seed=21, denoise_sigma=0.4)
    assert feats.shape == (4, 5)
    assert rewards.shape == (4,)
    assert np.all(rewards >= 0.0)
