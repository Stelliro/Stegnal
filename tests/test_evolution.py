"""Tests for the evolutionary search helpers."""
from __future__ import annotations

import logging

import numpy as np
import pytest

import umbra.decoding as decoding
import umbra.encoding as encoding
import umbra.gpu_runtime as gpu_runtime
import umbra.reconstruction as reconstruction
from umbra.codec import decode_wav_bytes_to_image, encode_image_to_wav_bytes
from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.evolution import EvolutionManager, _chaotic_seed_mix
from umbra.metrics import (
    audio_fidelity_score,
    composite_score,
    compute_metrics,
    partial_alignment_fraction,
    readability_score,
    team_cohesion_score,
)
from umbra.reconstruction import suggest_sample_rate, suggest_transmission_profile
from umbra.visualization import multiplicative_overlap


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
    image = np.zeros((8, 8), dtype=np.float32)
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


def test_generation_metrics_track_sound_alignment() -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    encoder = NoiseStreamEncoder(sigma=0.05)
    decoder = NoiseStreamDecoder(denoise_sigma=None)
    manager = EvolutionManager(
        original=image,
        encoder=encoder,
        decoder=decoder,
        population_size=2,
        base_seed=17,
        autosave_interval=1,
    )

    record = manager.run_generation()
    candidate = record.best_candidate
    reference = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    reconstruction = np.clip(np.asarray(candidate.reconstruction, dtype=np.float32), 0.0, 1.0)
    packet_metrics = compute_metrics(reference, reconstruction)
    _, packet_overlap = multiplicative_overlap(reference, reconstruction)

    assert candidate.metrics.psnr == pytest.approx(
        packet_metrics.psnr, rel=1e-5, abs=1e-5
    )
    assert candidate.metrics.ssim == pytest.approx(
        packet_metrics.ssim, rel=1e-5, abs=1e-5
    )

    bootstrap_payload = manager._get_sound_reference_payload()
    if bootstrap_payload and "image" in bootstrap_payload:
        baseline_image = np.clip(
            np.asarray(bootstrap_payload["image"], dtype=np.float32), 0.0, 1.0
        )
        _, bootstrap_overlap = multiplicative_overlap(baseline_image, reconstruction)
        expected_overlap = float(
            np.clip(
                np.sqrt(float(packet_overlap) * float(bootstrap_overlap)),
                0.0,
                100.0,
            )
        )
        assert candidate.sound_bootstrap_overlap == pytest.approx(
            float(bootstrap_overlap), rel=1e-5, abs=1e-5
        )
    else:
        expected_overlap = float(packet_overlap)
        assert candidate.sound_bootstrap_overlap is None

    assert candidate.reference_overlap == pytest.approx(
        float(packet_overlap), rel=1e-5, abs=1e-5
    )
    assert candidate.overlap_score == pytest.approx(
        float(expected_overlap), rel=1e-5, abs=1e-5
    )

    sample_rate = suggest_sample_rate(reference)
    segments, marker_duration = suggest_transmission_profile(reference)
    wav_bytes = encode_image_to_wav_bytes(
        reconstruction,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    sound_image, metadata = decode_wav_bytes_to_image(
        wav_bytes,
        resolution=reference.shape[:2],
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
        return_metadata=True,
    )
    sound_clipped = np.clip(np.asarray(sound_image, dtype=np.float32), 0.0, 1.0)

    assert candidate.waveform_reconstruction is not None
    np.testing.assert_allclose(
        np.clip(np.asarray(candidate.waveform_reconstruction, dtype=np.float32), 0.0, 1.0),
        sound_clipped,
        atol=1e-4,
        rtol=1e-4,
    )

    expected_reference_metrics = compute_metrics(reference, sound_clipped)
    _, expected_reference_overlap = multiplicative_overlap(reference, sound_clipped)
    assert candidate.waveform_reference_metrics is not None
    assert candidate.waveform_reference_metrics.psnr == pytest.approx(
        expected_reference_metrics.psnr, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_reference_metrics.ssim == pytest.approx(
        expected_reference_metrics.ssim, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_reference_overlap == pytest.approx(
        float(expected_reference_overlap), rel=1e-5, abs=1e-5
    )

    expected_packet_metrics = compute_metrics(reconstruction, sound_clipped)
    _, expected_packet_overlap = multiplicative_overlap(reconstruction, sound_clipped)
    assert candidate.waveform_packet_metrics is not None
    assert candidate.waveform_packet_metrics.psnr == pytest.approx(
        expected_packet_metrics.psnr, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_packet_metrics.ssim == pytest.approx(
        expected_packet_metrics.ssim, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_packet_overlap == pytest.approx(
        float(expected_packet_overlap), rel=1e-5, abs=1e-5
    )

    assert candidate.waveform_sample_rate == int(metadata.sample_rate)
    assert candidate.waveform_segments == int(metadata.segments)
    assert candidate.waveform_marker_duration == pytest.approx(
        metadata.marker_duration, rel=1e-6, abs=1e-6
    )
    expected_reference_partial = partial_alignment_fraction(reference, sound_clipped)
    expected_packet_partial = partial_alignment_fraction(reconstruction, sound_clipped)
    expected_sound_score = audio_fidelity_score(
        float(expected_reference_overlap),
        expected_reference_metrics.psnr,
        expected_reference_metrics.ssim,
        partial_credit=expected_reference_partial,
    )
    assert candidate.waveform_sound_score == pytest.approx(
        expected_sound_score, rel=1e-5, abs=1e-5
    )
    expected_readability = readability_score(
        float(expected_reference_overlap),
        expected_reference_metrics.psnr,
        expected_reference_metrics.ssim,
    )
    assert candidate.waveform_readability_score == pytest.approx(
        expected_readability, rel=1e-5, abs=1e-5
    )
    expected_alignment_score = audio_fidelity_score(
        float(expected_packet_overlap),
        expected_packet_metrics.psnr,
        expected_packet_metrics.ssim,
        partial_credit=expected_packet_partial,
    )
    assert candidate.waveform_alignment_score == pytest.approx(
        expected_alignment_score, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_reference_partial == pytest.approx(
        expected_reference_partial, rel=1e-5, abs=1e-5
    )
    assert candidate.waveform_alignment_partial == pytest.approx(
        expected_packet_partial, rel=1e-5, abs=1e-5
    )
    expected_team_score = team_cohesion_score(
        float(packet_overlap),
        packet_metrics.psnr,
        packet_metrics.ssim,
        sound_reference_overlap=float(expected_reference_overlap),
        sound_reference_psnr=expected_reference_metrics.psnr,
        sound_reference_ssim=expected_reference_metrics.ssim,
        sound_alignment_overlap=float(expected_packet_overlap),
        sound_alignment_psnr=expected_packet_metrics.psnr,
        sound_alignment_ssim=expected_packet_metrics.ssim,
        sound_reference_partial=expected_reference_partial,
        sound_alignment_partial=expected_packet_partial,
        readability=expected_readability,
    )
    assert candidate.team_score == pytest.approx(
        expected_team_score, rel=1e-5, abs=1e-5
    )
    expected_ai_score = composite_score(
        float(packet_overlap), packet_metrics.psnr, packet_metrics.ssim
    )
    assert candidate.ai_score == pytest.approx(
        expected_ai_score, rel=1e-5, abs=1e-5
    )


def test_evolution_falls_back_to_cpu_when_gpu_unavailable(monkeypatch, caplog) -> None:
    monkeypatch.setattr(gpu_runtime, "cp", None, raising=False)
    monkeypatch.setattr(reconstruction, "cp", None, raising=False)
    monkeypatch.setattr(encoding, "cp", None, raising=False)
    monkeypatch.setattr(decoding, "cp", None, raising=False)

    image = np.zeros((8, 8), dtype=np.float32)
    manager = EvolutionManager(
        original=image,
        encoder=NoiseStreamEncoder(sigma=0.1),
        decoder=NoiseStreamDecoder(denoise_sigma=None),
        population_size=1,
        base_seed=1,
        autosave_interval=1,
    )

    with caplog.at_level(logging.WARNING):
        record = manager.run_generation()

    assert manager._gpu_warning_emitted is True
    assert record.candidates, "CPU fallback should still evaluate candidates"
