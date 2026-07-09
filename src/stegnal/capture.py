# capture.py

"""Acoustic (speaker -> air -> mic) batch capture and training-data prep.

This is the "through air" path: the encoded noise is played out of a speaker and
recorded back through a microphone, so each capture is a *different* physical
realisation of the same signal. Recording the same noise N times and combining
the takes (the project's "messy key" idea) averages out room/mic noise, and the
individual takes become training examples for the reward model.

The actual play/record is delegated to :meth:`stegnal.audio_mixer.AudioEngine.
transmit_and_record`, so a real run needs working audio hardware. The functions
here are hardware-agnostic — pass any object exposing ``transmit_and_record`` —
which keeps them unit-testable with a stub engine.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .audio import audio_to_image_data, image_data_to_audio

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _wav_stats(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return {"samples": 0, "peak": 0.0, "rms": 0.0}
    return {
        "samples": int(samples.size),
        "peak": float(np.max(np.abs(samples))),
        "rms": float(np.sqrt(np.mean(samples ** 2))),
    }


def record_batch(
    audio_engine,
    encoded: np.ndarray,
    *,
    out_dir: str | Path,
    repeats: int = 10,
    sample_rate: int = 48000,
    idx_out: int = 0,
    idx_in: int = 0,
    label: str = "take",
    use_sync_pulse: bool = True,
) -> dict:
    """Play the encoded noise and record it ``repeats`` times through the mic.

    Each recording is written to ``out_dir/<label>_NNN.wav`` and described in
    ``out_dir/manifest.json``. Returns the manifest dict. Recordings that fail
    (``transmit_and_record`` returns ``None``) are skipped but still noted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav, sr = image_data_to_audio(np.asarray(encoded), sample_rate)
    takes: list[dict] = []

    for i in range(int(repeats)):
        rec = audio_engine.transmit_and_record(
            wav, sr, idx_out, idx_in, use_sync_pulse=use_sync_pulse
        )
        entry = {"index": i, "timestamp": _now(), "file": None, "ok": False}
        if rec is not None:
            rec = np.asarray(rec, dtype=np.float32)
            path = out_dir / f"{label}_{i:03d}.wav"
            # store as normalized int16, matching image_data_to_audio's convention
            pcm = np.clip(rec, -1.0, 1.0)
            wavfile.write(path, sr, (pcm * 32767.0).astype(np.int16))
            entry.update(file=path.name, ok=True, **_wav_stats(rec))
            logger.info("Captured %s (%d/%d)", path.name, i + 1, repeats)
        else:
            logger.warning("Capture %d/%d returned no audio", i + 1, repeats)
        takes.append(entry)

    manifest = {
        "label": label,
        "created_at": _now(),
        "sample_rate": int(sr),
        "repeats": int(repeats),
        "successful": sum(1 for t in takes if t["ok"]),
        "source": "acoustic",
        "encoded_size": int(np.asarray(encoded).size),
        "takes": takes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_recording(path: str | Path, target_shape=(256, 256, 3)) -> np.ndarray:
    """Read a captured WAV back into image-data of ``target_shape``."""
    return audio_to_image_data(str(path), target_shape=target_shape)


def average_recordings(paths, target_shape=(256, 256, 3)) -> np.ndarray:
    """Average several takes of the same noise to suppress per-capture noise.

    This is the "messy key" stabilisation: independent room/mic noise averages
    toward zero across takes, leaving the shared underlying signal.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no recordings to average")
    acc = None
    for p in paths:
        data = load_recording(p, target_shape)
        acc = data if acc is None else acc + data
    return (acc / len(paths)).astype(np.float32)


def score_recordings(
    paths,
    reference: np.ndarray,
    *,
    seed: int,
    decoder=None,
    denoise_sigma: float = 0.6,
    feature_names=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode each recording and return ``(features, rewards)`` for training.

    Features follow :data:`stegnal.checkpoint.DEFAULT_FEATURE_NAMES`; the reward is
    the structure-gated ``composite_score`` of the decoded image vs ``reference``.
    """
    from .checkpoint import DEFAULT_FEATURE_NAMES
    from .decoding import NoiseStreamDecoder
    from .encoding import NoisePacket
    from .metrics import composite_score, compute_metrics
    from .visualization import multiplicative_overlap

    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    ref = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
    if ref.ndim == 2:
        ref = np.stack([ref] * 3, axis=-1)
    shape = ref.shape
    decoder = decoder or NoiseStreamDecoder(denoise_sigma=denoise_sigma)

    feats: list[list[float]] = []
    rewards: list[float] = []
    for p in paths:
        data = load_recording(p, shape)
        packet = NoisePacket(
            encoded=data.reshape(-1), permutation_seed=seed,
            image_shape=shape, sigma=0.1,
        )
        recon = np.clip(np.asarray(decoder.decode(packet, seed=seed), dtype=np.float32), 0.0, 1.0)
        metrics = compute_metrics(ref, recon)
        _, overlap = multiplicative_overlap(ref, recon)
        reward = composite_score(float(overlap), metrics.psnr, metrics.ssim) * 100.0
        values = {
            "overlap": float(overlap), "psnr": float(metrics.psnr),
            "ssim": float(metrics.ssim), "denoise_sigma": float(denoise_sigma),
            "difficulty": 0.1,
        }
        feats.append([values.get(name, 0.0) for name in feature_names])
        rewards.append(reward)

    return np.asarray(feats, dtype=np.float32), np.asarray(rewards, dtype=np.float32)


__all__ = ["record_batch", "load_recording", "average_recordings", "score_recordings"]
