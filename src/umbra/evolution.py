# evolution.py

"""Evolutionary search utilities for Project Umbra."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .codec import (
    decode_wav_bytes_to_image,
    encode_image_to_wav_bytes,
)
from .decoding import NoiseStreamDecoder
from .encoding import NoiseStreamEncoder
from .gpu_runtime import cp, ensure_nvrtc_configured
from .metrics import (
    AI_PSNR_BASELINE,
    ReconstructionMetrics,
    audio_fidelity_score,
    composite_score,
    compute_metrics,
    compute_ms_ssim,
    dct_band_correlation,
    partial_alignment_fraction,
    readability_score,
    team_cohesion_score,
)
from .reconstruction import (
    GPUAccelerationRequiredError,
    suggest_sample_rate,
    suggest_transmission_profile,
)
from .runs import append_history, get_run_paths, load_history, new_run
from .visualization import multiplicative_overlap

if TYPE_CHECKING:  # pragma: no cover - optional neural advisor import
    from .neural import NeuralRewardModel

logger = logging.getLogger(__name__)


class EvolutionLimitReached(RuntimeError):
    """Raised when an evolution run reaches its configured generation limit."""

    def __init__(self, attempted_generation: int, limit: int) -> None:
        super().__init__(
            f"generation limit of {limit} reached at index {attempted_generation}"
        )
        self.attempted_generation = int(attempted_generation)
        self.limit = int(limit)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid int for {name}: {value}; using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}: {value}; using default {default}")
        return default


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as a clipped three-channel float array."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] > 3:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected image with shape (H, W, 3)")
    return np.clip(array, 0.0, 1.0)


def _chaotic_seed_mix(values: Sequence[int], noise: int, logistic: float) -> int:
    """Blend ``values`` with ``noise`` via a keyed hash for seed diversity."""

    buffer = bytearray()
    buffer.extend(int(noise & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    logistic_bits = int(abs(logistic) * (1 << 32)) & 0xFFFFFFFF
    buffer.extend(logistic_bits.to_bytes(4, "little", signed=False))
    for value in values:
        buffer.extend(int(value & 0x7FFFFFFF).to_bytes(8, "little", signed=False))
    digest = hashlib.blake2s(buffer, person=b"umbChaos").hexdigest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


_LINEAGE_RETENTION_FACTOR = 3


PLATEAU_CFG_DEFAULTS = {
    "window": 6,
    "delta_threshold": 0.002,
    "boost_step": 2,
    "boost_cap_factor": 3,
    "log": True,
}

PLATEAU_CFG = {
    "window": _env_int("UMBRA_PLATEAU_WINDOW", PLATEAU_CFG_DEFAULTS["window"]),
    "delta_threshold": _env_float(
        "UMBRA_PLATEAU_DELTA_THRESHOLD", PLATEAU_CFG_DEFAULTS["delta_threshold"]
    ),
    "boost_step": _env_int("UMBRA_PLATEAU_BOOST_STEP", PLATEAU_CFG_DEFAULTS["boost_step"]),
    "boost_cap_factor": _env_int(
        "UMBRA_PLATEAU_BOOST_CAP_FACTOR", PLATEAU_CFG_DEFAULTS["boost_cap_factor"]
    ),
    "log": os.getenv("UMBRA_PLATEAU_LOG", "1") not in {"0", "false", "False"},
}


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "0")
    return value.lower() not in {"0", "false", "off", "no"}


DIFFICULTY_LADDER_DEFAULTS = {
    "enabled": False,
    "base": 0.48,
    "spike": 0.60,
    "period_gens": 12,
    "spike_len_gens": 2,
}

DIFFICULTY_LADDER = {
    "enabled": _env_flag("UMBRA_DIFFICULTY_ENABLED"),
    "base": _env_float("UMBRA_DIFFICULTY_BASE", DIFFICULTY_LADDER_DEFAULTS["base"]),
    "spike": _env_float("UMBRA_DIFFICULTY_SPIKE", DIFFICULTY_LADDER_DEFAULTS["spike"]),
    "period_gens": _env_int("UMBRA_DIFFICULTY_PERIOD_GENS", DIFFICULTY_LADDER_DEFAULTS["period_gens"]),
    "spike_len_gens": _env_int("UMBRA_DIFFICULTY_SPIKE_LEN_GENS", DIFFICULTY_LADDER_DEFAULTS["spike_len_gens"]),
}


@dataclass(frozen=True)
class CandidateResult:
    """Outcome from evaluating a single candidate."""

    seed: int
    sigma: float
    reconstruction: np.ndarray
    metrics: ReconstructionMetrics
    overlap: float
    reward: float
    waveform: bytes | None = None


@dataclass(frozen=True)
class GenerationRecord:
    """Snapshot of a complete generation."""

    generation: int
    candidates: tuple[CandidateResult, ...]
    best_candidate: CandidateResult
    mean_reward: float
    difficulty: float
    duration: float


@dataclass(frozen=True)
class ParentLineage:
    """Ancestor chain for a candidate seed."""

    seed: int
    parent_seed: int | None = None
    generation: int | None = None
    reward: float | None = None


@dataclass
class HyperPerformanceProfile:
    """Tunable hyperparameters for evolution runs."""

    batch_size: int = 8
    autosave_interval: int = 5
    mean_duration: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "autosave_interval": self.autosave_interval,
            "mean_duration": self.mean_duration,
            "throughput": self.throughput,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperPerformanceProfile:
        return cls(
            batch_size=int(data.get("batch_size", 8)),
            autosave_interval=int(data.get("autosave_interval", 5)),
            mean_duration=float(data.get("mean_duration", 0.0)),
            throughput=float(data.get("throughput", 0.0)),
        )


@dataclass
class EvolutionSession:
    """Serializable snapshot of an evolution run."""

    original: np.ndarray
    population_size: int
    base_seed: int
    autosave_interval: int
    generations: list[GenerationRecord]
    rng_state: dict[str, Any]
    parent_lineage: list[ParentLineage]
    elite_seeds: list[int]
    best_overlap: float
    lifetime_reward: float
    reward_trace: list[float]
    difficulty_trace: list[float]
    plateau_generations: int
    mutation_boost: int
    hyper_profile: dict[str, Any]
    run_id: str | None = None
    next_generation_index: int | None = None
    enable_waveform: bool = True
    advisor_state: dict[str, Any] | None = None


class EvolutionManager:
    """Manage evolutionary search for optimal noise parameters."""

    def __init__(
        self,
        original: np.ndarray,
        *,
        encoder: NoiseStreamEncoder | None = None,
        decoder: NoiseStreamDecoder | None = None,
        population_size: int = 8,
        base_seed: int | None = None,
        autosave_interval: int = 5,
        advisor: NeuralRewardModel | None = None,
        run_id: str | None = None,
        next_generation_index: int | None = None,
        enable_waveform: bool = True,
    ) -> None:
        self.original = _ensure_three_channel(original)
        self.encoder = encoder or NoiseStreamEncoder()
        self.decoder = decoder or NoiseStreamDecoder()
        self.population_size = max(1, population_size)
        self.base_seed = base_seed or int(time.time())
        self.autosave_interval = max(1, autosave_interval)
        self._reward_model = advisor
        self.generations: list[GenerationRecord] = []
        self.rng = np.random.default_rng(self.base_seed)
        self._parent_lineage: dict[int, ParentLineage] = {}
        self.lifetime_reward = 0.0
        self.reward_trace: list[float] = []
        self.difficulty_trace: list[float] = []
        self._elite_pool: list[int] = []
        self._best_overlap = 0.0
        self._plateau_generations = 0
        self._mutation_boost = 0
        self._hyper_enabled = True
        self._hyper_profile = self.default_hyper_profile()
        self._duration_ema = 0.0
        self._throughput_ema = 0.0
        self.run_id = run_id or new_run()
        self.next_generation_index = next_generation_index or 0
        self.enable_waveform = enable_waveform

    def default_hyper_profile(self) -> HyperPerformanceProfile:
        return HyperPerformanceProfile(
            batch_size=self.population_size,
            autosave_interval=self.autosave_interval,
        )

    def evolve_generation(self, difficulty: float = 0.5, max_iters: int = 10) -> GenerationRecord:
        candidates = []
        start_time = time.time()

        for _ in range(self.population_size):
            seed = self.rng.integers(0, 2**31)
            sigma = np.clip(difficulty + self.rng.normal(0, 0.05), 0.1, 1.0)
            packet = self.encoder.encode(self.original, seed, sigma=sigma)
            recon = self.decoder.decode(packet, seed)
            metrics = compute_metrics(self.original, recon)
            overlap = multiplicative_overlap(self.original, recon)
            reward = composite_score(metrics, overlap)
            waveform = encode_image_to_wav_bytes(recon) if self.enable_waveform else None
            candidates.append(CandidateResult(seed=seed, sigma=sigma, reconstruction=recon, metrics=metrics, overlap=overlap, reward=reward, waveform=waveform))

        best = max(candidates, key=lambda c: c.reward)
        mean_reward = float(np.mean([c.reward for c in candidates]))
        duration = time.time() - start_time

        record = GenerationRecord(
            generation=self.next_generation_index,
            candidates=tuple(candidates),
            best_candidate=best,
            mean_reward=mean_reward,
            difficulty=difficulty,
            duration=duration,
        )

        self.append_generation_record(record)
        return record

    def append_generation_record(self, record: GenerationRecord, persist: bool = True, use_next_index: bool = True) -> None:
        if use_next_index:
            record = GenerationRecord(generation=self.next_generation_index, **vars(record))
            self.next_generation_index += 1
        self.generations.append(record)
        self.lifetime_reward += record.mean_reward
        self.reward_trace.append(record.mean_reward)
        self.difficulty_trace.append(record.difficulty)
        if persist and self.run_id:
            append_history(self.run_id, record)

    def sync_history(self) -> None:
        if not self.run_id:
            return
        for record in self.generations:
            append_history(self.run_id, record)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> EvolutionManager:
        with open(path, "rb") as f:
            session = pickle.load(f)
        restored_generations = session.generations
        original = session.original

        logger.info(
            "Loaded evolution session from %s with %d generations",
            path,
            len(restored_generations),
        )
        advisor_state = getattr(session, "advisor_state", None)
        advisor: NeuralRewardModel | None = None
        if advisor_state is not None:
            try:
                from .neural import NeuralRewardModel

                advisor = NeuralRewardModel.from_state(advisor_state)
            except Exception:  # pragma: no cover - advisor restoration is optional
                logger.debug("Failed to restore neural advisor", exc_info=True)

        manager = cls(
            original=original,
            population_size=session.population_size,
            base_seed=session.base_seed,
            autosave_interval=session.autosave_interval,
            advisor=advisor,
            run_id=getattr(session, "run_id", None),
            next_generation_index=getattr(session, "next_generation_index", None),
            enable_waveform=getattr(session, "enable_waveform", True),
        )
        for record in restored_generations:
            manager.append_generation_record(record, persist=False, use_next_index=False)
        manager.rng.bit_generator.state = session.rng_state
        lineage = getattr(session, "parent_lineage", [])
        manager._parent_lineage = {entry.seed: entry for entry in lineage}
        manager.lifetime_reward = float(getattr(session, "lifetime_reward", 0.0))
        manager.reward_trace = list(getattr(session, "reward_trace", []))
        manager.difficulty_trace = list(getattr(session, "difficulty_trace", []))
        manager._elite_pool = list(getattr(session, "elite_seeds", []))
        manager._best_overlap = float(getattr(session, "best_overlap", 0.0))
        manager._plateau_generations = int(getattr(session, "plateau_generations", 0))
        manager._mutation_boost = int(getattr(session, "mutation_boost", 0))
        saved_profile = getattr(session, "hyper_profile", None)
        if saved_profile and manager._hyper_enabled:
            try:
                profile = HyperPerformanceProfile.from_dict(dict(saved_profile))
            except Exception:  # pragma: no cover - corrupted saves fall back to defaults
                logger.debug("Failed to restore hyper profile", exc_info=True)
                profile = manager.default_hyper_profile()
            manager._hyper_profile = profile
            if profile.batch_size > 0:
                manager.population_size = max(1, profile.batch_size)
            if profile.autosave_interval > 0:
                manager.autosave_interval = max(1, profile.autosave_interval)
        manager._duration_ema = float(getattr(manager._hyper_profile, "mean_duration", 0.0))
        manager._throughput_ema = float(getattr(manager._hyper_profile, "throughput", 0.0))
        if advisor is None and advisor_state is None:
            manager._reward_model = None
        stored_next = getattr(session, "next_generation_index", None)
        if stored_next is not None:
            manager.next_generation_index = max(
                manager.next_generation_index, int(stored_next)
            )
        if manager.run_id:
            try:
                history_frame = load_history(manager.run_id)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to load history for run %s", manager.run_id, exc_info=True)
                history_frame = None
            if history_frame is not None and not history_frame.empty:
                try:
                    last_index = int(max(history_frame["generation"]))
                except Exception:  # pragma: no cover - defensive
                    last_index = None
                if last_index is not None:
                    manager.next_generation_index = max(
                        manager.next_generation_index, last_index + 1
                    )
        manager.sync_history()
        return manager


__all__ = [
    "CandidateResult",
    "EvolutionManager",
    "EvolutionSession",
    "GenerationRecord",
    "ParentLineage",
    "HyperPerformanceProfile",
]