"""Evolutionary search utilities for Project Umbra."""

from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import logging
import os
import pickle
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .codec import decode_wav_bytes_to_image, encode_image_to_wav_bytes
from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .gpu_runtime import (
    configure_device_memory_pool,
    cp,
    ensure_nvrtc_configured,
)
from .metrics import (
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
from .runs import append_history, get_run_paths, new_run
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
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "0")
    return value.lower() not in {"0", "false", "off", "no"}


def _batched(iterable: Sequence[int], size: int) -> Iterable[list[int]]:
    """Yield successive lists of up to *size* elements from *iterable*."""

    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(iterable), size):
        yield list(iterable[start : start + size])


class _LoopProfiler:
    """Collect timing samples for major evolution loop phases."""

    def __init__(self) -> None:
        self._samples: list[tuple[str, float]] = []

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._samples.append((name, max(0.0, float(duration))))

    def add(self, name: str, duration: float) -> None:
        self._samples.append((name, max(0.0, float(duration))))

    def aggregated(self) -> list[tuple[str, float]]:
        totals: dict[str, float] = {}
        for name, duration in self._samples:
            totals[name] = totals.get(name, 0.0) + float(duration)
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)

    def as_dicts(self) -> list[dict[str, float]]:
        return [
            {"name": name, "seconds": float(duration)}
            for name, duration in self.aggregated()
        ]


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
    "spike_len_gens": _env_int(
        "UMBRA_DIFFICULTY_SPIKE_LEN_GENS", DIFFICULTY_LADDER_DEFAULTS["spike_len_gens"]
    ),
}


@dataclass
class HyperPerformanceProfile:
    """Lightweight hyper-performance recommendation snapshot."""

    enabled: bool = False
    target_subjects: int = 0
    batch_size: int = 0
    dwell_generations: int = 0
    autosave_interval: int = 0
    queue_generations: int = 0
    mean_duration: float = 0.0
    throughput: float = 0.0
    last_update: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_subjects": self.target_subjects,
            "batch_size": self.batch_size,
            "dwell_generations": self.dwell_generations,
            "autosave_interval": self.autosave_interval,
            "queue_generations": self.queue_generations,
            "mean_duration": self.mean_duration,
            "throughput": self.throughput,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HyperPerformanceProfile:
        profile = cls(enabled=bool(payload.get("enabled", False)))
        profile.target_subjects = int(payload.get("target_subjects", 0))
        profile.batch_size = int(payload.get("batch_size", 0))
        profile.dwell_generations = int(payload.get("dwell_generations", 0))
        profile.autosave_interval = int(payload.get("autosave_interval", 0))
        profile.queue_generations = int(payload.get("queue_generations", 0))
        profile.mean_duration = float(payload.get("mean_duration", 0.0))
        profile.throughput = float(payload.get("throughput", 0.0))
        profile.last_update = int(payload.get("last_update", -1))
        return profile


@dataclass
class CandidateResult:
    """Summary of a single AI attempt within a generation."""

    seed: int
    reconstruction: np.ndarray
    metrics: ReconstructionMetrics
    overlap_score: float
    ai_score: float
    reward: float
    execution_backend: str
    sigma: float
    frame_time_ms: float | None = None
    predicted_reward: float | None = None
    feature_vector: tuple[float, ...] = field(default_factory=tuple)
    reward_components: dict[str, float] = field(default_factory=dict)
    waveform_reconstruction: np.ndarray | None = None
    waveform_reference_metrics: ReconstructionMetrics | None = None
    waveform_reference_overlap: float | None = None
    waveform_packet_metrics: ReconstructionMetrics | None = None
    waveform_packet_overlap: float | None = None
    waveform_reference_partial: float | None = None
    waveform_alignment_partial: float | None = None
    waveform_sample_rate: int | None = None
    waveform_segments: int | None = None
    waveform_marker_duration: float | None = None
    waveform_sound_score: float | None = None
    waveform_readability_score: float | None = None
    waveform_alignment_score: float | None = None
    team_score: float | None = None


@dataclass
class GenerationRecord:
    """Collection of candidates evaluated for a specific generation."""

    index: int
    candidates: list[CandidateResult] = field(default_factory=list)
    reward_summary: float = 0.0
    reward_peak: float = 0.0
    cumulative_reward: float = 0.0
    difficulty_level: float = 0.0
    difficulty_raw: float = 0.0
    improvement: float = 0.0
    checkpoint_tag: str | None = None

    @property
    def best_candidate(self) -> CandidateResult:
        return max(
            self.candidates,
            key=lambda cand: (cand.reward, cand.overlap_score, cand.metrics.ssim),
        )

    @property
    def difficulty_normalized(self) -> float:
        return float(np.clip(self.difficulty_level, 0.0, 1.0))


@dataclass
class ParentLineage:
    """History entry tracking a candidate seed across generations."""

    seed: int
    origin_generation: int
    metrics: ReconstructionMetrics
    overlap_score: float
    appearances: int = 1
    cumulative_reward: float = 0.0
    peak_reward: float = 0.0
    last_generation: int = -1


@dataclass
class EvolutionSession:
    """Serializable snapshot of an :class:`EvolutionManager`."""

    image_signature: str
    original: np.ndarray
    encoder_config: dict[str, Any]
    decoder_config: dict[str, Any]
    population_size: int
    base_seed: int
    autosave_interval: int
    generations: list[GenerationRecord]
    rng_state: dict[str, Any]
    run_id: str | None
    next_generation_index: int
    parent_lineage: list[ParentLineage]
    lifetime_reward: float
    reward_trace: list[float]
    difficulty_trace: list[float]
    elite_seeds: list[int]
    advisor_state: dict[str, Any] | None
    best_overlap: float
    plateau_generations: int
    mutation_boost: int
    hyper_profile: dict[str, Any]
    enable_waveform: bool
    max_generations: int | None


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("expected image with shape (H, W, 3)")
    return np.clip(array[..., :3], 0.0, 1.0)


def _chaotic_seed_mix(values: Sequence[int], noise: int, logistic: float) -> int:
    buffer = bytearray()
    buffer.extend(int(noise & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    logistic_bits = int(abs(logistic) * (1 << 32)) & 0xFFFFFFFF
    buffer.extend(logistic_bits.to_bytes(4, "little", signed=False))
    for value in values:
        buffer.extend(int(value & 0x7FFFFFFF).to_bytes(8, "little", signed=False))
    digest = hashlib.blake2s(buffer, person=b"umbChaos").hexdigest()
    return int(digest[:16], 16) & 0x7FFFFFFF


def compute_image_signature(array: np.ndarray) -> str:
    arr = np.asarray(array, dtype=np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


class EvolutionManager:
    """Manage an evolutionary search over encoder/decoder seeds."""

    def __init__(
        self,
        original: np.ndarray,
        encoder: NoiseStreamEncoder,
        decoder: NoiseStreamDecoder,
        population_size: int,
        base_seed: int,
        autosave_interval: int = 5,
        advisor: NeuralRewardModel | None = None,
        *,
        run_id: str | None = None,
        next_generation_index: int | None = None,
        enable_waveform: bool = True,
        max_generations: int | None = None,
    ) -> None:
        self.original = _ensure_three_channel(original)
        self.encoder = encoder
        self.decoder = decoder
        self.base_seed = int(base_seed)
        self.enable_waveform = bool(enable_waveform)
        self.max_generations = (
            int(max_generations)
            if max_generations is not None and int(max_generations) > 0
            else None
        )
        self._hyper_enabled = self.hyper_mode_enabled()
        self._hyper_profile = self.default_hyper_profile(population_size, autosave_interval)
        if self._hyper_profile.enabled:
            initial_population = max(
                1, int(self._hyper_profile.batch_size or population_size)
            )
            self.autosave_interval = max(
                1, int(self._hyper_profile.autosave_interval or autosave_interval)
            )
        else:
            initial_population = max(1, int(population_size))
            self.autosave_interval = max(1, int(autosave_interval))
        self.population_size = initial_population
        if run_id is None:
            run_id, run_dir = new_run()
        else:
            run_id = str(run_id)
            run_dir, _ = get_run_paths(run_id)
        self.run_id = run_id
        self._run_directory = run_dir
        self._reward_model: NeuralRewardModel | None = advisor
        self.generations: list[GenerationRecord] = []
        self.rng = np.random.default_rng(self.base_seed)
        self.next_generation_index = int(next_generation_index or 0)
        self.lifetime_reward = 0.0
        self.reward_trace: list[float] = []
        self.difficulty_trace: list[float] = []
        self._parent_lineage: dict[int, ParentLineage] = {}
        self._elite_pool: list[int] = []
        self._overlap_history: list[float] = []
        self._best_overlap = 0.0
        self._plateau_generations = 0
        self._mutation_boost = 0
        self._difficulty_bias = 0.5
        self._gpu_warning_emitted = False
        self._state_lock = threading.Lock()
        self._reward_lock = threading.Lock()
        self._parallel_workers = max(
            1,
            min(
                _env_int("UMBRA_EVOLUTION_WORKERS", os.cpu_count() or 1),
                32,
            ),
        )
        default_active = max(
            1,
            min(
                self.population_size,
                self._parallel_workers * 2,
            ),
        )
        self._active_batch_size = max(
            1,
            _env_int("UMBRA_EVOLUTION_ACTIVE_BATCH", default_active),
        )
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._base_population = self.population_size
        self._population_floor = max(
            1,
            _env_int("UMBRA_EVOLUTION_MIN_POPULATION", min(self.population_size, 6)),
        )
        self._population_ceiling = max(
            self.population_size,
            _env_int("UMBRA_EVOLUTION_MAX_POPULATION", self.population_size),
        )
        if self._population_floor > self._population_ceiling:
            self._population_floor = self._population_ceiling
        self._target_generation_seconds = max(
            1.0,
            _env_float("UMBRA_EVOLUTION_TARGET_SECONDS", 25.0),
        )
        self._duration_ema = 0.0
        self._throughput_ema = 0.0
        self._set_population_size(self.population_size)

        if cp is not None:
            try:
                configured = configure_device_memory_pool()
                if configured:
                    logger.debug(
                        "Configured GPU memory pool target to %.2f GiB",
                        configured / (1024 ** 3),
                    )
            except Exception:  # pragma: no cover - diagnostic path for GPU hosts
                logger.debug("Failed to configure GPU memory pool", exc_info=True)

    def _set_population_size(self, value: int) -> bool:
        """Clamp and apply a new population size, returning ``True`` if it changed."""

        desired = int(value)
        if desired <= 0:
            desired = 1
        if self._population_ceiling >= self._population_floor:
            desired = int(np.clip(desired, self._population_floor, self._population_ceiling))
        changed = desired != getattr(self, "population_size", desired)
        self.population_size = desired
        if self._active_batch_size > self.population_size:
            self._active_batch_size = self.population_size
        if self._hyper_profile.enabled:
            self._hyper_profile.batch_size = self.population_size
        if changed:
            logger.debug("Population size adjusted to %d", self.population_size)
        return changed

    # ------------------------------------------------------------------ basic properties
    @property
    def mutation_boost(self) -> int:
        return self._mutation_boost

    @property
    def parent_lineage(self) -> list[ParentLineage]:
        return list(self._parent_lineage.values())

    @property
    def image_signature(self) -> str:
        return compute_image_signature(self.original)

    @property
    def reward_advisor(self) -> NeuralRewardModel | None:
        return self._reward_model

    def set_advisor(self, advisor: NeuralRewardModel | None) -> None:
        self._reward_model = advisor

    @property
    def hyper_profile(self) -> HyperPerformanceProfile:
        return self._hyper_profile

    @property
    def run_directory(self) -> Path:
        return self._run_directory

    # ------------------------------------------------------------------ configuration helpers
    @staticmethod
    def hyper_mode_enabled() -> bool:
        return _env_flag("UMBRA_HYPER_MODE")

    @staticmethod
    def default_hyper_profile(population: int, autosave_interval: int) -> HyperPerformanceProfile:
        if not EvolutionManager.hyper_mode_enabled():
            return HyperPerformanceProfile(enabled=False, batch_size=population, autosave_interval=autosave_interval)
        return HyperPerformanceProfile(
            enabled=True,
            target_subjects=150,
            batch_size=max(5, int(population)),
            dwell_generations=30,
            autosave_interval=max(10, int(autosave_interval)),
            queue_generations=30,
            mean_duration=0.0,
            throughput=0.0,
            last_update=-1,
        )

    def update_settings(
        self,
        *,
        population_size: int | None = None,
        autosave_interval: int | None = None,
    ) -> None:
        if population_size is not None:
            self.population_size = max(1, int(population_size))
        if autosave_interval is not None:
            self.autosave_interval = max(1, int(autosave_interval))

    # ------------------------------------------------------------------ persistence helpers
    def _history_payload(self, generation: GenerationRecord) -> dict[str, object]:
        best_seed: int | None = None
        best_overlap: float | None = None
        best_ssim: float | None = None
        if generation.candidates:
            best = generation.best_candidate
            best_seed = int(best.seed)
            best_overlap = float(best.overlap_score)
            best_ssim = float(best.metrics.ssim)
        return {
            "generation": int(generation.index),
            "reward": float(generation.reward_summary),
            "difficulty": float(generation.difficulty_level),
            "best_seed": best_seed,
            "best_overlap": best_overlap,
            "best_ssim": best_ssim,
        }

    def _persist_generation_history(self, generation: GenerationRecord) -> None:
        if not self.run_id:
            return
        try:
            append_history(self.run_id, self._history_payload(generation))
        except Exception:  # pragma: no cover - defensive persistence
            logger.debug("Failed to persist generation history", exc_info=True)

    def sync_history(self) -> None:
        if not self.run_id:
            return
        rows = [self._history_payload(record) for record in self.generations]
        if not rows:
            return
        try:
            append_history(self.run_id, rows, replace=True)
        except Exception:  # pragma: no cover - defensive persistence
            logger.debug("Failed to rebuild generation history", exc_info=True)

    # ------------------------------------------------------------------ parent/seed helpers
    def _spawn_child_seed(self, anchors: Iterable[int]) -> int:
        anchor_list = list(int(seed) & 0x7FFFFFFF for seed in anchors)
        if not anchor_list:
            return int(self.rng.integers(0, 2**31))
        sample_size = min(3, len(anchor_list))
        selected = np.array(self.rng.choice(anchor_list, size=sample_size, replace=False), dtype=np.int64)
        logistic_source = float(self.rng.random())
        logistic = 3.999 * logistic_source * (1.0 - logistic_source)
        combined = 0
        for idx, parent_seed in enumerate(selected):
            shift = (idx * 17) % 31
            combined ^= (int(parent_seed) << shift) & 0x7FFFFFFF
        walsh = int(np.bitwise_xor.reduce(selected ^ np.roll(selected, 1))) & 0x7FFFFFFF
        noise = int(self.rng.integers(0, 2**31))
        chaotic = _chaotic_seed_mix(selected.tolist(), noise, logistic)
        logistic_component = int(abs(logistic) * 0x7FFFFFFF) & 0x7FFFFFFF
        mutation = int(self.rng.integers(0, 2**31))
        return (combined ^ walsh ^ chaotic ^ logistic_component ^ mutation) & 0x7FFFFFFF

    def _select_seed_pool(self, parent_selection: Sequence[int] | None, extra: int) -> list[int]:
        anchors: list[int] = []
        if parent_selection:
            seen: set[int] = set()
            for seed in parent_selection:
                seed_int = int(seed) & 0x7FFFFFFF
                if seed_int not in seen:
                    seen.add(seed_int)
                    anchors.append(seed_int)
        elif self._elite_pool:
            anchors.extend(self._elite_pool[-self.population_size :])

        seeds: list[int] = []
        total_needed = self.population_size + max(0, extra)
        if anchors:
            keep = min(len(anchors), max(self.population_size - 1, 1))
            seeds.extend(anchors[:keep])
            if len(seeds) < self.population_size:
                seeds.append(self._spawn_child_seed(anchors))
        while len(seeds) < total_needed:
            if anchors:
                seeds.append(self._spawn_child_seed(anchors))
            else:
                seeds.append(int(self.rng.integers(0, 2**31)))
        return seeds[:total_needed]

    def _update_parent_lineage(
        self,
        seed: int,
        generation_index: int,
        metrics: ReconstructionMetrics,
        overlap: float,
        reward: float,
    ) -> None:
        entry = self._parent_lineage.get(seed)
        if entry is None:
            entry = ParentLineage(
                seed=seed,
                origin_generation=generation_index,
                metrics=metrics,
                overlap_score=overlap,
                cumulative_reward=reward,
                peak_reward=reward,
                last_generation=generation_index,
            )
            self._parent_lineage[seed] = entry
            return
        entry.metrics = metrics
        entry.overlap_score = overlap
        entry.appearances += 1
        entry.cumulative_reward += reward
        entry.peak_reward = max(entry.peak_reward, reward)
        entry.last_generation = generation_index

    def _mark_gpu_warning(self) -> bool:
        """Mark that the GPU warning has been emitted, returning ``True`` once."""

        with self._state_lock:
            if self._gpu_warning_emitted:
                return False
            self._gpu_warning_emitted = True
            return True

    def _ensure_parallel_executor(
        self, workers: int
    ) -> concurrent.futures.ThreadPoolExecutor:
        """Return a thread pool sized for *workers* using daemon worker threads."""

        with self._state_lock:
            current = self._executor
            if current is not None and getattr(current, "_max_workers", None) == workers:
                return current
            if current is not None:
                current.shutdown(wait=False, cancel_futures=True)
            # ThreadPoolExecutor already spawns daemon threads; we simply reuse it with a
            # descriptive prefix so diagnostics stay readable.
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="umbra-evo",
            )
            return self._executor

    # ------------------------------------------------------------------ candidate evaluation
    def _evaluate_candidate(self, seed: int, difficulty: float) -> CandidateResult:
        reference = self.original
        start = time.time()
        prefer_gpu = cp is not None
        allow_cpu_fallback = True
        reconstruction: np.ndarray
        packet: NoisePacket | None = None
        backend = "cpu"

        if not prefer_gpu and self._mark_gpu_warning():
            logger.warning(
                "GPU acceleration unavailable; falling back to CPU execution for evolution runs."
            )

        if prefer_gpu:
            try:
                ensure_nvrtc_configured()
            except Exception:  # pragma: no cover - optional diagnostics
                logger.debug("NVRTC configuration failed; falling back to CPU", exc_info=True)
                prefer_gpu = False

        if self.encoder.sigma <= 0:
            reconstruction = reference.copy()
        else:
            attempt_gpu = prefer_gpu
            while True:
                try:
                    packet = self.encoder.encode(
                        reference,
                        seed,
                        allow_cpu_fallback=allow_cpu_fallback,
                        prefer_gpu=attempt_gpu,
                    )
                    backend = getattr(packet, "encoded_backend", "gpu" if attempt_gpu else "cpu")
                    if attempt_gpu and backend != "gpu" and self._mark_gpu_warning():
                        logger.warning(
                            "GPU acceleration unavailable; falling back to CPU execution for evolution runs."
                        )
                    reconstruction = self.decoder.decode(
                        packet,
                        seed,
                        allow_cpu_fallback=allow_cpu_fallback,
                    )
                    break
                except GPUAccelerationRequiredError:
                    raise
                except ValueError as exc:
                    if "Sigma must be positive" in str(exc):
                        reconstruction = reference.copy()
                        packet = None
                        backend = "cpu"
                        break
                    raise
                except Exception as exc:  # pragma: no cover - GPU fallback path
                    if attempt_gpu and allow_cpu_fallback:
                        if self._mark_gpu_warning():
                            logger.warning(
                                "GPU acceleration unavailable; falling back to CPU execution for evolution runs."
                            )
                        attempt_gpu = False
                        continue
                    raise exc

        recon = np.clip(np.asarray(reconstruction, dtype=np.float32), 0.0, 1.0)
        metrics = compute_metrics(reference, recon)
        _, overlap = multiplicative_overlap(reference, recon)
        reward = composite_score(overlap, metrics.psnr, metrics.ssim)
        ai_score = reward
        predicted_reward: float | None = None
        channel_axis = -1 if reference.ndim == 3 else None
        feature_vector: tuple[float, ...] = (
            float(overlap),
            metrics.psnr,
            metrics.ssim,
            compute_ms_ssim(reference, recon, channel_axis),
            dct_band_correlation(reference, recon),
        )
        reward_components = {
            "overlap": float(overlap),
            "psnr": metrics.psnr,
            "ssim": metrics.ssim,
        }

        if self._reward_model is not None:
            try:
                feature_array = np.asarray(feature_vector, dtype=np.float32)
                with self._reward_lock:
                    predicted_reward = float(self._reward_model.predict(feature_array))
                    self._reward_model.update(
                        feature_array.reshape(1, -1), np.array([reward], dtype=np.float32)
                    )
            except Exception:  # pragma: no cover - advisory failure should not abort evolution
                logger.debug("Neural advisor update failed", exc_info=True)
                predicted_reward = None

        waveform_image: np.ndarray | None = None
        waveform_reference_metrics: ReconstructionMetrics | None = None
        waveform_reference_overlap: float | None = None
        waveform_packet_metrics: ReconstructionMetrics | None = None
        waveform_packet_overlap: float | None = None
        waveform_reference_partial: float | None = None
        waveform_alignment_partial: float | None = None
        waveform_sound_score: float | None = None
        waveform_readability_score: float | None = None
        waveform_alignment_score: float | None = None
        waveform_sample_rate: int | None = None
        waveform_segments: int | None = None
        waveform_marker_duration: float | None = None
        team_score_value: float | None = None

        if self.enable_waveform:
            try:
                waveform_sample_rate = suggest_sample_rate(reference)
                waveform_segments, waveform_marker_duration = suggest_transmission_profile(reference)
                wav_bytes = encode_image_to_wav_bytes(
                    recon,
                    sample_rate=waveform_sample_rate,
                    segments=waveform_segments,
                    marker_duration=waveform_marker_duration,
                )
                waveform_image, metadata = decode_wav_bytes_to_image(
                    wav_bytes,
                    resolution=recon.shape[:2],
                    sample_rate=waveform_sample_rate,
                    segments=waveform_segments,
                    marker_duration=waveform_marker_duration,
                    return_metadata=True,
                )
                waveform_sample_rate = metadata.sample_rate or waveform_sample_rate
                waveform_segments = metadata.segments or waveform_segments
                waveform_marker_duration = metadata.marker_duration or waveform_marker_duration
                waveform_image = _ensure_three_channel(waveform_image)
                waveform_reference_metrics = compute_metrics(reference, waveform_image)
                _, waveform_reference_overlap = multiplicative_overlap(reference, waveform_image)
                waveform_packet_metrics = compute_metrics(recon, waveform_image)
                _, waveform_packet_overlap = multiplicative_overlap(recon, waveform_image)
                try:
                    waveform_reference_partial = partial_alignment_fraction(reference, waveform_image)
                except ValueError:
                    waveform_reference_partial = None
                try:
                    waveform_alignment_partial = partial_alignment_fraction(recon, waveform_image)
                except ValueError:
                    waveform_alignment_partial = None
                if waveform_packet_metrics is not None and waveform_packet_overlap is not None:
                    waveform_alignment_score = audio_fidelity_score(
                        float(waveform_packet_overlap),
                        waveform_packet_metrics.psnr,
                        waveform_packet_metrics.ssim,
                        partial_credit=waveform_alignment_partial,
                    )
                if waveform_reference_metrics is not None and waveform_reference_overlap is not None:
                    waveform_sound_score = audio_fidelity_score(
                        float(waveform_reference_overlap),
                        waveform_reference_metrics.psnr,
                        waveform_reference_metrics.ssim,
                        partial_credit=waveform_reference_partial,
                    )
                    waveform_readability_score = readability_score(
                        float(waveform_reference_overlap),
                        waveform_reference_metrics.psnr,
                        waveform_reference_metrics.ssim,
                    )
                team_score_value = team_cohesion_score(
                    float(overlap),
                    metrics.psnr,
                    metrics.ssim,
                    sound_reference_overlap=None
                    if waveform_reference_overlap is None
                    else float(waveform_reference_overlap),
                    sound_reference_psnr=None
                    if waveform_reference_metrics is None
                    else waveform_reference_metrics.psnr,
                    sound_reference_ssim=None
                    if waveform_reference_metrics is None
                    else waveform_reference_metrics.ssim,
                    sound_alignment_overlap=None
                    if waveform_packet_overlap is None
                    else float(waveform_packet_overlap),
                    sound_alignment_psnr=None
                    if waveform_packet_metrics is None
                    else waveform_packet_metrics.psnr,
                    sound_alignment_ssim=None
                    if waveform_packet_metrics is None
                    else waveform_packet_metrics.ssim,
                    sound_reference_partial=waveform_reference_partial,
                    sound_alignment_partial=waveform_alignment_partial,
                    readability=waveform_readability_score,
                )
            except GPUAccelerationRequiredError:
                raise
            except Exception:  # pragma: no cover - waveform diagnostics
                logger.debug("Waveform reconstruction failed", exc_info=True)
                waveform_image = None

        duration_ms = (time.time() - start) * 1000.0
        return CandidateResult(
            seed=int(seed),
            reconstruction=recon,
            metrics=metrics,
            overlap_score=float(overlap),
            ai_score=float(ai_score),
            reward=float(reward),
            execution_backend=backend,
            sigma=float(self.encoder.sigma),
            frame_time_ms=duration_ms,
            predicted_reward=predicted_reward,
            feature_vector=feature_vector,
            reward_components=reward_components,
            waveform_reconstruction=waveform_image,
            waveform_reference_metrics=waveform_reference_metrics,
            waveform_reference_overlap=waveform_reference_overlap,
            waveform_packet_metrics=waveform_packet_metrics,
            waveform_packet_overlap=waveform_packet_overlap,
            waveform_reference_partial=waveform_reference_partial,
            waveform_alignment_partial=waveform_alignment_partial,
            waveform_sample_rate=waveform_sample_rate,
            waveform_segments=waveform_segments,
            waveform_marker_duration=waveform_marker_duration,
            waveform_sound_score=waveform_sound_score,
            waveform_readability_score=waveform_readability_score,
            waveform_alignment_score=waveform_alignment_score,
            team_score=team_score_value,
        )

    # ------------------------------------------------------------------ difficulty management
    def _current_difficulty(self, generation_index: int) -> float:
        if not DIFFICULTY_LADDER["enabled"]:
            return self._difficulty_bias
        period = max(1, int(DIFFICULTY_LADDER["period_gens"]))
        spike_len = max(1, int(DIFFICULTY_LADDER["spike_len_gens"]))
        base = float(DIFFICULTY_LADDER["base"])
        spike = float(DIFFICULTY_LADDER["spike"])
        cycle_pos = generation_index % period
        if cycle_pos < spike_len:
            return float(np.clip(spike, 0.1, 1.0))
        return float(np.clip(base, 0.1, 1.0))

    def _update_plateau(self, best_overlap: float) -> None:
        self._overlap_history.append(best_overlap)
        window = max(1, int(PLATEAU_CFG["window"]))
        if len(self._overlap_history) < window:
            return
        recent = self._overlap_history[-window:]
        improvement = float(max(recent) - min(recent))
        threshold = float(PLATEAU_CFG["delta_threshold"]) * 100.0
        if improvement < threshold:
            self._plateau_generations += 1
            if self.encoder.sigma > 0.05:
                old_sigma = float(self.encoder.sigma)
                self.encoder.sigma = max(0.01, self.encoder.sigma * 0.9)
                if PLATEAU_CFG["log"]:
                    logger.debug(
                        "Plateau detected; reducing sigma from %.3f to %.3f", old_sigma, self.encoder.sigma
                    )
            boost_cap = max(1, int(PLATEAU_CFG["boost_cap_factor"]))
            boost_step = max(1, int(PLATEAU_CFG["boost_step"]))
            self._mutation_boost = min(self._mutation_boost + boost_step, boost_cap)
        else:
            self._plateau_generations = 0
            self._mutation_boost = max(0, self._mutation_boost - 1)

    def _adapt_population(
        self,
        duration: float,
        evaluated: int,
        worker_count: int,
        batch_size: int,
    ) -> dict[str, Any]:
        """Adjust population and batch sizing heuristics for the next generation."""

        adjustments: dict[str, Any] = {
            "evaluated": int(evaluated),
            "previous_population": int(self.population_size),
            "previous_batch_size": int(batch_size),
            "target_duration": float(self._target_generation_seconds),
        }

        duration = float(max(duration, 1e-6))
        smoothing = 0.25
        if self._duration_ema <= 0.0:
            self._duration_ema = duration
        else:
            self._duration_ema = (
                (1.0 - smoothing) * self._duration_ema + smoothing * duration
            )
        throughput = float(evaluated) / max(duration, 1e-6)
        if self._throughput_ema <= 0.0:
            self._throughput_ema = throughput
        else:
            self._throughput_ema = (
                (1.0 - smoothing) * self._throughput_ema + smoothing * throughput
            )

        adjustments["duration_ema"] = float(self._duration_ema)
        adjustments["throughput_ema"] = float(self._throughput_ema)

        population_before = int(self.population_size)
        reason: str | None = None
        slow_threshold = self._target_generation_seconds * 1.15
        fast_threshold = self._target_generation_seconds * 0.6

        if (
            self._duration_ema > slow_threshold
            and self.population_size > self._population_floor
        ):
            proposed = max(
                self._population_floor,
                int(round(self.population_size * 0.75)),
            )
            if proposed >= self.population_size:
                proposed = max(self._population_floor, self.population_size - 1)
            if self._set_population_size(proposed):
                reason = "slow"
        elif (
            self._duration_ema < fast_threshold
            and self.population_size < self._population_ceiling
        ):
            proposed = min(
                self._population_ceiling,
                max(self.population_size + 1, int(round(self.population_size * 1.15))),
            )
            if self._set_population_size(proposed):
                reason = "fast"

        max_batch = min(self.population_size, self._parallel_workers * 2)
        desired_batch = self._active_batch_size
        if self._duration_ema > slow_threshold:
            desired_batch = max(1, min(desired_batch, worker_count))
        elif self._duration_ema < fast_threshold:
            desired_batch = max(desired_batch, min(max_batch, worker_count + 1))

        if self._throughput_ema > 0.0:
            throughput_batch = int(
                max(
                    1,
                    min(
                        self.population_size,
                        round(
                            self._throughput_ema
                            * min(self._target_generation_seconds * 0.5, slow_threshold)
                            / max(worker_count or 1, 1)
                        ),
                    ),
                )
            )
            if self._duration_ema > slow_threshold:
                desired_batch = min(desired_batch, throughput_batch)
            elif self._duration_ema < fast_threshold:
                desired_batch = max(desired_batch, throughput_batch)

        desired_batch = max(1, min(desired_batch, max_batch))
        if desired_batch < self._active_batch_size:
            self._active_batch_size = max(1, (self._active_batch_size + desired_batch) // 2)
        else:
            self._active_batch_size = min(max_batch, desired_batch)

        adjustments["population"] = int(self.population_size)
        adjustments["population_changed"] = self.population_size != population_before
        if reason is not None and adjustments["population_changed"]:
            adjustments["population_adjustment_reason"] = reason
            adjustments["population_adjusted_to"] = int(self.population_size)
        adjustments["next_batch_size"] = int(self._active_batch_size)
        return adjustments

    def _apply_hyper_feedback(self, generation: GenerationRecord, duration: float) -> None:
        if not self._hyper_profile.enabled:
            return
        profile = self._hyper_profile
        previous_capacity = profile.batch_size * max(profile.dwell_generations, 1)
        profile.batch_size = max(profile.batch_size, self.population_size)
        if generation.reward_peak > 0 and profile.batch_size < self._population_ceiling:
            profile.batch_size = min(self._population_ceiling, profile.batch_size + 1)
        profile.batch_size = max(self._population_floor, profile.batch_size)
        profile.dwell_generations = max(profile.dwell_generations, 10)
        profile.dwell_generations += 1
        profile.autosave_interval = max(profile.autosave_interval, self.autosave_interval)
        if duration > 0 and generation.candidates:
            throughput = len(generation.candidates) / duration
            profile.mean_duration = (
                0.8 * profile.mean_duration + 0.2 * duration
                if profile.mean_duration
                else duration
            )
            profile.throughput = (
                0.8 * profile.throughput + 0.2 * throughput
                if profile.throughput
                else throughput
            )
        profile.last_update = generation.index
        current_capacity = profile.batch_size * profile.dwell_generations
        if current_capacity < previous_capacity:
            profile.dwell_generations = max(profile.dwell_generations, previous_capacity // max(profile.batch_size, 1))
        self._set_population_size(profile.batch_size)
        self.autosave_interval = max(self.autosave_interval, profile.autosave_interval)

    # ------------------------------------------------------------------ main evolution loop
    def run_generation(self, parent_selection: Sequence[int] | None = None) -> GenerationRecord:
        if self.max_generations is not None and self.next_generation_index >= self.max_generations:
            raise EvolutionLimitReached(self.next_generation_index, self.max_generations)

        generation_index = self.next_generation_index
        difficulty = self._current_difficulty(generation_index)
        mutation_extra = self._mutation_boost if self._mutation_boost > 0 else 0
        profiler = _LoopProfiler()
        with profiler.track("seed_selection"):
            seeds = self._select_seed_pool(parent_selection, mutation_extra)

        record = GenerationRecord(index=generation_index)
        record.difficulty_raw = difficulty
        start = time.time()
        best_overlap = 0.0
        warmup_penalty = 0.0
        if generation_index == 0 and float(self.encoder.sigma) >= 0.1:
            warmup_penalty = min(0.5, float(self.encoder.sigma) * 0.05)

        worker_count = min(len(seeds), self._parallel_workers)
        batch_size = max(1, min(self._active_batch_size, len(seeds)))

        def _ingest(candidate: CandidateResult) -> None:
            nonlocal best_overlap
            if warmup_penalty:
                candidate.overlap_score = max(
                    0.0, candidate.overlap_score - warmup_penalty
                )
                candidate.reward = max(0.0, candidate.reward - warmup_penalty)
            record.candidates.append(candidate)
            record.reward_summary += candidate.reward
            record.reward_peak = max(record.reward_peak, candidate.reward)
            best_overlap = max(best_overlap, candidate.overlap_score)
            self._update_parent_lineage(
                candidate.seed,
                generation_index,
                candidate.metrics,
                candidate.overlap_score,
                candidate.reward,
            )

        with profiler.track("candidate_evaluation"):
            if worker_count > 1:
                executor = self._ensure_parallel_executor(worker_count)
                for chunk in _batched(seeds, batch_size):
                    if len(chunk) == 1:
                        _ingest(self._evaluate_candidate(chunk[0], difficulty))
                        continue
                    for candidate in executor.map(
                        self._evaluate_candidate,
                        chunk,
                        itertools.repeat(difficulty, len(chunk)),
                        chunksize=1,
                    ):
                        _ingest(candidate)
            else:
                for seed in seeds:
                    _ingest(self._evaluate_candidate(seed, difficulty))

        with profiler.track("postprocess"):
            if not record.candidates:
                raise RuntimeError("No candidates evaluated; population size may be zero")

            best_candidate = record.best_candidate
            record.improvement = max(best_candidate.overlap_score - self._best_overlap, 0.0)
            self._best_overlap = max(self._best_overlap, best_candidate.overlap_score)
            record.difficulty_level = float(np.clip(best_candidate.overlap_score / 100.0, 0.0, 1.0))
            record.cumulative_reward = self.lifetime_reward + record.reward_summary

        duration = time.time() - start
        adapt_start = time.perf_counter()
        adjustments = self._adapt_population(
            duration, len(record.candidates), worker_count, batch_size
        )
        population_before_hyper = self.population_size
        self._apply_hyper_feedback(record, duration)
        if self.population_size != population_before_hyper:
            adjustments["population"] = int(self.population_size)
            adjustments["population_changed"] = True
            adjustments["population_adjusted_to"] = int(self.population_size)
            adjustments["population_adjustment_reason"] = "hyper"
        adjustments["next_batch_size"] = int(self._active_batch_size)
        self._update_plateau(best_overlap)
        profiler.add("adaptive_controls", time.perf_counter() - adapt_start)

        persist_start = time.perf_counter()
        self._persist_generation_history(record)
        persist_duration = time.perf_counter() - persist_start
        profiler.add("history_persist", persist_duration)

        loop_data = profiler.as_dicts()
        loop_map = {entry["name"]: float(entry["seconds"]) for entry in loop_data}
        record.loop_timings = loop_data

        frame_times = [
            float(candidate.frame_time_ms)
            for candidate in record.candidates
            if candidate.frame_time_ms is not None
        ]
        avg_frame_ms = float(np.mean(frame_times)) if frame_times else 0.0

        record.timing_summary = {
            "total_seconds": float(duration),
            "candidate_count": len(record.candidates),
            "average_frame_ms": float(avg_frame_ms),
            "throughput": float(len(record.candidates) / max(duration, 1e-6)),
            "best_overlap": float(best_candidate.overlap_score),
            "reward_peak": float(record.reward_peak),
            "improvement": float(record.improvement),
            "ema_duration": float(self._duration_ema),
            "ema_throughput": float(self._throughput_ema),
            "seed_selection_seconds": float(loop_map.get("seed_selection", 0.0)),
            "evaluation_seconds": float(loop_map.get("candidate_evaluation", 0.0)),
            "postprocess_seconds": float(loop_map.get("postprocess", 0.0)),
            "adaptive_seconds": float(loop_map.get("adaptive_controls", 0.0)),
            "persist_seconds": float(loop_map.get("history_persist", persist_duration)),
        }

        population_adjusted_to = adjustments.get("population_adjusted_to")
        if population_adjusted_to is not None:
            population_adjusted_to = int(population_adjusted_to)

        record.worker_summary = {
            "evaluated": len(record.candidates),
            "seed_count": len(seeds),
            "workers": int(worker_count),
            "batch_size": int(batch_size),
            "next_batch_size": int(adjustments.get("next_batch_size", self._active_batch_size)),
            "previous_batch_size": int(adjustments.get("previous_batch_size", batch_size)),
            "population": int(self.population_size),
            "previous_population": int(
                adjustments.get("previous_population", self.population_size)
            ),
            "population_changed": bool(adjustments.get("population_changed", False)),
            "population_adjusted_to": population_adjusted_to,
            "population_adjustment_reason": adjustments.get(
                "population_adjustment_reason"
            ),
            "duration": float(duration),
            "duration_ema": float(self._duration_ema),
            "throughput_ema": float(self._throughput_ema),
            "target_duration": float(self._target_generation_seconds),
            "warmup_penalty": float(warmup_penalty),
            "history_persist_seconds": float(loop_map.get("history_persist", persist_duration)),
        }

        self.generations.append(record)
        self.next_generation_index += 1
        self.lifetime_reward += record.reward_summary
        self.reward_trace.append(record.reward_summary)
        self.difficulty_trace.append(record.difficulty_level)

        if best_candidate.seed not in self._elite_pool:
            self._elite_pool.append(best_candidate.seed)
            if len(self._elite_pool) > self.population_size * 4:
                self._elite_pool = self._elite_pool[-self.population_size * 4 :]

        return record

    # ------------------------------------------------------------------ persistence API
    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        session = EvolutionSession(
            image_signature=self.image_signature,
            original=self.original,
            encoder_config={
                "sigma": getattr(self.encoder, "sigma", 0.2),
                "waveform_plugin": getattr(getattr(self.encoder, "_plugin", None), "name", "dsss"),
            },
            decoder_config={
                "denoise_sigma": getattr(self.decoder, "denoise_sigma", None),
            },
            population_size=self.population_size,
            base_seed=self.base_seed,
            autosave_interval=self.autosave_interval,
            generations=self.generations,
            rng_state=self.rng.bit_generator.state,
            run_id=self.run_id,
            next_generation_index=self.next_generation_index,
            parent_lineage=self.parent_lineage,
            lifetime_reward=self.lifetime_reward,
            reward_trace=self.reward_trace,
            difficulty_trace=self.difficulty_trace,
            elite_seeds=self._elite_pool,
            advisor_state=self._reward_model.to_state() if self._reward_model else None,
            best_overlap=self._best_overlap,
            plateau_generations=self._plateau_generations,
            mutation_boost=self._mutation_boost,
            hyper_profile=self._hyper_profile.to_dict(),
            enable_waveform=self.enable_waveform,
            max_generations=self.max_generations,
        )
        target = path / "evolution_session.pkl"
        with target.open("wb") as handle:
            pickle.dump(session, handle)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> EvolutionManager:
        path = Path(directory)
        if path.is_dir():
            file_path = path / "evolution_session.pkl"
        else:
            file_path = path
        with file_path.open("rb") as handle:
            session: EvolutionSession = pickle.load(handle)
        encoder = NoiseStreamEncoder(
            sigma=float(session.encoder_config.get("sigma", 0.2)),
            waveform_plugin=session.encoder_config.get("waveform_plugin", "dsss"),
        )
        decoder = NoiseStreamDecoder(
            denoise_sigma=session.decoder_config.get("denoise_sigma", None)
        )
        manager = cls(
            original=session.original,
            encoder=encoder,
            decoder=decoder,
            population_size=session.population_size,
            base_seed=session.base_seed,
            autosave_interval=session.autosave_interval,
            run_id=session.run_id,
            next_generation_index=session.next_generation_index,
            enable_waveform=session.enable_waveform,
            max_generations=session.max_generations,
        )
        manager.generations = session.generations
        manager.rng.bit_generator.state = session.rng_state
        manager._parent_lineage = {entry.seed: entry for entry in session.parent_lineage}
        manager.lifetime_reward = session.lifetime_reward
        manager.reward_trace = list(session.reward_trace)
        manager.difficulty_trace = list(session.difficulty_trace)
        manager._elite_pool = list(session.elite_seeds)
        manager._best_overlap = float(session.best_overlap)
        manager._plateau_generations = int(session.plateau_generations)
        manager._mutation_boost = int(session.mutation_boost)
        if session.hyper_profile:
            try:
                manager._hyper_profile = HyperPerformanceProfile.from_dict(dict(session.hyper_profile))
            except Exception:  # pragma: no cover - corrupted save fallback
                logger.debug("Failed to restore hyper profile", exc_info=True)
                manager._hyper_profile = manager.default_hyper_profile(
                    session.population_size, session.autosave_interval
                )
        if session.advisor_state is not None:
            try:
                from .neural import NeuralRewardModel

                manager._reward_model = NeuralRewardModel.from_state(session.advisor_state)
            except Exception:  # pragma: no cover - advisor restoration optional
                logger.debug("Failed to restore neural advisor", exc_info=True)
                manager._reward_model = None
        manager.next_generation_index = max(
            manager.next_generation_index, int(session.next_generation_index)
        )
        return manager


__all__ = [
    "CandidateResult",
    "EvolutionManager",
    "EvolutionSession",
    "EvolutionLimitReached",
    "GenerationRecord",
    "ParentLineage",
    "HyperPerformanceProfile",
    "_chaotic_seed_mix",
]
