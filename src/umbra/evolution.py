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

from .codec import decode_waveform_to_image, encode_image_to_waveform
from .decoding import NoiseStreamDecoder
from .encoding import NoiseStreamEncoder
from .metrics import (
    ReconstructionMetrics,
    compute_metrics,
    compute_ms_ssim,
    dct_band_correlation,
)
from .reconstruction import suggest_sample_rate, suggest_transmission_profile
from .runs import append_history, get_run_paths, load_history, new_run
from .visualization import multiplicative_overlap

if TYPE_CHECKING:  # pragma: no cover - optional neural advisor import
    from .neural import NeuralRewardModel

logger = logging.getLogger(__name__)


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
    digest = hashlib.blake2s(buffer, person=b"umbChaos").digest()
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
    "enabled": os.getenv("UMBRA_DIFFICULTY_LADDER_ENABLED", "0")
    not in {"0", "false", "False"},
    "base": _env_float("UMBRA_DIFFICULTY_LADDER_BASE", DIFFICULTY_LADDER_DEFAULTS["base"]),
    "spike": _env_float(
        "UMBRA_DIFFICULTY_LADDER_SPIKE", DIFFICULTY_LADDER_DEFAULTS["spike"]
    ),
    "period_gens": _env_int(
        "UMBRA_DIFFICULTY_LADDER_PERIOD", DIFFICULTY_LADDER_DEFAULTS["period_gens"]
    ),
    "spike_len_gens": _env_int(
        "UMBRA_DIFFICULTY_LADDER_SPIKE_LEN",
        DIFFICULTY_LADDER_DEFAULTS["spike_len_gens"],
    ),
}


REWARD_MODE = os.getenv("UMBRA_REWARD_MODE", "strict").strip().lower()
if REWARD_MODE not in {"strict", "perceptual_mix"}:
    REWARD_MODE = "strict"

REWARD_CFG = {
    "strict": {"alpha_overlap": 0.25, "beta_msssim": 0.0, "gamma_dct_corr": 0.0},
    "perceptual_mix": {"alpha_overlap": 0.8, "beta_msssim": 0.15, "gamma_dct_corr": 0.05},
}


def normalize_difficulty(raw_value: float) -> float:
    """Map the internal difficulty score to the [0, 1] display scale."""

    return float(np.clip(raw_value, 0.0, 1.0))


@dataclass
class HyperPerformanceProfile:
    """Summary of hyper performance tuning recommendations."""

    enabled: bool = False
    target_subjects: int = 0
    batch_size: int = 0
    dwell_generations: int = 0
    autosave_interval: int = 0
    queue_generations: int = 0
    mean_duration: float = 0.0
    throughput: float = 0.0
    last_update: int = -1

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "target_subjects": int(self.target_subjects),
            "batch_size": int(self.batch_size),
            "dwell_generations": int(self.dwell_generations),
            "autosave_interval": int(self.autosave_interval),
            "queue_generations": int(self.queue_generations),
            "mean_duration": float(self.mean_duration),
            "throughput": float(self.throughput),
            "last_update": int(self.last_update),
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
    waveform_reconstruction: np.ndarray | None = None
    waveform_reference_metrics: ReconstructionMetrics | None = None
    waveform_reference_overlap: float | None = None
    waveform_packet_metrics: ReconstructionMetrics | None = None
    waveform_packet_overlap: float | None = None
    waveform_sample_rate: int | None = None
    waveform_segments: int | None = None
    waveform_marker_duration: float | None = None
    reward: float = 0.0
    predicted_reward: float | None = None
    feature_vector: tuple[float, ...] = field(default_factory=tuple)
    reward_components: dict[str, float] = field(default_factory=dict)


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
            key=lambda cand: (cand.overlap_score, cand.metrics.ssim, cand.reward),
        )

    @property
    def difficulty_normalized(self) -> float:
        base = self.difficulty_level if self.difficulty_level else self.difficulty_raw
        return normalize_difficulty(base)


@dataclass
class EvolutionSession:
    """Serializable snapshot of an :class:`EvolutionManager`."""

    image_signature: str
    original: np.ndarray
    encoder_config: dict[str, float]
    decoder_config: dict[str, float | None]
    population_size: int
    base_seed: int
    autosave_interval: int
    generations: list[GenerationRecord]
    rng_state: dict
    run_id: str | None = None
    next_generation_index: int = 0
    parent_lineage: list[ParentLineage] = field(default_factory=list)
    lifetime_reward: float = 0.0
    reward_trace: list[float] = field(default_factory=list)
    difficulty_trace: list[float] = field(default_factory=list)
    elite_seeds: list[int] = field(default_factory=list)
    advisor_state: dict[str, Any] | None = None
    best_overlap: float = 0.0
    plateau_generations: int = 0
    mutation_boost: int = 0
    hyper_profile: dict[str, Any] | None = None
    enable_waveform: bool = True


@dataclass
class ParentLineage:
    """History entry recording a seed's performance within the lineage."""

    seed: int
    origin_generation: int
    metrics: ReconstructionMetrics
    overlap_score: float
    appearances: int = 0
    cumulative_reward: float = 0.0
    peak_reward: float = 0.0
    last_generation: int = -1


def compute_image_signature(array: np.ndarray) -> str:
    """Compute a stable hash for the provided image array."""

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
    ) -> None:
        self.original = np.asarray(original, dtype=np.float32)
        self.encoder = encoder
        self.decoder = decoder
        self._hyper_enabled = self.hyper_mode_enabled()
        if self._hyper_enabled:
            baseline = self.default_hyper_profile()
            batch_size = max(1, baseline.batch_size or int(population_size))
            autosave = max(1, baseline.autosave_interval or int(autosave_interval))
            self.population_size = batch_size
            self.autosave_interval = autosave
            self._hyper_profile = baseline
        else:
            self.population_size = max(1, int(population_size))
            self.autosave_interval = max(1, int(autosave_interval))
            self._hyper_profile = HyperPerformanceProfile(enabled=False)
        self.base_seed = int(base_seed)
        if run_id is None:
            run_id, run_dir = new_run()
        else:
            run_id = str(run_id)
            run_dir, _ = get_run_paths(run_id)
        self.run_id = run_id
        self._run_directory = run_dir
        self.enable_waveform = bool(enable_waveform)
        if next_generation_index is None:
            self.next_generation_index = 0
        else:
            self.next_generation_index = max(0, int(next_generation_index))
        self.generations: list[GenerationRecord] = []
        self.rng = np.random.default_rng(self.base_seed)
        self._parent_lineage: dict[int, ParentLineage] = {}
        self._carryover_generations = 3
        self._elite_pool: list[int] = []
        self._reward_model: NeuralRewardModel | None = advisor
        self.lifetime_reward: float = 0.0
        self.reward_trace: list[float] = []
        self.difficulty_trace: list[float] = []
        self._best_overlap: float = 0.0
        self._plateau_generations: int = 0
        self._mutation_boost: int = 0
        self._duration_ema: float = 0.0
        self._throughput_ema: float = 0.0

    @property
    def mutation_boost(self) -> int:
        """Expose the number of additional exploratory candidates."""

        return self._mutation_boost

    @staticmethod
    def hyper_mode_enabled() -> bool:
        """Return whether hyper performance mode is active via environment."""

        return _env_flag("UMBRA_HYPER_MODE")

    @staticmethod
    def default_hyper_profile() -> HyperPerformanceProfile:
        """Return the baseline hyper performance profile."""

        if not EvolutionManager.hyper_mode_enabled():
            return HyperPerformanceProfile(enabled=False)
        return HyperPerformanceProfile(
            enabled=True,
            target_subjects=150,
            batch_size=5,
            dwell_generations=30,
            autosave_interval=10,
            queue_generations=30,
            mean_duration=0.0,
            throughput=0.0,
            last_update=-1,
        )

    @property
    def hyper_profile(self) -> HyperPerformanceProfile:
        """Expose the most recent hyper performance recommendations."""

        return self._hyper_profile

    @property
    def parent_lineage(self) -> list[ParentLineage]:
        """Return a snapshot of the current parent lineage."""

        return list(self._parent_lineage.values())

    @property
    def image_signature(self) -> str:
        return compute_image_signature(self.original)

    @property
    def reward_advisor(self) -> NeuralRewardModel | None:
        return self._reward_model

    def set_advisor(self, advisor: NeuralRewardModel | None) -> None:
        """Attach or detach a neural reward advisor."""

        self._reward_model = advisor

    @property
    def run_directory(self) -> Path:
        """Return the filesystem directory backing this evolution run."""

        return self._run_directory

    def append_generation_record(
        self,
        generation: GenerationRecord,
        *,
        persist: bool = True,
        use_next_index: bool = False,
    ) -> None:
        """Append ``generation`` to the history and optionally persist it."""

        if use_next_index or generation.index < 0:
            generation.index = int(self.next_generation_index)
        else:
            generation.index = int(generation.index)
        self.generations.append(generation)
        self.next_generation_index = max(self.next_generation_index, generation.index + 1)
        if persist:
            self._persist_generation_history(generation)

    def sync_history(self) -> None:
        """Ensure the on-disk history matches the in-memory generations."""

        if not self.run_id:
            return

        rows = [self._history_payload(record) for record in self.generations]
        run_dir, history_path = get_run_paths(self.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        if not rows:
            return

        try:
            history = load_history(self.run_id)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to read existing history; forcing rebuild", exc_info=True)
            history = None

        needs_replace = True
        if history is not None and hasattr(history, "empty") and not history.empty:
            try:
                existing_generations = list(history["generation"])
            except Exception:  # pragma: no cover - defensive
                existing_generations = []
            desired_generations = [row["generation"] for row in rows]
            if existing_generations == desired_generations:
                needs_replace = False

        if needs_replace:
            try:
                append_history(self.run_id, rows, replace=True)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to rebuild run history", exc_info=True)

    def _history_payload(self, generation: GenerationRecord) -> dict[str, object]:
        best_seed: int | None = None
        best_overlap: float | None = None
        best_ssim: float | None = None
        if generation.candidates:
            try:
                best = generation.best_candidate
            except ValueError:  # pragma: no cover - defensive
                best = None
            if best is not None:
                best_seed = int(best.seed)
                best_overlap = float(best.overlap_score)
                best_ssim = float(best.metrics.ssim)

        return {
            "generation": int(generation.index),
            "reward_summary": float(getattr(generation, "reward_summary", 0.0)),
            "reward_peak": float(getattr(generation, "reward_peak", 0.0)),
            "cumulative_reward": float(getattr(generation, "cumulative_reward", 0.0)),
            "difficulty_level": float(getattr(generation, "difficulty_level", 0.0)),
            "difficulty_raw": float(getattr(generation, "difficulty_raw", 0.0)),
            "improvement": float(getattr(generation, "improvement", 0.0)),
            "checkpoint_tag": generation.checkpoint_tag,
            "best_seed": best_seed,
            "best_overlap": best_overlap,
            "best_ssim": best_ssim,
        }

    def _persist_generation_history(self, generation: GenerationRecord) -> None:
        if not self.run_id:
            return
        payload = self._history_payload(generation)
        try:
            append_history(self.run_id, payload)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to append generation history", exc_info=True)

    def update_settings(
        self,
        *,
        original: np.ndarray | None = None,
        encoder: NoiseStreamEncoder | None = None,
        decoder: NoiseStreamDecoder | None = None,
        population_size: int | None = None,
        autosave_interval: int | None = None,
    ) -> None:
        """Update runtime settings without discarding existing history."""

        if original is not None:
            self.original = np.asarray(original, dtype=np.float32)
        if encoder is not None:
            self.encoder = encoder
        if decoder is not None:
            self.decoder = decoder
        if population_size is not None and not self._hyper_enabled:
            self.population_size = max(1, int(population_size))
        if autosave_interval is not None and not self._hyper_enabled:
            self.autosave_interval = max(1, int(autosave_interval))
        logger.info(
            "Updated manager settings: population=%d autosave=%d",
            self.population_size,
            self.autosave_interval,
        )

    def _spawn_child_seed(self, anchors: Sequence[int]) -> int:
        """Combine anchor seeds with fresh noise to produce a child seed."""

        if not anchors:
            return int(self.rng.integers(0, np.iinfo(np.int32).max))

        choices = self.rng.choice(anchors, size=min(3, len(anchors)), replace=False)
        selected = np.asarray(choices, dtype=np.int64)
        combined = 0
        for idx, parent_seed in enumerate(selected):
            shift = (idx * 17) % 31
            combined ^= (int(parent_seed) << shift) & 0x7FFFFFFF

        logistic = float(self.rng.random())
        logistic = (3.999 * logistic * (1.0 - logistic)) or float(self.rng.random())
        logistic_component = int(abs(logistic) * 0x7FFFFFFF) & 0x7FFFFFFF

        if selected.size:
            xor_mix = selected ^ np.roll(selected, 1)
            walsh = int(np.bitwise_xor.reduce(xor_mix)) & 0x7FFFFFFF
        else:  # pragma: no cover - defensive
            walsh = 0

        noise = int(self.rng.integers(0, np.iinfo(np.int32).max))
        chaotic = _chaotic_seed_mix(selected.tolist(), noise, logistic)

        mutation = int(self.rng.integers(0, np.iinfo(np.int32).max))
        combined ^= walsh
        combined ^= chaotic
        combined ^= logistic_component
        combined ^= mutation
        return combined & 0x7FFFFFFF

    def _evaluate_candidate(
        self,
        seed: int,
        reference_clipped: np.ndarray,
        *,
        waveform_payload: dict[str, Any] | None,
    ) -> CandidateResult:
        """Run the packet pipeline and optional waveform reconstruction for ``seed``."""

        packet = self.encoder.encode(self.original, int(seed))
        reconstruction = self.decoder.decode(packet, int(seed))
        try:
            recon_image = _ensure_three_channel(reconstruction)
        except ValueError:
            logger.debug(
                "Falling back to reference alignment for seed %d", seed, exc_info=True
            )
            recon_image = reference_clipped

        recon_image = np.asarray(recon_image, dtype=np.float32)
        packet_metrics = compute_metrics(reference_clipped, recon_image)
        _, packet_overlap = multiplicative_overlap(reference_clipped, recon_image)

        waveform_image: np.ndarray | None = None
        waveform_reference_metrics: ReconstructionMetrics | None = None
        waveform_reference_overlap: float | None = None
        waveform_packet_metrics: ReconstructionMetrics | None = None
        waveform_packet_overlap: float | None = None
        waveform_sample_rate: int | None = None
        waveform_segments: int | None = None
        waveform_marker_duration: float | None = None

        if self.enable_waveform and waveform_payload is not None:
            try:
                logger.info("Generating WAV reconstruction for seed %d", seed)
                waveform = np.asarray(waveform_payload["waveform"], dtype=np.float32)
                waveform_sample_rate = int(waveform_payload["sample_rate"])
                waveform_segments = int(waveform_payload["segments"])
                waveform_marker_duration = float(waveform_payload["marker_duration"])
                waveform_image = decode_waveform_to_image(
                    waveform,
                    sample_rate=waveform_sample_rate,
                    resolution=reference_clipped.shape[:2],
                    segments=waveform_segments,
                    marker_duration=waveform_marker_duration,
                )
                waveform_image = _ensure_three_channel(waveform_image).astype(
                    np.float32
                )
                waveform_reference_metrics = compute_metrics(
                    reference_clipped, waveform_image
                )
                _, waveform_reference_overlap = multiplicative_overlap(
                    reference_clipped, waveform_image
                )
                waveform_packet_metrics = compute_metrics(recon_image, waveform_image)
                _, waveform_packet_overlap = multiplicative_overlap(
                    recon_image, waveform_image
                )
            except Exception as exc:  # pragma: no cover - diagnostic logging path
                logger.debug(
                    "Waveform reconstruction failed for seed %d: %s",
                    seed,
                    exc,
                    exc_info=True,
                )
                waveform_image = None
                waveform_reference_metrics = None
                waveform_reference_overlap = None
                waveform_packet_metrics = None
                waveform_packet_overlap = None
                waveform_sample_rate = None
                waveform_segments = None
                waveform_marker_duration = None

        return CandidateResult(
            seed=int(seed),
            reconstruction=recon_image.astype(np.float32, copy=True),
            metrics=packet_metrics,
            overlap_score=float(packet_overlap),
            waveform_reconstruction=None
            if waveform_image is None
            else waveform_image.astype(np.float32, copy=True),
            waveform_reference_metrics=waveform_reference_metrics,
            waveform_reference_overlap=None
            if waveform_reference_overlap is None
            else float(waveform_reference_overlap),
            waveform_packet_metrics=waveform_packet_metrics,
            waveform_packet_overlap=None
            if waveform_packet_overlap is None
            else float(waveform_packet_overlap),
            waveform_sample_rate=waveform_sample_rate,
            waveform_segments=waveform_segments,
            waveform_marker_duration=waveform_marker_duration,
        )

    def _candidate_features(
        self,
        candidate: CandidateResult,
        generation_index: int,
        previous_best_ssim: float,
    ) -> np.ndarray:
        ssim = float(np.clip(candidate.metrics.ssim, 0.0, 1.0))
        psnr_norm = float(np.clip(candidate.metrics.psnr / 50.0, 0.0, 1.0))
        overlap_norm = float(np.clip(candidate.overlap_score / 100.0, 0.0, 1.0))
        improvement = float(np.clip(ssim - previous_best_ssim, -1.0, 1.0))
        depth = float(np.clip(generation_index / (generation_index + 6.0), 0.0, 1.0))
        return np.array([ssim, psnr_norm, overlap_norm, improvement, depth], dtype=np.float32)

    def _compute_reward(
        self,
        features: np.ndarray,
        *,
        overlap_norm: float,
        msssim: float | None = None,
        dct_corr: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        ssim, psnr_norm, overlap_feature, improvement, depth = [float(v) for v in features]
        overlap_norm = float(np.clip(overlap_norm, 0.0, 1.0))
        overlap_feature = float(np.clip(overlap_feature, 0.0, 1.0))
        positive_improvement = max(improvement, 0.0)
        high_overlap_bonus = max(overlap_norm - 0.4, 0.0) * 1.6
        base_terms = (
            0.45 * ssim
            + 0.2 * psnr_norm
            + 0.2 * positive_improvement
            + 0.1 * depth
        )

        if REWARD_MODE == "perceptual_mix":
            cfg = REWARD_CFG["perceptual_mix"]
            overlap_component = cfg["alpha_overlap"] * overlap_norm + high_overlap_bonus
            msssim_component = cfg["beta_msssim"] * float(np.clip(msssim or 0.0, 0.0, 1.0))
            dct_component = cfg["gamma_dct_corr"] * float(np.clip(dct_corr or 0.0, 0.0, 1.0))
            reward = base_terms + overlap_component + msssim_component + dct_component
            components = {
                "overlap": overlap_component,
                "msssim": msssim_component,
                "dct_corr": dct_component,
            }
        else:
            cfg = REWARD_CFG["strict"]
            overlap_component = cfg["alpha_overlap"] * overlap_feature + high_overlap_bonus
            reward = base_terms + overlap_component
            components = {"overlap": overlap_component}

        return float(np.clip(reward, 0.0, 5.0)), components

    def _update_elite_pool(self, generation: GenerationRecord) -> None:
        ranked = sorted(generation.candidates, key=lambda cand: cand.reward, reverse=True)
        elite_count = max(3, self.population_size // 2)
        self._elite_pool = [candidate.seed for candidate in ranked[:elite_count]]

    def _prune_parent_lineage(self) -> None:
        """Retain only the fittest lineage entries to favour elite parents."""

        max_entries = max(self.population_size * _LINEAGE_RETENTION_FACTOR, self.population_size)
        if len(self._parent_lineage) <= max_entries:
            return

        ranked = sorted(
            self._parent_lineage.values(),
            key=lambda record: (
                float(record.cumulative_reward),
                float(record.metrics.ssim),
                float(record.overlap_score),
                -float(record.last_generation),
            ),
            reverse=True,
        )
        survivors = {entry.seed for entry in ranked[:max_entries]}
        removed = 0
        for seed in list(self._parent_lineage.keys()):
            if seed not in survivors:
                del self._parent_lineage[seed]
                removed += 1
        if removed:
            logger.debug("Pruned %d low-fitness parents from lineage", removed)

    def _difficulty_from_generation(
        self,
        generation: GenerationRecord,
        previous_best_ssim: float,
        previous_best_overlap: float,
    ) -> tuple[float, float]:
        best = generation.best_candidate
        overlap_norm = float(np.clip(best.overlap_score / 100.0, 0.0, 1.0))
        ssim = float(np.clip(best.metrics.ssim, 0.0, 1.0))
        overlap_improvement = float(
            max(best.overlap_score - previous_best_overlap, 0.0) / 100.0
        )
        ssim_improvement = float(max(ssim - previous_best_ssim, 0.0))
        improvement = max(overlap_improvement, ssim_improvement)
        reward_signal = float(np.clip(generation.reward_peak / 6.0, 0.0, 1.0))

        difficulty = (
            0.45 * overlap_norm
            + 0.25 * ssim
            + 0.2 * reward_signal
            + 0.1 * improvement
        )

        if overlap_norm >= 0.4:
            difficulty += 0.15 * (overlap_norm - 0.4)
        if overlap_improvement > 0.0:
            difficulty += 0.1 * min(overlap_improvement, 0.2)

        previous_difficulty = self.difficulty_trace[-1] if self.difficulty_trace else 0.0
        if best.overlap_score >= previous_best_overlap:
            difficulty = max(difficulty, previous_difficulty)
        else:
            difficulty = max(difficulty, previous_difficulty * 0.9)

        return float(np.clip(difficulty, 0.0, 1.25)), improvement

    def run_generation(self, parent_selection: Sequence[int] | None = None) -> GenerationRecord:
        """Evaluate a new generation and append it to the history.

        Parameters
        ----------
        parent_selection:
            Optional iterable of seed values that should persist as parents for
            this generation. When omitted, the full stored lineage is used.
        """

        lineage_seeds = list(self._parent_lineage.keys())
        anchors = list(dict.fromkeys(int(seed) for seed in (parent_selection or lineage_seeds)))
        recent_records = self.generations[-self._carryover_generations :]
        for record in recent_records:
            for candidate in record.candidates:
                anchors.append(int(candidate.seed))
        anchors.extend(self._elite_pool)
        anchors = list(dict.fromkeys(anchors))
        target_candidates = self.population_size + len(anchors) + self._mutation_boost
        seen: set[int] = set()
        seeds: list[int] = []

        def _add_seed(raw: int) -> None:
            candidate_seed = int(raw) & 0x7FFFFFFF
            if candidate_seed not in seen:
                seen.add(candidate_seed)
                seeds.append(candidate_seed)

        for parent_seed in anchors:
            _add_seed(parent_seed)

        while len(seeds) < target_candidates:
            _add_seed(self._spawn_child_seed(anchors or seeds))

        generation_index = int(self.next_generation_index)
        generation = GenerationRecord(index=generation_index)
        logger.info(
            "Running generation %d with %d parents and %d children",
            generation_index,
            len(anchors),
            len(seeds) - len(anchors),
        )

        previous_best_candidate = self.generations[-1].best_candidate if self.generations else None
        previous_best_ssim = (
            previous_best_candidate.metrics.ssim if previous_best_candidate else 0.0
        )
        previous_best_overlap = (
            previous_best_candidate.overlap_score if previous_best_candidate else 0.0
        )
        feature_vectors: list[np.ndarray] = []
        rewards: list[float] = []

        channel_axis = -1 if self.original.ndim == 3 else None

        reference_clipped = _ensure_three_channel(self.original)
        waveform_payload: dict[str, Any] | None = None
        if self.enable_waveform:
            try:
                sample_rate = suggest_sample_rate(reference_clipped)
                segments, marker_duration = suggest_transmission_profile(reference_clipped)
                waveform = encode_image_to_waveform(
                    reference_clipped,
                    sample_rate=sample_rate,
                    segments=segments,
                    marker_duration=marker_duration,
                )
                waveform_payload = {
                    "waveform": waveform,
                    "sample_rate": int(sample_rate),
                    "segments": int(segments),
                    "marker_duration": float(marker_duration),
                }
            except Exception as exc:  # pragma: no cover - diagnostic waveform prep
                logger.debug(
                    "Failed to prepare waveform baseline for generation %d: %s",
                    generation_index,
                    exc,
                    exc_info=True,
                )
                waveform_payload = None
        start_time = time.perf_counter()
        for seed in seeds:
            candidate = self._evaluate_candidate(
                int(seed),
                reference_clipped,
                waveform_payload=waveform_payload,
            )
            reconstruction = np.asarray(candidate.reconstruction, dtype=np.float32)
            features = self._candidate_features(candidate, generation.index, previous_best_ssim)
            overlap_norm = float(np.clip(candidate.overlap_score / 100.0, 0.0, 1.0))
            msssim_value: float | None = None
            dct_corr_value: float | None = None
            if REWARD_MODE == "perceptual_mix":
                msssim_value = compute_ms_ssim(
                    self.original,
                    reconstruction,
                    channel_axis=channel_axis,
                )
                dct_corr_value = dct_band_correlation(self.original, reconstruction)
            base_reward, components = self._compute_reward(
                features,
                overlap_norm=overlap_norm,
                msssim=msssim_value,
                dct_corr=dct_corr_value,
            )
            predicted = None
            if self._reward_model is not None:
                try:
                    predicted = float(self._reward_model.predict(features))
                except Exception:  # pragma: no cover - advisor failures are non-fatal
                    logger.debug("Neural reward prediction failed", exc_info=True)
                    predicted = None
            reward = base_reward if predicted is None else float(0.65 * base_reward + 0.35 * predicted)
            reward = float(np.clip(reward, 0.0, 6.0))
            candidate.reward = reward
            candidate.predicted_reward = predicted
            candidate.feature_vector = tuple(float(value) for value in features)
            candidate.reward_components = components
            feature_vectors.append(features)
            rewards.append(reward)
            generation.candidates.append(candidate)

        if rewards:
            generation.reward_summary = float(np.mean(rewards))
            generation.reward_peak = float(np.max(rewards))
            self.lifetime_reward += float(generation.reward_summary)
        generation.cumulative_reward = float(self.lifetime_reward)
        self.reward_trace.append(float(generation.reward_summary))
        self._update_elite_pool(generation)
        difficulty_raw, improvement = self._difficulty_from_generation(
            generation, previous_best_ssim, previous_best_overlap
        )
        normalized_difficulty = normalize_difficulty(difficulty_raw)
        scheduled = normalized_difficulty
        if DIFFICULTY_LADDER["enabled"]:
            period = max(1, int(DIFFICULTY_LADDER["period_gens"]))
            spike_len = max(1, int(DIFFICULTY_LADDER["spike_len_gens"]))
            if generation.index % period < spike_len:
                scheduled = float(np.clip(DIFFICULTY_LADDER["spike"], 0.0, 1.0))
            else:
                scheduled = float(np.clip(DIFFICULTY_LADDER["base"], 0.0, 1.0))
        generation.difficulty_raw = difficulty_raw
        generation.difficulty_level = scheduled
        generation.improvement = improvement
        self.difficulty_trace.append(scheduled)
        duration = time.perf_counter() - start_time
        self._record_generation_duration(
            duration=duration,
            candidate_count=len(seeds),
            difficulty=normalized_difficulty,
            generation=generation,
        )
        if self._reward_model is not None and feature_vectors:
            try:
                feature_batch = np.vstack(feature_vectors).astype(np.float32)
                reward_batch = np.asarray(rewards, dtype=np.float32)
                self._reward_model.update(feature_batch, reward_batch)
            except Exception:  # pragma: no cover - advisor training is optional
                logger.debug("Neural reward update failed", exc_info=True)

        for candidate in generation.candidates:
            record = self._parent_lineage.get(candidate.seed)
            if record is None:
                self._parent_lineage[candidate.seed] = ParentLineage(
                    seed=candidate.seed,
                    origin_generation=generation.index,
                    metrics=candidate.metrics,
                    overlap_score=candidate.overlap_score,
                    appearances=1,
                    cumulative_reward=candidate.reward,
                    peak_reward=candidate.reward,
                    last_generation=generation.index,
                )
                continue

            if candidate.metrics.ssim >= record.metrics.ssim:
                record.metrics = candidate.metrics
                record.overlap_score = candidate.overlap_score
            record.appearances += 1
            record.cumulative_reward = float(record.cumulative_reward + candidate.reward)
            record.peak_reward = float(max(record.peak_reward, candidate.reward))
            record.last_generation = generation.index
        self._prune_parent_lineage()
        self.append_generation_record(generation, persist=False, use_next_index=True)
        if generation.candidates:
            best = generation.best_candidate
            logger.info(
                "Completed generation %d; best seed %d with SSIM %.3f and overlap %.2f",
                generation.index,
                best.seed,
                best.metrics.ssim,
                best.overlap_score,
            )
        else:
            logger.info("Completed generation %d with no evaluated candidates", generation.index)
        self._handle_plateau(generation)
        self._persist_generation_history(generation)
        return generation

    def _record_generation_duration(
        self,
        *,
        duration: float,
        candidate_count: int,
        difficulty: float,
        generation: GenerationRecord,
    ) -> None:
        """Track throughput and update hyper tuning when enabled."""

        if not self._hyper_enabled:
            return

        window = 0.2
        if self._duration_ema <= 0.0:
            self._duration_ema = duration
        else:
            self._duration_ema = (1.0 - window) * self._duration_ema + window * duration

        subjects_per_second = candidate_count / max(duration, 1e-6)
        if self._throughput_ema <= 0.0:
            self._throughput_ema = subjects_per_second
        else:
            self._throughput_ema = (
                (1.0 - window) * self._throughput_ema + window * subjects_per_second
            )

        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        target_subjects = int(np.clip(round(140 + 60 * difficulty), 90, 220))
        throughput_factor = float(np.clip(self._throughput_ema / 6.0, 0.0, 4.0))
        batch_size = int(
            np.clip(
                round(5 + difficulty * 7 + throughput_factor),
                5,
                24,
            )
        )
        dwell_generations = int(
            np.clip(
                np.ceil(target_subjects / max(batch_size, 1)),
                18,
                90,
            )
        )
        autosave_interval = int(np.clip(max(dwell_generations // 4, 6), 6, 40))
        queue_generations = int(np.clip(dwell_generations, 6, 120))

        self._hyper_profile = HyperPerformanceProfile(
            enabled=True,
            target_subjects=target_subjects,
            batch_size=batch_size,
            dwell_generations=dwell_generations,
            autosave_interval=autosave_interval,
            queue_generations=queue_generations,
            mean_duration=self._duration_ema,
            throughput=self._throughput_ema,
            last_update=generation.index,
        )

        self.population_size = max(1, batch_size)
        self.autosave_interval = max(1, autosave_interval)

    def _handle_plateau(self, generation: GenerationRecord) -> None:
        """Adjust exploration and precision when progress stalls."""

        best_overlap = float(generation.best_candidate.overlap_score)
        tolerance = 1e-3
        if best_overlap > self._best_overlap + tolerance:
            self._best_overlap = best_overlap
            self._plateau_generations = 0
            if self._mutation_boost > 0:
                self._mutation_boost = max(0, self._mutation_boost - 1)
            logger.debug(
                "Plateau reset at generation %d with overlap %.3f", generation.index, best_overlap
            )
            return

        recent = self.generations[-max(1, PLATEAU_CFG["window"]) :]
        overlaps = [
            float(record.best_candidate.overlap_score)
            for record in recent
            if record.candidates
        ]
        overlap_range = 0.0
        if overlaps:
            overlap_range = (max(overlaps) - min(overlaps)) / 100.0

        self._plateau_generations += 1
        plateau_limit = self._carryover_generations + 2
        if (
            overlap_range < PLATEAU_CFG["delta_threshold"]
            and self._plateau_generations >= plateau_limit
        ):
            self._plateau_generations = 0
            cap = PLATEAU_CFG["boost_cap_factor"] * self.population_size
            self._mutation_boost = min(self._mutation_boost + PLATEAU_CFG["boost_step"], cap)
            if PLATEAU_CFG["log"]:
                logger.info(
                    "[plateau] range=%.4f -> mutation_boost=%d; precision ramp applied",
                    overlap_range,
                    self._mutation_boost,
                )
            generation.checkpoint_tag = "plateau_kick"
            self._apply_precision_ramp()
        else:
            logger.debug(
                "Plateau range %.4f with counter %d (overlap %.3f)",
                overlap_range,
                self._plateau_generations,
                best_overlap,
            )

    def _apply_precision_ramp(self) -> None:
        """Reduce encode/decode noise to approach full reconstruction."""

        encoder_sigma = getattr(self.encoder, "sigma", None)
        if encoder_sigma is not None:
            current_sigma = float(encoder_sigma)
            target_sigma = max(current_sigma * 0.85, 0.01)
            if target_sigma < current_sigma - 1e-5:
                logger.info(
                    "Precision ramp: encoder sigma adjusted from %.4f to %.4f",
                    current_sigma,
                    target_sigma,
                )
                encoder_cls = type(self.encoder)
                try:
                    self.encoder = encoder_cls(sigma=target_sigma)  # type: ignore[arg-type]
                except TypeError:
                    self.encoder = NoiseStreamEncoder(sigma=target_sigma)

        decoder_sigma = getattr(self.decoder, "denoise_sigma", None)
        if decoder_sigma is not None:
            current_denoise = float(decoder_sigma)
            target_denoise = max(current_denoise * 0.9, 0.03)
            if target_denoise < current_denoise - 1e-5:
                logger.info(
                    "Precision ramp: decoder denoise sigma adjusted from %.4f to %.4f",
                    current_denoise,
                    target_denoise,
                )
                decoder_cls = type(self.decoder)
                try:
                    self.decoder = decoder_cls(denoise_sigma=target_denoise)  # type: ignore[arg-type]
                except TypeError:
                    self.decoder = NoiseStreamDecoder(denoise_sigma=target_denoise)

    def to_session(self) -> EvolutionSession:
        """Create a serializable snapshot of the manager."""

        compact_generations: list[GenerationRecord] = []
        for record in self.generations:
            compact_candidates: list[CandidateResult] = []
            for candidate in record.candidates:
                compact_candidates.append(
                    CandidateResult(
                        seed=candidate.seed,
                        reconstruction=np.asarray(
                            candidate.reconstruction, dtype=np.float16
                        ),
                        metrics=candidate.metrics,
                        overlap_score=candidate.overlap_score,
                        waveform_reconstruction=None
                        if getattr(candidate, "waveform_reconstruction", None) is None
                        else np.asarray(
                            candidate.waveform_reconstruction, dtype=np.float16
                        ),
                        waveform_reference_metrics=getattr(
                            candidate, "waveform_reference_metrics", None
                        ),
                        waveform_reference_overlap=(
                            None
                            if getattr(candidate, "waveform_reference_overlap", None)
                            is None
                            else float(candidate.waveform_reference_overlap)
                        ),
                        waveform_packet_metrics=getattr(
                            candidate, "waveform_packet_metrics", None
                        ),
                        waveform_packet_overlap=(
                            None
                            if getattr(candidate, "waveform_packet_overlap", None) is None
                            else float(candidate.waveform_packet_overlap)
                        ),
                        waveform_sample_rate=(
                            None
                            if getattr(candidate, "waveform_sample_rate", None) is None
                            else int(candidate.waveform_sample_rate)
                        ),
                        waveform_segments=(
                            None
                            if getattr(candidate, "waveform_segments", None) is None
                            else int(candidate.waveform_segments)
                        ),
                        waveform_marker_duration=(
                            None
                            if getattr(candidate, "waveform_marker_duration", None) is None
                            else float(candidate.waveform_marker_duration)
                        ),
                        reward=float(getattr(candidate, "reward", 0.0)),
                        predicted_reward=(
                            None
                            if getattr(candidate, "predicted_reward", None) is None
                            else float(candidate.predicted_reward)
                        ),
                        feature_vector=tuple(
                            float(value)
                            for value in getattr(candidate, "feature_vector", tuple())
                        ),
                        reward_components=dict(
                            getattr(candidate, "reward_components", {})
                        ),
                    )
                )
            compact_generations.append(
                GenerationRecord(
                    index=record.index,
                    candidates=compact_candidates,
                    reward_summary=float(getattr(record, "reward_summary", 0.0)),
                    reward_peak=float(getattr(record, "reward_peak", 0.0)),
                    cumulative_reward=float(getattr(record, "cumulative_reward", 0.0)),
                    difficulty_level=float(getattr(record, "difficulty_level", 0.0)),
                    difficulty_raw=float(
                        getattr(record, "difficulty_raw", getattr(record, "difficulty_level", 0.0))
                    ),
                    improvement=float(getattr(record, "improvement", 0.0)),
                    checkpoint_tag=getattr(record, "checkpoint_tag", None),
                )
            )

        return EvolutionSession(
            image_signature=self.image_signature,
            original=np.asarray(self.original, dtype=np.float16),
            encoder_config=self.encoder.to_config(),
            decoder_config=self.decoder.to_config(),
            population_size=self.population_size,
            base_seed=self.base_seed,
            autosave_interval=self.autosave_interval,
            run_id=self.run_id,
            next_generation_index=int(self.next_generation_index),
            generations=compact_generations,
            rng_state=self.rng.bit_generator.state,
            parent_lineage=list(self._parent_lineage.values()),
            lifetime_reward=float(self.lifetime_reward),
            reward_trace=list(self.reward_trace),
            difficulty_trace=list(self.difficulty_trace),
            elite_seeds=list(self._elite_pool),
            advisor_state=(
                self._reward_model.to_state() if self._reward_model is not None else None
            ),
            best_overlap=float(self._best_overlap),
            plateau_generations=int(self._plateau_generations),
            mutation_boost=int(self._mutation_boost),
            hyper_profile=(
                self._hyper_profile.as_dict() if self._hyper_profile.enabled else None
            ),
            enable_waveform=self.enable_waveform,
        )

    def save(self, directory: str | Path) -> Path:
        """Persist the current session to ``directory``."""

        self.sync_history()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "evolution_state.pkl"
        with path.open("wb") as handle:
            pickle.dump(self.to_session(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saved evolution session with %d generations to %s",
            len(self.generations),
            path,
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> EvolutionManager:
        """Restore a manager from :meth:`save` output."""

        directory = Path(directory)
        path = directory
        if directory.is_dir():
            path = directory / "evolution_state.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved evolution session at {path}")

        with path.open("rb") as handle:
            session: EvolutionSession = pickle.load(handle)

        encoder = NoiseStreamEncoder.from_config(session.encoder_config)
        decoder = NoiseStreamDecoder.from_config(session.decoder_config)
        original = np.asarray(session.original, dtype=np.float32)

        restored_generations: list[GenerationRecord] = []
        for record in session.generations:
            restored_candidates: list[CandidateResult] = []
            for candidate in record.candidates:
                restored_candidates.append(
                    CandidateResult(
                        seed=candidate.seed,
                        reconstruction=np.asarray(
                            candidate.reconstruction, dtype=np.float32
                        ),
                        metrics=candidate.metrics,
                        overlap_score=candidate.overlap_score,
                        waveform_reconstruction=(
                            None
                            if getattr(candidate, "waveform_reconstruction", None) is None
                            else np.asarray(
                                candidate.waveform_reconstruction, dtype=np.float32
                            )
                        ),
                        waveform_reference_metrics=getattr(
                            candidate, "waveform_reference_metrics", None
                        ),
                        waveform_reference_overlap=(
                            None
                            if getattr(candidate, "waveform_reference_overlap", None)
                            is None
                            else float(candidate.waveform_reference_overlap)
                        ),
                        waveform_packet_metrics=getattr(
                            candidate, "waveform_packet_metrics", None
                        ),
                        waveform_packet_overlap=(
                            None
                            if getattr(candidate, "waveform_packet_overlap", None) is None
                            else float(candidate.waveform_packet_overlap)
                        ),
                        waveform_sample_rate=(
                            None
                            if getattr(candidate, "waveform_sample_rate", None) is None
                            else int(candidate.waveform_sample_rate)
                        ),
                        waveform_segments=(
                            None
                            if getattr(candidate, "waveform_segments", None) is None
                            else int(candidate.waveform_segments)
                        ),
                        waveform_marker_duration=(
                            None
                            if getattr(candidate, "waveform_marker_duration", None) is None
                            else float(candidate.waveform_marker_duration)
                        ),
                        reward=float(getattr(candidate, "reward", 0.0)),
                        predicted_reward=(
                            None
                            if getattr(candidate, "predicted_reward", None) is None
                            else float(candidate.predicted_reward)
                        ),
                        feature_vector=tuple(
                            float(value)
                            for value in getattr(candidate, "feature_vector", tuple())
                        ),
                        reward_components=dict(
                            getattr(candidate, "reward_components", {})
                        ),
                    )
                )
            restored_generations.append(
                GenerationRecord(
                    index=record.index,
                    candidates=restored_candidates,
                    reward_summary=float(getattr(record, "reward_summary", 0.0)),
                    reward_peak=float(getattr(record, "reward_peak", 0.0)),
                    cumulative_reward=float(getattr(record, "cumulative_reward", 0.0)),
                    difficulty_level=float(getattr(record, "difficulty_level", 0.0)),
                    difficulty_raw=float(
                        getattr(record, "difficulty_raw", getattr(record, "difficulty_level", 0.0))
                    ),
                    improvement=float(getattr(record, "improvement", 0.0)),
                    checkpoint_tag=getattr(record, "checkpoint_tag", None),
                )
            )

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
            encoder=encoder,
            decoder=decoder,
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
            if history_frame is not None and hasattr(history_frame, "empty") and not history_frame.empty:
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
    "normalize_difficulty",
    "compute_image_signature",
    "ParentLineage",
    "HyperPerformanceProfile",
]
