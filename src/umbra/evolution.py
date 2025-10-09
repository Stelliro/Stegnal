"""Evolutionary search utilities for Project Umbra."""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .decoding import NoiseStreamDecoder
from .encoding import NoiseStreamEncoder
from .metrics import ReconstructionMetrics, compute_metrics
from .visualization import multiplicative_overlap

if TYPE_CHECKING:  # pragma: no cover - optional neural advisor import
    from .neural import NeuralRewardModel

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Summary of a single AI attempt within a generation."""

    seed: int
    reconstruction: np.ndarray
    metrics: ReconstructionMetrics
    overlap_score: float
    reward: float = 0.0
    predicted_reward: float | None = None
    feature_vector: tuple[float, ...] = field(default_factory=tuple)


@dataclass
class GenerationRecord:
    """Collection of candidates evaluated for a specific generation."""

    index: int
    candidates: list[CandidateResult] = field(default_factory=list)
    reward_summary: float = 0.0
    reward_peak: float = 0.0
    cumulative_reward: float = 0.0
    difficulty_level: float = 0.0
    improvement: float = 0.0

    @property
    def best_candidate(self) -> CandidateResult:
        return max(self.candidates, key=lambda cand: cand.metrics.ssim)


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
    parent_lineage: list[ParentLineage] = field(default_factory=list)
    lifetime_reward: float = 0.0
    reward_trace: list[float] = field(default_factory=list)
    difficulty_trace: list[float] = field(default_factory=list)
    elite_seeds: list[int] = field(default_factory=list)
    advisor_state: dict[str, Any] | None = None


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
    ) -> None:
        self.original = np.asarray(original, dtype=np.float32)
        self.encoder = encoder
        self.decoder = decoder
        self.population_size = max(1, int(population_size))
        self.base_seed = int(base_seed)
        self.autosave_interval = max(1, int(autosave_interval))
        self.generations: list[GenerationRecord] = []
        self.rng = np.random.default_rng(self.base_seed)
        self._parent_lineage: dict[int, ParentLineage] = {}
        self._carryover_generations = 3
        self._elite_pool: list[int] = []
        self._reward_model: NeuralRewardModel | None = advisor
        self.lifetime_reward: float = 0.0
        self.reward_trace: list[float] = []
        self.difficulty_trace: list[float] = []

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
        if population_size is not None:
            self.population_size = max(1, int(population_size))
        if autosave_interval is not None:
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
        combined = 0
        for idx, parent_seed in enumerate(np.asarray(choices, dtype=np.int64)):
            shift = (idx * 17) % 31
            combined ^= (int(parent_seed) << shift) & 0x7FFFFFFF
        mutation = int(self.rng.integers(0, np.iinfo(np.int32).max))
        return (combined ^ mutation) & 0x7FFFFFFF

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

    def _base_reward(self, features: np.ndarray) -> float:
        ssim, psnr_norm, overlap_norm, improvement, depth = [float(v) for v in features]
        positive_improvement = max(improvement, 0.0)
        high_overlap_bonus = max(overlap_norm - 0.4, 0.0) * 1.6
        reward = (
            0.45 * ssim
            + 0.2 * psnr_norm
            + 0.25 * overlap_norm
            + 0.2 * positive_improvement
            + 0.1 * depth
            + high_overlap_bonus
        )
        return float(np.clip(reward, 0.0, 5.0))

    def _update_elite_pool(self, generation: GenerationRecord) -> None:
        ranked = sorted(generation.candidates, key=lambda cand: cand.reward, reverse=True)
        elite_count = max(3, self.population_size // 2)
        self._elite_pool = [candidate.seed for candidate in ranked[:elite_count]]

    def _difficulty_from_generation(
        self, generation: GenerationRecord, previous_best_ssim: float
    ) -> tuple[float, float]:
        best = generation.best_candidate
        overlap_norm = float(np.clip(best.overlap_score / 100.0, 0.0, 1.0))
        ssim = float(np.clip(best.metrics.ssim, 0.0, 1.0))
        improvement = float(max(ssim - previous_best_ssim, 0.0))
        reward_signal = float(np.clip(generation.reward_peak / 5.0, 0.0, 1.5))
        difficulty = 0.3 * overlap_norm + 0.35 * ssim + 0.2 * reward_signal + 0.15 * improvement
        if overlap_norm >= 0.4:
            difficulty += 0.2 * (overlap_norm - 0.4)
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
        target_candidates = self.population_size + len(anchors)
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

        generation = GenerationRecord(index=len(self.generations))
        logger.info(
            "Running generation %d with %d parents and %d children",
            generation.index,
            len(anchors),
            len(seeds) - len(anchors),
        )

        previous_best_ssim = (
            self.generations[-1].best_candidate.metrics.ssim if self.generations else 0.0
        )
        feature_vectors: list[np.ndarray] = []
        rewards: list[float] = []

        for seed in seeds:
            packet = self.encoder.encode(self.original, int(seed))
            reconstruction = self.decoder.decode(packet, int(seed))
            metrics = compute_metrics(self.original, reconstruction)
            _, overlap_score = multiplicative_overlap(self.original, reconstruction)
            candidate = CandidateResult(
                seed=int(seed),
                reconstruction=np.asarray(reconstruction, dtype=np.float16),
                metrics=metrics,
                overlap_score=float(overlap_score),
            )
            features = self._candidate_features(candidate, generation.index, previous_best_ssim)
            base_reward = self._base_reward(features)
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
            feature_vectors.append(features)
            rewards.append(reward)
            generation.candidates.append(candidate)

        self.generations.append(generation)
        if rewards:
            generation.reward_summary = float(np.mean(rewards))
            generation.reward_peak = float(np.max(rewards))
            self.lifetime_reward += float(generation.reward_summary)
        generation.cumulative_reward = float(self.lifetime_reward)
        self.reward_trace.append(float(generation.reward_summary))
        self._update_elite_pool(generation)
        difficulty_level, improvement = self._difficulty_from_generation(
            generation, previous_best_ssim
        )
        generation.difficulty_level = difficulty_level
        generation.improvement = improvement
        self.difficulty_trace.append(difficulty_level)
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
        best = generation.best_candidate
        logger.info(
            "Completed generation %d; best seed %d with SSIM %.3f and overlap %.2f",
            generation.index,
            best.seed,
            best.metrics.ssim,
            best.overlap_score,
        )
        return generation

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
                    improvement=float(getattr(record, "improvement", 0.0)),
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
        )

    def save(self, directory: str | Path) -> Path:
        """Persist the current session to ``directory``."""

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
                    improvement=float(getattr(record, "improvement", 0.0)),
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
        )
        manager.generations = restored_generations
        manager.rng.bit_generator.state = session.rng_state
        lineage = getattr(session, "parent_lineage", [])
        manager._parent_lineage = {entry.seed: entry for entry in lineage}
        manager.lifetime_reward = float(getattr(session, "lifetime_reward", 0.0))
        manager.reward_trace = list(getattr(session, "reward_trace", []))
        manager.difficulty_trace = list(getattr(session, "difficulty_trace", []))
        manager._elite_pool = list(getattr(session, "elite_seeds", []))
        if advisor is None and advisor_state is None:
            manager._reward_model = None
        return manager


__all__ = [
    "CandidateResult",
    "EvolutionManager",
    "EvolutionSession",
    "GenerationRecord",
    "compute_image_signature",
    "ParentLineage",
]
