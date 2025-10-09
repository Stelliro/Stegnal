"""Evolutionary search utilities for Project Umbra."""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decoding import NoiseStreamDecoder
from .encoding import NoiseStreamEncoder
from .metrics import ReconstructionMetrics, compute_metrics
from .visualization import multiplicative_overlap

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Summary of a single AI attempt within a generation."""

    seed: int
    reconstruction: np.ndarray
    metrics: ReconstructionMetrics
    overlap_score: float


@dataclass
class GenerationRecord:
    """Collection of candidates evaluated for a specific generation."""

    index: int
    candidates: list[CandidateResult] = field(default_factory=list)

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


@dataclass
class ParentLineage:
    """History entry recording a seed's performance within the lineage."""

    seed: int
    origin_generation: int
    metrics: ReconstructionMetrics
    overlap_score: float


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

    @property
    def parent_lineage(self) -> list[ParentLineage]:
        """Return a snapshot of the current parent lineage."""

        return list(self._parent_lineage.values())

    @property
    def image_signature(self) -> str:
        return compute_image_signature(self.original)

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

    def run_generation(self, parent_selection: Sequence[int] | None = None) -> GenerationRecord:
        """Evaluate a new generation and append it to the history.

        Parameters
        ----------
        parent_selection:
            Optional iterable of seed values that should persist as parents for
            this generation. When omitted, the full stored lineage is used.
        """

        anchors = list(
            dict.fromkeys(
                int(seed) for seed in (parent_selection or self._parent_lineage.keys())
            )
        )
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
            generation.candidates.append(candidate)

        self.generations.append(generation)
        for candidate in generation.candidates:
            record = self._parent_lineage.get(candidate.seed)
            if record is None or candidate.metrics.ssim >= record.metrics.ssim:
                self._parent_lineage[candidate.seed] = ParentLineage(
                    seed=candidate.seed,
                    origin_generation=generation.index,
                    metrics=candidate.metrics,
                    overlap_score=candidate.overlap_score,
                )
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
                        reconstruction=np.asarray(candidate.reconstruction, dtype=np.float16),
                        metrics=candidate.metrics,
                        overlap_score=candidate.overlap_score,
                    )
                )
            compact_generations.append(GenerationRecord(index=record.index, candidates=compact_candidates))

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
                        reconstruction=np.asarray(candidate.reconstruction, dtype=np.float32),
                        metrics=candidate.metrics,
                        overlap_score=candidate.overlap_score,
                    )
                )
            restored_generations.append(GenerationRecord(index=record.index, candidates=restored_candidates))

        logger.info(
            "Loaded evolution session from %s with %d generations",
            path,
            len(restored_generations),
        )
        manager = cls(
            original=original,
            encoder=encoder,
            decoder=decoder,
            population_size=session.population_size,
            base_seed=session.base_seed,
            autosave_interval=session.autosave_interval,
        )
        manager.generations = restored_generations
        manager.rng.bit_generator.state = session.rng_state
        lineage = getattr(session, "parent_lineage", [])
        manager._parent_lineage = {entry.seed: entry for entry in lineage}
        return manager


__all__ = [
    "CandidateResult",
    "EvolutionManager",
    "EvolutionSession",
    "GenerationRecord",
    "compute_image_signature",
    "ParentLineage",
]
