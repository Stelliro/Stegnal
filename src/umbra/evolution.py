"""Evolutionary search utilities for Project Umbra."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decoding import NoiseStreamDecoder
from .encoding import NoiseStreamEncoder
from .metrics import ReconstructionMetrics, compute_metrics
from .visualization import multiplicative_overlap


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

    def run_generation(self) -> GenerationRecord:
        """Evaluate a new generation and append it to the history."""

        seeds = self.rng.integers(0, np.iinfo(np.int32).max, size=self.population_size, dtype=np.int64)
        generation = GenerationRecord(index=len(self.generations))

        for seed in seeds.tolist():
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
        )

    def save(self, directory: str | Path) -> Path:
        """Persist the current session to ``directory``."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "evolution_state.pkl"
        with path.open("wb") as handle:
            pickle.dump(self.to_session(), handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        return manager


__all__ = [
    "CandidateResult",
    "EvolutionManager",
    "EvolutionSession",
    "GenerationRecord",
    "compute_image_signature",
]

