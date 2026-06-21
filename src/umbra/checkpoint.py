# checkpoint.py

"""Persistent, metadata-rich model checkpoints for Project Umbra.

An :class:`UmbraModel` bundles a trainable :class:`~umbra.neural.NeuralRewardModel`
with the context needed to *resume* training later instead of starting cold:

* **who its peers were** — the best candidate seeds/rewards it has seen (lineage),
* **how well it did** — a per-session performance history,
* **where its data came from** — training sessions, including acoustic
  speaker->mic batches and the recording files behind them.

Everything serialises to a single self-describing JSON file (``*.umbra.json``) so
the metadata always travels with the weights. Load a checkpoint to train further
(the optimiser momentum and feature normalisation are preserved) or just to use
it for inference, then save again.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .neural import NeuralRewardModel

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Default per-candidate feature vector fed to the reward model.
DEFAULT_FEATURE_NAMES = ("overlap", "psnr", "ssim", "denoise_sigma", "difficulty")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git may be absent
        pass
    return "unknown"


@dataclass
class TrainingSession:
    """One increment of training applied to the model."""

    timestamp: str
    source: str               # "simulation" | "acoustic" | ...
    samples: int
    mean_reward: float = 0.0
    best_reward: float = 0.0
    recordings: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PeerRecord:
    """A standout candidate the model learned from (lineage / 'peers')."""

    seed: int
    reward: float
    generation: int
    competence: float = 0.0   # hardest channel difficulty this lineage cleared


def candidate_feature_vector(candidate, difficulty: float,
                             feature_names=DEFAULT_FEATURE_NAMES) -> list[float]:
    """Extract the model's feature vector from an evolution ``Candidate``."""
    metrics = getattr(candidate, "metrics", None)
    values = {
        "overlap": float(getattr(candidate, "overlap_score", 0.0) or 0.0),
        "psnr": float(getattr(metrics, "psnr", 0.0) or 0.0),
        "ssim": float(getattr(metrics, "ssim", 0.0) or 0.0),
        "denoise_sigma": float(getattr(candidate.genes, "denoise_sigma", 0.0)),
        "difficulty": float(difficulty),
    }
    return [values.get(name, 0.0) for name in feature_names]


class UmbraModel:
    """A trainable reward model plus the metadata needed to resume training."""

    def __init__(self, reward_model: NeuralRewardModel | None = None, *,
                 name: str = "umbra-model", feature_names=DEFAULT_FEATURE_NAMES):
        self.feature_names = tuple(feature_names)
        self.reward_model = reward_model or NeuralRewardModel(input_dim=len(self.feature_names))
        self.name = name
        self.created_at = _now()
        self.updated_at = self.created_at
        self.total_samples = 0
        self.generations_trained = 0
        self.sessions: list[TrainingSession] = []
        self.peers: list[PeerRecord] = []         # best candidates seen, by reward
        self.performance: list[dict] = []          # per-session score summary
        self.champion: dict | None = None          # best candidate ever (hardest difficulty)
        self.provenance: dict = {"format_version": FORMAT_VERSION, "git": _git_hash()}

    # -- training -------------------------------------------------------
    def train(self, features, rewards, *, source: str = "simulation",
              recordings=None, peers=None, notes: str = "") -> dict:
        """Apply one training increment and record it in the metadata.

        ``features`` is (N, F); ``rewards`` is (N,). The underlying model keeps
        its optimiser momentum and feature statistics, so repeated calls (across
        save/load) continue training rather than restart it.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        rewards = np.asarray(rewards, dtype=np.float32).ravel()
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {features.shape[1]}"
            )

        self.reward_model.update(features, rewards)

        n = int(features.shape[0])
        mean_r = float(np.mean(rewards)) if rewards.size else 0.0
        best_r = float(np.max(rewards)) if rewards.size else 0.0
        self.total_samples += n
        self.generations_trained += 1
        self.sessions.append(TrainingSession(
            timestamp=_now(), source=source, samples=n,
            mean_reward=mean_r, best_reward=best_r,
            recordings=list(recordings or []), notes=notes,
        ))
        self.performance.append({
            "timestamp": _now(), "source": source, "samples": n,
            "mean_reward": mean_r, "best_reward": best_r,
        })
        if peers:
            for p in peers:
                self.peers.append(PeerRecord(
                    seed=int(p.get("seed", 0)),
                    reward=float(p.get("reward", 0.0)),
                    generation=int(p.get("generation", self.generations_trained)),
                    competence=float(p.get("competence", 0.0)),
                ))
            # Keep the strongest peers as lineage context.
            self.peers = sorted(self.peers, key=lambda r: r.reward, reverse=True)[:64]
        self.updated_at = _now()
        return {"samples": n, "mean_reward": mean_r, "best_reward": best_r}

    def train_from_generation(self, manager, *, source: str = "simulation",
                              recordings=None, notes: str = "") -> dict:
        """Train on the latest candidates of an ``EvolutionManager``.

        Works with both the legacy ``population`` field and the new-API
        ``generations[-1].candidates``.
        """
        population = list(getattr(manager, "population", []) or [])
        if not population:
            generations = getattr(manager, "generations", None) or []
            if generations:
                population = list(getattr(generations[-1], "candidates", []) or [])
        if not population:
            return {"samples": 0, "mean_reward": 0.0, "best_reward": 0.0}
        difficulty = float(
            getattr(manager, "difficulty", 0.0)
            or getattr(getattr(manager, "_encoder", None), "sigma", 0.0)
            or 0.0
        )
        feats = [candidate_feature_vector(c, difficulty, self.feature_names) for c in population]
        rewards = [float(c.reward) for c in population]
        peers = [
            {"seed": int(c.seed), "reward": float(c.reward),
             "generation": int(getattr(manager, "generation_count", 0)),
             "competence": float(getattr(getattr(c, "heritage", None), "competence", 0.0) or 0.0)}
            for c in sorted(population, key=lambda c: c.reward, reverse=True)[:5]
        ]
        result = self.train(feats, rewards, source=source, recordings=recordings,
                            peers=peers, notes=notes)
        # Track the run's best-ever candidate (hardest difficulty cleared).
        champ = getattr(manager, "champion", None)
        if champ is not None and (
            self.champion is None
            or champ.get("reward", 0.0) > self.champion.get("reward", -1.0)
        ):
            self.champion = dict(champ)
        return result

    def predict(self, features) -> np.ndarray:
        return self.reward_model.predict(np.asarray(features, dtype=np.float32))

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "feature_names": list(self.feature_names),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_samples": self.total_samples,
            "generations_trained": self.generations_trained,
            "sessions": [asdict(s) for s in self.sessions],
            "peers": [asdict(p) for p in self.peers],
            "performance": self.performance,
            "champion": self.champion,
            "provenance": self.provenance,
            "reward_model": self.reward_model.to_state(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> UmbraModel:
        feature_names = tuple(data.get("feature_names", DEFAULT_FEATURE_NAMES))
        model = cls(
            reward_model=NeuralRewardModel.from_state(data["reward_model"]),
            name=data.get("name", "umbra-model"),
            feature_names=feature_names,
        )
        model.created_at = data.get("created_at", model.created_at)
        model.updated_at = data.get("updated_at", model.updated_at)
        model.total_samples = int(data.get("total_samples", 0))
        model.generations_trained = int(data.get("generations_trained", 0))
        model.sessions = [TrainingSession(**s) for s in data.get("sessions", [])]
        model.peers = [PeerRecord(**p) for p in data.get("peers", [])]
        model.performance = list(data.get("performance", []))
        model.champion = data.get("champion")
        model.provenance = dict(data.get("provenance", model.provenance))
        return model

    def save(self, path: str | Path) -> Path:
        """Write the checkpoint to ``path`` (``.umbra.json`` if no suffix given)."""
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".umbra.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved Umbra model checkpoint to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> UmbraModel:
        path = Path(path)
        if not path.exists() and path.suffix == "":
            path = path.with_suffix(".umbra.json")
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        last = self.performance[-1] if self.performance else {}
        champ = ""
        if self.champion:
            champ = (f"; champion cleared difficulty "
                     f"{self.champion.get('difficulty_cleared', 0.0):.3f} "
                     f"@ reward {self.champion.get('reward', 0.0):.1f}")
        return (
            f"{self.name}: {self.generations_trained} train steps, "
            f"{self.total_samples} samples, {len(self.peers)} peers tracked; "
            f"last best_reward={last.get('best_reward', 0.0):.2f}{champ} "
            f"(updated {self.updated_at})"
        )


__all__ = ["UmbraModel", "TrainingSession", "PeerRecord", "candidate_feature_vector",
           "DEFAULT_FEATURE_NAMES"]
