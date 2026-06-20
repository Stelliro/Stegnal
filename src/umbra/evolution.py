# evolution.py - PURIST MODE (NO COLOR HACKING)
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Project Imports
from .audio import AUDIO_SCALE_FACTOR, image_data_to_audio

logger = logging.getLogger("evolution")

_HYPER_MODE = os.environ.get("UMBRA_HYPER_MODE", "").strip() not in ("", "0", "false")

_VALID_COMPUTE_MODES = ("cpu", "gpu", "hybrid")


def _env_compute_mode() -> str:
    """Default compute mode from ``UMBRA_COMPUTE_MODE`` (cpu|gpu|hybrid)."""
    mode = os.environ.get("UMBRA_COMPUTE_MODE", "cpu").strip().lower()
    return mode if mode in _VALID_COMPUTE_MODES else "cpu"


def _env_gpu_fraction() -> float:
    """Default GPU share for hybrid mode from ``UMBRA_GPU_FRACTION`` (0.0-1.0)."""
    try:
        value = float(os.environ.get("UMBRA_GPU_FRACTION", "0.5"))
    except ValueError:
        value = 0.5
    return min(1.0, max(0.0, value))


def _cupy_available() -> bool:
    """True when CuPy is importable and a GPU backend is present."""
    try:
        from . import gpu_runtime

        return getattr(gpu_runtime, "cp", None) is not None
    except Exception:
        return False


def _env_unlock_genes() -> bool:
    """Default for color/gamma gene unlocking from ``UMBRA_UNLOCK_GENES``."""
    return os.environ.get("UMBRA_UNLOCK_GENES", "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Adaptive-difficulty defaults. Difficulty is the channel noise level (encoder
# sigma): higher = harder. It only ratchets up once the population is
# *recognizable* (best SSIM >= MIN), so harder is earned, not imposed.
_DIFFICULTY_MIN_SSIM = 0.5     # must reconstruct this well before difficulty rises
_DIFFICULTY_STEP = 0.02        # per-generation difficulty change
_DIFFICULTY_REWARD_WEIGHT = 1.0  # how much surviving harder boosts the reward


# Neutral ("purist") color genes vs. modest randomized ranges when unlocked.
_NEUTRAL_COLOR_GENES = {
    "brightness_shift": 0.0, "contrast_scale": 1.0, "gamma": 1.0,
    "r_gain": 1.0, "g_gain": 1.0, "b_gain": 1.0,
}


def _sample_color_genes(unlock: bool) -> dict[str, float]:
    """Neutral colors in purist mode; small random perturbations when unlocked."""
    if not unlock:
        return dict(_NEUTRAL_COLOR_GENES)
    return {
        "brightness_shift": random.uniform(-0.08, 0.08),
        "contrast_scale": random.uniform(0.85, 1.15),
        "gamma": random.uniform(0.8, 1.25),
        "r_gain": random.uniform(0.9, 1.1),
        "g_gain": random.uniform(0.9, 1.1),
        "b_gain": random.uniform(0.9, 1.1),
    }


def _chaotic_seed_mix(seeds: list[int], noise: int, logistic: float) -> int:
    """Deterministic chaotic mixing of parent seeds with noise and logistic map."""

    combined = 0
    for idx, s in enumerate(seeds):
        shift = (idx * 13) % 31
        combined ^= (int(s) << shift) & 0x7FFFFFFF
    logistic_bits = int(abs(logistic) * 0x7FFFFFFF) & 0x7FFFFFFF
    noise_bits = int(noise) & 0x7FFFFFFF
    return (combined ^ logistic_bits ^ noise_bits) & 0x7FFFFFFF

class EvolutionLimitReached(Exception):
    def __init__(self, limit: int, attempted_generation: int) -> None:
        self.limit = limit
        self.attempted_generation = attempted_generation
        super().__init__(f"Evolution limit of {limit} reached at generation {attempted_generation}")


@dataclass
class LineageEntry:
    seed: int
    appearances: int = 0
    cumulative_reward: float = 0.0


@dataclass
class Heritage:
    """Heritable record carried by a candidate's bloodline.

    ``competence`` is the highest channel difficulty this lineage has actually
    cleared (SSIM >= the run's threshold) — a ratchet that only goes up and is
    passed to offspring, so each bloodline faces a channel matched to what it has
    proven it can handle.
    """

    competence: float = 0.01          # hardest difficulty cleared so far
    birth_generation: int = 0
    depth: int = 0                     # generations of ancestry
    parents: list = field(default_factory=list)   # parent seeds
    cumulative_reward: float = 0.0
    wins: int = 0                      # times this lineage topped a generation

    def child(self, other: "Heritage", *, generation: int, parent_seeds: list) -> "Heritage":
        """Breed a child heritage: inherit the stronger competence + deepen."""
        return Heritage(
            competence=max(self.competence, other.competence),
            birth_generation=int(generation),
            depth=max(self.depth, other.depth) + 1,
            parents=list(parent_seeds),
            cumulative_reward=0.0,
        )

    def to_dict(self) -> dict:
        return {
            "competence": self.competence, "birth_generation": self.birth_generation,
            "depth": self.depth, "parents": list(self.parents),
            "cumulative_reward": self.cumulative_reward, "wins": self.wins,
        }


@dataclass
class HyperProfile:
    enabled: bool = False
    batch_size: int = 5
    dwell_generations: int = 3
    last_update: int = 0


@dataclass
class Gene:
    seed: int
    sigma: float
    denoise_sigma: float
    inpainter_steps: int
    guidance_scale: float
    sharpness_factor: float
    clahe_clip_limit: float
    h_shift: float = 0.0
    brightness_shift: float = 0.0
    contrast_scale: float = 1.0
    gamma: float = 1.0
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0
    line_length: int = 256
    start_sample: int = 0
    scan_mode: int = 0

    def to_dict(self):
        return self.__dict__

@dataclass
class Candidate:
    genes: Gene
    reconstruction: np.ndarray = field(default=None, repr=False)
    metrics: Any = field(default=None, repr=False)
    reward: float = 0.0
    # New-API fields for sound alignment tracking
    sound_bootstrap_overlap: float | None = None
    reference_overlap: float = 0.0
    overlap_score: float = 0.0
    waveform_reconstruction: np.ndarray | None = field(default=None, repr=False)
    waveform_reference_metrics: Any = field(default=None, repr=False)
    waveform_reference_overlap: float = 0.0
    waveform_packet_metrics: Any = field(default=None, repr=False)
    waveform_packet_overlap: float = 0.0
    waveform_sample_rate: int = 0
    waveform_segments: int = 0
    waveform_marker_duration: float = 0.0
    waveform_sound_score: float = 0.0
    waveform_readability_score: float = 0.0
    waveform_alignment_score: float = 0.0
    waveform_reference_partial: float = 0.0
    waveform_alignment_partial: float = 0.0
    team_score: float = 0.0
    ai_score: float = 0.0
    heritage: Any = field(default=None, repr=False)
    # Per-evaluation bookkeeping (set during scoring; not persisted).
    eval_ssim: float = 0.0
    eval_difficulty: float = 0.0

    @property
    def seed(self) -> int:
        return self.genes.seed

@dataclass
class GenerationRecord:
    generation: int
    difficulty: float
    best_candidate: Candidate
    mean_reward: float
    candidates: list = field(default_factory=list)
    index: int = 0
    reward_summary: float = 0.0
    difficulty_level: float = 0.0


class ProxyPacket:
    def __init__(self, orig, raw_wave_1d):
        self._original = orig
        self.raw_wave = raw_wave_1d
        self.permutation_seed = orig.permutation_seed
        self.image_shape = orig.image_shape
        self.sigma = orig.sigma

class EvolutionManager:
    def __init__(
        self,
        reference_image=None,
        population_size=50,
        data_mode=False,
        audio_engine=None,
        audio_out_idx=0,
        audio_in_idx=0,
        simulation_mode=True,
        *,
        original=None,
        encoder=None,
        decoder=None,
        base_seed=None,
        autosave_interval=0,
        enable_waveform: bool = True,
        max_generations: int | None = None,
        compute_mode: str | None = None,
        gpu_fraction: float | None = None,
        unlock_genes: bool | None = None,
        difficulty_min_ssim: float | None = None,
        difficulty_step: float | None = None,
        difficulty_reward_weight: float | None = None,
    ):
        # Support both old (reference_image) and new (original=) APIs
        self.reference_image = original if original is not None else reference_image
        self.population_size = population_size
        self.audio_engine = audio_engine
        self.simulation_mode = simulation_mode
        self.audio_out, self.audio_in = audio_out_idx, audio_in_idx
        self.population, self.history = [], []
        self.generation_count = 0
        self.interference_cache = []
        self.enable_waveform = enable_waveform
        self.max_generations = max_generations

        # New-API attributes
        self._encoder = encoder
        self._decoder = decoder
        self.autosave_interval = autosave_interval
        self.generations: list[GenerationRecord] = []
        self.rng = np.random.default_rng(base_seed)
        self._base_seed = base_seed
        self._gpu_warning_emitted = False

        # Compute placement: "cpu" (default), "gpu" (all on GPU), or "hybrid"
        # (split each generation's population across a GPU worker + CPU workers).
        mode = (compute_mode or _env_compute_mode()).strip().lower()
        self._compute_mode = mode if mode in _VALID_COMPUTE_MODES else "cpu"
        self._gpu_fraction = (
            _env_gpu_fraction() if gpu_fraction is None
            else min(1.0, max(0.0, float(gpu_fraction)))
        )
        # Purist mode (default) locks color/gamma genes to neutral; unlocking lets
        # the search also tune brightness/contrast/gamma/RGB gains.
        self.unlock_genes = (
            _env_unlock_genes() if unlock_genes is None else bool(unlock_genes)
        )

        # Adaptive difficulty: gated on SSIM so we only make it harder once the
        # population is doing well, and credit the difficulty back into the reward.
        self.difficulty_min_ssim = (
            _env_float("UMBRA_DIFFICULTY_MIN_SSIM", _DIFFICULTY_MIN_SSIM)
            if difficulty_min_ssim is None else float(difficulty_min_ssim)
        )
        self.difficulty_step = (
            _env_float("UMBRA_DIFFICULTY_STEP", _DIFFICULTY_STEP)
            if difficulty_step is None else float(difficulty_step)
        )
        self.difficulty_reward_weight = (
            _env_float("UMBRA_DIFFICULTY_REWARD_WEIGHT", _DIFFICULTY_REWARD_WEIGHT)
            if difficulty_reward_weight is None else float(difficulty_reward_weight)
        )
        self.difficulty = 0.1            # current channel difficulty (encoder sigma)
        self._current_difficulty = 0.1   # difficulty of the generation being scored
        self._last_best_ssim = 0.0

        # Smart, learned difficulty controller (smooth transitions). Per-lineage
        # difficulty in sim mode advances each bloodline from its own competence.
        from .difficulty import DifficultyController
        self.difficulty_controller = DifficultyController(
            target_ssim=self.difficulty_min_ssim, max_step=self.difficulty_step * 2.5,
        )

        # Reward / lineage tracking
        self.lifetime_reward: float = 0.0
        self.mutation_boost: int = 0
        self.parent_lineage: list[LineageEntry] = []
        self._plateau_count: int = 0
        self._last_best_overlap: float = 0.0

        # Hyper mode
        self._hyper_profile = HyperProfile(
            enabled=_HYPER_MODE,
            batch_size=max(5, population_size) if _HYPER_MODE else population_size,
            dwell_generations=3,
            last_update=0,
        )
        if _HYPER_MODE:
            self.population_size = self._hyper_profile.batch_size

        # Compute a stable signature from the image content
        img_bytes = np.asarray(self.reference_image, dtype=np.float32).tobytes()
        self.image_signature = hashlib.sha256(img_bytes).hexdigest()

        # Only initialise old-style population when using legacy API
        if original is None and reference_image is not None:
            self._initialize_population()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def hyper_profile(self) -> HyperProfile:
        return self._hyper_profile

    # ------------------------------------------------------------------
    # New API: run_generation / save / load / update_settings
    # ------------------------------------------------------------------

    def _split_map(self, n: int, work) -> None:
        """Run ``work(i, use_gpu)`` for ``i`` in ``range(n)`` across CPU/GPU.

        Placement follows ``self._compute_mode``:
          * ``cpu``    – serial on CPU.
          * ``gpu``    – serial, every item on the GPU.
          * ``hybrid`` – a single dedicated worker thread runs the GPU share
            (CuPy is only ever touched from that one thread) while a CPU thread
            pool runs the rest concurrently. NumPy/skimage and CuPy both release
            the GIL during heavy work, so the two devices genuinely overlap.

        ``work`` must be independent per ``i`` (it writes its own slot / mutates
        its own candidate); ordering of side effects therefore does not matter.
        """
        mode = self._compute_mode
        gpu_ok = _cupy_available()
        if mode in ("gpu", "hybrid") and not gpu_ok:
            if not self._gpu_warning_emitted:
                logger.warning("GPU unavailable – evaluating on CPU")
                self._gpu_warning_emitted = True
            mode = "cpu"

        if mode == "cpu" or n <= 1:
            for i in range(n):
                work(i, mode == "gpu")
            return
        if mode == "gpu":
            for i in range(n):
                work(i, True)
            return

        # --- hybrid: GPU worker thread + CPU thread pool, concurrently ---
        import concurrent.futures as _cf
        import threading as _th

        n_gpu = max(0, min(n, int(round(n * self._gpu_fraction))))
        gpu_error: list[BaseException] = []

        def _run_gpu_share():
            try:
                for i in range(n_gpu):
                    work(i, True)
            except BaseException as exc:  # surface after join
                gpu_error.append(exc)

        gpu_thread = _th.Thread(target=_run_gpu_share, daemon=True)
        gpu_thread.start()

        if n_gpu < n:
            workers = max(1, min(n - n_gpu, os.cpu_count() or 2))
            with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, i, False) for i in range(n_gpu, n)]
                for fut in _cf.as_completed(futures):
                    fut.result()  # propagate CPU-side exceptions

        gpu_thread.join()
        if gpu_error:
            raise gpu_error[0]

    def _evaluate_population(self, genes_list, evaluate):
        """Map candidate genes -> Candidates using the CPU/GPU split scheduler."""
        results: list[Candidate | None] = [None] * len(genes_list)

        def work(i, use_gpu):
            results[i] = evaluate(genes_list[i], use_gpu)

        self._split_map(len(genes_list), work)
        return results

    def run_generation(self, *, parent_selection: list | None = None) -> GenerationRecord:
        """Run one evolutionary generation using the encoder/decoder API."""
        # Enforce generation limit
        if self.max_generations is not None and len(self.generations) >= self.max_generations:
            raise EvolutionLimitReached(self.max_generations, len(self.generations))

        from .codec import decode_wav_bytes_to_image, encode_image_to_wav_bytes
        from .metrics import (
            audio_fidelity_score,
            composite_score,
            compute_metrics,
            partial_alignment_fraction,
            readability_score,
            team_cohesion_score,
        )
        from .reconstruction import suggest_sample_rate, suggest_transmission_profile
        from .visualization import multiplicative_overlap

        reference = np.clip(
            np.asarray(self.reference_image, dtype=np.float32), 0.0, 1.0
        )
        # Ensure 3-channel image for the encoder
        if reference.ndim == 2:
            reference = np.stack([reference] * 3, axis=-1)

        # Check GPU availability and warn once
        try:
            from . import gpu_runtime
            if getattr(gpu_runtime, "cp", None) is None and not self._gpu_warning_emitted:
                logger.warning("GPU unavailable – falling back to CPU")
                self._gpu_warning_emitted = True
        except Exception:
            if not self._gpu_warning_emitted:
                logger.warning("GPU unavailable – falling back to CPU")
                self._gpu_warning_emitted = True

        seed = int(self.rng.integers(0, 2**31))
        packet = self._encoder.encode(reference, seed=seed)

        # Build candidate seeds: reuse parent_selection seeds + new offspring
        if parent_selection:
            # Keep at most half the population from parents; rest are fresh offspring
            max_parents = max(1, self.population_size // 2)
            parent_seeds = list(parent_selection)[:max_parents]
        else:
            parent_seeds = []

        # --- Phase A: build the gene population deterministically ---------
        # All RNG draws happen here, in order, so results are independent of
        # how the candidates are later distributed across CPU/GPU workers.
        genes_list: list[Gene] = []
        for i in range(self.population_size):
            if i < len(parent_seeds):
                cand_seed = int(parent_seeds[i])
            else:
                cand_seed = int(self.rng.integers(0, 2**31))

            # Mutate gene parameters so candidates produce diverse reconstructions
            denoise_sigma = float(np.clip(self.rng.normal(0.5, 0.25), 0.0, 2.0))
            brightness_shift = float(np.clip(self.rng.normal(0.0, 0.05), -0.15, 0.15))
            contrast_scale = float(np.clip(self.rng.normal(1.0, 0.1), 0.6, 1.6))
            gamma = float(np.clip(self.rng.normal(1.0, 0.15), 0.3, 2.5))
            r_gain = float(np.clip(self.rng.normal(1.0, 0.08), 0.7, 1.3))
            g_gain = float(np.clip(self.rng.normal(1.0, 0.08), 0.7, 1.3))
            b_gain = float(np.clip(self.rng.normal(1.0, 0.08), 0.7, 1.3))

            genes_list.append(Gene(
                seed=cand_seed, sigma=self._encoder.sigma,
                denoise_sigma=denoise_sigma, inpainter_steps=5, guidance_scale=1.0,
                sharpness_factor=1.0, clahe_clip_limit=0.1,
                brightness_shift=brightness_shift,
                contrast_scale=contrast_scale,
                gamma=gamma,
                r_gain=r_gain, g_gain=g_gain, b_gain=b_gain,
            ))

        # --- Phase B: evaluate candidates (CPU / GPU / hybrid split) -------
        def evaluate(gene: Gene, use_gpu: bool) -> Candidate:
            recon = self._decoder.decode(packet, seed=seed, genes=gene, use_gpu=use_gpu)
            recon = np.clip(np.asarray(recon, dtype=np.float32), 0.0, 1.0)
            pkt_metrics = compute_metrics(reference, recon)
            _, pkt_overlap = multiplicative_overlap(reference, recon)
            ai = composite_score(float(pkt_overlap), pkt_metrics.psnr, pkt_metrics.ssim)
            return Candidate(
                genes=gene,
                reconstruction=recon,
                metrics=pkt_metrics,
                reward=ai,
                reference_overlap=float(pkt_overlap),
                overlap_score=float(pkt_overlap),
                ai_score=ai,
            )

        candidates: list[Candidate] = self._evaluate_population(genes_list, evaluate)

        # Populate waveform fields when enabled
        if self.enable_waveform:
            wf_sample_rate = suggest_sample_rate(reference)
            wf_segments, wf_marker_dur = suggest_transmission_profile(reference)
            for cand in candidates:
                recon = np.clip(np.asarray(cand.reconstruction, dtype=np.float32), 0.0, 1.0)
                try:
                    wav_bytes = encode_image_to_wav_bytes(
                        recon,
                        sample_rate=wf_sample_rate,
                        segments=wf_segments,
                        marker_duration=wf_marker_dur,
                    )
                    sound_image, wf_meta = decode_wav_bytes_to_image(
                        wav_bytes,
                        resolution=reference.shape[:2],
                        sample_rate=wf_sample_rate,
                        segments=wf_segments,
                        marker_duration=wf_marker_dur,
                        return_metadata=True,
                    )
                except Exception:
                    continue
                sound_clipped = np.clip(np.asarray(sound_image, dtype=np.float32), 0.0, 1.0)

                ref_metrics = compute_metrics(reference, sound_clipped)
                _, ref_overlap = multiplicative_overlap(reference, sound_clipped)
                pkt_wf_metrics = compute_metrics(recon, sound_clipped)
                _, pkt_wf_overlap = multiplicative_overlap(recon, sound_clipped)
                ref_partial = partial_alignment_fraction(reference, sound_clipped)
                pkt_partial = partial_alignment_fraction(recon, sound_clipped)

                cand.waveform_reconstruction = sound_clipped
                cand.waveform_sample_rate = int(wf_meta.sample_rate)
                cand.waveform_segments = int(wf_meta.segments)
                cand.waveform_marker_duration = float(wf_meta.marker_duration)
                cand.waveform_reference_metrics = ref_metrics
                cand.waveform_reference_overlap = float(ref_overlap)
                cand.waveform_packet_metrics = pkt_wf_metrics
                cand.waveform_packet_overlap = float(pkt_wf_overlap)
                cand.waveform_reference_partial = ref_partial
                cand.waveform_alignment_partial = pkt_partial
                cand.waveform_sound_score = audio_fidelity_score(
                    float(ref_overlap), ref_metrics.psnr, ref_metrics.ssim,
                    partial_credit=ref_partial,
                )
                cand.waveform_readability_score = readability_score(
                    float(ref_overlap), ref_metrics.psnr, ref_metrics.ssim,
                )
                cand.waveform_alignment_score = audio_fidelity_score(
                    float(pkt_wf_overlap), pkt_wf_metrics.psnr, pkt_wf_metrics.ssim,
                    partial_credit=pkt_partial,
                )
                cand.team_score = team_cohesion_score(
                    cand.reference_overlap,
                    cand.metrics.psnr,
                    cand.metrics.ssim,
                    sound_reference_overlap=float(ref_overlap),
                    sound_reference_psnr=ref_metrics.psnr,
                    sound_reference_ssim=ref_metrics.ssim,
                    sound_alignment_overlap=float(pkt_wf_overlap),
                    sound_alignment_psnr=pkt_wf_metrics.psnr,
                    sound_alignment_ssim=pkt_wf_metrics.ssim,
                    sound_reference_partial=ref_partial,
                    sound_alignment_partial=pkt_partial,
                    readability=cand.waveform_readability_score,
                )

        candidates.sort(key=lambda c: c.reward, reverse=True)
        best = candidates[0]
        mean_rwd = float(np.mean([c.reward for c in candidates]))

        # Update lifetime reward
        self.lifetime_reward += sum(c.reward for c in candidates)

        # Update lineage
        lineage_map = {entry.seed: entry for entry in self.parent_lineage}
        for cand in candidates:
            if cand.seed in lineage_map:
                lineage_map[cand.seed].appearances += 1
                lineage_map[cand.seed].cumulative_reward += cand.reward
            else:
                entry = LineageEntry(seed=cand.seed, appearances=1, cumulative_reward=cand.reward)
                lineage_map[cand.seed] = entry
                self.parent_lineage.append(entry)

        # Plateau detection and sigma reduction
        improvement = best.overlap_score - self._last_best_overlap
        if improvement < 0.5:  # insignificant improvement threshold
            self._plateau_count += 1
        else:
            self._plateau_count = 0
        self._last_best_overlap = max(self._last_best_overlap, best.overlap_score)

        if self._plateau_count >= 2 and self._encoder.sigma > 0.01:
            self._encoder.sigma *= 0.9
            self.mutation_boost += 1
            self._plateau_count = 0

        gen_index = len(self.generations)
        record = GenerationRecord(
            generation=self.generation_count + 1,
            difficulty=self._encoder.sigma,
            best_candidate=best,
            mean_reward=mean_rwd,
            candidates=candidates,
            index=gen_index,
            reward_summary=mean_rwd,
            difficulty_level=best.overlap_score,
        )
        self.generation_count += 1
        self.generations.append(record)
        self.history.append(record)

        # Update hyper profile after generation
        if self._hyper_profile.enabled:
            self._hyper_profile.last_update = gen_index
            new_batch = max(self._hyper_profile.batch_size, self.population_size + 1)
            self._hyper_profile.batch_size = new_batch
            self.population_size = new_batch
            self._hyper_profile.dwell_generations = max(
                self._hyper_profile.dwell_generations,
                gen_index + 1,
            )

        return record

    def save(self, directory: str | Path) -> Path:
        """Persist evolution state to *directory* and return the JSON path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "evolution_state.json"
        data = {
            "image_signature": self.image_signature,
            "population_size": self.population_size,
            "autosave_interval": self.autosave_interval,
            "generation_count": self.generation_count,
            "generations": [
                {
                    "generation": rec.generation,
                    "difficulty": rec.difficulty,
                    "mean_reward": rec.mean_reward,
                    "best_reward": rec.best_candidate.reward,
                }
                for rec in self.generations
            ],
        }
        dest.write_text(json.dumps(data, indent=2))
        return dest

    @classmethod
    def load(cls, directory: str | Path) -> EvolutionManager:
        """Restore an EvolutionManager from a previously saved directory."""
        directory = Path(directory)
        src = directory / "evolution_state.json"
        raw = json.loads(src.read_text())

        # Create a minimal placeholder image (actual image not persisted)
        placeholder = np.zeros((8, 8), dtype=np.float32)
        from .decoding import NoiseStreamDecoder
        from .encoding import NoiseStreamEncoder

        mgr = cls(
            original=placeholder,
            encoder=NoiseStreamEncoder(sigma=0.1),
            decoder=NoiseStreamDecoder(denoise_sigma=None),
            population_size=raw["population_size"],
            autosave_interval=raw["autosave_interval"],
        )
        mgr.image_signature = raw["image_signature"]
        mgr.generation_count = raw["generation_count"]
        # Reconstruct lightweight generation records
        for g in raw["generations"]:
            dummy_gene = Gene(
                seed=0, sigma=0.1, denoise_sigma=0.0,
                inpainter_steps=5, guidance_scale=1.0,
                sharpness_factor=1.0, clahe_clip_limit=0.1,
            )
            dummy_cand = Candidate(genes=dummy_gene, reward=g["best_reward"])
            rec = GenerationRecord(
                generation=g["generation"],
                difficulty=g["difficulty"],
                best_candidate=dummy_cand,
                mean_reward=g["mean_reward"],
            )
            mgr.generations.append(rec)
            mgr.history.append(rec)
        return mgr

    def update_settings(self, **kwargs: Any) -> None:
        """Update manager settings while preserving history."""
        if "population_size" in kwargs:
            self.population_size = kwargs["population_size"]
        if "autosave_interval" in kwargs:
            self.autosave_interval = kwargs["autosave_interval"]
        if "compute_mode" in kwargs:
            mode = str(kwargs["compute_mode"]).strip().lower()
            if mode in _VALID_COMPUTE_MODES:
                self._compute_mode = mode
        if "gpu_fraction" in kwargs:
            self._gpu_fraction = min(1.0, max(0.0, float(kwargs["gpu_fraction"])))
        if "unlock_genes" in kwargs:
            self.unlock_genes = bool(kwargs["unlock_genes"])
        if "difficulty_min_ssim" in kwargs:
            self.difficulty_min_ssim = float(kwargs["difficulty_min_ssim"])
        if "difficulty_step" in kwargs:
            self.difficulty_step = float(kwargs["difficulty_step"])
        if "difficulty_reward_weight" in kwargs:
            self.difficulty_reward_weight = float(kwargs["difficulty_reward_weight"])

    def _spawn_child_seed(self, anchors: list[int]) -> int:
        """Derive a new seed from *anchors* using chaotic + Walsh mixing."""
        anchor_arr = np.array(anchors, dtype=np.int64)
        selected = self.rng.choice(anchor_arr, size=3, replace=False)

        random_val = self.rng.random()
        logistic = 3.999 * random_val * (1.0 - random_val)

        # Shifted XOR combination
        combined = 0
        for idx, parent_seed in enumerate(selected):
            shift = (idx * 17) % 31
            combined ^= (int(parent_seed) << shift) & 0x7FFFFFFF

        # Walsh-Hadamard XOR
        walsh = int(np.bitwise_xor.reduce(selected ^ np.roll(selected, 1))) & 0x7FFFFFFF

        noise = int(self.rng.integers(0, 2**31))
        chaotic = _chaotic_seed_mix(selected.tolist(), noise, logistic)
        logistic_component = int(abs(logistic) * 0x7FFFFFFF) & 0x7FFFFFFF
        mutation = int(self.rng.integers(0, 2**31))

        return (combined ^ walsh ^ chaotic ^ logistic_component ^ mutation) & 0x7FFFFFFF

    def _get_sound_reference_payload(self) -> dict | None:
        """Return the sound bootstrap payload, or None if unavailable."""
        return None

    # ------------------------------------------------------------------
    # Legacy API (used by ui.py)
    # ------------------------------------------------------------------

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            cand = Candidate(genes=self._random_gene())
            cand.heritage = Heritage(birth_generation=0)
            self.population.append(cand)

    def _random_gene(self) -> Gene:
        st_min, st_max = (0, 2) if self.simulation_mode else (0, 15000)
        
        return Gene(
            seed=random.randint(0, 2**32-1), 
            sigma=0.1, 
            denoise_sigma=random.uniform(0.0, 0.3),
            inpainter_steps=5,
            guidance_scale=1.0,
            sharpness_factor=1.0,
            clahe_clip_limit=0.1,
            h_shift=0.0,

            # Neutral in purist mode; randomized when genes are unlocked.
            **_sample_color_genes(self.unlock_genes),

            line_length=256,
            start_sample=random.randint(st_min, st_max),
            scan_mode=0
        )

    def recommend_difficulty(self, current: float, best_ssim: float) -> float:
        """Adaptive difficulty: only ratchet up once the run is recognizable.

        Difficulty rises (a *reward* for doing well) when ``best_ssim`` clears
        ``difficulty_min_ssim``; otherwise it eases back gently so the population
        can recover. The result self-balances at the hardest channel the run can
        currently reconstruct above the threshold.
        """
        current = float(current)
        if best_ssim >= self.difficulty_min_ssim:
            new = current + self.difficulty_step
        else:
            new = current - self.difficulty_step * 0.5
        self.difficulty = float(min(1.0, max(0.01, new)))
        return self.difficulty

    def _reference_3ch(self) -> np.ndarray:
        ref = np.clip(np.asarray(self.reference_image, dtype=np.float32), 0.0, 1.0)
        if ref.ndim == 2:
            ref = np.stack([ref] * 3, axis=-1)
        return ref

    def _score_recon(self, recon, reference, difficulty):
        """Structure-gated composite reward with the difficulty bonus."""
        from .metrics import composite_score, compute_metrics
        from .visualization import multiplicative_overlap

        recon = np.clip(np.asarray(recon, dtype=np.float32), 0.0, 1.0)
        metrics = compute_metrics(reference, recon)
        _, overlap = multiplicative_overlap(reference, recon)
        quality = composite_score(float(overlap), metrics.psnr, metrics.ssim)
        # Credit the difficulty being survived, but only once recognizable.
        bonus = 1.0
        if metrics.ssim >= self.difficulty_min_ssim:
            bonus += self.difficulty_reward_weight * float(difficulty)
        return float(quality * bonus * 100.0), metrics

    def _decode_from_wave(self, raw_wave, image_shape, perm_seed, genes, use_gpu):
        """Decode a recovered image from a (normalized) recorded waveform."""
        from .decoding import NoiseStreamDecoder
        from .encoding import NoisePacket

        req = image_shape[0] * image_shape[1] * 3
        start = int(max(0, getattr(genes, "start_sample", 0)))
        pixels = np.asarray(raw_wave, dtype=np.float32)[start:] * AUDIO_SCALE_FACTOR
        if len(pixels) < req:
            pixels = np.pad(pixels, (0, req - len(pixels)), constant_values=0.0)
        pkt = NoisePacket(encoded=pixels[:req], permutation_seed=perm_seed,
                          image_shape=image_shape, sigma=0.1)
        decoder = NoiseStreamDecoder(denoise_sigma=genes.denoise_sigma)
        return decoder.decode(pkt, seed=perm_seed, genes=genes, use_gpu=use_gpu)

    def _evaluate_at_difficulty(self, cand, reference, clean_encoded, image_shape,
                                perm_seed, difficulty, *, noise_seed, use_gpu=False):
        """Per-lineage sim evaluation: noise the shared key at this difficulty,
        push it through the audio channel, decode, and score. Returns
        ``(reward, ssim)``. Uses a private RNG so it is thread-safe under the
        hybrid CPU/GPU scheduler."""
        try:
            rng = np.random.default_rng(int(noise_seed) & 0xFFFFFFFF)
            if difficulty > 0:
                encoded = np.clip(clean_encoded + rng.normal(0, difficulty, clean_encoded.shape), 0.0, 1.0)
            else:
                encoded = clean_encoded
            wav, _ = image_data_to_audio(encoded.astype(np.float32), 48000)
            wav_norm = wav.astype(np.float32) / 32767.0
            if difficulty > 0:
                wav_norm = np.clip(wav_norm + rng.normal(0, difficulty * 0.1, len(wav_norm)), -1.0, 1.0)
            recon = self._decode_from_wave(wav_norm, image_shape, perm_seed, cand.genes, use_gpu)
            reward, metrics = self._score_recon(recon, reference, difficulty)
            cand.reconstruction = np.clip(np.asarray(recon, dtype=np.float32), 0.0, 1.0)
            cand.metrics = metrics
            return reward, float(metrics.ssim)
        except Exception:
            return 0.0, 0.0

    def evolve_generation(self, difficulty: float) -> GenerationRecord:
        from .encoding import NoiseStreamEncoder

        self._current_difficulty = float(difficulty)
        master_seed = random.randint(0, 2**32 - 1)
        reference = self._reference_3ch()

        for cand in self.population:
            cand.genes.seed = master_seed
            if cand.heritage is None:
                cand.heritage = Heritage(birth_generation=self.generation_count)

        if self.simulation_mode:
            # --- Per-lineage curriculum: shared permutation key, per-bloodline
            # difficulty advanced from each lineage's earned competence by a
            # learned, smooth step. ---
            clean_packet = NoiseStreamEncoder(sigma=0.0).encode(self.reference_image, seed=master_seed)
            clean_encoded = clean_packet.encoded
            image_shape = clean_packet.image_shape
            step = self.difficulty_controller.propose_step(self._last_best_ssim)

            def _score(i, use_gpu):
                cand = self.population[i]
                diff_i = float(np.clip(cand.heritage.competence + step, 0.01, 1.0))
                reward, ssim = self._evaluate_at_difficulty(
                    cand, reference, clean_encoded, image_shape, master_seed, diff_i,
                    noise_seed=master_seed + i, use_gpu=use_gpu,
                )
                cand.reward = reward
                cand.eval_ssim = ssim
                cand.eval_difficulty = diff_i
                cand.heritage.cumulative_reward += reward
                # Competence ratchet: clearing the bar at a harder channel sticks.
                if ssim >= self.difficulty_min_ssim and diff_i > cand.heritage.competence:
                    cand.heritage.competence = diff_i

            self._split_map(len(self.population), _score)

            # Representative channel for the UI air-signal view.
            med = float(np.median([c.heritage.competence for c in self.population]))
            self.difficulty = med
            viz_noise = np.random.default_rng(master_seed).normal(0, med, clean_encoded.shape)
            viz = np.clip(clean_encoded + viz_noise, 0.0, 1.0).astype(np.float32)
            vwav, _ = image_data_to_audio(viz, 48000)
            self.interference_cache = [vwav.astype(np.float32) / 32767.0]
        else:
            # --- Hardware: a single recorded channel shared by all candidates
            # (per-lineage is impractical when each take is a real play+record). ---
            base_packet = NoiseStreamEncoder(sigma=difficulty).encode(self.reference_image, seed=master_seed)
            self.interference_cache = []
            wav, sr = image_data_to_audio(base_packet.encoded, 48000)
            rec = self.audio_engine.transmit_and_record(wav, sr, self.audio_out, self.audio_in, use_sync_pulse=False)
            if rec is not None:
                self.interference_cache.append(np.asarray(rec, dtype=np.float32))
            batch_packet = ProxyPacket(base_packet, self.interference_cache[-1]) if self.interference_cache else base_packet

            def _score(i, use_gpu):
                cand = self.population[i]
                cand.reward = self._evaluate_single_candidate(cand, batch_packet, master_seed, use_gpu=use_gpu)
                cand.eval_ssim = float(getattr(cand.metrics, "ssim", 0.0) or 0.0)
                cand.eval_difficulty = float(difficulty)

            self._split_map(len(self.population), _score)

        self.population.sort(key=lambda c: c.reward, reverse=True)
        best = self.population[0]
        self._last_best_ssim = float(best.eval_ssim)
        self.difficulty_controller.observe(best.eval_difficulty, best.eval_ssim)
        if best.heritage is not None:
            best.heritage.wins += 1

        next_gen = [best]
        for _ in range(5):
            child = Candidate(genes=self._random_gene())
            child.heritage = Heritage(birth_generation=self.generation_count + 1)
            next_gen.append(child)

        while len(next_gen) < self.population_size:
            p1, p2 = random.sample(self.population[:15], 2)
            traits = {k: (v if random.random() > 0.5 else getattr(p2.genes, k)) for k, v in p1.genes.to_dict().items()}

            if random.random() < 0.3:
                if self.simulation_mode:
                    traits['start_sample'] = 0
                else:
                    traits['start_sample'] += random.randint(-10, 10)

                traits['denoise_sigma'] = np.clip(traits['denoise_sigma'] + random.uniform(-0.05, 0.05), 0.0, 1.0)

                if self.unlock_genes:
                    traits['contrast_scale'] = float(np.clip(traits['contrast_scale'] + random.uniform(-0.07, 0.07), 0.6, 1.6))
                    traits['gamma'] = float(np.clip(traits['gamma'] + random.uniform(-0.1, 0.1), 0.5, 2.0))
                    traits['brightness_shift'] = float(np.clip(traits['brightness_shift'] + random.uniform(-0.04, 0.04), -0.2, 0.2))
                    traits['r_gain'] = float(np.clip(traits['r_gain'] + random.uniform(-0.05, 0.05), 0.7, 1.3))
                    traits['g_gain'] = float(np.clip(traits['g_gain'] + random.uniform(-0.05, 0.05), 0.7, 1.3))
                    traits['b_gain'] = float(np.clip(traits['b_gain'] + random.uniform(-0.05, 0.05), 0.7, 1.3))
                else:
                    traits['contrast_scale'] = 1.0
                    traits['gamma'] = 1.0
                    traits['brightness_shift'] = 0.0
                    traits['r_gain'] = 1.0
                    traits['g_gain'] = 1.0
                    traits['b_gain'] = 1.0

            traits['seed'] = master_seed
            child = Candidate(genes=Gene(**traits))
            # Inherit the stronger parent's competence (heritage ratchet).
            child.heritage = p1.heritage.child(
                p2.heritage, generation=self.generation_count + 1,
                parent_seeds=[p1.seed, p2.seed],
            )
            next_gen.append(child)

        self.population = next_gen
        self.generation_count += 1
        rec = GenerationRecord(self.generation_count, self.difficulty, best, 0.0)
        self.history.append(rec)
        return rec

    def _evaluate_single_candidate(self, cand, packet, seed, *, use_gpu: bool = False) -> float:
        from .decoding import NoiseStreamDecoder
        from .encoding import NoisePacket
        try:
            if not hasattr(packet, 'raw_wave'):
                return 0.0

            reference = self._reference_3ch()

            best_score = -1.0
            best_recon = None
            best_metrics = None

            offsets = [0] if self.simulation_mode else [0, -1, 1]

            for offset in offsets:
                start = int(max(0, cand.genes.start_sample + offset))
                raw = packet.raw_wave[start:]
                pixels = raw * AUDIO_SCALE_FACTOR
                req = packet.image_shape[0] * packet.image_shape[1] * 3

                if len(pixels) < req:
                    pixels = np.pad(pixels, (0, req - len(pixels)), constant_values=0.0)

                test_pkt = NoisePacket(encoded=pixels[:req], permutation_seed=packet.permutation_seed, image_shape=packet.image_shape, sigma=0.1)

                decoder = NoiseStreamDecoder(denoise_sigma=cand.genes.denoise_sigma)
                recon = decoder.decode(test_pkt, seed=seed, genes=cand.genes, use_gpu=use_gpu)

                # Structure-gated composite reward + difficulty bonus (shared logic).
                score, metrics = self._score_recon(recon, reference, self._current_difficulty)
                if score > best_score:
                    best_score = score
                    best_recon = np.clip(np.asarray(recon, dtype=np.float32), 0.0, 1.0)
                    best_metrics = metrics

            cand.reconstruction = best_recon
            cand.metrics = best_metrics
            return max(0.0, best_score)
        except Exception:
            return 0.0

    def export_history_json(self, path: str) -> bool:
        if not self.history:
            return False
        data = []
        for rec in self.history:
            ssim_val = getattr(rec.best_candidate.metrics, 'ssim', 0)
            if hasattr(ssim_val, 'item'):
                ssim_val = ssim_val.item()
            reward_val = float(rec.best_candidate.reward)
            m_data = {"ssim": float(ssim_val)}
            data.append({
                "generation": int(rec.generation),
                "reward": reward_val,
                "metrics": m_data,
                "genes": rec.best_candidate.genes.to_dict()
            })
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Export Error: {e}")
            return False