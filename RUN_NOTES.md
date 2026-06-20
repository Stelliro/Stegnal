# Run Notes

This release introduces provenance breadcrumbs, reward instrumentation, and
adaptive plateau controls. The defaults remain unchanged; the toggles below are
optional and disabled unless explicitly enabled.

## Environment toggles

| Variable | Default | Description |
| --- | --- | --- |
| `UMBRA_PLATEAU_WINDOW` | `6` | Generations considered when measuring overlap drift. |
| `UMBRA_PLATEAU_DELTA_THRESHOLD` | `0.002` | Normalized overlap range that triggers a plateau kick. |
| `UMBRA_PLATEAU_BOOST_STEP` | `2` | Increment applied to the mutation boost when a plateau is detected. |
| `UMBRA_PLATEAU_BOOST_CAP_FACTOR` | `3` | Caps the mutation boost relative to the population size. |
| `UMBRA_PLATEAU_LOG` | `1` | Set to `0` to silence plateau log messages. |
| `UMBRA_DIFFICULTY_LADDER_ENABLED` | `0` | Enable the periodic difficulty ladder (two-generation spike every 12 generations). |
| `UMBRA_DIFFICULTY_LADDER_BASE` | `0.48` | Base normalized difficulty when the ladder is active. |
| `UMBRA_DIFFICULTY_LADDER_SPIKE` | `0.60` | Spike difficulty value during ladder cycles. |
| `UMBRA_DIFFICULTY_LADDER_PERIOD` | `12` | Ladder period, in generations. |
| `UMBRA_DIFFICULTY_LADDER_SPIKE_LEN` | `2` | Number of generations that use the spike value within each ladder period. |
| `UMBRA_REWARD_MODE` | `strict` | Choose `strict` (overlap-weighted) or `perceptual_mix` (adds MS-SSIM and DCT correlation terms). |
| `UMBRA_HYPER_MODE` | `0` | When set to `1`, enables hyper performance tuning that auto-sizes populations and dwell windows from runtime throughput and difficulty. |
| `UMBRA_COMPUTE_MODE` | `cpu` | Candidate-evaluation placement: `cpu`, `gpu` (all on the GPU via CuPy), or `hybrid` (split each generation across a GPU worker + CPU thread pool, running concurrently). Falls back to CPU when CuPy is unavailable. |
| `UMBRA_GPU_FRACTION` | `0.5` | In `hybrid` mode, the fraction (0.0–1.0) of each generation routed to the GPU; the rest run on CPU at the same time. |
| `UMBRA_UNLOCK_GENES` | `0` | Set to `1` to let the search tune color/gamma/contrast genes too (default "purist" mode locks them to neutral). Note: benchmarks show this usually *lowers* the score in short runs — the extra dimensions dilute convergence — so leave it off unless experimenting. |
| `UMBRA_DIFFICULTY_MIN_SSIM` | `0.5` | Auto-difficulty only raises the channel difficulty once the best SSIM clears this bar; below it, difficulty eases off. So harder is *earned*, never imposed on a struggling run. |
| `UMBRA_DIFFICULTY_STEP` | `0.02` | Per-generation difficulty change applied by the adaptive controller. |
| `UMBRA_DIFFICULTY_REWARD_WEIGHT` | `1.0` | How much the current difficulty boosts a *recognizable* candidate's reward (`reward = quality·(1 + weight·difficulty)·100` when SSIM ≥ min). Makes "good at a harder channel" score higher, so progress isn't punished. |

## Training, checkpoints & acoustic capture

* **Model checkpoints** (`umbra.checkpoint.UmbraModel`): wraps the trainable
  `NeuralRewardModel` with metadata — training sessions, lineage/"peers" (best
  candidate seeds + rewards), per-session performance, and provenance — in one
  self-describing `*.umbra.json` file. `UmbraModel.load(path)` restores the
  weights **and** optimiser momentum / feature stats, so you can keep training
  where you left off, then `save()` again. The desktop UI's **SAVE MODEL** /
  **LOAD MODEL** buttons drive this; it trains from each generation automatically.
* **Acoustic capture** (`umbra.capture`): `record_batch(...)` plays the encoded
  noise and records it back through the mic `repeats` times into separate WAVs +
  a `manifest.json` (the "through-air" path). `average_recordings(...)` combines
  takes to suppress per-capture noise (the "messy key" idea); `score_recordings(...)`
  decodes each take into `(features, rewards)` to train the model further. In the
  UI: uncheck **SILENT SIM**, pick speaker/mic, and hit **CAPTURE x10 → TRAIN**.
  (Real capture needs working audio hardware.)

## CPU / GPU split processing

Candidate decoding/scoring can run on CPU, GPU, or a concurrent **hybrid** mix.
Set it per run via `UMBRA_COMPUTE_MODE` / `UMBRA_GPU_FRACTION`, the
`EvolutionManager(compute_mode=..., gpu_fraction=...)` constructor args, or live
from the desktop UI's **COMPUTE** dropdown.

* GPU acceleration requires CuPy — install with `pip install -e ".[gpu]"`
  (this machine runs `cupy-cuda11x`; the driver also supports `cupy-cuda12x`).
* The first GPU generation pays a one-time CuPy kernel-compile cost (cached on
  disk afterwards). GPU/hybrid pays off on larger images; for tiny images the
  per-call transfer overhead means plain `cpu` can be faster.
* Numerical note: GPU and CPU Gaussian-denoise implementations differ by a tiny
  amount, so reconstructions (and thus scores) can vary negligibly between modes.

## Heritage & per-lineage difficulty (simulation mode)

In simulation mode each candidate carries a **Heritage** record whose
`competence` is the hardest channel difficulty its bloodline has actually cleared
(SSIM ≥ `difficulty_min_ssim`). Competence is a ratchet and is inherited
(children take the stronger parent's), so every lineage faces a channel matched
to what it has proven — a **curriculum per bloodline** rather than one global
difficulty. Offspring also record their parents, birth generation, and depth;
the top peers (with the difficulty they mastered) are saved into the model
checkpoint's metadata.

How far past a lineage's competence to probe each generation is set by
`umbra.difficulty.DifficultyController`, a tiny self-learning controller: it
estimates the local `dSSIM/dDifficulty` slope online and proposes a damped,
capped step toward the target SSIM, so difficulty transitions stay **smooth**
instead of using a fixed step. (Hardware/acoustic mode still uses one shared
recorded channel for the whole population — per-lineage channels aren't practical
when each take is a real play+record.)

## Export additions

* `session_summary.json` now contains a `provenance` block with git hash,
  binary versions, device information, and deterministic random seeds.
* The `metrics` section distinguishes `global_pooled` and
  `per_candidate_strict` summaries.
* The `difficulty` object exposes `raw`, `normalized`, and `target` values.
* Hyper performance mode (when enabled) adds a `hyper_performance` block with throughput-aware recommendations.
* `generation_progress.csv` includes `difficulty_raw`, `difficulty_normalized`,
  `reward_total`, `reward_overlap`, `reward_msssim`, `reward_dct_corr`, and
  `checkpoint_tag` (string markers for plateau kicks).

## Reproducing a smoke run

```bash
pip install -e ".[ui,dev]"
pytest
python -m umbra ui   # optional UI smoke test (Tkinter desktop explorer)
```
