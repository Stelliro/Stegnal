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
pip install -e .[ui,dev]
pytest
streamlit run src/umbra/ui.py  # optional UI smoke test
```
