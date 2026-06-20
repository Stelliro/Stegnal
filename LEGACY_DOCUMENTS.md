# LEGACY DOCUMENTS

Consolidated documentation harvested from the duplicate / sibling Project-Umbra
working copies before they were archived. The full copies were compressed into
`TOP SECRET/Backups/` (one `.zip` each) and the originals removed on
**2026-06-20**. This file preserves the parts of their docs that were *not*
already present in this repo.

## Archive manifest

| Archived copy | VCS state | Notes |
| --- | --- | --- |
| `Project-Umbra - Grok` | branch `codex/check-for-errors-and-integrate-functions-d9ikll` @ `18cf9ba` | Exact duplicate of this repo's commit. Docs identical to current `README.md` / `RUN_NOTES.md` / `Project Notes.txt` (modulo line endings) — nothing unique to preserve. |
| `Project-Umbra - old` | same branch @ `18cf9ba` | Identical to `- Grok`. |
| `Project-Umbra-1` | branch `main` @ `878d46d` (2025-10-01) | Older Streamlit-era snapshot. Only unique content captured below. |
| `UmbraFusion` | no git repo (2025-11-02) | Distinct V4/V5/V6 "fusion" experiment. Docs captured in full below. |

> The current canonical docs live in `README.md`, `RUN_NOTES.md`, and
> `Project Notes.txt` in this repo. Use this file only for history/recovery.

---

## 1. Project-Umbra-1 — Streamlit-era delta (historical)

The `main`-branch snapshot predates the Tkinter desktop UI. It documented the
explorer as a **Streamlit web app** rather than a native window:

```
Launch the visual explorer UI (uses Streamlit):

    pip install -e .[ui]
    umbra ui --port 8501
```

This is the origin of the now-removed `streamlit` dependency and the
`streamlit run src/umbra/ui.py` line that used to be in `RUN_NOTES.md`. The UI
was later rewritten in Tkinter (`src/umbra/ui.py`) and customtkinter (`app.py`),
so the Streamlit path is obsolete and intentionally not carried forward.

---

## 2. UmbraFusion — fusion scaffold (distinct experiment)

UmbraFusion was a separate scaffold that merged Project-Umbra V4/V5/V6 and aimed
at a *learned* (ML) image↔wave round-trip. It is the most forward-looking of the
archived copies. Its key ideas are summarized in `Project Notes.txt`
("Project lineage") and reproduced verbatim here.

### 2a. `UmbraFusion/README.md`

```markdown
# UmbraFusion

A unified scaffold merging Project-Umbra V4/V5/V6 while preserving originals in `/legacy`.
Objective: image -> wave and wave -> image round-trip that resembles the original image via ML.
This scaffold exposes learning hooks without losing prior behavior.

## Layout
- core/       Transforms, pipeline entry points
- models/     Model definitions (AE/VAE/UNet/etc.)
- training/   Training loops, datasets, evaluation
- io/         Dataset I/O utils
- scripts/    Launchers
- legacy/     Untouched V4/V5/V6 copies
- docs/       Notes
- data/       Place sample inputs here

## Large Images & Compatibility
- Public API is stable: `image_to_wave(img, sr=...)` and `wave_to_image(wav, sr=..., target_size=(W,H))`.
- Under the hood we use a PRP (Feistel) so we can map pixels <-> audio without building giant index arrays.
- Pass the same `seed` both ways. To handle enormous images on constrained RAM, use `scale < 1.0`
  temporarily and train models to close the gap later.
- For quick checks on huge inputs, run: `python scripts\bench_large.py`
```

### 2b. `UmbraFusion/CHANGELOG.md`

```
- 2025-11-02 — Initial fusion scaffold created from V4/V5/V6.
  - Preserved all source files under `legacy/`.
  - Promoted likely modules into `core/`, `models/`, `training/`, `scripts/`, `docs/`.
  - Added `core/pipeline_demo.py` + `core/transforms.py` for a deterministic baseline round-trip.
  - Added `run.ps1`, `requirements.txt`, and `fusion_manifest.json`.
```

### 2c. `UmbraFusion/PATCH_NOTES.txt`

```
UmbraFusion Patch — 2025-11-02
Files: umbra_io/multi_downloader.py, core/transforms.py, ui/app.py, tools/validate_dataset.py
Preserves core image -> wave -> image. Adds retries, truncation checks,
verified raster transcode, logging of failures, and curriculum auto-fetch/promote.
```

> The downloader/curriculum tooling (`umbra_io/multi_downloader.py`,
> `pinterest_downloader.py`) and the `fusion_manifest.json` were left intact in
> the archived `UmbraFusion.zip` if that direction is ever revived.
