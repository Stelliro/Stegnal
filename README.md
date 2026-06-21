# Project Umbra — Test Build 0.1.0

> **Status: early prototype — help wanted.** This is an experimental research
> build. Testers and contributors are welcome; see
> [CONTRIBUTING.md](CONTRIBUTING.md). Licensed for **noncommercial use only**
> under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

This repository contains the first toy build of the Project Umbra pipeline. The goal of the build is to demonstrate the end-to-end flow of transforming an input image into a noise-like carrier and reconstructing a recognizable image using the same secret seed.

The implementation is intentionally lightweight and self-contained so that the team can iterate quickly before integrating heavier AI models.

## Features

- **Noise-stream encoder** that permutes image pixels based on a shared seed and injects Gaussian noise to mimic a noisy channel.
- **Correlation-based decoder** that reverses the permutation, applies a denoising filter, and clips the results into image space.
- **Quality metrics** (PSNR and SSIM) that provide an approximate measure of reconstruction fidelity.
- **Command-line interface** that provides `encode`, `decode`, `pipeline`, `evaluate`, and `ui` commands.
- **Interactive desktop explorer** (Tkinter-based) that compares the original signal, encoded packet, reconstruction, and multiplicative overlap score side-by-side.
- **Generational evolution playground** with configurable AI attempt counts, infinite/finite runs, and autosave/load support for overnight experiments.
- **Automated test** verifying that the toy pipeline can recover a synthetic image with reasonable fidelity.

## Getting Started

### Quick start (Windows)

Double-click one of the launchers. On first run they create a dedicated `.venv`
(Python 3.12), install everything, and start the app; subsequent runs just launch:

- **`launch_umbra_ui.bat`** — the Tkinter desktop explorer / evolution playground (`umbra ui`).
- **`launch_terminal.bat`** — the standalone customtkinter "Terminal" (`app.py`).

### Manual setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui]"      # ".[ui]" pulls in the desktop-UI deps (customtkinter, sounddevice)
```

On Windows PowerShell:

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[ui]"
```

> Use Python **3.12** for this project. (Python 3.10 — which the old bundled
> venvs were built on — has been removed from this machine, and the system
> default `python` is 3.14.)

### Optional GPU acceleration

The waveform reconstruction pipeline can leverage NVIDIA GPUs via CuPy. Install the
CUDA-enabled dependencies with:

```bash
pip install -e .[gpu]
```

or, on PowerShell:

```powershell
pip install -e .[gpu]
```

If you already have CuPy installed, ensure it matches your CUDA toolkit version. The
runtime surfaces a recommendation such as `pip install -U "cupy-cuda12x"` whenever the
NVRTC runtime is missing.

When NVRTC ships with another application (for example, a bundled PyTorch runtime),
set the `UMBRA_NVRTC_PATH_HINTS` environment variable to point at the directory or DLL
path (multiple entries are separated with the platform path separator). The helper will
automatically wire the hint into `CUPY_NVRTC_PATH` before CuPy initializes.

## Usage

Encode an image:

```bash
umbra encode --image path/to/input.png --output packet.npz --seed 1234 --sigma 0.25
```

Decode a packet:

```bash
umbra decode --packet packet.npz --output recovered.png --seed 1234
```

Run the full pipeline and report metrics:

```bash
umbra pipeline --image path/to/input.png --seed 1234 --sigma 0.2 --packet packet.npz --reconstruction recon.png
```

Evaluate two images:

```bash
umbra evaluate --reference path/to/input.png --candidate recon.png
```

Launch the desktop visual explorer (Tkinter):

```bash
umbra ui
```

The command opens a native window that renders the original image, the encoded noise packet, the decoder's reconstruction, and a multiplicative overlap map that highlights shared signal energy. The sidebar controls expose the evolution playground with AI/sound composite scoring, Pinterest inspiration downloads, autosave, and long-running evolution support.

On Windows you can launch either desktop app directly by double-clicking its
helper script (each bootstraps the `.venv` on first run):

```bat
launch_umbra_ui.bat     :: Tkinter explorer  (python -m umbra ui)
launch_terminal.bat     :: customtkinter "Terminal"  (python app.py)
```

Both scripts call `python -m umbra ui`, so any extra arguments are forwarded to the CLI wrapper.

## Next Steps

This build establishes the scaffolding for more sophisticated experiments. Follow-up iterations should focus on:

1. Capturing richer channel effects (jitter, frequency offsets, burst errors).
2. Integrating a learned decoder (e.g., convolutional autoencoder).
3. Scaling the training dataset and benchmarking across different success metrics.
4. Documenting experimental results in the "Umbra Codex" logbook.
5. Add CI with linting and tests: `ruff`, `mypy`, `pytest`.
