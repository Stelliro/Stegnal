# Project Umbra — Test Build 0.1.0

This repository contains the first toy build of the Project Umbra pipeline. The goal of the build is to demonstrate the end-to-end flow of transforming an input image into a noise-like carrier and reconstructing a recognizable image using the same secret seed.

The implementation is intentionally lightweight and self-contained so that the team can iterate quickly before integrating heavier AI models.

## Features

- **Noise-stream encoder** that permutes image pixels based on a shared seed and injects Gaussian noise to mimic a noisy channel.
- **Correlation-based decoder** that reverses the permutation, applies a denoising filter, and clips the results into image space.
- **Quality metrics** (PSNR and SSIM) that provide an approximate measure of reconstruction fidelity.
- **Command-line interface** that provides `encode`, `decode`, `pipeline`, `evaluate`, and `ui` commands.
- **Interactive visual explorer** that compares the original signal, encoded packet, reconstruction, and multiplicative overlap score side-by-side.
- **Generational evolution playground** with configurable AI attempt counts, infinite/finite runs, and autosave/load support for overnight experiments.
- **Automated test** verifying that the toy pipeline can recover a synthetic image with reasonable fidelity.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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

Launch the visual explorer UI (uses [Streamlit](https://streamlit.io/)):

```bash
umbra ui --port 8501
```

The command spawns a local Streamlit server that renders the original image, the encoded noise packet, the decoder's reconstruction, and a multiplicative overlap map that highlights shared signal energy. Use the sidebar controls to adjust the encoder/decoder hyperparameters and experiment with either uploaded images or bundled grayscale samples.

The visual explorer now includes an **evolution playground** that lets you:

- Pick how many AI attempts run each generation and how many generations execute before stopping.
- Toggle an infinite mode that keeps evolving until you hit **Stop evolution**.
- Inspect every generation's reconstructions, pick individual candidates for overlap analysis, and review per-candidate metrics.
- Automatically save checkpoints to `~/.umbra_autosave/evolution_state.pkl` (or a custom directory) and resume from the latest snapshot on launch.

On Windows you can launch the UI directly with the provided helper scripts:

```powershell
./launch_umbra_ui.ps1
```

or

```bat
launch_umbra_ui.bat
```

Both scripts forward additional arguments to `streamlit run`, so you can override ports or Streamlit options if needed.

## Next Steps

This build establishes the scaffolding for more sophisticated experiments. Follow-up iterations should focus on:

1. Capturing richer channel effects (jitter, frequency offsets, burst errors).
2. Integrating a learned decoder (e.g., convolutional autoencoder).
3. Scaling the training dataset and benchmarking across different success metrics.
4. Documenting experimental results in the "Umbra Codex" logbook.
