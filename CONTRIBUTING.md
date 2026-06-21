# Contributing to Project Umbra

**Project Umbra is an early-stage prototype, and help is genuinely wanted.**
Whether you want to run it and report what happens, fix a bug, or push the
research forward, contributions of all sizes are welcome.

> By contributing, you agree that your contributions are licensed under the
> project's [PolyForm Noncommercial License 1.0.0](LICENSE.md) — i.e. the project
> and everything in it remain free for noncommercial use only.

## Ways to help

- **Test it and report back.** Run the app on your own images and tell us how the
  reconstruction looks and what scores you get. Bugs, crashes, confusing UI, and
  "this number seems wrong" reports are all useful.
- **Improve the decoder.** The current decoder is a simple seed-based un-permute
  plus a Gaussian denoiser. It plateaus at a modest channel-noise level. A learned
  decoder (autoencoder / U-Net / diffusion) is the most promising direction for
  pushing the difficulty curriculum higher.
- **Improve the evolutionary search / difficulty curriculum.** See
  `src/umbra/evolution.py` and `src/umbra/difficulty.py`.
- **Hardware acoustic path.** Test the speaker→microphone capture
  (`src/umbra/capture.py`) on real hardware and report results.
- **Docs, tests, packaging, and tooling.** Always appreciated.

## Getting started (from source)

Requires **Python 3.12**.

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate
pip install -e ".[ui,dev]"
pytest                  # run the test suite
umbra smoke-test        # quick end-to-end sanity check
umbra ui                # the desktop evolution explorer
python app.py           # the standalone "Terminal" encode/decode tool
```

On Windows you can also just double-click `launch_umbra_ui.bat` (explorer) or
`launch_terminal.bat` (Terminal); they create the virtual environment on first run.

## Try the prebuilt app

If you don't want to set up Python, grab the latest **early build** `.exe` from
the [Releases](../../releases) page and run it.

## Reporting issues

Please open a [GitHub Issue](../../issues) and include:

- What you did and what you expected vs. what happened.
- Your OS and Python version (`python --version`).
- For reconstruction-quality reports: the input image size and the PSNR / SSIM /
  reward you saw, plus a screenshot if you can.
- Full error text / traceback for crashes.

## Pull requests

1. Fork and create a branch.
2. Keep changes focused; match the surrounding code style.
3. Run `ruff check .` and `pytest` before opening the PR.
4. Describe what you changed and why.

This is a research prototype — expect rough edges, and don't hesitate to ask
questions in an issue. Thanks for helping build it.
