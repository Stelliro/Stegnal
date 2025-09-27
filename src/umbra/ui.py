"""Streamlit-based visual explorer for Project Umbra."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoisePacket, NoiseStreamEncoder
from umbra.evolution import EvolutionManager, compute_image_signature
from umbra.metrics import compute_metrics
from umbra.sound import ShapeGuess, generate_sound_art, guess_shapes
from umbra.visualization import (
    colorize_comparison,
    multiplicative_overlap,
    normalize_for_display,
    to_uint8_image,
)


DEFAULT_AUTOSAVE_DIR = Path.home() / ".umbra_autosave"


def _autosave_path(directory: Path) -> Path:
    return directory / "evolution_state.pkl"


def _attempt_autoload(autosave_dir: Path) -> None:
    state = st.session_state
    try:
        manager = EvolutionManager.load(autosave_dir)
    except FileNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.sidebar.warning(f"Failed to load autosave: {exc}")
        return

    state["evolution_manager"] = manager
    state["evolution_signature"] = (
        manager.image_signature,
        float(manager.encoder.sigma),
        float(manager.decoder.denoise_sigma or 0.0),
        int(manager.base_seed),
    )
    state["shared_seed"] = manager.base_seed
    state["encoder_sigma"] = float(manager.encoder.sigma)
    state["decoder_sigma"] = float(manager.decoder.denoise_sigma or 0.0)
    state["population_size"] = manager.population_size
    state["autosave_interval"] = manager.autosave_interval
    st.sidebar.success("Loaded autosaved evolution session.")


def _ensure_manager(
    original: np.ndarray,
    encoder: NoiseStreamEncoder,
    decoder: NoiseStreamDecoder,
    population_size: int,
    seed: int,
    autosave_interval: int,
) -> EvolutionManager:
    state = st.session_state
    signature = (
        compute_image_signature(original),
        float(encoder.sigma),
        float(decoder.denoise_sigma or 0.0),
        int(seed),
    )

    manager: EvolutionManager | None = state.get("evolution_manager")
    if manager is None or state.get("evolution_signature") != signature:
        manager = EvolutionManager(
            original=original,
            encoder=encoder,
            decoder=decoder,
            population_size=population_size,
            base_seed=seed,
            autosave_interval=autosave_interval,
        )
        state["evolution_manager"] = manager
        state["evolution_signature"] = signature
        state["pending_generations"] = 0
        state["run_infinite"] = False
    else:
        manager.update_settings(
            original=original,
            encoder=encoder,
            decoder=decoder,
            population_size=population_size,
            autosave_interval=autosave_interval,
        )
    return manager


def _trigger_rerun() -> None:
    rerun = getattr(st, "experimental_rerun", None)
    if rerun is None:
        rerun = getattr(st, "rerun")
    rerun()


def _predict_noise_map(image: np.ndarray, packet: NoisePacket) -> np.ndarray:
    """Recover the exact noise contribution used during encoding."""

    flat_image = np.asarray(image, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(packet.permutation_seed)
    permutation = rng.permutation(flat_image.size)
    permuted = flat_image[permutation]
    noise = packet.encoded - permuted
    return noise.reshape(packet.image_shape).astype(np.float32)


def _build_color_template(color: np.ndarray, grayscale: np.ndarray) -> np.ndarray:
    """Create a colour template that re-applies the original hues to reconstructions."""

    rgb = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    gray = np.clip(np.asarray(grayscale, dtype=np.float32), 0.0, 1.0)
    template = np.zeros_like(rgb)
    denom = np.where(gray[..., None] > 1e-6, gray[..., None], 1.0)
    template = np.where(gray[..., None] > 1e-6, rgb / denom, rgb)
    return template.astype(np.float32)


def _apply_color_template(grayscale: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Colourize ``grayscale`` using the ratios captured in ``template``."""

    gray = np.clip(np.asarray(grayscale, dtype=np.float32), 0.0, 1.0)
    tinted = gray[..., None] * template
    return np.clip(tinted, 0.0, 1.0).astype(np.float32)


def run() -> None:
    """Entry-point for the Streamlit application."""
    st.set_page_config(page_title="Project Umbra Visual Explorer", layout="wide")
    st.title("Project Umbra Visual Explorer")
    st.markdown(
        """
        Use this dashboard to inspect how the Project Umbra toy pipeline encodes
        images into apparent noise and reconstructs them. Adjust the parameters to
        explore how the stochastic encoder and decoder behave, and inspect the
        overlap score that multiplies the generated and detected imagery.
        """
    )

    state = st.session_state
    state.setdefault("pending_generations", 0)
    state.setdefault("run_infinite", False)
    state.setdefault("autosave_dir", str(DEFAULT_AUTOSAVE_DIR))
    state.setdefault("last_autosave_dir", str(Path(state["autosave_dir"]).expanduser()))
    state.setdefault("autosave_checked", False)
    state.setdefault("population_size", 4)
    state.setdefault("autosave_interval", 5)
    state.setdefault("generations_to_queue", 5)
    state.setdefault("evolution_mode", "Finite")
    state.setdefault("shared_seed", 1234)
    state.setdefault("encoder_sigma", 0.2)
    state.setdefault("decoder_sigma", 1.0)
    state.setdefault("sound_seed", 4321)
    state.setdefault("sound_sample_rate", 48_000)
    state.setdefault("sound_resolution", 192)

    st.sidebar.header("Input & Parameters")
    autosave_input = st.sidebar.text_input(
        "Autosave directory",
        value=state.get("autosave_dir", str(DEFAULT_AUTOSAVE_DIR)),
        help="Evolution checkpoints are saved here as evolution_state.pkl.",
        key="autosave_dir",
    )
    autosave_dir = Path(autosave_input).expanduser()
    normalized_autosave_dir = str(autosave_dir)
    if state.get("last_autosave_dir") != normalized_autosave_dir:
        state["last_autosave_dir"] = normalized_autosave_dir
        state["autosave_checked"] = False

    autosave_path = _autosave_path(autosave_dir)
    if not state.get("autosave_checked"):
        if autosave_path.exists():
            _attempt_autoload(autosave_dir)
        state["autosave_checked"] = True

    seed = int(
        st.sidebar.number_input(
            "Shared seed",
            min_value=0,
            value=int(state.get("shared_seed", 1234)),
            step=1,
            key="shared_seed",
        )
    )
    sigma = float(
        st.sidebar.slider(
            "Encoder noise σ",
            min_value=0.0,
            max_value=1.0,
            value=float(state.get("encoder_sigma", 0.2)),
            step=0.01,
            key="encoder_sigma",
        )
    )
    denoise_sigma = float(
        st.sidebar.slider(
            "Decoder denoise σ",
            min_value=0.0,
            max_value=5.0,
            value=float(state.get("decoder_sigma", 1.0)),
            step=0.1,
            help="Gaussian blur strength applied after decoding",
            key="decoder_sigma",
        )
    )

    st.sidebar.subheader("Sound synthesis")
    sound_seed = int(
        st.sidebar.number_input(
            "Sound seed",
            min_value=0,
            value=int(state.get("sound_seed", seed)),
            step=1,
            key="sound_seed",
        )
    )
    sample_rate = int(
        st.sidebar.slider(
            "Sound sample rate",
            min_value=8_000,
            max_value=96_000,
            value=int(state.get("sound_sample_rate", 48_000)),
            step=1_000,
            help="Controls the number of random samples driving the colour volumes.",
            key="sound_sample_rate",
        )
    )
    resolution = int(
        st.sidebar.select_slider(
            "Generated image resolution",
            options=[128, 192, 256],
            value=int(state.get("sound_resolution", 192)),
            key="sound_resolution",
        )
    )

    original_color, original, sound_clip, shape_specs = generate_sound_art(
        seed=sound_seed,
        image_size=(resolution, resolution),
        sample_rate=sample_rate,
    )
    source_label = f"Sound seed {sound_seed}"
    color_template = _build_color_template(original_color, original)

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma if denoise_sigma > 0 else None)

    packet = encoder.encode(original, int(seed))
    reconstructed = decoder.decode(packet, int(seed))

    noise_map = _predict_noise_map(original, packet)
    noise_display = normalize_for_display(noise_map)
    packet_display = normalize_for_display(packet.encoded.reshape(original.shape))

    sound_packet = NoisePacket(
        encoded=np.asarray(noise_map.reshape(-1), dtype=np.float32),
        image_shape=packet.image_shape,
        permutation_seed=packet.permutation_seed,
        sigma=packet.sigma,
    )
    sound_reconstruction = decoder.decode(sound_packet, int(seed))

    colored_original = np.clip(original_color, 0.0, 1.0).astype(np.float32)
    ai_colored = _apply_color_template(reconstructed, color_template)
    sound_colored = _apply_color_template(sound_reconstruction, color_template)

    overlap_map, overlap_score = multiplicative_overlap(original, reconstructed)
    ai_overlap_color = colorize_comparison(original, reconstructed)
    _, sound_overlap_score = multiplicative_overlap(original, sound_reconstruction)
    sound_overlap_color = colorize_comparison(original, sound_reconstruction)
    cross_overlap_color = colorize_comparison(reconstructed, sound_reconstruction)

    metrics = compute_metrics(colored_original, ai_colored)
    sound_metrics = compute_metrics(colored_original, sound_colored)
    ai_sound_alignment = compute_metrics(ai_colored, sound_colored)

    st.subheader("Sound profile")
    volume_cols = st.columns(3)
    for idx, color in enumerate(("red", "green", "blue")):
        volume_cols[idx].metric(
            f"{color.title()} volume",
            f"{sound_clip.band_volumes[color]:.2f}",
            help="Relative energy detected in the sound clip for this colour band.",
        )
    st.caption(
        f"Waveform length: {sound_clip.samples.size} samples @ {sound_clip.sample_rate} Hz."
    )

    st.subheader("Reconstruction quality")
    ai_metrics_cols = st.columns(3)
    ai_metrics_cols[0].metric("AI colour PSNR", f"{metrics.psnr:.2f} dB")
    ai_metrics_cols[1].metric("AI colour SSIM", f"{metrics.ssim:.3f}")
    ai_metrics_cols[2].metric("AI overlap", f"{overlap_score:.1f}%")

    sound_metrics_cols = st.columns(3)
    sound_metrics_cols[0].metric("Sound colour PSNR", f"{sound_metrics.psnr:.2f} dB")
    sound_metrics_cols[1].metric("Sound colour SSIM", f"{sound_metrics.ssim:.3f}")
    sound_metrics_cols[2].metric("Sound overlap", f"{sound_overlap_score:.1f}%")

    st.metric("AI ↔ Sound colour SSIM", f"{ai_sound_alignment.ssim:.3f}")

    st.write(
        "The overlap score multiplies the normalized original and reconstructed pixels," \
        " providing a quick proxy for how much of the signal is mutually present."
    )

    st.subheader("Shape guessing AI")
    ai_guess_map: Dict[str, ShapeGuess] = {guess.color: guess for guess in guess_shapes(ai_colored)}
    sound_guess_map: Dict[str, ShapeGuess] = {
        guess.color: guess for guess in guess_shapes(sound_colored)
    }
    guess_rows = []
    for spec in shape_specs:
        ai_guess = ai_guess_map.get(spec.color)
        sound_guess = sound_guess_map.get(spec.color)
        guess_rows.append(
            {
                "Colour": spec.color.title(),
                "Target shape": spec.shape.title(),
                "Target volume": f"{spec.volume:.2f}",
                "AI guess": ai_guess.guess.title() if ai_guess else "None",
                "AI confidence": f"{ai_guess.confidence:.2f}" if ai_guess else "0.00",
                "AI match": "✅" if ai_guess and ai_guess.guess == spec.shape else "❌",
                "Sound guess": sound_guess.guess.title() if sound_guess else "None",
                "Sound confidence": f"{sound_guess.confidence:.2f}" if sound_guess else "0.00",
                "Sound match": "✅" if sound_guess and sound_guess.guess == spec.shape else "❌",
            }
        )
    st.table(guess_rows)
    st.caption(
        "Both AIs operate on the colourised reconstructions. Matching guesses indicate that "
        "the generated patterns retained the intended geometric cues."
    )

    st.subheader("Visual comparisons")
    overview_row = [
        (to_uint8_image(colored_original), f"Sound-derived image ({source_label})"),
        (to_uint8_image(packet_display), "Encoded packet (noise + signal)"),
        (to_uint8_image(ai_colored), "AI reconstruction (colourised)"),
        (to_uint8_image(sound_colored), "Sound-only reconstruction (colourised)"),
    ]
    overlay_row = [
        (to_uint8_image(noise_display), "Predicted noise contribution"),
        (to_uint8_image(ai_overlap_color), "Colour overlap: AI vs original"),
        (to_uint8_image(sound_overlap_color), "Colour overlap: Sound vs original"),
        (to_uint8_image(cross_overlap_color), "Colour overlap: AI vs sound"),
    ]

    for columns, content in ((st.columns(4), overview_row), (st.columns(4), overlay_row)):
        for col, (image, caption) in zip(columns, content):
            col.image(image, caption=caption, width="stretch", clamp=True)

    st.caption(
        "Red highlights information present only in the generated candidate, blue marks"
        " reference-only structure, and neutral grayscale indicates shared content."
    )

    st.sidebar.subheader("Evolution settings")
    population_size = int(
        st.sidebar.number_input(
            "AI attempts per generation",
            min_value=1,
            max_value=32,
            value=int(state.get("population_size", 4)),
            step=1,
            key="population_size",
        )
    )
    generations_to_queue = int(
        st.sidebar.number_input(
            "Generations to queue",
            min_value=1,
            value=int(state.get("generations_to_queue", 5)),
            step=1,
            key="generations_to_queue",
        )
    )
    evolution_mode = st.sidebar.selectbox(
        "Evolution length",
        options=["Finite", "Infinite"],
        index=0 if state.get("evolution_mode", "Finite") == "Finite" else 1,
        key="evolution_mode",
    )
    autosave_interval = int(
        st.sidebar.number_input(
            "Autosave every N generations",
            min_value=1,
            value=int(state.get("autosave_interval", 5)),
            step=1,
            key="autosave_interval",
        )
    )

    run_button = st.sidebar.button("Start evolution")
    stop_button = st.sidebar.button("Stop evolution")
    reset_button = st.sidebar.button("Reset evolution")
    save_button = st.sidebar.button("Save snapshot now")
    reload_button = st.sidebar.button("Reload autosave")

    if reset_button:
        state.pop("evolution_manager", None)
        state.pop("evolution_signature", None)
        state["pending_generations"] = 0
        state["run_infinite"] = False
        st.sidebar.info("Cleared evolution history.")

    if reload_button:
        state.pop("evolution_manager", None)
        state.pop("evolution_signature", None)
        state["autosave_checked"] = False

    manager = _ensure_manager(
        original=original,
        encoder=encoder,
        decoder=decoder,
        population_size=population_size,
        seed=seed,
        autosave_interval=autosave_interval,
    )

    if save_button:
        save_path = manager.save(autosave_dir)
        st.sidebar.success(f"Saved evolution session to {save_path}")

    if run_button:
        if evolution_mode == "Finite":
            state["pending_generations"] = generations_to_queue
            state["run_infinite"] = False
        else:
            state["run_infinite"] = True

    if stop_button:
        state["run_infinite"] = False
        state["pending_generations"] = 0

    generations_ran = 0
    if state.get("pending_generations", 0) > 0:
        manager.run_generation()
        state["pending_generations"] -= 1
        generations_ran = 1
    elif state.get("run_infinite", False):
        manager.run_generation()
        generations_ran = 1

    trigger_rerun = False
    if generations_ran:
        if len(manager.generations) % manager.autosave_interval == 0:
            save_path = manager.save(autosave_dir)
            st.sidebar.success(f"Autosaved evolution session to {save_path}")
        if state.get("pending_generations", 0) > 0 or state.get("run_infinite", False):
            trigger_rerun = True

    if manager.generations:
        st.header("Evolution progress")
        progress_rows = []
        for record in manager.generations:
            best = record.best_candidate
            progress_rows.append(
                {
                    "Generation": record.index,
                    "Best SSIM": best.metrics.ssim,
                    "Best PSNR": best.metrics.psnr,
                    "Best overlap": best.overlap_score,
                }
            )

        if progress_rows:
            progress_df = pd.DataFrame(progress_rows).set_index("Generation")
            st.subheader("Best-of-generation trend")
            st.line_chart(progress_df)

        gen_indices = [record.index for record in manager.generations]
        default_gen = gen_indices[-1]
        selected_generation = st.select_slider(
            "Select generation",
            options=gen_indices,
            value=default_gen,
            key="selected_generation",
            format_func=lambda idx: f"Generation {idx}",
        )
        generation = manager.generations[selected_generation]
        best_candidate = generation.best_candidate

        st.subheader("Best candidate metrics")
        best_cols = st.columns(3)
        best_cols[0].metric("Seed", str(best_candidate.seed))
        best_cols[1].metric("PSNR", f"{best_candidate.metrics.psnr:.2f} dB")
        best_cols[2].metric("SSIM", f"{best_candidate.metrics.ssim:.3f}")
        st.metric("Best overlap", f"{best_candidate.overlap_score:.1f}%")

        st.subheader("Generation gallery")
        cols_per_row = min(4, len(generation.candidates))
        for offset in range(0, len(generation.candidates), cols_per_row):
            row = st.columns(cols_per_row)
            for col, candidate in zip(row, generation.candidates[offset : offset + cols_per_row]):
                caption = (
                    f"Seed {candidate.seed}\nPSNR {candidate.metrics.psnr:.2f} dB\nSSIM {candidate.metrics.ssim:.3f}"
                )
                col.image(
                    to_uint8_image(_apply_color_template(candidate.reconstruction, color_template)),
                    caption=caption,
                    width="stretch",
                    clamp=True,
                )

        st.subheader("Candidate inspector")
        option_labels = [
            f"AI {idx + 1}: Seed {cand.seed} – SSIM {cand.metrics.ssim:.3f}"
            for idx, cand in enumerate(generation.candidates)
        ]
        candidate_indices = list(range(len(generation.candidates)))
        default_candidate = next(
            (i for i, cand in enumerate(generation.candidates) if cand.seed == best_candidate.seed),
            0,
        )
        selected_index = st.selectbox(
            "Choose a candidate to inspect",
            options=candidate_indices,
            index=default_candidate,
            format_func=lambda idx: option_labels[idx],
            key="candidate_selector",
        )
        inspected = generation.candidates[selected_index]
        overlap_map, overlap_score = multiplicative_overlap(manager.original, inspected.reconstruction)
        inspected_color = colorize_comparison(manager.original, inspected.reconstruction)

        inspect_cols = st.columns(4)
        inspect_cols[0].image(
            to_uint8_image(colored_original), caption="Evolution reference", width="stretch"
        )
        inspect_cols[1].image(
            to_uint8_image(_apply_color_template(inspected.reconstruction, color_template)),
            caption=f"Candidate seed {inspected.seed}",
            width="stretch",
        )
        inspect_cols[2].image(
            to_uint8_image(overlap_map),
            caption=f"Overlap map ({overlap_score:.1f}%)",
            width="stretch",
        )
        inspect_cols[3].image(
            to_uint8_image(inspected_color),
            caption="Colour overlap vs reference",
            width="stretch",
        )

        st.subheader("Generation summary")
        summary_rows = [
            {
                "AI": idx + 1,
                "Seed": cand.seed,
                "PSNR (dB)": f"{cand.metrics.psnr:.2f}",
                "SSIM": f"{cand.metrics.ssim:.3f}",
                "Overlap (%)": f"{cand.overlap_score:.1f}",
            }
            for idx, cand in enumerate(generation.candidates)
        ]
        st.table(summary_rows)

    if trigger_rerun:
        _trigger_rerun()

    st.markdown(
        """
        ### Next steps
        * Iterate on encoder/decoder designs and plug in learning-based components.
        * Compare overlap metrics across different seeds and hyperparameters.
        * Capture packets from real channels and replay them here for offline study.
        """
    )


def main() -> None:  # pragma: no cover - delegated to Streamlit runtime
    run()


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
