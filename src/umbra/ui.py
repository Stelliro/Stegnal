"""Streamlit-based visual explorer for Project Umbra."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
import zipfile

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
    state["encoder_sigma_base"] = float(manager.encoder.sigma)
    state["decoder_sigma_base"] = float(manager.decoder.denoise_sigma or 0.0)
    state["active_encoder_sigma"] = float(manager.encoder.sigma)
    state["active_decoder_sigma"] = float(manager.decoder.denoise_sigma or 0.0)
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


def _migrate_legacy_state(state: st.session_state) -> None:
    """Remove legacy widget-driven keys that conflict with automated controls."""

    legacy_seed = state.pop("sound_seed", None)
    if legacy_seed is not None and "active_sound_seed" not in state:
        try:
            state["active_sound_seed"] = int(legacy_seed)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    for noisy_key in (
        "encoder_sigma",
        "decoder_sigma",
        "encoder_noise",
        "decoder_noise",
    ):
        state.pop(noisy_key, None)

    bounds_low = state.pop("sound_sample_rate_min", None)
    bounds_high = state.pop("sound_sample_rate_max", None)
    if (
        bounds_low is not None
        and bounds_high is not None
        and "sound_sample_rate_bounds" not in state
    ):
        try:
            state["sound_sample_rate_bounds"] = (int(bounds_low), int(bounds_high))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    res_low = state.pop("sound_resolution_min", None)
    res_high = state.pop("sound_resolution_max", None)
    if res_low is not None and res_high is not None and "sound_resolution_bounds" not in state:
        try:
            state["sound_resolution_bounds"] = (int(res_low), int(res_high))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass


_SOUND_RESOLUTION_OPTIONS: tuple[int, ...] = (128, 192, 256)
_PERFORMANCE_HISTORY = 60
_RECENT_PERFORMANCE = 8


def _detect_hardware_backend() -> str:
    """Attempt to detect accelerated compute backends available to the app."""

    backends: list[str] = []

    try:  # pragma: no cover - optional dependency
        import cupy as cp  # type: ignore

        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except cp.cuda.runtime.CUDARuntimeError:  # pragma: no cover - defensive
            device_count = 0
        if device_count > 0:
            suffix = "s" if device_count > 1 else ""
            backends.append(f"CuPy CUDA ({device_count} device{suffix})")
    except Exception:  # pragma: no cover - optional dependency
        pass

    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            backends.append(f"PyTorch CUDA ({name})")
    except Exception:  # pragma: no cover - optional dependency
        pass

    if backends:
        return ", ".join(backends)
    return "CPU (NumPy)"


def _random_sample_rate(
    rng: np.random.Generator, bounds: tuple[int, int], difficulty: float
) -> int:
    """Sample a sound rate biased by ``difficulty`` towards the upper bound."""

    low, high = bounds
    if low >= high:
        return int(low)

    span = max(high - low, 1)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    exponent = max(0.3, 2.5 - 2.0 * difficulty)
    sample = rng.random() ** exponent
    value = low + sample * span
    value = int(1_000 * round(value / 1_000))
    return int(np.clip(value, low, high))


def _random_resolution(
    rng: np.random.Generator, bounds: tuple[int, int], difficulty: float
) -> int:
    """Choose a resolution that unlocks larger sizes as ``difficulty`` increases."""

    minimum, maximum = bounds
    available = [
        res for res in _SOUND_RESOLUTION_OPTIONS if minimum <= res <= maximum
    ]
    if not available:
        available = [res for res in _SOUND_RESOLUTION_OPTIONS if res >= minimum]
    if not available:
        available = list(_SOUND_RESOLUTION_OPTIONS)

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    unlocked = max(1, min(len(available), int(np.floor(difficulty * len(available))) + 1))
    return int(rng.choice(available[:unlocked]))


def _randomize_sound_parameters(
    rng: np.random.Generator,
    sample_bounds: tuple[int, int],
    resolution_bounds: tuple[int, int],
    difficulty: float,
) -> tuple[int, int]:
    """Randomly choose sound synthesis parameters within provided bounds."""

    sample_rate = _random_sample_rate(rng, sample_bounds, difficulty)
    resolution = _random_resolution(rng, resolution_bounds, difficulty)
    return sample_rate, resolution


def _compute_adaptive_noise(
    base_encoder_sigma: float, base_decoder_sigma: float, difficulty: float
) -> tuple[float, float]:
    """Scale encoder/decoder sigmas based on the adaptive difficulty."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    encoder_sigma = base_encoder_sigma * (1.0 + 0.75 * difficulty)
    if base_decoder_sigma <= 0:
        decoder_sigma = 0.0
    else:
        decoder_sigma = max(base_decoder_sigma * (1.0 - 0.5 * difficulty), 0.05)
    return float(encoder_sigma), float(decoder_sigma)


def _adaptive_sample_bounds(
    difficulty: float,
    previous: tuple[int, int] | None,
    improvement: float = 0.0,
    volatility: float = 0.0,
) -> tuple[int, int]:
    """Derive a difficulty-weighted sample-rate window with gentle inertia."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))

    base_low = float(np.interp(difficulty, [0.0, 1.0], [12_000, 8_000]))
    base_high = float(np.interp(difficulty, [0.0, 1.0], [24_000, 96_000]))
    spread = float(np.interp(difficulty, [0.0, 1.0], [3_000, 26_000]))

    spread *= float(np.clip(1.0 + 0.8 * improvement - 0.6 * volatility, 0.5, 1.8))

    rng = np.random.default_rng()
    jitter_low = float((rng.random() - 0.5) * spread)
    jitter_high = float((rng.random() - 0.5) * spread)

    low = int(np.clip(base_low + jitter_low, 8_000, 96_000))
    high = int(np.clip(base_high + jitter_high, low + 1_000, 96_000))

    if previous is not None:
        prev_low, prev_high = int(previous[0]), int(previous[1])
        blend = 0.6
        low = int(np.clip(blend * prev_low + (1.0 - blend) * low, 8_000, 96_000))
        high = int(
            np.clip(blend * prev_high + (1.0 - blend) * high, low + 1_000, 96_000)
        )

    return low, high


def _adaptive_resolution_bounds(
    difficulty: float,
    previous: tuple[int, int] | None,
    improvement: float = 0.0,
    volatility: float = 0.0,
) -> tuple[int, int]:
    """Unlock larger image sizes as difficulty progresses."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))

    options = sorted(_SOUND_RESOLUTION_OPTIONS)
    unlocked = max(1, min(len(options), 1 + int(np.floor(difficulty * len(options)))))

    if improvement > 0.05 and unlocked < len(options):
        unlocked += 1

    rng = np.random.default_rng()
    if unlocked < len(options) and rng.random() > 0.85:
        unlocked = min(len(options), unlocked + 1)
    if volatility > 0.08 and unlocked > 1:
        unlocked -= 1

    lower = options[0]
    upper = options[unlocked - 1]

    if previous is not None:
        prev_low, prev_high = int(previous[0]), int(previous[1])
        lower = int(np.clip(0.65 * prev_low + 0.35 * lower, options[0], options[-1]))
        upper = int(np.clip(0.65 * prev_high + 0.35 * upper, lower, options[-1]))

    return lower, upper


def _update_noise_bases(
    state: st.session_state,
    difficulty: float,
    improvement: float,
    volatility: float,
) -> None:
    """Gently steer encoder/decoder noise levels in response to difficulty."""

    rng = np.random.default_rng()
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))

    prev_encoder = float(state.get("encoder_sigma_base", 0.2))
    exploration_gain = 1.0 + 0.7 * improvement
    stability_pull = 1.0 - min(0.5, 1.4 * volatility)
    target_encoder = float(
        np.interp(difficulty, [0.0, 1.0], [0.16, 0.65]) * exploration_gain * stability_pull
    )
    encoder_jitter = float(
        rng.normal(0.0, 0.02 + 0.05 * difficulty + 0.03 * improvement)
    )
    encoder_sigma = float(np.clip(target_encoder + encoder_jitter, 0.05, 0.9))
    state["encoder_sigma_base"] = float(0.6 * prev_encoder + 0.4 * encoder_sigma)

    prev_decoder = float(state.get("decoder_sigma_base", 1.0))
    denoise_target = float(np.interp(difficulty, [0.0, 1.0], [1.4, 0.25]))
    denoise_target *= float(np.clip(1.0 - 0.6 * improvement, 0.35, 1.0))
    denoise_target *= float(np.clip(1.0 + 0.8 * volatility, 0.5, 1.6))
    decoder_jitter = float(
        rng.normal(0.0, 0.05 + 0.04 * (1.0 - difficulty) + 0.03 * volatility)
    )
    decoder_sigma = float(np.clip(denoise_target + decoder_jitter, 0.05, 2.5))
    state["decoder_sigma_base"] = float(0.6 * prev_decoder + 0.4 * decoder_sigma)


def _record_performance_history(
    state: st.session_state,
    ai_overlap: float,
    ai_ssim: float,
    ai_psnr: float,
    sound_overlap: float,
) -> list[Dict[str, float]]:
    """Track recent reconstruction metrics for adaptive scheduling."""

    history: list[Dict[str, float]] = list(state.get("performance_history", []))
    history.append(
        {
            "ai_overlap": float(ai_overlap),
            "ai_ssim": float(ai_ssim),
            "ai_psnr": float(ai_psnr),
            "sound_overlap": float(sound_overlap),
        }
    )
    if len(history) > _PERFORMANCE_HISTORY:
        history = history[-_PERFORMANCE_HISTORY:]
    state["performance_history"] = history
    return history


def _derive_difficulty_metrics(
    history: list[Dict[str, float]]
) -> tuple[float, float, float]:
    """Compute difficulty progress, improvement, and volatility signals."""

    if not history:
        return 0.0, 0.0, 0.0

    overlaps = np.asarray([entry["ai_overlap"] for entry in history], dtype=np.float32)
    recent_window = int(min(len(history), _RECENT_PERFORMANCE))
    recent = overlaps[-recent_window:]
    recent_mean = float(np.mean(recent)) / 100.0
    best = float(np.max(overlaps)) / 100.0
    long_term = float(np.mean(overlaps[:-recent_window])) / 100.0 if len(history) > recent_window else recent_mean
    improvement = float(np.clip(recent_mean - long_term, 0.0, 1.0))
    volatility = float(np.std(recent) / 100.0)

    coverage = float(min(len(history) / _PERFORMANCE_HISTORY, 1.0))
    difficulty_target = float(
        np.clip(0.45 * best + 0.35 * recent_mean + 0.2 * coverage + 0.4 * improvement, 0.0, 1.0)
    )
    return difficulty_target, improvement, float(np.clip(volatility, 0.0, 1.0))


def _refresh_sound_scene(
    state: st.session_state,
    difficulty: float,
    target_dwell: int,
    *,
    record_event: bool = True,
    improvement: float = 0.0,
    volatility: float = 0.0,
) -> tuple[int, int, int]:
    """Randomise the sound target and associated hyper-parameters."""

    sample_bounds = _adaptive_sample_bounds(
        difficulty,
        state.get("sound_sample_rate_bounds"),
        improvement,
        volatility,
    )
    resolution_bounds = _adaptive_resolution_bounds(
        difficulty,
        state.get("sound_resolution_bounds"),
        improvement,
        volatility,
    )

    rng = np.random.default_rng()
    sample_rate, resolution = _randomize_sound_parameters(
        rng,
        sample_bounds,
        resolution_bounds,
        difficulty,
    )

    new_sound_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    new_shared_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    state["active_sound_seed"] = new_sound_seed
    state["current_sound_sample_rate"] = int(sample_rate)
    state["current_sound_resolution"] = int(resolution)
    state["sound_sample_rate_bounds"] = (int(sample_bounds[0]), int(sample_bounds[1]))
    state["sound_resolution_bounds"] = (
        int(resolution_bounds[0]),
        int(resolution_bounds[1]),
    )
    state["sound_generations_left"] = int(target_dwell)
    state["shared_seed"] = new_shared_seed

    if record_event:
        state["sound_reseed_count"] = int(state.get("sound_reseed_count", 0) + 1)
    else:
        state.setdefault("sound_reseed_count", 0)

    _update_noise_bases(state, difficulty, improvement, volatility)
    return new_sound_seed, int(sample_rate), int(resolution)


def _build_export_bundle(payload: Dict[str, Any], progress_rows: list[Dict[str, Any]]) -> BytesIO:
    """Create a zipped export containing session metrics and progress curves."""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session_summary.json", json.dumps(payload, indent=2))
        if progress_rows:
            export_df = pd.DataFrame(progress_rows)
            archive.writestr("generation_progress.csv", export_df.to_csv(index=False))
    buffer.seek(0)
    return buffer


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
    _migrate_legacy_state(state)
    state.setdefault("pending_generations", 0)
    state.setdefault("run_infinite", False)
    state.setdefault("autosave_dir", str(DEFAULT_AUTOSAVE_DIR))
    state.setdefault("last_autosave_dir", str(Path(state["autosave_dir"]).expanduser()))
    state.setdefault("autosave_checked", False)
    state.setdefault("population_size", 4)
    state.setdefault("autosave_interval", 5)
    state.setdefault("generations_to_queue", 5)
    state.setdefault("evolution_mode", "Finite")
    if "shared_seed" not in state:
        state["shared_seed"] = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
    state.setdefault("encoder_sigma_base", 0.2)
    state.setdefault("decoder_sigma_base", 1.0)
    legacy_sound_seed = state.pop("sound_seed", None)
    if "active_sound_seed" not in state:
        if legacy_sound_seed is not None:
            state["active_sound_seed"] = int(legacy_sound_seed)
        else:
            state["active_sound_seed"] = int(
                np.random.default_rng().integers(0, np.iinfo(np.int32).max)
            )
    state.setdefault("sound_target_dwell", 10)
    state.setdefault("last_sound_target_dwell", int(state["sound_target_dwell"]))
    state.setdefault("sound_generations_left", int(state["sound_target_dwell"]))
    state.setdefault("sound_reseed_count", 0)
    state.setdefault("difficulty_progress", 0.0)
    state.setdefault("max_overlap_seen", 0.0)
    state.setdefault("performance_history", [])
    state.setdefault("difficulty_improvement", 0.0)
    state.setdefault("difficulty_volatility", 0.0)
    state.setdefault("hardware_backend", _detect_hardware_backend())

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

    difficulty_progress = float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0))

    st.sidebar.subheader("Sound cadence")
    target_dwell = int(
        st.sidebar.number_input(
            "Generations per sound target",
            min_value=1,
            max_value=500,
            value=int(state.get("sound_target_dwell", 10)),
            step=1,
            key="sound_target_dwell",
            help=(
                "Number of evolution steps to spend matching the current sound-derived "
                "image before refreshing it with a new randomised scene."
            ),
        )
    )

    if state.get("last_sound_target_dwell") != target_dwell:
        state["last_sound_target_dwell"] = target_dwell
        state["sound_generations_left"] = target_dwell

    if "sound_sample_rate_bounds" not in state or "sound_resolution_bounds" not in state:
        _refresh_sound_scene(
            state,
            difficulty_progress,
            target_dwell,
            record_event=False,
            improvement=float(state.get("difficulty_improvement", 0.0)),
            volatility=float(state.get("difficulty_volatility", 0.0)),
        )

    if state.get("sound_generations_left", target_dwell) > target_dwell:
        state["sound_generations_left"] = target_dwell

    manual_refresh = st.sidebar.button(
        "Refresh sound scene",
        key="refresh_sound_scene",
        help="Force an immediate reseed using the current difficulty profile.",
    )
    if manual_refresh:
        new_seed, new_rate, new_resolution = _refresh_sound_scene(
            state,
            difficulty_progress,
            target_dwell,
            improvement=float(state.get("difficulty_improvement", 0.0)),
            volatility=float(state.get("difficulty_volatility", 0.0)),
        )
        st.sidebar.info(
            "Forced refresh triggered new scene "
            f"(seed {new_seed}, {new_rate:,} Hz, {new_resolution}×{new_resolution} px)."
        )

    seed = int(state.get("shared_seed", 0))
    sound_seed = int(state.get("active_sound_seed", seed))
    sample_rate_range = tuple(
        int(v) for v in state.get("sound_sample_rate_bounds", (24_000, 48_000))
    )
    resolution_range = tuple(
        int(v)
        for v in state.get(
            "sound_resolution_bounds",
            (_SOUND_RESOLUTION_OPTIONS[0], _SOUND_RESOLUTION_OPTIONS[0]),
        )
    )

    base_encoder_sigma = float(state.get("encoder_sigma_base", 0.2))
    base_decoder_sigma = float(state.get("decoder_sigma_base", 1.0))
    encoder_sigma, denoise_sigma = _compute_adaptive_noise(
        base_encoder_sigma, base_decoder_sigma, difficulty_progress
    )
    state["active_encoder_sigma"] = encoder_sigma
    state["active_decoder_sigma"] = denoise_sigma

    st.sidebar.subheader("Adaptive configuration")
    st.sidebar.metric("Hardware backend", state.get("hardware_backend", "CPU (NumPy)"))
    st.sidebar.metric("Shared seed", str(seed))
    st.sidebar.metric("Sound seed", str(sound_seed))
    st.sidebar.metric(
        "Sound sample window", f"{sample_rate_range[0]:,}–{sample_rate_range[1]:,} Hz"
    )
    st.sidebar.metric(
        "Image resolution window",
        f"{resolution_range[0]}–{resolution_range[1]} px",
    )

    noise_cols = st.sidebar.columns(2)
    noise_cols[0].metric("Active encoder σ", f"{encoder_sigma:.3f}")
    noise_cols[1].metric("Active denoise σ", f"{denoise_sigma:.3f}")
    st.sidebar.caption(
        "Adaptive noise scales increase encoder randomness while tempering decoder blur "
        "as the system improves."
    )

    state["sound_generations_left"] = int(
        max(0, min(state.get("sound_generations_left", target_dwell), target_dwell))
    )
    remaining_before = int(state.get("sound_generations_left", target_dwell))

    current_sample_rate = int(state.get("current_sound_sample_rate", sample_rate_range[0]))
    current_resolution = int(state.get("current_sound_resolution", resolution_range[0]))

    rng_params = np.random.default_rng()
    if not (sample_rate_range[0] <= current_sample_rate <= sample_rate_range[1]):
        current_sample_rate = _random_sample_rate(
            rng_params, sample_rate_range, difficulty_progress
        )
        state["current_sound_sample_rate"] = current_sample_rate
    if not (resolution_range[0] <= current_resolution <= resolution_range[1]):
        current_resolution = _random_resolution(
            rng_params, resolution_range, difficulty_progress
        )
        state["current_sound_resolution"] = current_resolution

    st.sidebar.metric("Active sample rate", f"{current_sample_rate:,} Hz")
    st.sidebar.metric(
        "Active image resolution",
        f"{current_resolution}×{current_resolution} px",
    )

    original_color, original, sound_clip, shape_specs = generate_sound_art(
        seed=sound_seed,
        image_size=(current_resolution, current_resolution),
        sample_rate=current_sample_rate,
    )
    source_label = f"Sound seed {sound_seed}"
    color_template = _build_color_template(original_color, original)

    encoder = NoiseStreamEncoder(sigma=encoder_sigma)
    decoder = NoiseStreamDecoder(
        denoise_sigma=denoise_sigma if denoise_sigma > 0 else None
    )

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

    _, ai_overlap_score = multiplicative_overlap(original, reconstructed)
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
    ai_metrics_cols[2].metric("AI overlap", f"{ai_overlap_score:.1f}%")

    sound_metrics_cols = st.columns(3)
    sound_metrics_cols[0].metric("Sound colour PSNR", f"{sound_metrics.psnr:.2f} dB")
    sound_metrics_cols[1].metric("Sound colour SSIM", f"{sound_metrics.ssim:.3f}")
    sound_metrics_cols[2].metric("Sound overlap", f"{sound_overlap_score:.1f}%")

    st.metric("AI ↔ Sound colour SSIM", f"{ai_sound_alignment.ssim:.3f}")

    overlap_pct = float(ai_overlap_score)
    state["max_overlap_seen"] = max(state.get("max_overlap_seen", 0.0), overlap_pct)
    history = _record_performance_history(
        state,
        overlap_pct,
        float(metrics.ssim),
        float(metrics.psnr),
        float(sound_overlap_score),
    )
    target_progress, improvement_signal, volatility_signal = _derive_difficulty_metrics(history)
    reseed_progress = min(1.0, state.get("sound_reseed_count", 0) / 10.0)
    blended_target = max(target_progress, reseed_progress)
    previous_progress = float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0))
    updated_progress = float(
        np.clip(0.6 * previous_progress + 0.4 * blended_target, 0.0, 1.0)
    )
    state["difficulty_progress"] = updated_progress
    state["difficulty_improvement"] = float(improvement_signal)
    state["difficulty_volatility"] = float(volatility_signal)
    difficulty_progress = updated_progress

    st.sidebar.metric("Adaptive difficulty", f"{difficulty_progress * 100:.0f}%")
    momentum = float(state.get("difficulty_improvement", 0.0)) * 100.0
    variability = float(state.get("difficulty_volatility", 0.0)) * 100.0
    trend_cols = st.sidebar.columns(2)
    trend_cols[0].metric("Difficulty momentum", f"{momentum:.1f} pts")
    trend_cols[1].metric("Difficulty range", f"{variability:.1f} pts")

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
                "Target size (px)": f"{spec.size}",
                "Target rotation (°)": f"{spec.rotation:.1f}",
                "Target centre (y, x)": f"({spec.center[0]}, {spec.center[1]})",
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
        remaining_after = max(remaining_before - 1, 0)
        state["sound_generations_left"] = remaining_after
        reseeded = False

        if remaining_after == 0:
            new_seed, next_rate, next_resolution = _refresh_sound_scene(
                state,
                float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0)),
                target_dwell,
                improvement=float(state.get("difficulty_improvement", 0.0)),
                volatility=float(state.get("difficulty_volatility", 0.0)),
            )
            seed = int(state.get("shared_seed", seed))
            sound_seed = int(state.get("active_sound_seed", new_seed))
            current_sample_rate = next_rate
            current_resolution = next_resolution
            remaining_after = target_dwell
            reseeded = True
            st.sidebar.info(
                "Auto-randomised sound scene after completing the dwell window "
                f"(seed {sound_seed}, {next_rate:,} Hz, {next_resolution}×{next_resolution} px)."
            )

        if len(manager.generations) % manager.autosave_interval == 0:
            save_path = manager.save(autosave_dir)
            st.sidebar.success(f"Autosaved evolution session to {save_path}")
        if reseeded or state.get("pending_generations", 0) > 0 or state.get("run_infinite", False):
            trigger_rerun = True
    else:
        state["sound_generations_left"] = remaining_before

    st.sidebar.metric(
        "Generations remaining on sound target",
        int(state.get("sound_generations_left", target_dwell)),
    )

    generation_progress_rows: list[Dict[str, Any]] = []
    best_candidate_summary: Dict[str, Any] | None = None

    if manager.generations:
        st.header("Evolution progress")
        generation_progress_rows = [
            {
                "Generation": record.index,
                "Best SSIM": record.best_candidate.metrics.ssim,
                "Best PSNR": record.best_candidate.metrics.psnr,
                "Best overlap": record.best_candidate.overlap_score,
            }
            for record in manager.generations
        ]

        if generation_progress_rows:
            progress_df = pd.DataFrame(generation_progress_rows).set_index("Generation")
            st.subheader("Best-of-generation trend")
            st.line_chart(progress_df, width="stretch")

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

        best_candidate_summary = {
            "generation": generation.index,
            "seed": best_candidate.seed,
            "psnr": best_candidate.metrics.psnr,
            "ssim": best_candidate.metrics.ssim,
            "overlap": best_candidate.overlap_score,
        }

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
        inspect_overlap_map, inspect_overlap_score = multiplicative_overlap(
            manager.original, inspected.reconstruction
        )
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
            to_uint8_image(inspect_overlap_map),
            caption=f"Overlap map ({inspect_overlap_score:.1f}%)",
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

    export_payload = {
        "hardware_backend": state.get("hardware_backend", "CPU (NumPy)"),
        "difficulty": {
            "current": float(state.get("difficulty_progress", 0.0)),
            "max_overlap": float(state.get("max_overlap_seen", 0.0)),
            "sound_reseed_count": int(state.get("sound_reseed_count", 0)),
            "dwell_generations": int(target_dwell),
            "momentum": float(state.get("difficulty_improvement", 0.0)),
            "volatility": float(state.get("difficulty_volatility", 0.0)),
        },
        "seeds": {
            "shared": int(seed),
            "sound": int(sound_seed),
        },
        "noise": {
            "base_encoder_sigma": float(base_encoder_sigma),
            "base_decoder_sigma": float(base_decoder_sigma),
            "active_encoder_sigma": float(encoder_sigma),
            "active_decoder_sigma": float(denoise_sigma),
        },
        "sound": {
            "current_sample_rate": int(current_sample_rate),
            "current_resolution": int(current_resolution),
            "sample_rate_window": [int(sample_rate_range[0]), int(sample_rate_range[1])],
            "resolution_window": [int(resolution_range[0]), int(resolution_range[1])],
            "band_volumes": sound_clip.band_volumes,
            "generator_seed": int(sound_clip.seed),
            "generator_sample_rate": int(sound_clip.sample_rate),
        },
        "metrics": {
            "ai_vs_reference": metrics.as_dict(),
            "sound_vs_reference": sound_metrics.as_dict(),
            "ai_vs_sound": ai_sound_alignment.as_dict(),
            "overlap": {
                "ai_vs_reference": float(ai_overlap_score),
                "sound_vs_reference": float(sound_overlap_score),
            },
        },
        "performance_history": list(state.get("performance_history", [])),
        "manager": {
            "population_size": int(manager.population_size),
            "autosave_interval": int(manager.autosave_interval),
            "generation_count": len(manager.generations),
            "best_candidate": best_candidate_summary,
        },
    }

    export_bundle = _build_export_bundle(export_payload, generation_progress_rows)
    st.sidebar.download_button(
        "Download session snapshot",
        data=export_bundle.getvalue(),
        file_name="umbra_session_snapshot.zip",
        mime="application/zip",
    )

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
