"""Streamlit-based visual explorer for Project Umbra."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

from umbra.adversarial import AdversarialManager, apply_generator
from umbra.chart_export import export_chart_png
from umbra.codec import (
    decode_waveform_to_image,
    encode_image_to_wav_bytes,
    encode_image_to_waveform,
)
from umbra.decoding import NoiseStreamDecoder
from umbra.demo_packager import build_demo_package
from umbra.encoding import NoisePacket, NoiseStreamEncoder
from umbra.evolution import (
    EvolutionManager,
    GenerationRecord,
    HyperPerformanceProfile,
    compute_image_signature,
    normalize_difficulty,
)
from umbra.logging_utils import collect_provenance, configure_logging
from umbra.metrics import ReconstructionMetrics, compute_metrics
from umbra.neural import NeuralRewardModel
from umbra.predictor import predict_image_from_waveform
from umbra.progress import prepare_metrics_chart, prepare_trend_chart, sanitize_progress_rows
from umbra.reconstruction import (
    ReconstructionResult,
    run_reconstruction_cycle,
    waveform_to_wav_bytes,
)
from umbra.run_helpers import ensure_run_paths
from umbra.sound import (
    ShapeGuess,
    SyntheticSound,
    generate_sound_art,
    guess_shapes,
    load_waveform_from_wav,
)
from umbra.visualization import (
    colorize_comparison,
    multiplicative_overlap,
    normalize_for_display,
    to_uint8_image,
)

logger = logging.getLogger(__name__)


DEFAULT_AUTOSAVE_DIR = Path.home() / ".umbra_autosave"

_PAGE_CONFIGURED = False

_DARK_STYLE = """
<style>
:root {
    color-scheme: dark;
}
.stApp, .block-container {
    background: #0d1017 !important;
    color: #f5f6fa !important;
}
.stMarkdown, .stMetric, .stSelectbox, .stButton, .stSlider, .stTabs {
    color: #f5f6fa !important;
}
.stButton>button, .stSlider>div>div>div>button {
    background: #1f2937 !important;
    color: #f5f6fa !important;
    border: 1px solid #374151 !important;
}
.stSelectbox>div>div>button, .stSelectbox>div>div>div {
    background: #111827 !important;
    color: #f5f6fa !important;
}
.stExpander, .stTextInput, .stNumberInput, .stMultiSelect, .stFileUploader {
    background: transparent !important;
    color: #f5f6fa !important;
}
.stTable, .stDataFrame {
    color: #f5f6fa !important;
}
</style>
"""


download_cache: dict[str, dict[str, Any]] = {}


_PINTEREST_DEFAULT_FEEDS: tuple[str, ...] = (
    "https://au.pinterest.com/search/pins/?q=space&rs=typed",
    "https://www.pinterest.com/pinterest/official-news.rss",
    "https://www.pinterest.com/pinterest/engagement.rss",
    "https://www.pinterest.com/pinterest/creative.rss",
)

_PINTEREST_IMAGE_PATTERN = re.compile(
    r"(?:src|data-pin-media)=\"(https://i\.pinimg\.com[^\"]+)\"",
    flags=re.IGNORECASE,
)
_PINTEREST_IMAGE_TAG_PATTERN = re.compile(
    r"<img[^>]+(?:data-pin-media|data-src|src)=['\"](?P<url>https://i\.pinimg\.com[^'\"]+)['\"][^>]*"
    r"(?:alt=['\"](?P<title>[^'\"]+)['\"])?",
    flags=re.IGNORECASE | re.DOTALL,
)
_PINTEREST_JSON_SCRIPT_PATTERN = re.compile(
    r"<script[^>]*type=['\"]application/json['\"][^>]*>(.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)

_PINTEREST_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


class RunStore:
    """Persist run-specific artifacts to disk."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self) -> str:
        """Create a new run folder and return its identifier."""

        base = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        counter = 0
        while True:
            run_id = base if counter == 0 else f"{base}-{counter:02d}"
            run_dir = self.root / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
                return run_id
            except FileExistsError:
                counter += 1

    def artifacts_directory(self, run_id: str) -> Path:
        path = self.root / run_id / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_bytes(self, run_id: str, filename: str, data: bytes) -> Path:
        path = self.artifacts_directory(run_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save_image(self, run_id: str, filename: str, image: np.ndarray) -> Path:
        png_bytes = _image_to_png_bytes(image)
        return self.save_bytes(run_id, filename, png_bytes)


def _sanitize_filename(label: str, suffix: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label.strip().lower()
    )
    safe = safe.strip("._") or "artifact"
    return f"{safe}{suffix}"


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    with BytesIO() as buffer:
        Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB").save(
            buffer, format="PNG"
        )
        return buffer.getvalue()


def _resize_image(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    rows, cols = resolution
    if rows <= 0 or cols <= 0:
        raise ValueError("resolution must contain positive dimensions")
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.shape[0] == rows and array.shape[1] == cols:
        return array
    pil_image = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    resized = pil_image.resize((cols, rows), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _load_uploaded_image(file) -> np.ndarray:
    try:
        with Image.open(file) as uploaded:
            array = np.asarray(uploaded.convert("RGB"), dtype=np.float32) / 255.0
        return np.clip(array, 0.0, 1.0)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to read image: {exc}") from exc


def _download_bytes(url: str, *, timeout: float = 10.0) -> bytes:
    """Download ``url`` returning the raw bytes."""

    request = urllib.request.Request(url, headers={"User-Agent": _PINTEREST_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network safeguard
        raise RuntimeError(f"Unable to download {url}: {exc}") from exc


def _normalize_pinterest_source(source: str | None) -> str:
    """Return a usable Pinterest RSS feed URL from ``source``."""

    if source:
        trimmed = source.strip()
        if trimmed:
            parsed = urllib.parse.urlparse(trimmed)
            if parsed.scheme and parsed.netloc:
                return trimmed
            path = trimmed.strip("/ ")
            if path:
                return f"https://www.pinterest.com/{path}.rss"
    return _PINTEREST_DEFAULT_FEEDS[0]


def _parse_pinterest_feed(feed_data: bytes | str) -> list[tuple[str, str]]:
    """Extract ``(image_url, title)`` pairs from Pinterest RSS or HTML content."""

    if isinstance(feed_data, bytes):
        text = feed_data.decode("utf-8", errors="replace")
    else:
        text = feed_data

    candidates = _parse_pinterest_feed_xml(text)
    if not candidates:
        candidates = _parse_pinterest_html(text)

    if not candidates:
        raise ValueError("Unable to parse Pinterest feed XML")

    deduped: OrderedDict[str, str] = OrderedDict()
    for url, label in candidates:
        normalized_url = _normalize_pinterest_url(url)
        if normalized_url not in deduped:
            deduped[normalized_url] = label
    return list(deduped.items())


def _parse_pinterest_feed_xml(text: str) -> list[tuple[str, str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else root.findall(".//item")

    candidates: list[tuple[str, str]] = []
    for item in items:
        title = (item.findtext("title") or "Pinterest inspiration").strip()
        for child in item:
            tag = child.tag.split("}")[-1]
            if tag in {"content", "thumbnail"}:
                url = child.attrib.get("url")
                if url and url.startswith("http"):
                    candidates.append((url, title))
        description = item.findtext("description") or ""
        for match in _PINTEREST_IMAGE_PATTERN.findall(description):
            candidates.append((match, title))
    return candidates


def _parse_pinterest_html(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    for snippet in _PINTEREST_JSON_SCRIPT_PATTERN.findall(text):
        snippet = snippet.strip()
        if not snippet:
            continue
        try:
            payload = json.loads(html.unescape(snippet))
        except json.JSONDecodeError:
            continue
        for node in _collect_pinterest_nodes(payload):
            if not isinstance(node, Mapping):
                continue
            images = node.get("images")
            if isinstance(images, Mapping):
                orig = images.get("orig")
                if isinstance(orig, Mapping):
                    url_value = orig.get("url")
                    if isinstance(url_value, str):
                        title_value = node.get("title") or node.get("name") or node.get("description")
                        label = str(title_value).strip() if title_value else "Pinterest inspiration"
                        candidates.append((_normalize_pinterest_url(url_value), label))

    for match in _PINTEREST_IMAGE_TAG_PATTERN.finditer(text):
        url = _normalize_pinterest_url(match.group("url"))
        title = match.group("title")
        label = html.unescape(title).strip() if title else "Pinterest inspiration"
        candidates.append((url, label))

    # Fallback: generic pinimg URLs without metadata
    if not candidates:
        for url in re.findall(r"https://i\.pinimg\.com/[^'\"\s>]+", text):
            candidates.append((_normalize_pinterest_url(url), "Pinterest inspiration"))

    return candidates


def _normalize_pinterest_url(url: str) -> str:
    cleaned = html.unescape(url)
    return cleaned.replace("\\u002F", "/").replace("\\/", "/")


def _collect_pinterest_nodes(payload: object) -> list[Mapping[str, object]]:
    stack = [payload]
    collected: list[Mapping[str, object]] = []
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            collected.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return collected


def _fetch_random_pinterest_image(
    source: str | None,
    *,
    timeout: float = 10.0,
    download: Callable[[str, float], bytes] | None = None,
) -> tuple[np.ndarray, str]:
    """Download and decode a random image from a Pinterest feed."""

    downloader = download or (lambda url, to: _download_bytes(url, timeout=to))
    feed_url = _normalize_pinterest_source(source)
    try:
        feed_bytes = downloader(feed_url, timeout)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to retrieve Pinterest feed: {exc}") from exc

    try:
        candidates = _parse_pinterest_feed(feed_bytes)
    except ValueError as exc:
        raise RuntimeError(f"Pinterest feed at {feed_url} is invalid: {exc}") from exc

    image_url, title = random.choice(candidates)
    try:
        image_bytes = downloader(image_url, timeout)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to download Pinterest image: {exc}") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as downloaded:
            array = np.asarray(downloaded.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to decode Pinterest image from {image_url}") from exc

    label = (title or "Pinterest inspiration").strip() or "Pinterest inspiration"
    return np.clip(array, 0.0, 1.0), f"{label} (Pinterest)"


def _auto_refresh_pinterest_reference(
    state: MutableMapping[str, Any],
    *,
    timeout: float = 10.0,
) -> bool:
    """Refresh the active Pinterest inspiration image if Pinterest is selected."""

    active_source = state.get("quick_start_media_source")
    reference_source = state.get("quick_start_reference_source")
    if active_source != "pinterest" and reference_source != "pinterest":
        return False

    board_input = state.get("quick_start_pinterest_source") or None
    try:
        image, label = _fetch_random_pinterest_image(board_input, timeout=timeout)
    except RuntimeError as exc:
        logger.warning("Pinterest auto-refresh failed: %s", exc)
        return False

    state["quick_start_reference_image"] = image
    state["quick_start_reference_label"] = label
    state["quick_start_reference_source"] = "pinterest"
    state["quick_start_step1_timestamp"] = time.time()
    state["_last_pinterest_refresh"] = time.time()
    return True


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image in ``[0, 1]`` to a single-channel grayscale array."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image shaped [H, W, 3]")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grayscale = np.tensordot(image.astype(np.float32), weights, axes=([-1], [0]))
    return np.clip(grayscale, 0.0, 1.0).astype(np.float32)


def _ensure_page_config() -> None:
    """Apply a compact dark configuration once for the Streamlit page."""

    global _PAGE_CONFIGURED
    if _PAGE_CONFIGURED:
        return
    st.set_page_config(
        page_title="Project Umbra",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_DARK_STYLE, unsafe_allow_html=True)
    _PAGE_CONFIGURED = True


def _render_metric_visual(
    label: str,
    value: float,
    *,
    value_display: str,
    scale_min: float,
    scale_max: float,
    good_range: tuple[float, float],
    caption: str,
    tooltip: str,
) -> None:
    """Render a beginner-friendly visual summary for a reconstruction metric."""

    label_html = (
        f'<div class="metric-label" title="{html.escape(tooltip)}">{html.escape(label)}</div>'
    )
    st.markdown(label_html, unsafe_allow_html=True)

    if not math.isfinite(value) or not math.isfinite(scale_min) or not math.isfinite(scale_max):
        st.markdown("**⚠️ Not available**")
        st.caption(caption)
        return

    span = scale_max - scale_min
    if span <= 0:
        progress = 0.0
    else:
        progress = float(np.clip((value - scale_min) / span, 0.0, 1.0))

    if value >= good_range[1]:
        emoji, verdict = "🚀", "Excellent"
    elif value >= good_range[0]:
        emoji, verdict = "🙂", "Solid"
    else:
        emoji, verdict = "🛠️", "Needs work"

    st.progress(progress)
    st.markdown(f"**{emoji} {verdict}: {value_display}**")
    st.caption(caption)


def _quantize_slider_value(
    value: float,
    *,
    min_value: float,
    max_value: float,
    step: float,
) -> float:
    """Clip ``value`` to the slider domain and align it with the provided step."""

    if step <= 0:
        raise ValueError("Slider step must be positive")

    clipped = float(np.clip(value, min_value, max_value))
    steps = round((clipped - min_value) / step)
    quantized = min_value + steps * step
    quantized = float(np.clip(quantized, min_value, max_value))
    # Guard against floating point drift so Streamlit accepts the default value.
    decimals = max(0, -int(math.floor(math.log10(step))) if step < 1 else 0)
    return round(quantized, decimals)


def _metric_badge_html(label: str, psnr: float, ssim: float) -> str:
    """Return a styled HTML badge summarising PSNR/SSIM metrics."""

    safe_label = html.escape(label)
    return (
        "<div style=\"background-color: var(--secondary-background-color,#f0f2f6);"
        "padding:0.5rem 0.75rem;border-radius:0.5rem;font-size:0.85rem;"
        "line-height:1.4;\">"
        f"<strong>{safe_label}</strong><br>"
        f"PSNR {psnr:.2f} dB · SSIM {ssim:.3f}"
        "</div>"
    )


def _blend_range(bounds: tuple[int, int], weight: float) -> int:
    """Interpolate ``bounds`` using ``weight`` in ``[0, 1]`` and return an int."""

    low, high = bounds
    return int(round(np.interp(np.clip(weight, 0.0, 1.0), [0.0, 1.0], [low, high])))


def _auto_run_parameters(
    *,
    difficulty_progress: float,
    profile: dict[str, Any],
    hyper_profile: HyperPerformanceProfile | None,
) -> dict[str, int]:
    """Derive compact run parameters from the chosen difficulty profile."""

    base_target = float(profile.get("target", 0.5))
    weight = float(np.clip(0.35 * difficulty_progress + 0.65 * base_target, 0.0, 1.0))

    subjects_goal = int(profile.get("subjects", 120))
    if hyper_profile is not None and hyper_profile.target_subjects:
        subjects_goal = int(max(subjects_goal, hyper_profile.target_subjects))
    else:
        subjects_goal = int(round(subjects_goal * (0.85 + 0.3 * weight)))

    dwell = _blend_range(tuple(profile.get("dwell", (12, 20))), weight)
    population = _blend_range(tuple(profile.get("population", (4, 8))), weight)
    if dwell > 0:
        ideal_population = max(3, int(round(subjects_goal / dwell)))
        population = int(np.clip(ideal_population, profile["population"][0], profile["population"][1]))

    queue = _blend_range(tuple(profile.get("queue", (4, 8))), weight)
    autosave = _blend_range(tuple(profile.get("autosave", (6, 10))), weight)
    shapes = _blend_range(tuple(profile.get("shapes", (3, 6))), weight)

    return {
        "population_size": max(1, population),
        "generations_to_queue": max(1, queue),
        "autosave_interval": max(1, autosave),
        "target_dwell": max(1, dwell),
        "subjects_goal": max(1, subjects_goal),
        "shape_count": max(3, shapes),
        "pause_threshold": float(profile.get("pause_threshold", 0.9)),
        "difficulty_target": base_target,
    }


def _base_sound_score_target(
    dwell: float,
    *,
    difficulty_progress: float,
    difficulty_target: float | None = None,
) -> float:
    """Estimate the score budget required before refreshing the sound scene."""

    progress = float(np.clip(difficulty_progress, 0.0, 1.0))
    target = (
        float(np.clip(difficulty_target, 0.0, 1.0))
        if difficulty_target is not None
        else progress
    )

    progress_factor = 0.9 + 0.8 * progress
    difficulty_factor = 1.2 - 0.5 * target

    return float(max(1.0, dwell * progress_factor * difficulty_factor))


def _next_sound_budget_rng(state: st.session_state) -> np.random.Generator:
    """Return a reproducible RNG for score budgeting with updated seed."""

    seed = int(state.get("_sound_score_seed", 0))
    if seed <= 0:
        seed = int(np.random.default_rng().integers(1, np.iinfo(np.int32).max))
    rng = np.random.default_rng(seed)
    state["_sound_score_seed"] = int(rng.integers(1, np.iinfo(np.int32).max))
    return rng


def _compute_sound_score_target(
    auto_settings: Mapping[str, Any],
    *,
    difficulty_progress: float,
    rng: np.random.Generator,
) -> float:
    dwell = float(auto_settings.get("target_dwell", 10))
    base_target = _base_sound_score_target(
        dwell,
        difficulty_progress=difficulty_progress,
        difficulty_target=float(auto_settings.get("difficulty_target", difficulty_progress)),
    )
    jitter = float(rng.uniform(0.85, 1.25))
    return float(max(1.0, base_target * jitter))


def _ensure_sound_score_budget(
    state: st.session_state,
    auto_settings: Mapping[str, Any],
    *,
    difficulty_progress: float,
    force: bool = False,
) -> float:
    """Synchronise the target score budget with the current difficulty profile."""

    dwell = float(auto_settings.get("target_dwell", 10))
    baseline = _base_sound_score_target(
        dwell,
        difficulty_progress=difficulty_progress,
        difficulty_target=float(auto_settings.get("difficulty_target", difficulty_progress)),
    )

    previous_baseline = float(state.get("_sound_target_baseline", 0.0))
    current_target = float(state.get("sound_target_score", 0.0))

    if force or current_target <= 0.0:
        rng = _next_sound_budget_rng(state)
        new_target = _compute_sound_score_target(
            auto_settings,
            difficulty_progress=difficulty_progress,
            rng=rng,
        )
        state["sound_target_score"] = new_target
        state["sound_score_remaining"] = new_target
        state["_sound_target_baseline"] = baseline
        return new_target

    if previous_baseline <= 0.0:
        state["_sound_target_baseline"] = baseline
        return current_target

    change_ratio = abs(baseline - previous_baseline) / max(previous_baseline, 1.0)
    if change_ratio >= 0.25:
        scale = baseline / previous_baseline
        updated_target = max(1.0, current_target * scale)
        remaining = float(state.get("sound_score_remaining", current_target))
        state["sound_target_score"] = updated_target
        state["sound_score_remaining"] = max(0.0, remaining * scale)
        state["_sound_target_baseline"] = baseline
        return updated_target

    state["_sound_target_baseline"] = baseline
    return current_target


def _generation_sound_score(generation: GenerationRecord) -> float:
    """Quantify how much a generation should consume from the sound budget."""

    try:
        reward_peak = max(0.0, float(generation.reward_peak))
    except (TypeError, ValueError):
        reward_peak = 0.0

    try:
        reward_summary = max(0.0, float(generation.reward_summary))
    except (TypeError, ValueError):
        reward_summary = 0.0

    overlap = 0.0
    best_candidate = getattr(generation, "best_candidate", None)
    overlap_score = getattr(best_candidate, "overlap_score", None)
    if overlap_score is not None:
        try:
            overlap = float(np.clip(overlap_score, 0.0, 100.0)) / 100.0
        except (TypeError, ValueError):
            overlap = 0.0

    weighted = 0.45 * reward_peak + 0.3 * reward_summary + 1.5 * overlap + 0.2
    return float(np.clip(weighted, 0.1, 8.0))


def _aggregate_reward_components(record: GenerationRecord) -> dict[str, float | None]:
    """Average reward component contributions for ``record``."""

    def _mean(values: list[float | None]) -> float | None:
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not finite:
            return None
        return float(np.mean(finite))

    overlap_vals = [
        candidate.reward_components.get("overlap") for candidate in record.candidates
    ]
    msssim_vals = [
        candidate.reward_components.get("msssim") for candidate in record.candidates
    ]
    dct_vals = [
        candidate.reward_components.get("dct_corr") for candidate in record.candidates
    ]

    return {
        "overlap": _mean(overlap_vals),
        "msssim": _mean(msssim_vals),
        "dct_corr": _mean(dct_vals),
    }


def _apply_auto_pause(
    state: MutableMapping[str, Any],
    *,
    difficulty_progress: float,
    pause_threshold: float,
) -> str | None:
    """Update run state if adaptive difficulty requests a pause.

    Returns a sidebar message describing the action taken. When ``None`` is
    returned the caller should not display an alert.
    """

    if difficulty_progress < pause_threshold:
        state["auto_pause_acknowledged"] = False
        return None

    if state.get("auto_pause_acknowledged", False):
        return None

    state["auto_pause_acknowledged"] = True

    infinite_requested = bool(state.get("run_infinite", False)) or state.get(
        "evolution_mode"
    ) == "Infinite"

    if infinite_requested:
        return (
            "Difficulty spike detected – Keep Improving will continue running; "
            "refresh the scene manually if desired."
        )

    state["run_infinite"] = False
    state["pending_generations"] = 0
    return "Difficulty spike reached – evolution paused so a new scene can be prepared."


def _render_quick_start_wizard(
    state: st.session_state,
    container: DeltaGenerator | None = None,
    *,
    difficulty_progress: float,
    hyper_profile: HyperPerformanceProfile | None,
) -> tuple[bool, bool, bool, bool, bool, dict[str, int]]:
    """Render the staged quick-start controls inside ``container``."""

    target_container = container or st.container()
    easy_mode = bool(state.get("easy_mode", True))
    state.setdefault("quick_start_infinite_preference", False)

    easy_mode = target_container.toggle(
        "Easy mode",
        value=easy_mode,
        help=(
            "Keep Umbra in guided mode with balanced difficulty and autosave pacing. "
            "Disable to unlock manual control over evolution settings."
        ),
        key="quick_start_easy_mode",
    )
    state["easy_mode"] = easy_mode
    state["show_advanced_controls"] = not easy_mode
    mode_names = list(_DIFFICULTY_MODE_PRESETS.keys())
    if "difficulty_mode" not in state or state.get("difficulty_mode") not in mode_names:
        default_idx = 1 if len(mode_names) > 1 else 0
        state["difficulty_mode"] = mode_names[default_idx]

    media_source = state.get("quick_start_media_source", "auto")
    if media_source not in {"auto", "upload", "pinterest"}:
        media_source = "auto"
    state["quick_start_media_source"] = media_source

    target_container.markdown("### Quick start wizard")
    step_container = target_container.container()
    step_cols = step_container.columns(3)
    previous_status = state.get("quick_start_step_status", {})
    step_status = {
        "upload": bool(previous_status.get("upload", False)),
        "mode": bool(previous_status.get("mode", False)),
        "run": bool(previous_status.get("run", False)),
    }

    upload_col = step_cols[0]
    mode_col = step_cols[1]
    run_col = step_cols[2]

    upload_col.markdown(
        ("✅" if step_status["upload"] else "①") + " **Step 1** · Upload media"
    )
    source_choice = upload_col.radio(
        "Media source",
        options=("auto", "upload", "pinterest"),
        index={"auto": 0, "upload": 1, "pinterest": 2}[media_source],
        format_func=lambda val: {
            "auto": "Use generated scene",
            "upload": "Upload image",
            "pinterest": "Pinterest inspiration",
        }[val],
        key="quick_start_media_selector",
    )
    state["quick_start_media_source"] = source_choice
    if source_choice == "upload":
        uploaded = upload_col.file_uploader(
            "Choose an image",
            type=("png", "jpg", "jpeg", "bmp"),
            key="quick_start_image_upload",
        )
        if uploaded is not None:
            try:
                uploaded.seek(0)
                image = _load_uploaded_image(uploaded)
            except ValueError as exc:  # pragma: no cover - defensive
                upload_col.error(str(exc))
            else:
                state["quick_start_reference_image"] = image
                state["quick_start_reference_label"] = uploaded.name or "Uploaded image"
                state["quick_start_reference_source"] = "upload"
                state["quick_start_step1_timestamp"] = time.time()
                step_status["upload"] = True
        elif state.get("quick_start_reference_image") is not None:
            step_status["upload"] = True
        else:
            state.pop("quick_start_reference_source", None)
    elif source_choice == "pinterest":
        pinterest_hint = state.get("quick_start_pinterest_source", "")
        board_input = upload_col.text_input(
            "Pinterest board or RSS feed",
            value=pinterest_hint,
            help=(
                "Enter a board path such as 'umbresearch/inspirations' or paste a Pinterest RSS "
                "URL. Leave blank to use Umbra's curated feeds."
            ),
            key="quick_start_pinterest_board",
        )
        state["quick_start_pinterest_source"] = board_input

        fetch_button = upload_col.button(
            "Fetch Pinterest inspiration",
            key="quick_start_pinterest_fetch",
            help="Download a random Pinterest image using the selected feed.",
        )

        needs_fetch = (
            fetch_button
            or state.get("quick_start_reference_source") != "pinterest"
            or state.get("quick_start_reference_image") is None
        )
        if needs_fetch:
            try:
                image, label = _fetch_random_pinterest_image(board_input or None)
            except RuntimeError as exc:
                logger.warning("Pinterest fetch failed: %s", exc)
                upload_col.error(str(exc))
                state.pop("quick_start_reference_image", None)
                state.pop("quick_start_reference_label", None)
                state.pop("quick_start_reference_source", None)
                step_status["upload"] = False
            else:
                state["quick_start_reference_image"] = image
                state["quick_start_reference_label"] = label
                state["quick_start_reference_source"] = "pinterest"
                state["quick_start_step1_timestamp"] = time.time()
                step_status["upload"] = True
        elif state.get("quick_start_reference_source") == "pinterest":
            step_status["upload"] = True
        upload_col.caption(
            "Pinterest images refresh automatically when targets are met or when you press "
            "the fetch button."
        )
    else:
        state.pop("quick_start_reference_image", None)
        state.pop("quick_start_reference_label", None)
        state.pop("quick_start_reference_source", None)
        step_status["upload"] = True

    mode_col.markdown(
        ("✅" if step_status["mode"] else "②") + " **Step 2** · Pick your pace"
    )
    mode_names = list(_DIFFICULTY_MODE_PRESETS.keys())
    current_mode = state.get("difficulty_mode", mode_names[0])
    if current_mode not in mode_names:
        current_mode = mode_names[0]
        state["difficulty_mode"] = current_mode
    difficulty_choice = mode_col.select_slider(
        "Difficulty focus",
        options=mode_names,
        value=current_mode,
        help="Single dial that sets cadence, autosave, and dwell automatically.",
        key="quick_start_difficulty",
    )
    state["difficulty_mode"] = difficulty_choice
    step_status["mode"] = True
    mode_col.caption(
        "Difficulty drives every setting—no advanced mode required. Sound scenes refresh "
        "automatically once their score target is reached."
    )

    profile = _DIFFICULTY_MODE_PRESETS[state["difficulty_mode"]]
    auto_settings = _auto_run_parameters(
        difficulty_progress=difficulty_progress,
        profile=profile,
        hyper_profile=hyper_profile,
    )

    stats_cols = mode_col.columns(3)
    stats_cols[0].metric(
        "AI attempts/gen",
        auto_settings["population_size"],
    )
    stats_cols[1].metric(
        "Gen queue",
        auto_settings["generations_to_queue"],
    )
    stats_cols[2].metric("Scene shapes", auto_settings["shape_count"])
    estimated_score = _base_sound_score_target(
        float(auto_settings["target_dwell"]),
        difficulty_progress=difficulty_progress,
        difficulty_target=float(auto_settings.get("difficulty_target", difficulty_progress)),
    )
    mode_col.metric("Sound score target", f"{estimated_score:.1f}")
    if hyper_profile is None:
        state["population_size"] = auto_settings["population_size"]
        state["generations_to_queue"] = auto_settings["generations_to_queue"]
        state["autosave_interval"] = auto_settings["autosave_interval"]
        state["sound_target_dwell"] = auto_settings["target_dwell"]
        state["last_sound_target_dwell"] = auto_settings["target_dwell"]
    state["target_shape_count"] = int(auto_settings["shape_count"])

    desired_target = auto_settings["difficulty_target"]
    state["difficulty_target_override"] = desired_target

    step_cols[2].markdown(
        ("✅" if step_status["run"] else "③") + " **Step 3** · Launch evolution"
    )
    with step_cols[2]:
        controls_row = st.columns(2)
        run_button = controls_row[0].button("Start", use_container_width=True, key="quick_start_run")
        stop_button = controls_row[1].button("Pause", use_container_width=True, key="quick_start_pause")
        secondary_row = st.columns(2)
        reset_button = secondary_row[0].button("Reset", use_container_width=True, key="quick_start_reset")
        save_button = secondary_row[1].button("Save", use_container_width=True, key="quick_start_save")
        reload_button = st.button("Reload autosave", use_container_width=True, key="quick_start_reload")

        infinite_preference = st.toggle(
            "Keep running until I press Pause",
            value=bool(state.get("quick_start_infinite_preference", False)),
            help=(
                "Schedule new generations automatically without a queue so Umbra runs continuously"
                " until you hit Pause."
            ),
            key="quick_start_infinite_toggle",
        )
        state["quick_start_infinite_preference"] = bool(infinite_preference)
        if not infinite_preference and state.get("run_infinite", False):
            state["run_infinite"] = False
            state["evolution_mode"] = "Finite"

        metrics_cols = st.columns(2)
        metrics_cols[0].metric("Adaptive progress", f"{difficulty_progress * 100:.0f}%")
        metrics_cols[1].metric("Difficulty target", f"{desired_target * 100:.0f}%")
        st.caption(
            f"Auto-tuned for about {auto_settings['subjects_goal']} subjects with "
            f"{auto_settings['population_size']} evolving in parallel."
        )

        pause_threshold = auto_settings.get("pause_threshold", 0.9)
        pause_message = _apply_auto_pause(
            state,
            difficulty_progress=difficulty_progress,
            pause_threshold=pause_threshold,
        )
        if pause_message:
            st.info(pause_message)
        step_status["run"] = step_status["run"] or run_button
        if not step_status["run"]:
            step_status["run"] = bool(state.get("pending_generations", 0) > 0)
        if not step_status["run"]:
            step_status["run"] = bool(state.get("active_run_id"))

    balanced_name = "Balanced climb"
    profile = _DIFFICULTY_MODE_PRESETS.get(balanced_name)
    if profile is None:
        # Fall back to the first available profile to avoid runtime errors.
        profile = next(iter(_DIFFICULTY_MODE_PRESETS.values()))
    auto_settings = _auto_run_parameters(
        difficulty_progress=difficulty_progress,
        profile=profile,
        hyper_profile=hyper_profile,
    )

    if easy_mode:
        auto_settings["population_size"] = 10
        auto_settings["difficulty_target"] = 0.5
        state["difficulty_mode"] = balanced_name
        state["population_size"] = auto_settings["population_size"]
        state["difficulty_target_override"] = auto_settings["difficulty_target"]
        state["_easy_mode_defaults_applied"] = True
        if hyper_profile is None:
            state["generations_to_queue"] = auto_settings["generations_to_queue"]
            state["autosave_interval"] = auto_settings["autosave_interval"]
            state["sound_target_dwell"] = auto_settings["target_dwell"]
            state["last_sound_target_dwell"] = auto_settings["target_dwell"]
        guidance_text = (
            "Easy Mode prepares balanced defaults with 10 candidates per generation."
        )
    else:
        state["_easy_mode_defaults_applied"] = False
        mode_names = list(_DIFFICULTY_MODE_PRESETS.keys())
        default_mode = state.get("difficulty_mode")
        if default_mode not in mode_names:
            default_mode = mode_names[1] if len(mode_names) > 1 else mode_names[0]
        difficulty_mode = st.sidebar.select_slider(
            "Difficulty focus",
            options=mode_names,
            value=default_mode,
            help="Single dial that steers all modelling settings and adaptive tuning.",
        )
        state["difficulty_mode"] = difficulty_mode
        profile = _DIFFICULTY_MODE_PRESETS[difficulty_mode]
        auto_settings = _auto_run_parameters(
            difficulty_progress=difficulty_progress,
            profile=profile,
            hyper_profile=hyper_profile,
        )
        state["difficulty_target_override"] = auto_settings["difficulty_target"]
        guidance_text = (
            f"Auto-tuned for about {auto_settings['subjects_goal']} subjects with "
            f"{auto_settings['population_size']} evolving in parallel."
        )

    desired_target = auto_settings["difficulty_target"]
    metrics_cols = st.sidebar.columns(2)
    metrics_cols[0].metric("Adaptive progress", f"{difficulty_progress * 100:.0f}%")
    metrics_cols[1].metric("Difficulty target", f"{desired_target * 100:.0f}%")
    st.sidebar.caption(guidance_text)

    pause_threshold = auto_settings.get("pause_threshold", 0.9)
    pause_message = _apply_auto_pause(
        state,
        difficulty_progress=difficulty_progress,
        pause_threshold=pause_threshold,
    )
    if pause_message:
        run_col.info(pause_message)
    step_status["run"] = step_status["run"] or run_button
    if not step_status["run"]:
        step_status["run"] = bool(state.get("pending_generations", 0) > 0)
    if not step_status["run"]:
        step_status["run"] = bool(state.get("active_run_id"))

    state["quick_start_step_status"] = step_status

    return (
        easy_mode,
        run_button,
        stop_button,
        reset_button,
        save_button,
        reload_button,
        auto_settings,
    )


def _render_control_panel(
    state: st.session_state,
    container: DeltaGenerator | None = None,
    *,
    difficulty_progress: float,
    hyper_profile: HyperPerformanceProfile | None,
) -> tuple[bool, bool, bool, bool, bool, dict[str, int]]:
    """Compatibility shim delegating legacy control calls to the wizard layout."""

    target_container = container or st.sidebar
    return _render_quick_start_wizard(
        state,
        target_container,
        difficulty_progress=difficulty_progress,
        hyper_profile=hyper_profile,
    )


def _render_demo_lab(state: st.session_state, container: DeltaGenerator | None = None) -> None:
    """Render the portable demo controls with packaging helpers."""

    target = container or st.container()

    target.header("Codec demo package")
    target.write(
        "Bundle a minimal Umbra codec demo or try the encode/decode loop directly below."
    )

    package_col, demo_col = target.columns([1, 2])

    package_container = package_col.container()
    package_container.subheader("Share the model")
    package_name = state.get("demo_package_name")
    package_blob: bytes | None = state.get("demo_package_blob")

    if package_container.button("Package demo executable", key="demo_package_button"):
        with package_container.spinner("Building demo archive..."):
            try:
                name, blob = build_demo_package()
            except Exception as exc:  # pragma: no cover - defensive
                package_container.error(f"Packaging failed: {exc}")
            else:
                state["demo_package_name"] = name
                state["demo_package_blob"] = blob
                state["demo_package_timestamp"] = time.time()
                package_name = name
                package_blob = blob
                package_container.success(
                    "Demo packaged! Use the download button below to share it."
                )

    if package_blob:
        _render_download_link(
            package_container,
            label="Download demo archive",
            data=package_blob,
            file_name=package_name or "umbra_demo.pyz",
            mime="application/octet-stream",
            key="demo_package_download",
        )
    else:
        package_container.caption(
            "No package built yet. Press the button above to generate a shareable .pyz archive."
        )

    demo_container = demo_col.container()
    demo_container.subheader("In-app translator")

    uploaded = demo_container.file_uploader(
        "Image input",
        type=("png", "jpg", "jpeg", "bmp"),
        key="demo_image_upload",
    )

    demo_image: np.ndarray | None = state.get("demo_image_array")
    if uploaded is not None:
        try:
            uploaded.seek(0)
            demo_image = _load_uploaded_image(uploaded)
        except ValueError as exc:  # pragma: no cover - defensive
            demo_container.error(str(exc))
            demo_image = None
        else:
            state["demo_image_array"] = demo_image
            state["demo_image_label"] = uploaded.name or "Uploaded image"
            state["demo_resolution"] = demo_image.shape[:2]
            state["demo_sample_rate"] = 48_000
            try:
                wav_bytes = encode_image_to_wav_bytes(
                    demo_image,
                    sample_rate=int(state["demo_sample_rate"]),
                )
            except Exception as exc:  # pragma: no cover - defensive
                demo_container.error(f"Failed to encode image: {exc}")
            else:
                state["demo_wav_bytes"] = wav_bytes
                state["demo_wav_label"] = (
                    Path(state.get("demo_image_label", "demo"))
                    .with_suffix(".wav")
                    .name
                )

    if demo_image is not None:
        _render_image(
            demo_container,
            normalize_for_display(demo_image),
            state.get("demo_image_label", "Uploaded image"),
        )

    wav_bytes = state.get("demo_wav_bytes")
    sample_rate = int(state.get("demo_sample_rate", 48_000))
    resolution = state.get("demo_resolution")

    if wav_bytes is not None and isinstance(wav_bytes, (bytes, bytearray)):
        _render_audio(demo_container, bytes(wav_bytes))
        _render_download_link(
            demo_container,
            label="Download WAV",
            data=wav_bytes,
            file_name=state.get("demo_wav_label", "demo.wav"),
            mime="audio/wav",
            key="demo_wav_download",
        )

        if demo_container.button("Translate", key="demo_translate_button"):
            try:
                waveform, detected_rate = load_waveform_from_wav(bytes(wav_bytes))
                active_rate = sample_rate or int(detected_rate)
                if resolution is None:
                    raise ValueError(
                        "Unknown resolution. Upload an image first to establish dimensions."
                    )
                translated = decode_waveform_to_image(
                    waveform,
                    sample_rate=int(active_rate),
                    resolution=tuple(int(v) for v in resolution),
                )
            except Exception as exc:  # pragma: no cover - defensive
                demo_container.error(f"Translation failed: {exc}")
            else:
                state["demo_translated_image"] = translated
                state["demo_translation_timestamp"] = time.time()

    translated = state.get("demo_translated_image")
    if isinstance(translated, np.ndarray):
        _render_image(
            demo_container,
            normalize_for_display(translated),
            "Translated image",
        )

def _render_control_panel(
    state: st.session_state,
    container: DeltaGenerator | None = None,
    *,
    difficulty_progress: float,
    hyper_profile: HyperPerformanceProfile | None,
) -> tuple[bool, bool, bool, bool, bool, dict[str, int]]:
    """Compatibility shim delegating legacy control calls to the wizard layout."""

    target_container = container or st.sidebar
    return _render_quick_start_wizard(
        state,
        target_container,
        difficulty_progress=difficulty_progress,
        hyper_profile=hyper_profile,
    )


_IMAGE_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "Geometric Echo (balanced)": {
        "encoder_sigma": 0.2,
        "decoder_sigma": 1.0,
        "description": "Balanced encoder/decoder pair tuned for the classic Umbra evolution flow.",
    },
    "Nocturne Bloom (soft focus)": {
        "encoder_sigma": 0.28,
        "decoder_sigma": 0.85,
        "description": "Higher encoder variance with a softer decoder for painterly reconstructions.",
    },
    "Solar Cascade (high contrast)": {
        "encoder_sigma": 0.38,
        "decoder_sigma": 0.65,
        "description": "Aggressive noise for vivid contrasts balanced by a sharper decoder output.",
    },
}

_SOUND_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "Ambient Skyline": {
        "sample_rate": (20_000, 48_000),
        "resolution": (160, 224),
        "target_dwell": 12,
        "description": "Mid-range rates with generous dwell time for slow evolving ambient textures.",
    },
    "Pulse Forge": {
        "sample_rate": (28_000, 64_000),
        "resolution": (192, 256),
        "target_dwell": 8,
        "description": "Higher sample rates and resolutions for rhythmic, fast-changing scenes.",
    },
    "Aurora Sweep": {
        "sample_rate": (16_000, 56_000),
        "resolution": (128, 224),
        "target_dwell": 16,
        "description": "Wide spectrum exploration with longer dwell for evolving harmonic washes.",
    },
    "Mythic Resonance": {
        "sample_rate": (36_000, 110_000),
        "resolution": (224, 320),
        "target_dwell": 6,
        "description": "Unrealistic studio rates with towering canvases; risky but capable of sudden clarity breakthroughs.",
    },
    "Oblivion Fracture": {
        "sample_rate": (12_000, 96_000),
        "resolution": (96, 288),
        "target_dwell": 4,
        "description": "Chaotic low-to-ultra bandwidth swings that court disasters while leaving room for serendipitous control wins.",
    },
}

_DIFFICULTY_PRESET_LINKS: dict[str, dict[str, str]] = {
    "Calm focus": {
        "image": "Geometric Echo (balanced)",
        "sound": "Ambient Skyline",
        "engine": "Lineage & noise (classic)",
    },
    "Balanced climb": {
        "image": "Nocturne Bloom (soft focus)",
        "sound": "Aurora Sweep",
        "engine": "Lineage & noise (classic)",
    },
    "Edge runner": {
        "image": "Solar Cascade (high contrast)",
        "sound": "Pulse Forge",
        "engine": "Neural lineage fusion",
    },
    "Apex trial": {
        "image": "Solar Cascade (high contrast)",
        "sound": "Pulse Forge",
        "engine": "Neural lineage fusion",
    },
}

_DIFFICULTY_MODE_PRESETS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "Calm focus",
            {
                "target": 0.3,
                "subjects": 90,
                "population": (3, 6),
                "queue": (3, 6),
                "autosave": (5, 8),
                "dwell": (12, 18),
                "shapes": (3, 6),
                "pause_threshold": 0.82,
            },
        ),
        (
            "Balanced climb",
            {
                "target": 0.5,
                "subjects": 120,
                "population": (4, 8),
                "queue": (4, 9),
                "autosave": (6, 10),
                "dwell": (14, 22),
                "shapes": (8, 16),
                "pause_threshold": 0.86,
            },
        ),
        (
            "Edge runner",
            {
                "target": 0.68,
                "subjects": 150,
                "population": (5, 9),
                "queue": (5, 10),
                "autosave": (6, 11),
                "dwell": (18, 28),
                "shapes": (16, 24),
                "pause_threshold": 0.9,
            },
        ),
        (
            "Apex trial",
            {
                "target": 0.82,
                "subjects": 180,
                "population": (6, 12),
                "queue": (6, 12),
                "autosave": (7, 12),
                "dwell": (20, 34),
                "shapes": (22, 30),
                "pause_threshold": 0.92,
            },
        ),
    ]
)


def _generate_tree_id() -> str:
    return uuid4().hex


def _default_tree_label(sound_seed: int, resolution: int, sample_rate: int) -> str:
    rate_khz = sample_rate / 1000.0
    return f"Seed {sound_seed} • {resolution}px @ {rate_khz:.1f} kHz"


def _format_tree_label(entry: dict[str, Any]) -> str:
    label = entry.get("label") or "Evolution tree"
    manager: EvolutionManager | None = entry.get("manager")
    generation_count = len(manager.generations) if manager is not None else 0
    suffix = "s" if generation_count != 1 else ""
    return f"{label} ({generation_count} generation{suffix})"


def _activate_tree(state: st.session_state, tree_id: str) -> None:
    forest: dict[str, dict[str, Any]] = state.setdefault("evolution_trees", {})
    entry = forest.get(tree_id)
    if entry is None:
        return
    state["active_tree_id"] = tree_id
    state["evolution_manager"] = entry["manager"]
    state["manager"] = entry["manager"]
    state["evolution_signature"] = entry["signature"]
    state["run_infinite"] = entry.get("run_infinite", False)
    state["pending_generations"] = entry.get("pending", 0)
    meta = dict(entry.get("metadata", {}))
    if "shared_seed" in meta:
        state["shared_seed"] = int(meta["shared_seed"])
    else:
        state["shared_seed"] = int(entry["manager"].base_seed)
    if "sound_seed" in meta:
        state["active_sound_seed"] = int(meta["sound_seed"])
    if "sample_rate" in meta:
        state["current_sound_sample_rate"] = int(meta["sample_rate"])
    if "resolution" in meta:
        state["current_sound_resolution"] = int(meta["resolution"])
    if "parent_seeds" in meta:
        parent_meta = meta.get("parent_seeds", [])
        if isinstance(parent_meta, (list, tuple)):
            state["active_parent_seeds"] = [int(seed) for seed in parent_meta]


def _register_tree(
    state: st.session_state,
    manager: EvolutionManager,
    *,
    label: str,
    run_infinite: bool,
    pending: int = 0,
    activate: bool = True,
    unique: bool = False,
    metadata: dict[str, Any] | None = None,
) -> str:
    forest: dict[str, dict[str, Any]] = state.setdefault("evolution_trees", {})
    signature = (manager.image_signature, int(manager.base_seed))

    if not unique:
        for existing_id, entry in forest.items():
            if entry.get("signature") == signature:
                entry["manager"] = manager
                entry["label"] = label
                entry["run_infinite"] = bool(run_infinite)
                entry["pending"] = int(pending)
                entry["metadata"] = dict(metadata or entry.get("metadata", {}))
                if activate or state.get("active_tree_id") is None:
                    _activate_tree(state, existing_id)
                return existing_id

    tree_id = _generate_tree_id()
    forest[tree_id] = {
        "manager": manager,
        "signature": signature,
        "label": label,
        "created": time.time(),
        "run_infinite": bool(run_infinite),
        "pending": int(pending),
        "metadata": dict(metadata or {}),
    }

    if activate or state.get("active_tree_id") is None:
        _activate_tree(state, tree_id)
    return tree_id


def _sync_active_tree_state(state: st.session_state) -> None:
    forest: dict[str, dict[str, Any]] | None = state.get("evolution_trees")
    tree_id = state.get("active_tree_id")
    if not forest or tree_id not in forest:
        return
    entry = forest[tree_id]
    entry["run_infinite"] = bool(state.get("run_infinite", False))
    entry["pending"] = int(state.get("pending_generations", 0))
    entry["signature"] = tuple(state.get("evolution_signature", entry["signature"]))
    meta = dict(entry.get("metadata", {}))
    meta.update(
        {
            "shared_seed": int(state.get("shared_seed", meta.get("shared_seed", 0))),
            "sound_seed": int(state.get("active_sound_seed", meta.get("sound_seed", 0))),
            "sample_rate": int(
                state.get("current_sound_sample_rate", meta.get("sample_rate", 48_000))
            ),
            "resolution": int(
                state.get("current_sound_resolution", meta.get("resolution", 192))
            ),
            "parent_seeds": [
                int(seed)
                for seed in state.get(
                    "active_parent_seeds", meta.get("parent_seeds", [])
                )
            ],
        }
    )
    entry["metadata"] = meta


def _update_tree_label(state: st.session_state, label: str) -> None:
    forest: dict[str, dict[str, Any]] | None = state.get("evolution_trees")
    tree_id = state.get("active_tree_id")
    if not forest or tree_id not in forest:
        return
    forest[tree_id]["label"] = label


def _refresh_active_parents(
    state: st.session_state,
    manager: EvolutionManager,
    generation: GenerationRecord | None = None,
) -> None:
    """Ensure the session tracks lineage seeds for selective breeding."""

    available = {entry.seed for entry in manager.parent_lineage}
    selection = [
        int(seed)
        for seed in state.get("active_parent_seeds", [])
        if int(seed) in available
    ]
    if generation is not None and generation.candidates:
        selection.append(generation.best_candidate.seed)
    elif not selection and available:
        best_entry = max(manager.parent_lineage, key=lambda entry: entry.metrics.ssim)
        selection.append(best_entry.seed)
    state["active_parent_seeds"] = list(dict.fromkeys(selection))


def _get_reconstruction_cache(state: st.session_state) -> dict[str, Any]:
    """Access the memoized reconstruction lab state."""

    return state.setdefault("_reconstruction_lab", {})


def _store_reconstruction_result(
    state: st.session_state, result: ReconstructionResult
) -> None:
    cache = _get_reconstruction_cache(state)
    cache["result"] = result
    cache["stored_at"] = time.time()


def _render_reconstruction_result(result: ReconstructionResult) -> None:
    st.markdown("### Reconstruction outcomes")
    base_cols = st.columns(3)
    base_cols[0].image(
        to_uint8_image(result.base_image),
        caption="Ground truth collage",
        use_column_width=True,
    )
    _render_image(
        base_cols[1],
        result.ensemble_prediction,
        "Ensemble prediction",
    )
    _render_image(
        base_cols[2],
        result.hybrid_prediction,
        "Hybrid fill (ensemble + audio)",
    )

    comparison_cols = st.columns(2)
    _render_image(
        comparison_cols[0],
        result.audio_reconstruction,
        "Audio-derived reconstruction",
    )
    overlay = colorize_comparison(result.base_image, result.hybrid_prediction)
    _render_image(
        comparison_cols[1],
        overlay,
        "Overlay vs. ground truth",
    )

    diff = np.abs(result.base_image - result.hybrid_prediction)
    _render_image(
        st,
        normalize_for_display(diff),
        "Residual difference heatmap",
    )

    coverage_map = normalize_for_display(result.coverage)
    coverage_rgb = np.stack(
        [coverage_map, np.zeros_like(coverage_map), 1.0 - coverage_map],
        axis=2,
    )
    _render_image(
        st,
        coverage_rgb,
        "Coverage confidence (red = uncertain, blue = confident)",
    )

    variation_count = result.variations.shape[0]
    if variation_count > 0:
        index = st.slider(
            "Inspect noisy variation",
            min_value=0,
            max_value=variation_count - 1,
            value=0,
            key="reconstruction_variation_index",
        )
        _render_image(
            st,
            result.variations[index],
            f"Variation sample {index + 1} of {variation_count}",
        )

    try:
        wav_bytes = waveform_to_wav_bytes(result.waveform, result.sample_rate)
    except ValueError as exc:
        st.warning(f"Unable to render waveform audio: {exc}")
    else:
        _render_audio(st, wav_bytes)

    shape_rows = [
        {
            "Shape": item.shape,
            "Colour (RGB)": (
                f"{item.color[0]:.2f}, {item.color[1]:.2f}, {item.color[2]:.2f}"
            ),
            "Centre": f"({item.center[0]}, {item.center[1]})",
            "Rotation": f"{item.rotation:.1f}°",
            "Size": item.size,
        }
        for item in result.shapes
    ]
    if shape_rows:
        st.dataframe(pd.DataFrame(shape_rows), use_container_width=True)



def _render_reconstruction_lab(
    state: st.session_state, *, default_resolution: int, default_sample_rate: int
) -> None:
    cache = _get_reconstruction_cache(state)
    default_seed = int(state.get("shared_seed", 0)) or 1

    seed_value = int(cache.get("seed", default_seed))
    variation_value = int(cache.get("variation_count", 6))
    noise_value = float(cache.get("noise_sigma", 0.3))
    dropout_value = float(cache.get("dropout_probability", 0.35))
    sample_rate_value = int(cache.get("sample_rate", default_sample_rate))

    noise_default = _quantize_slider_value(
        noise_value, min_value=0.05, max_value=0.6, step=0.05
    )
    dropout_default = _quantize_slider_value(
        dropout_value, min_value=0.05, max_value=0.8, step=0.05
    )
    sample_rate_default = int(
        _quantize_slider_value(
            float(sample_rate_value),
            min_value=16_000,
            max_value=64_000,
            step=2_000,
        )
    )

    seed = st.number_input(
        "Collage seed",
        value=seed_value,
        min_value=1,
        step=1,
        key="reconstruction_seed",
        help="Controls the random collage generator used for the reconstruction challenge.",
    )
    variation_count = st.slider(
        "Number of noisy glimpses",
        min_value=3,
        max_value=15,
        value=min(max(variation_value, 3), 15),
        key="reconstruction_variations",
        help="How many corrupted observations to blend before predicting the missing pixels.",
    )
    noise_sigma = st.slider(
        "Noise amplitude",
        min_value=0.05,
        max_value=0.6,
        value=float(noise_default),
        step=0.05,
        key="reconstruction_noise",
        help="Standard deviation of the Gaussian noise added to each variation.",
    )
    dropout_probability = st.slider(
        "Dropout probability",
        min_value=0.05,
        max_value=0.8,
        value=float(dropout_default),
        step=0.05,
        key="reconstruction_dropout",
        help="Likelihood that a pixel is replaced with unrelated noise in each glimpse.",
    )
    sample_rate = st.slider(
        "Audio sample rate",
        min_value=16_000,
        max_value=64_000,
        step=2_000,
        value=sample_rate_default,
        key="reconstruction_sample_rate",
        help="Encoding rate for the collage-to-audio conversion stage.",
    )

    run_experiment = st.button(
        "Run reconstruction experiment",
        key="reconstruction_run",
    )

    if run_experiment:
        try:
            result = run_reconstruction_cycle(
                int(seed),
                resolution=(default_resolution, default_resolution),
                variation_count=int(variation_count),
                noise_sigma=float(noise_sigma),
                dropout_probability=float(dropout_probability),
                sample_rate=int(sample_rate),
            )
        except ValueError as exc:
            st.error(f"Failed to run reconstruction experiment: {exc}")
        else:
            cache.update(
                {
                    "seed": int(seed),
                    "variation_count": int(variation_count),
                    "noise_sigma": float(noise_sigma),
                    "dropout_probability": float(dropout_probability),
                    "sample_rate": int(sample_rate),
                }
            )
            _store_reconstruction_result(state, result)
            st.success("Reconstruction experiment completed; review the outcomes below.")

    cached_result = cache.get("result")
    if isinstance(cached_result, ReconstructionResult):
        _render_reconstruction_result(cached_result)
    else:
        st.caption("Run the reconstruction experiment to visualise predictions.")


def _update_difficulty(state: st.session_state, latest_overlap: float) -> float:
    target = float(np.clip(latest_overlap / 100.0, 0.0, 1.0))
    previous = float(state.get("difficulty", 0.0))
    difficulty = float(np.clip(0.7 * previous + 0.3 * target, 0.0, 1.0))
    state["difficulty"] = difficulty
    return difficulty


def _autosave_path(directory: Path) -> Path:
    return directory / "evolution_state.pkl"


def _finite_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when unavailable."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if math.isfinite(numeric):
        return numeric
    return None


def _attempt_autoload(autosave_dir: Path) -> None:
    state = st.session_state
    autosave_path = _autosave_path(autosave_dir)
    try:
        manager = EvolutionManager.load(autosave_dir)
    except FileNotFoundError:
        return
    except EOFError:
        logger.warning("Autosave at %s appears to be truncated; ignoring", autosave_path)
        try:
            autosave_path.unlink()
        except OSError:
            logger.debug("Failed to remove corrupted autosave at %s", autosave_path, exc_info=True)
        st.warning("Autosave data was incomplete and has been discarded.")
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load autosave from %s", autosave_dir)
        st.warning(f"Failed to load autosave: {exc}")
        return

    previous_scene = state.get("adaptive_scene")
    scene = dict(previous_scene) if isinstance(previous_scene, dict) else {}
    rng = np.random.default_rng()
    scene.setdefault("sound_seed", int(rng.integers(0, np.iinfo(np.int32).max)))
    scene.setdefault("sample_rate", 48_000)
    resolution = int(manager.original.shape[0]) if manager.original.ndim >= 2 else 192
    scene.setdefault("resolution", resolution)
    scene.setdefault("target_dwell", 10)
    scene.update(
        {
            "base_seed": int(manager.base_seed),
            "encoder_sigma": float(manager.encoder.sigma),
            "decoder_sigma": float(manager.decoder.denoise_sigma or 0.0),
        }
    )
    state["adaptive_scene"] = scene
    state["active_sound_seed"] = int(scene["sound_seed"])
    state["current_sound_sample_rate"] = int(scene["sample_rate"])
    state["current_sound_resolution"] = int(scene["resolution"])
    state["shared_seed"] = int(scene["base_seed"])
    state["population_size"] = manager.population_size
    state["autosave_interval"] = manager.autosave_interval
    if EvolutionManager.hyper_mode_enabled():
        state["_hyper_profile_cache"] = manager.hyper_profile
        state["generations_to_queue"] = int(
            max(1, manager.hyper_profile.queue_generations or state.get("generations_to_queue", 1))
        )
        dwell = int(
            max(1, manager.hyper_profile.dwell_generations or state.get("sound_target_dwell", scene["target_dwell"]))
        )
        state["sound_target_dwell"] = dwell
        state["last_sound_target_dwell"] = dwell
        scene["target_dwell"] = dwell
    label = f"Autosave • {time.strftime('%H:%M:%S')}"
    metadata = {
        "shared_seed": int(scene["base_seed"]),
        "sound_seed": int(scene["sound_seed"]),
        "sample_rate": int(scene["sample_rate"]),
        "resolution": int(scene["resolution"]),
        "parent_seeds": [entry.seed for entry in manager.parent_lineage],
    }
    _register_tree(
        state,
        manager,
        label=label,
        run_infinite=False,
        pending=0,
        activate=True,
        metadata=metadata,
    )
    state["evolution_signature"] = (manager.image_signature, int(manager.base_seed))
    state["pending_generations"] = 0
    state["run_infinite"] = False
    auto_settings = {
        "target_dwell": int(scene["target_dwell"]),
        "difficulty_target": float(state.get("difficulty", 0.0)),
    }
    _ensure_sound_score_budget(
        state,
        auto_settings,
        difficulty_progress=float(state.get("difficulty", 0.0)),
        force=True,
    )
    _sync_active_tree_state(state)
    st.success("Loaded autosaved evolution session.")
    logger.info(
        "Restored autosave from %s with %d generations",
        autosave_dir,
        len(manager.generations),
    )
    state["manager"] = manager


def _ensure_manager(
    original: np.ndarray,
    encoder: NoiseStreamEncoder,
    decoder: NoiseStreamDecoder,
    population_size: int,
    seed: int,
    autosave_interval: int,
    *,
    force_new_tree: bool = False,
    tree_label: str | None = None,
) -> EvolutionManager:
    state = st.session_state
    forest: dict[str, dict[str, Any]] = state.setdefault("evolution_trees", {}) 
    was_running_infinite = bool(state.get("run_infinite", False))
    signature = (compute_image_signature(original), int(seed))
    metadata = {
        "shared_seed": int(state.get("shared_seed", seed)),
        "sound_seed": int(state.get("active_sound_seed", seed)),
        "sample_rate": int(state.get("current_sound_sample_rate", 48_000)),
        "resolution": int(
            state.get(
                "current_sound_resolution",
                original.shape[0] if original.ndim >= 2 else int(np.sqrt(original.size)),
            )
        ),
        "parent_seeds": [int(seed) for seed in state.get("active_parent_seeds", [])],
    }

    active_id = state.get("active_tree_id")
    entry = forest.get(active_id)

    if force_new_tree or entry is None:
        if force_new_tree:
            was_running_infinite = False
        manager = EvolutionManager(
            original=original,
            encoder=encoder,
            decoder=decoder,
            population_size=population_size,
            base_seed=seed,
            autosave_interval=autosave_interval,
        )
        label = tree_label or _default_tree_label(
            int(state.get("active_sound_seed", seed)),
            int(original.shape[0] if original.ndim >= 2 else np.sqrt(original.size)),
            int(state.get("current_sound_sample_rate", 48_000)),
        )
        _register_tree(
            state,
            manager,
            label=label,
            run_infinite=was_running_infinite,
            pending=0,
            activate=True,
            unique=force_new_tree,
            metadata=metadata,
        )
        state["evolution_signature"] = signature
        state["pending_generations"] = 0
        state["run_infinite"] = was_running_infinite
        logger.info("Initialised evolution tree %s", label)
        state["manager"] = manager
        return manager

    manager = entry["manager"]
    if entry.get("signature") != signature:
        manager = EvolutionManager(
            original=original,
            encoder=encoder,
            decoder=decoder,
            population_size=population_size,
            base_seed=seed,
            autosave_interval=autosave_interval,
        )
        label = tree_label or entry.get("label") or _default_tree_label(
            int(state.get("active_sound_seed", seed)),
            int(original.shape[0] if original.ndim >= 2 else np.sqrt(original.size)),
            int(state.get("current_sound_sample_rate", 48_000)),
        )
        _register_tree(
            state,
            manager,
            label=label,
            run_infinite=was_running_infinite,
            pending=0,
            activate=True,
            unique=True,
            metadata=metadata,
        )
        state["evolution_signature"] = signature
        state["pending_generations"] = 0
        logger.info("Reinitialised evolution manager for new scene: %s", label)
        state["manager"] = manager
        return manager

    manager.update_settings(
        original=original,
        encoder=encoder,
        decoder=decoder,
        population_size=population_size,
        autosave_interval=autosave_interval,
    )
    engine_mode = state.get("_active_evolution_engine", "Lineage & noise (classic)")
    if engine_mode == "Neural lineage fusion":
        advisor = manager.reward_advisor
        if not isinstance(advisor, NeuralRewardModel):
            manager.set_advisor(NeuralRewardModel())
    else:
        manager.set_advisor(None)
    entry["manager"] = manager
    entry["signature"] = signature
    state["evolution_manager"] = manager
    state["manager"] = manager
    state["evolution_signature"] = signature
    if tree_label:
        _update_tree_label(state, tree_label)
    _sync_active_tree_state(state)
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


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an image array as PNG bytes for inline display."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        array = to_uint8_image(array)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] not in (3, 4):  # pragma: no cover - defensive
            raise ValueError("Expected 1, 3, or 4 channel image for PNG conversion")

    pil_image = Image.fromarray(array)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_data_url(image: np.ndarray) -> str:
    """Convert an image into a ``data:`` URL for stable inline rendering."""

    png_bytes = _image_to_png_bytes(image)
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _audio_to_data_url(data: bytes, *, mime: str = "audio/wav") -> str:
    """Convert raw audio ``data`` into a self-contained data URL."""

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_image(column: st.delta_generator.DeltaGenerator, image: np.ndarray, caption: str) -> None:
    """Render ``image`` inside ``column`` with a semantic caption."""

    data_url = _image_to_data_url(image)
    alt_text = html.escape(caption)
    caption_html = html.escape(caption).replace("\n", "<br />")
    column.markdown(
        f"""
        <figure style="margin:0;text-align:center;">
          <img src=\"{data_url}\" alt=\"{alt_text}\" style=\"width:100%;height:auto;border-radius:4px;\" />
          <figcaption style=\"font-size:0.8rem;color:var(--text-color,#666);\">{caption_html}</figcaption>
        </figure>
        """,
        unsafe_allow_html=True,
    )


def _render_audio(column: st.delta_generator.DeltaGenerator, data: bytes) -> None:
    """Render ``data`` as an inline HTML5 audio player."""

    audio_url = _audio_to_data_url(bytes(data))
    column.markdown(
        f"<audio controls style=\"width:100%;\">"
        f"<source src=\"{audio_url}\" type=\"audio/wav\" />"
        "</audio>",
        unsafe_allow_html=True,
    )


def _render_download_link(
    container: st.delta_generator.DeltaGenerator,
    *,
    label: str,
    data: bytes | bytearray,
    file_name: str,
    mime: str,
    key: str | None = None,
) -> None:
    """Render a Streamlit-styled download link backed by a data URL.

    Streamlit's :func:`download_button` stores payloads in an in-memory cache that
    is frequently invalidated during rapid reruns. When a rerun happens before the
    browser downloads the file the request fails with noisy ``MediaFileHandler``
    errors. Embedding the payload in a ``data:`` URL sidesteps the cache entirely
    while preserving the download experience.
    """

    if not isinstance(data, (bytes, bytearray)):
        container.caption("Download unavailable: no data generated yet.")
        return

    payload = bytes(data)
    if not payload:
        container.caption("Download unavailable: payload is empty.")
        return

    encoded = base64.b64encode(payload).decode("ascii")
    safe_label = html.escape(label)
    safe_name = html.escape(file_name or "download")
    element_id = html.escape(key) if key else f"download-{uuid4().hex}"

    container.markdown(
        (
            "<a id=\"{element_id}\" "
            "class=\"st-emotion-cache-3a2x2s e1nzilvr5\" "
            "style=\"display:inline-block;padding:0.4rem 0.9rem;margin:0.25rem 0;"
            "text-decoration:none;border-radius:0.5rem;background:#1f2937;color:#f5f6fa;"
            "border:1px solid #374151;font-weight:600;\" "
            "download=\"{safe_name}\" href=\"data:{mime};base64,{encoded}\">"
            "{safe_label}</a>"
        ).format(
            element_id=element_id,
            safe_name=safe_name,
            mime=html.escape(mime or "application/octet-stream"),
            encoded=encoded,
            safe_label=safe_label,
        ),
        unsafe_allow_html=True,
    )


def _reset_widget_key(state: st.session_state, key: str) -> None:
    """Safely clear a widget-managed session key if it exists."""

    reset = getattr(state, "reset_state_value", None)
    if callable(reset):  # pragma: no branch - Streamlit >= 1.32
        try:
            reset(key, None)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "reset_state_value failed for key '%s'", key, exc_info=True
            )

    setter = getattr(state, "_set_widget_state", None)
    if callable(setter):  # pragma: no branch - private Streamlit helper
        try:
            setter(key, None)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("_set_widget_state failed for key '%s'", key, exc_info=True)

    widget_state = getattr(state, "_new_widget_state", None)
    if isinstance(widget_state, dict):  # pragma: no branch - legacy internals
        widget_state.pop(key, None)

    if key in state:
        try:
            del state[key]
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to delete key '%s' directly", key, exc_info=True)
            try:
                state.pop(key, None)
            except Exception:
                logger.debug(
                    "Failed to remove key '%s' using pop", key, exc_info=True
                )


def _migrate_legacy_state(state: st.session_state) -> None:
    """Remove legacy widget-driven keys that conflict with automated controls."""

    legacy_seed = None
    try:
        if "sound_seed" in state:
            legacy_seed = state.get("sound_seed")
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Unable to read legacy sound_seed", exc_info=True)
        legacy_seed = None
    _reset_widget_key(state, "sound_seed")
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
        _reset_widget_key(state, noisy_key)

    bounds_low = state.get("sound_sample_rate_min")
    bounds_high = state.get("sound_sample_rate_max")
    if (
        bounds_low is not None
        and bounds_high is not None
        and "sound_sample_rate_bounds" not in state
    ):
        try:
            state["sound_sample_rate_bounds"] = (int(bounds_low), int(bounds_high))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    _reset_widget_key(state, "sound_sample_rate_min")
    _reset_widget_key(state, "sound_sample_rate_max")

    res_low = state.get("sound_resolution_min")
    res_high = state.get("sound_resolution_max")
    if res_low is not None and res_high is not None and "sound_resolution_bounds" not in state:
        try:
            state["sound_resolution_bounds"] = (int(res_low), int(res_high))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    _reset_widget_key(state, "sound_resolution_min")
    _reset_widget_key(state, "sound_resolution_max")


_SOUND_RESOLUTION_OPTIONS: tuple[int, ...] = (64, 96, 128, 160, 192, 224, 256)
_PERFORMANCE_HISTORY = 60
_PERFORMANCE_CHART_WINDOW = 48
_RECENT_PERFORMANCE = 8
_MAX_GENERATIONS_PER_TICK = 3


def _should_schedule_rerun(
    *,
    generations_ran: int,
    reseeded: bool,
    run_infinite: bool,
    pending_generations: int,
) -> bool:
    """Return ``True`` when the Streamlit app should immediately rerun.

    The previous implementation only considered Keep Improving when no finite work had
    been scheduled for the current tick. If a user switched to Keep Improving while a
    finite backlog was still draining, the rerun condition short-circuited and the UI
    stopped scheduling additional generations once the backlog reached zero. By
    checking the infinite flag independently of the finite queue we guarantee that the
    evolution loop keeps ticking after the queued work completes.
    """

    if generations_ran <= 0:
        return False

    if reseeded:
        return True

    if pending_generations > 0:
        return True

    return run_infinite


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
        logger.debug("CuPy backend check failed", exc_info=True)

    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            backends.append(f"PyTorch CUDA ({name})")
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("PyTorch backend check failed", exc_info=True)

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
    exponent = float(np.clip(np.interp(difficulty, [0.0, 1.0], [2.3, 0.6]), 0.4, 3.0))
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
    base_encoder_sigma: float,
    base_decoder_sigma: float,
    difficulty: float,
    max_overlap: float,
) -> tuple[float, float]:
    """Scale encoder/decoder sigmas based on the adaptive difficulty."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    overlap_push = float(np.clip(max_overlap / 100.0, 0.0, 1.0))
    effective = float(np.clip(0.5 * difficulty + 0.5 * overlap_push, 0.0, 1.0))

    encoder_sigma = base_encoder_sigma * float(
        np.interp(effective, [0.0, 1.0], [0.9, 1.45])
    )
    if base_decoder_sigma <= 0:
        decoder_sigma = 0.0
    else:
        decoder_sigma = max(
            base_decoder_sigma
            * float(np.interp(effective, [0.0, 1.0], [1.1, 0.55])),
            0.05,
        )
    return float(encoder_sigma), float(decoder_sigma)


def _adaptive_sample_bounds(
    difficulty: float,
    previous: tuple[int, int] | None,
    improvement: float = 0.0,
    volatility: float = 0.0,
    max_overlap: float = 0.0,
) -> tuple[int, int]:
    """Derive a difficulty-weighted sample-rate window with gentle inertia."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))
    overlap_push = float(np.clip(max_overlap / 100.0, 0.0, 1.0))

    effective = float(np.clip(0.55 * difficulty + 0.45 * overlap_push, 0.0, 1.0))

    base_low = float(np.interp(effective, [0.0, 1.0], [16_000, 52_000]))
    base_high = float(np.interp(effective, [0.0, 1.0], [28_000, 96_000]))
    spread = float(np.interp(effective, [0.0, 1.0], [4_000, 36_000]))

    spread *= float(np.clip(1.0 + 0.9 * improvement - 0.7 * volatility, 0.45, 1.9))

    rng = np.random.default_rng()
    jitter_low = float((rng.random() - 0.5) * spread * 0.7)
    jitter_high = float((rng.random() - 0.5) * spread)

    low = int(np.clip(base_low + jitter_low, 8_000, 96_000))
    high = int(np.clip(base_high + jitter_high, low + 1_000, 96_000))

    if previous is not None:
        prev_low, prev_high = int(previous[0]), int(previous[1])
        blend = 0.55
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
    max_overlap: float = 0.0,
) -> tuple[int, int]:
    """Unlock larger image sizes as difficulty progresses."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))
    overlap_push = float(np.clip(max_overlap / 100.0, 0.0, 1.0))

    effective = float(np.clip(0.5 * difficulty + 0.5 * overlap_push, 0.0, 1.0))

    options = sorted(_SOUND_RESOLUTION_OPTIONS)
    unlocked = max(1, min(len(options), int(np.floor(effective * len(options))) + 1))

    if improvement > 0.05 and unlocked < len(options):
        unlocked += 1

    rng = np.random.default_rng()
    if unlocked < len(options) and rng.random() > 0.82:
        unlocked = min(len(options), unlocked + 1)
    if volatility > 0.1 and unlocked > 1:
        unlocked -= 1

    lower = options[0]
    upper = options[unlocked - 1]

    if previous is not None:
        prev_low, prev_high = int(previous[0]), int(previous[1])
        lower = int(np.clip(0.6 * prev_low + 0.4 * lower, options[0], options[-1]))
        upper = int(np.clip(0.6 * prev_high + 0.4 * upper, lower, options[-1]))

    return lower, upper


def _update_noise_bases(
    state: st.session_state,
    difficulty: float,
    improvement: float,
    volatility: float,
    max_overlap: float,
) -> None:
    """Gently steer encoder/decoder noise levels in response to difficulty."""

    rng = np.random.default_rng()
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    improvement = float(np.clip(improvement, 0.0, 1.0))
    volatility = float(max(0.0, volatility))
    overlap_push = float(np.clip(max_overlap / 100.0, 0.0, 1.0))

    effective = float(np.clip(0.55 * difficulty + 0.45 * overlap_push, 0.0, 1.0))

    prev_encoder = float(state.get("encoder_sigma_base", 0.2))
    exploration_gain = 1.0 + 0.9 * improvement
    stability_pull = 1.0 - min(0.55, 1.6 * volatility)
    target_encoder = float(
        np.interp(effective, [0.0, 1.0], [0.14, 0.8]) * exploration_gain * stability_pull
    )
    encoder_jitter = float(
        rng.normal(0.0, 0.015 + 0.05 * effective + 0.03 * improvement)
    )
    encoder_sigma = float(np.clip(target_encoder + encoder_jitter, 0.05, 0.9))
    state["encoder_sigma_base"] = float(0.6 * prev_encoder + 0.4 * encoder_sigma)

    prev_decoder = float(state.get("decoder_sigma_base", 1.0))
    denoise_floor = 0.12
    denoise_target = float(np.interp(effective, [0.0, 1.0], [1.45, denoise_floor]))
    denoise_target *= float(np.clip(1.0 - 0.7 * improvement, 0.35, 1.0))
    denoise_target *= float(np.clip(1.0 + 0.9 * volatility, 0.45, 1.5))
    decoder_jitter = float(
        rng.normal(0.0, 0.04 + 0.05 * (1.0 - effective) + 0.03 * volatility)
    )
    decoder_sigma = float(np.clip(denoise_target + decoder_jitter, denoise_floor, 2.5))
    state["decoder_sigma_base"] = float(0.6 * prev_decoder + 0.4 * decoder_sigma)


def _record_performance_history(
    state: st.session_state,
    ai_overlap: float,
    ai_ssim: float,
    ai_psnr: float,
    sound_overlap: float,
    *,
    sound_reference_overlap: float | None = None,
) -> list[dict[str, float]]:
    """Track recent reconstruction metrics for adaptive scheduling."""

    history: list[dict[str, float]] = list(state.get("performance_history", []))
    observation = int(state.get("_performance_observation", 0)) + 1
    state["_performance_observation"] = observation
    record = {
        "ai_overlap": float(ai_overlap),
        "ai_ssim": float(ai_ssim),
        "ai_psnr": float(ai_psnr),
        "sound_overlap": float(sound_overlap),
        "step": float(observation),
    }
    if sound_reference_overlap is not None:
        record["sound_reference_overlap"] = float(sound_reference_overlap)
    history.append(record)
    if len(history) > _PERFORMANCE_HISTORY:
        history = history[-_PERFORMANCE_HISTORY:]
    earliest_step = float(history[0].get("step", 1.0)) if history else 1.0
    markers: list[int] = list(state.get("_sound_target_markers", []))
    if markers:
        state["_sound_target_markers"] = [
            int(marker) for marker in markers if marker >= earliest_step
        ]
    state["performance_history"] = history
    return history


def _append_global_progress_row(
    state: st.session_state,
    manager: EvolutionManager,
    row: Mapping[str, Any],
) -> None:
    """Persist a generation summary in a cumulative timeline across runs."""

    history: list[dict[str, Any]] = list(state.get("_global_progress_history", []))
    index_map: dict[str, int] = dict(state.get("_global_progress_index", {}))

    try:
        run_generation = int(row.get("Generation", len(history)))
    except (TypeError, ValueError):
        run_generation = len(history)

    progress_key = f"{manager.run_id}:{run_generation}"

    base_entry: dict[str, Any] = dict(row)
    base_entry["_run_id"] = manager.run_id
    base_entry["_run_generation"] = run_generation
    base_entry["_run_seed"] = int(getattr(manager, "base_seed", 0))

    existing_index = index_map.get(progress_key)
    if existing_index is not None and 0 <= existing_index < len(history):
        preserved_step = history[existing_index].get("Generation", existing_index + 1)
        merged = dict(history[existing_index])
        merged.update(base_entry)
        merged["Generation"] = preserved_step
        history[existing_index] = merged
        state["_global_progress_history"] = history
        state["_global_progress_index"] = index_map
        return
    if existing_index is not None:
        index_map.pop(progress_key, None)

    global_step = int(state.get("_global_progress_step", 0)) + 1
    state["_global_progress_step"] = global_step

    base_entry["Generation"] = float(global_step)
    history.append(base_entry)
    index_map[progress_key] = len(history) - 1

    state["_global_progress_history"] = history
    state["_global_progress_index"] = index_map


def _mark_sound_target_transition(state: st.session_state) -> None:
    """Record the observation index where the sound target changed."""

    observation = int(state.get("_performance_observation", 0))
    if observation <= 0:
        return

    markers: list[int] = list(state.get("_sound_target_markers", []))
    markers.append(observation)
    if len(markers) > _PERFORMANCE_HISTORY:
        markers = markers[-_PERFORMANCE_HISTORY:]
    state["_sound_target_markers"] = markers


def _derive_difficulty_metrics(
    history: list[dict[str, float]]
) -> tuple[float, float, float, float]:
    """Compute difficulty progress, improvement, volatility, and reward signals."""

    if not history:
        return 0.0, 0.0, 0.0, 0.0

    overlaps = np.asarray([entry["ai_overlap"] for entry in history], dtype=np.float32)
    recent_window = int(min(len(history), _RECENT_PERFORMANCE))
    recent = overlaps[-recent_window:]
    recent_mean = float(np.mean(recent)) / 100.0
    best = float(np.max(overlaps)) / 100.0
    long_term = float(np.mean(overlaps[:-recent_window])) / 100.0 if len(history) > recent_window else recent_mean
    improvement = float(np.clip(recent_mean - long_term, 0.0, 1.0))
    volatility = float(np.std(recent) / 100.0)

    normalized = np.clip(overlaps / 100.0, 0.0, 1.0)
    reward_curve = np.clip((normalized - 0.4) / 0.6, 0.0, 1.0)
    reward_recent = float(np.mean(reward_curve[-recent_window:])) if recent_window else 0.0
    reward_peak = float(np.max(reward_curve)) if reward_curve.size else 0.0
    reward_signal = float(np.clip(0.65 * reward_recent + 0.35 * reward_peak, 0.0, 1.0))

    coverage = float(min(len(history) / _PERFORMANCE_HISTORY, 1.0))
    difficulty_target = float(
        np.clip(
            0.35 * best
            + 0.3 * recent_mean
            + 0.15 * coverage
            + 0.3 * improvement
            + 0.25 * reward_signal,
            0.0,
            1.0,
        )
    )
    return difficulty_target, improvement, float(np.clip(volatility, 0.0, 1.0)), reward_signal


def _refresh_sound_scene(
    state: st.session_state,
    difficulty: float,
    target_dwell: int,
    *,
    record_event: bool = True,
    improvement: float = 0.0,
    volatility: float = 0.0,
    max_overlap: float = 0.0,
) -> tuple[int, int, int]:
    """Randomise the sound target and associated hyper-parameters."""

    # Ensure any lingering legacy widget keys are purged before mutating state.
    _reset_widget_key(state, "sound_seed")

    sample_bounds = _adaptive_sample_bounds(
        difficulty,
        state.get("sound_sample_rate_bounds"),
        improvement,
        volatility,
        max_overlap,
    )
    resolution_bounds = _adaptive_resolution_bounds(
        difficulty,
        state.get("sound_resolution_bounds"),
        improvement,
        volatility,
        max_overlap,
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
    state["shared_seed"] = new_shared_seed

    if record_event:
        state["sound_reseed_count"] = int(state.get("sound_reseed_count", 0) + 1)
    else:
        state.setdefault("sound_reseed_count", 0)

    _update_noise_bases(state, difficulty, improvement, volatility, max_overlap)
    return new_sound_seed, int(sample_rate), int(resolution)


def _session_export_payload(
    *,
    state: st.session_state,
    manager: EvolutionManager,
    metrics: ReconstructionMetrics,
    sound_metrics: ReconstructionMetrics,
    sound_reference_metrics: ReconstructionMetrics,
    ai_sound_alignment: ReconstructionMetrics,
    ai_overlap_score: float,
    sound_overlap_score: float,
    sound_reference_overlap: float,
    sound_clip: Any,
    base_encoder_sigma: float,
    base_decoder_sigma: float,
    encoder_sigma: float,
    denoise_sigma: float,
    current_sample_rate: int,
    current_resolution: int,
    sample_rate_range: tuple[int, int],
    resolution_range: tuple[int, int],
    seed: int,
    sound_seed: int,
    target_dwell: int,
    best_candidate_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the export payload used by the session snapshot."""

    hardware_backend = state.get("hardware_backend", "CPU (NumPy)")

    if manager.generations:
        last_generation = manager.generations[-1]
        difficulty_raw = float(last_generation.difficulty_raw)
        difficulty_target = float(np.clip(last_generation.difficulty_level, 0.0, 1.0))
    else:
        difficulty_raw = float(state.get("latest_generation_difficulty_raw", 0.0))
        difficulty_target = float(
            np.clip(state.get("latest_generation_difficulty", 0.0), 0.0, 1.0)
        )
    difficulty_normalized = normalize_difficulty(difficulty_raw)

    config_snapshot = {
        "population_size": manager.population_size,
        "autosave_interval": manager.autosave_interval,
        "encoder_sigma": float(encoder_sigma),
        "decoder_sigma": float(denoise_sigma),
        "base_encoder_sigma": float(base_encoder_sigma),
        "base_decoder_sigma": float(base_decoder_sigma),
        "sample_rate": int(current_sample_rate),
        "resolution": int(current_resolution),
        "difficulty_target": difficulty_target,
        "shared_seed": int(seed),
        "sound_seed": int(sound_seed),
    }

    config_hash_value = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()
    provenance = collect_provenance(config_hash=f"sha256:{config_hash_value}")
    provenance["random_seeds"] = {"session": int(seed), "sound": int(sound_seed)}

    best_psnr = float(metrics.psnr)
    best_ssim = float(metrics.ssim)
    if best_candidate_summary is not None:
        best_psnr = float(best_candidate_summary.get("psnr", best_psnr))
        best_ssim = float(best_candidate_summary.get("ssim", best_ssim))

    metrics_block = {
        "ai_vs_reference": metrics.as_dict(),
        "sound_vs_ai": sound_metrics.as_dict(),
        "sound_vs_reference": sound_reference_metrics.as_dict(),
        "ai_vs_sound": ai_sound_alignment.as_dict(),
        "overlap": {
            "ai_vs_reference": float(ai_overlap_score),
            "sound_vs_ai": float(sound_overlap_score),
            "sound_vs_reference": float(sound_reference_overlap),
        },
        "global_pooled": {
            "psnr": float(metrics.psnr),
            "ssim": float(metrics.ssim),
            "desc": "pooled/global comparator",
        },
        "per_candidate_strict": {
            "psnr": best_psnr,
            "ssim": best_ssim,
            "desc": "gallery/best-candidate strict comparator",
        },
    }

    payload = {
        "hardware_backend": hardware_backend,
        "difficulty": {
            "current": float(state.get("difficulty_progress", 0.0)),
            "max_overlap": float(state.get("max_overlap_seen", 0.0)),
            "sound_reseed_count": int(state.get("sound_reseed_count", 0)),
            "dwell_generations": int(target_dwell),
            "score_budget": float(state.get("sound_target_score", 0.0)),
            "score_remaining": float(state.get("sound_score_remaining", 0.0)),
            "momentum": float(state.get("difficulty_improvement", 0.0)),
            "volatility": float(state.get("difficulty_volatility", 0.0)),
            "raw": difficulty_raw,
            "normalized": difficulty_normalized,
            "target": difficulty_target,
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
            "band_volumes": dict(getattr(sound_clip, "band_volumes", {})),
            "generator_seed": int(getattr(sound_clip, "seed", sound_seed)),
            "generator_sample_rate": int(
                getattr(sound_clip, "sample_rate", current_sample_rate)
            ),
        },
        "metrics": metrics_block,
        "performance_history": list(state.get("performance_history", [])),
        "manager": {
            "population_size": int(manager.population_size),
            "autosave_interval": int(manager.autosave_interval),
            "generation_count": len(manager.generations),
            "best_candidate": best_candidate_summary,
            "mutation_boost": int(manager.mutation_boost),
        },
        "provenance": provenance,
    }
    if manager.hyper_profile.enabled:
        profile = manager.hyper_profile
        payload["hyper_performance"] = {
            "target_subjects": int(profile.target_subjects),
            "batch_size": int(profile.batch_size),
            "dwell_generations": int(profile.dwell_generations),
            "queue_generations": int(profile.queue_generations),
            "autosave_interval": int(profile.autosave_interval),
            "mean_generation_duration": float(profile.mean_duration),
            "subjects_per_second": float(profile.throughput),
        }
    return payload


def _build_export_bundle(payload: dict[str, Any], progress_rows: list[dict[str, Any]]) -> bytes:
    """Create a zipped export containing session metrics and progress curves.

    Returns raw bytes so downloads don't rely on Streamlit's transient media storage.
    """

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session_summary.json", json.dumps(payload, indent=2))
        if progress_rows:
            export_df = (
                pd.DataFrame(progress_rows)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how="all")
            )
            if not export_df.empty:
                archive.writestr("generation_progress.csv", export_df.to_csv(index=False))
    buffer.seek(0)
    return buffer.getvalue()


def run() -> None:
    """Entry-point for the Streamlit application."""
    _ensure_page_config()
    # Streamlit options: show detailed errors and reduce external telemetry
    try:
        st.set_option("client.showErrorDetails", True)
        st.set_option("browser.gatherUsageStats", False)
    except Exception:
        logger.debug("Failed to configure Streamlit options", exc_info=True)

    log_dir = configure_logging()
    state = st.session_state
    if state.get("_umbra_log_directory") != str(log_dir):
        state["_umbra_log_directory"] = str(log_dir)
        logger.info("Logging configured; writing UI diagnostics to %s", log_dir)

    run_id = state.get("_umbra_run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = uuid4().hex
        state["_umbra_run_id"] = run_id
    run_paths = ensure_run_paths(run_id)
    state["_umbra_run_directory"] = str(run_paths.root)
    state["_umbra_charts_directory"] = str(run_paths.charts)
    chart_files = state.get("_umbra_chart_files")
    if not isinstance(chart_files, dict):
        chart_files = {}
        state["_umbra_chart_files"] = chart_files

    st.title("Project Umbra · Compact evolution console")
    st.caption(
        "Difficulty steers every setting automatically—press start and let the system tune itself."
    )

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
    if "run_store" not in state:
        state["run_store"] = RunStore(Path("runs"))
    state.setdefault("active_run_id", None)
    if "shared_seed" not in state:
        state["shared_seed"] = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
    state.setdefault("encoder_sigma_base", 0.2)
    state.setdefault("decoder_sigma_base", 1.0)
    state.setdefault("_performance_observation", 0)
    state.setdefault("_sound_target_markers", [])
    state.setdefault("metrics_autofollow_toggle", True)
    if "active_sound_seed" not in state:
        state["active_sound_seed"] = int(
            np.random.default_rng().integers(0, np.iinfo(np.int32).max)
        )
    state.setdefault("sound_target_dwell", 10)
    state.setdefault("last_sound_target_dwell", int(state["sound_target_dwell"]))
    state.setdefault("sound_target_score", 0.0)
    state.setdefault("sound_score_remaining", float(state.get("sound_target_score", 0.0)))
    state.setdefault("sound_reseed_count", 0)
    state.setdefault("difficulty_progress", 0.0)
    state.setdefault("max_overlap_seen", 0.0)
    state.setdefault("performance_history", [])
    state.setdefault("difficulty_improvement", 0.0)
    state.setdefault("difficulty_volatility", 0.0)
    state.setdefault("difficulty_reward", 0.0)
    state.setdefault("difficulty_reward_points", 0.0)
    state.setdefault("easy_mode", True)
    state.setdefault("_easy_mode_defaults_applied", False)
    state.pop("show_advanced_controls", None)
    state.setdefault("latest_generation_reward", 0.0)
    state.setdefault("latest_generation_peak", 0.0)
    state.setdefault("latest_generation_difficulty", 0.0)
    state.setdefault("latest_generation_difficulty_raw", 0.0)
    state.setdefault("latest_generation_improvement", 0.0)
    state.setdefault("latest_lifetime_reward", 0.0)
    state.setdefault("hardware_backend", _detect_hardware_backend())
    state.setdefault("evolution_trees", {})
    state.setdefault("active_parent_seeds", [])
    if state.get("active_tree_id") not in state["evolution_trees"]:
        state.pop("active_tree_id", None)
    state.setdefault("_active_image_model", next(iter(_IMAGE_MODEL_PRESETS)))
    state.setdefault("_active_sound_model", next(iter(_SOUND_MODEL_PRESETS)))
    state.setdefault("_active_evolution_engine", "Lineage & noise (classic)")

    hyper_enabled = EvolutionManager.hyper_mode_enabled()
    hyper_profile: HyperPerformanceProfile | None = None
    if hyper_enabled:
        cached_profile = state.get("_hyper_profile_cache")
        if isinstance(cached_profile, HyperPerformanceProfile):
            hyper_profile = cached_profile
        else:
            hyper_profile = EvolutionManager.default_hyper_profile()
        state["_hyper_profile_cache"] = hyper_profile
        population_baseline = int(
            max(1, hyper_profile.batch_size or state.get("population_size", 5))
        )
        autosave_baseline = int(
            max(1, hyper_profile.autosave_interval or state.get("autosave_interval", 5))
        )
        dwell_baseline = int(
            max(1, hyper_profile.dwell_generations or state.get("sound_target_dwell", 10))
        )
        queue_baseline = int(
            max(1, hyper_profile.queue_generations or dwell_baseline)
        )
        state["population_size"] = population_baseline
        state["autosave_interval"] = autosave_baseline
        state["generations_to_queue"] = queue_baseline
        state["sound_target_dwell"] = dwell_baseline
        state["last_sound_target_dwell"] = dwell_baseline
        state["evolution_mode"] = "Finite"
    else:
        hyper_profile = None

    difficulty_progress = float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0))
    quick_tab, demo_tab, tune_tab, progress_tab = st.tabs(
        ["Quick Start", "Demo Lab", "Tune It", "Watch Progress"]
    )
    (
        easy_mode,
        run_button,
        stop_button,
        reset_button,
        save_button,
        reload_button,
        auto_settings,
    ) = _render_control_panel(
        state,
        quick_tab,
        difficulty_progress=difficulty_progress,
        hyper_profile=hyper_profile,
    )
    state["_latest_auto_settings"] = auto_settings
    show_advanced = bool(state.get("show_advanced_controls", not easy_mode))

    _render_demo_lab(state, demo_tab)

    population_size = int(state.get("population_size", auto_settings["population_size"]))
    generations_to_queue = int(
        state.get("generations_to_queue", auto_settings["generations_to_queue"])
    )
    autosave_interval = int(state.get("autosave_interval", auto_settings["autosave_interval"]))
    max_overlap_so_far = float(np.clip(state.get("max_overlap_seen", 0.0), 0.0, 100.0))

    advanced_mode = not easy_mode
    autosave_dir = Path(state.get("autosave_dir", str(DEFAULT_AUTOSAVE_DIR))).expanduser()
    if advanced_mode:
        with st.sidebar.expander("Session & autosave", expanded=False):
            st.text_input(
                "Autosave directory",
                value=str(autosave_dir),
                help="Evolution checkpoints are saved here as evolution_state.pkl.",
                key="autosave_dir",
            )
            if st.button("Load autosave"):
                autosave_dir = Path(state.get("autosave_dir", str(DEFAULT_AUTOSAVE_DIR))).expanduser()
                state["autosave_checked"] = True
                _attempt_autoload(autosave_dir)

    autosave_dir = Path(state.get("autosave_dir", str(DEFAULT_AUTOSAVE_DIR))).expanduser()
    normalized_autosave_dir = str(autosave_dir)
    if state.get("last_autosave_dir") != normalized_autosave_dir:
        state["last_autosave_dir"] = normalized_autosave_dir
        state["autosave_checked"] = False

    autosave_path = _autosave_path(autosave_dir)
    if not state.get("autosave_checked"):
        if autosave_path.exists():
            _attempt_autoload(autosave_dir)
        state["autosave_checked"] = True

    image_model_names = list(_IMAGE_MODEL_PRESETS.keys())
    current_image_model = state.get("_active_image_model", image_model_names[0])
    selected_image_model = current_image_model
    sound_model_names = list(_SOUND_MODEL_PRESETS.keys())
    current_sound_model = state.get("_active_sound_model", sound_model_names[0])
    selected_sound_model = current_sound_model
    evolution_engines = ["Lineage & noise (classic)", "Neural lineage fusion"]
    selected_engine = state.get("_active_evolution_engine", evolution_engines[0])

    if advanced_mode:
        with st.sidebar.expander("Model presets", expanded=False):
            engine_index = (
                evolution_engines.index(selected_engine)
                if selected_engine in evolution_engines
                else 0
            )
            selected_image_model = st.selectbox(
                "Image evolution preset",
                image_model_names,
                index=image_model_names.index(current_image_model),
                key="image_model_select",
            )
            st.caption(_IMAGE_MODEL_PRESETS[selected_image_model]["description"])
            selected_sound_model = st.selectbox(
                "Sound scene preset",
                sound_model_names,
                index=sound_model_names.index(current_sound_model),
                key="sound_model_select",
            )
            st.caption(_SOUND_MODEL_PRESETS[selected_sound_model]["description"])
            selected_engine = st.selectbox(
                "Evolution engine",
                evolution_engines,
                index=engine_index,
                key="evolution_engine_select",
            )
            if selected_engine == "Neural lineage fusion":
                st.caption(
                    "Neural advisor boosts high performers while preserving elite lineages for deep selective breeding."
                )
            else:
                st.caption(
                    "Classic stochastic evolution that prioritises lineage parents and their direct noise descendants."
                )

    if state.get("_active_image_model") != selected_image_model:
        preset = _IMAGE_MODEL_PRESETS[selected_image_model]
        state["_active_image_model"] = selected_image_model
        state["encoder_sigma_base"] = float(preset["encoder_sigma"])
        state["decoder_sigma_base"] = float(preset["decoder_sigma"])

    if state.get("_active_sound_model") != selected_sound_model:
        preset = _SOUND_MODEL_PRESETS[selected_sound_model]
        state["_active_sound_model"] = selected_sound_model
        state["sound_sample_rate_bounds"] = tuple(int(v) for v in preset["sample_rate"])
        state["sound_resolution_bounds"] = tuple(int(v) for v in preset["resolution"])
        state["sound_target_dwell"] = int(preset["target_dwell"])
        state["last_sound_target_dwell"] = int(preset["target_dwell"])
        sound_budget_settings = {
            "target_dwell": int(preset["target_dwell"]),
            "difficulty_target": float(auto_settings.get("difficulty_target", difficulty_progress)),
        }
        _ensure_sound_score_budget(
            state,
            sound_budget_settings,
            difficulty_progress=difficulty_progress,
            force=True,
        )

    if state.get("_active_evolution_engine") != selected_engine:
        state["_active_evolution_engine"] = selected_engine

    target_dwell = int(state.get("sound_target_dwell", auto_settings["target_dwell"]))
    if hyper_enabled and hyper_profile is not None:
        target_dwell = int(
            max(1, hyper_profile.dwell_generations or target_dwell)
        )
        state["sound_target_dwell"] = target_dwell
        state["last_sound_target_dwell"] = target_dwell
    elif not advanced_mode:
        target_dwell = int(auto_settings["target_dwell"])
        state["sound_target_dwell"] = target_dwell
        state["last_sound_target_dwell"] = target_dwell

    if advanced_mode:
        with st.sidebar.expander("Sound cadence", expanded=False):
            if hyper_enabled and hyper_profile is not None:
                subjects_per_cycle = int(
                    max(
                        hyper_profile.target_subjects,
                        target_dwell * max(state.get("population_size", 1), 1),
                    )
                )
                st.metric("Generations per sound target", target_dwell)
                st.metric("Subjects scheduled this cycle", subjects_per_cycle)
                st.caption(
                    "Hyper performance mode rotates the sound scene automatically once the dwell window completes."
                )
            else:
                target_dwell = int(
                    st.number_input(
                        "Generations per sound target",
                        min_value=1,
                        max_value=500,
                        value=int(state.get("sound_target_dwell", target_dwell)),
                        step=1,
                        key="sound_target_dwell_input",
                        help=(
                            "Number of evolution steps to spend matching the current sound-derived image before refreshing it with a new randomised scene."
                        ),
                    )
                )
                state["sound_target_dwell"] = target_dwell
            manual_refresh = st.button(
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
                    max_overlap=float(state.get("max_overlap_seen", max_overlap_so_far)),
                )
                st.info(
                    "Forced refresh triggered new scene "
                    f"(seed {new_seed}, {new_rate:,} Hz, {new_resolution}×{new_resolution} px)."
                )

    if state.get("last_sound_target_dwell") != target_dwell:
        state["last_sound_target_dwell"] = target_dwell

    sound_budget_settings = dict(auto_settings)
    sound_budget_settings["target_dwell"] = target_dwell
    target_score = _ensure_sound_score_budget(
        state,
        sound_budget_settings,
        difficulty_progress=difficulty_progress,
    )

    if "sound_sample_rate_bounds" not in state or "sound_resolution_bounds" not in state:
        _refresh_sound_scene(
            state,
            difficulty_progress,
            target_dwell,
            record_event=False,
            improvement=float(state.get("difficulty_improvement", 0.0)),
            volatility=float(state.get("difficulty_volatility", 0.0)),
            max_overlap=max_overlap_so_far,
        )
        target_score = _ensure_sound_score_budget(
            state,
            sound_budget_settings,
            difficulty_progress=difficulty_progress,
            force=True,
        )

    if "sound_generations_left" not in state:
        state["sound_generations_left"] = target_dwell
    elif state.get("sound_generations_left", target_dwell) > target_dwell:
        state["sound_generations_left"] = target_dwell
    remaining_before = int(state.get("sound_generations_left", target_dwell))
    remaining_after = remaining_before

    forest = state.setdefault("evolution_trees", {})
    new_tree_requested = False
    if advanced_mode:
        with st.sidebar.expander("Evolution trees", expanded=False):
            if forest:
                tree_ids = list(forest.keys())
                active_tree_id = state.get("active_tree_id", tree_ids[0])
                if active_tree_id not in forest:
                    active_tree_id = tree_ids[0]
                    _activate_tree(state, active_tree_id)
                selected_tree_id = st.selectbox(
                    "Stored branches",
                    tree_ids,
                    index=tree_ids.index(active_tree_id),
                    format_func=lambda tid: _format_tree_label(forest[tid]),
                )
                if selected_tree_id != active_tree_id:
                    _activate_tree(state, selected_tree_id)
                    forest = state.setdefault("evolution_trees", {})
                st.caption(
                    "Switch between stored evolution branches to revisit previous sound scenes."
                )
            else:
                st.caption("Branches will appear here after your first evolution run.")

            new_tree_requested = st.button(
                "Create new evolution tree",
                key="spawn_new_tree",
                help="Start a fresh branch with the current presets and sound scene.",
            )
    if new_tree_requested:
        state["_spawn_new_tree_requested"] = True
        state["pending_generations"] = 0
        state["run_infinite"] = False
        _sync_active_tree_state(state)
        if advanced_mode:
            st.sidebar.info("Queued a new tree; it will initialise on the next evolution tick.")

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
        base_encoder_sigma,
        base_decoder_sigma,
        difficulty_progress,
        float(state.get("max_overlap_seen", max_overlap_so_far)),
    )
    state["active_encoder_sigma"] = encoder_sigma
    state["active_decoder_sigma"] = denoise_sigma

    state["sound_score_remaining"] = float(
        max(0.0, state.get("sound_score_remaining", float(target_score)))
    )
    score_remaining_before = float(
        state.get("sound_score_remaining", float(target_score))
    )

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

    if advanced_mode:
        with st.sidebar.expander("Active configuration", expanded=False):
            st.metric("Hardware backend", state.get("hardware_backend", "CPU (NumPy)"))
            st.metric("Shared seed", str(seed))
            st.metric("Sound seed", str(sound_seed))
            st.metric(
                "Sound sample window", f"{sample_rate_range[0]:,}–{sample_rate_range[1]:,} Hz"
            )
            st.metric(
                "Image resolution window",
                f"{resolution_range[0]}–{resolution_range[1]} px",
            )
            st.metric("Active sample rate", f"{current_sample_rate:,} Hz")
            st.metric(
                "Active image resolution",
                f"{current_resolution}×{current_resolution} px",
            )

            noise_cols = st.columns(2)
            noise_cols[0].metric("Active encoder σ", f"{encoder_sigma:.3f}")
            noise_cols[1].metric("Active denoise σ", f"{denoise_sigma:.3f}")
            st.caption(
                "Adaptive noise scales increase encoder randomness while tempering decoder blur as the system improves."
            )
    _sync_active_tree_state(state)

    custom_media_sources = {"upload", "pinterest"}
    if (
        state.get("quick_start_media_source") in custom_media_sources
        and isinstance(state.get("quick_start_reference_image"), np.ndarray)
    ):
        uploaded_image = np.asarray(state.get("quick_start_reference_image"), dtype=np.float32)
        original_color = np.clip(uploaded_image, 0.0, 1.0)
        try:
            original = _rgb_to_grayscale(original_color)
        except ValueError:
            original = np.asarray(np.mean(original_color, axis=-1), dtype=np.float32)
        band_volumes = {
            "red": float(np.mean(original_color[..., 0])),
            "green": float(np.mean(original_color[..., 1])),
            "blue": float(np.mean(original_color[..., 2])),
        }
        sound_clip = SyntheticSound(
            seed=int(sound_seed),
            sample_rate=int(current_sample_rate),
            samples=np.zeros(1, dtype=np.float32),
            band_volumes=band_volumes,
        )
        shape_specs = []
        default_label = (
            "Uploaded image"
            if state.get("quick_start_media_source") == "upload"
            else "Pinterest inspiration"
        )
        source_label = state.get("quick_start_reference_label", default_label)
    else:
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
    _, sound_reference_overlap = multiplicative_overlap(original, sound_reconstruction)
    sound_reference_overlap_color = colorize_comparison(original, sound_reconstruction)
    _, sound_overlap_score = multiplicative_overlap(reconstructed, sound_reconstruction)
    sound_overlap_color = colorize_comparison(reconstructed, sound_reconstruction)

    metrics = compute_metrics(colored_original, ai_colored)
    ai_sound_alignment = compute_metrics(ai_colored, sound_colored)
    sound_reference_metrics = compute_metrics(colored_original, sound_colored)
    sound_metrics = ai_sound_alignment

    with progress_tab:
        st.subheader("Sound profile")
        volume_cols = st.columns(3)
        band_volumes = getattr(sound_clip, "band_volumes", {})
        if not isinstance(band_volumes, MutableMapping):
            band_volumes = {}
        for idx, color in enumerate(("red", "green", "blue")):
            volume_cols[idx].metric(
                f"{color.title()} volume",
                f"{float(band_volumes.get(color, 0.0)):.2f}",
                help="Relative energy detected in the sound clip for this colour band.",
            )
        samples = getattr(sound_clip, "samples", np.zeros(0, dtype=np.float32))
        sample_count = samples.size if isinstance(samples, np.ndarray) else 0
        sample_rate_display = int(getattr(sound_clip, "sample_rate", current_sample_rate))
        st.caption(
            f"Waveform length: {sample_count} samples @ {sample_rate_display} Hz."
        )

        st.subheader("Reconstruction quality")
        ai_metrics_cols = st.columns(3)
        with ai_metrics_cols[0]:
            _render_metric_visual(
                "How clear the AI picture looks",
                float(metrics.psnr),
                value_display=f"{metrics.psnr:.2f} dB",
                scale_min=10.0,
                scale_max=50.0,
                good_range=(28.0, 35.0),
                caption="Aim for 30 dB or more for a crisp-looking render.",
                tooltip="PSNR compares brightness differences; higher numbers mean less noise.",
            )
        with ai_metrics_cols[1]:
            _render_metric_visual(
                "How closely the AI matches fine details",
                float(metrics.ssim),
                value_display=f"{metrics.ssim:.3f}",
                scale_min=0.0,
                scale_max=1.0,
                good_range=(0.75, 0.9),
                caption="Scores near 1.0 mean the textures and edges line up well.",
                tooltip="SSIM checks structure and contrast; 1.0 is a perfect match.",
            )
        with ai_metrics_cols[2]:
            _render_metric_visual(
                "How much the AI shapes line up",
                float(ai_overlap_score),
                value_display=f"{ai_overlap_score:.1f}%",
                scale_min=0.0,
                scale_max=100.0,
                good_range=(50.0, 70.0),
                caption="Above 60% means the main forms overlap reliably.",
                tooltip="Overlap highlights shared bright areas between original and AI images.",
            )

        if state.get("adversarial_enabled", False):
            adv: AdversarialManager | None = state.get("adversarial")
            if adv is None:
                adv = AdversarialManager()
                state["adversarial"] = adv
            pred_image = apply_generator(original, adv.state.generator)
            _, pred_overlap_score = multiplicative_overlap(original, pred_image)
            gen, best_score, dec_sigma = adv.step(original, reconstructed)
            state["decoder_sigma_base"] = dec_sigma
            st.subheader("Adversarial generator")
            gen_cols = st.columns(4)
            gen_cols[0].metric("Gen blur σ", f"{gen.blur_sigma:.2f}")
            gen_cols[1].metric("Gen contrast", f"{gen.contrast:.2f}")
            gen_cols[2].metric("Gen brightness", f"{gen.brightness:.2f}")
            gen_cols[3].metric("Gen score", f"{best_score:.3f}")
            st.metric("Predicted overlap", f"{pred_overlap_score:.1f}%")

        sound_metrics_cols = st.columns(3)
        with sound_metrics_cols[0]:
            _render_metric_visual(
                "How closely the sound picture tracks the AI",
                float(sound_metrics.psnr),
                value_display=f"{sound_metrics.psnr:.2f} dB",
                scale_min=10.0,
                scale_max=50.0,
                good_range=(26.0, 32.0),
                caption="Higher PSNR means the sound-driven reconstruction preserves the AI's brightness cues.",
                tooltip="PSNR between the sound-only reconstruction and the AI reconstruction.",
            )
        with sound_metrics_cols[1]:
            _render_metric_visual(
                "How closely sound details follow the AI",
                float(sound_metrics.ssim),
                value_display=f"{sound_metrics.ssim:.3f}",
                scale_min=0.0,
                scale_max=1.0,
                good_range=(0.75, 0.9),
                caption="Staying near 0.9 means the audio reconstruction mirrors the AI's structure.",
                tooltip="SSIM between the sound-only reconstruction and the AI reconstruction.",
            )
        with sound_metrics_cols[2]:
            _render_metric_visual(
                "How much the sound shapes line up with the AI",
                float(sound_overlap_score),
                value_display=f"{sound_overlap_score:.1f}%",
                scale_min=0.0,
                scale_max=100.0,
                good_range=(60.0, 85.0),
                caption="Higher overlap means the sonic cues align with the AI prediction.",
                tooltip="Overlap measures shared highlights between the sound-only and AI reconstructions.",
            )

        _render_metric_visual(
            "How similar the AI and sound colours feel",
            float(ai_sound_alignment.ssim),
            value_display=f"{ai_sound_alignment.ssim:.3f}",
            scale_min=0.0,
            scale_max=1.0,
            good_range=(0.75, 0.9),
            caption="Use this to judge if both methods agree on colour placement.",
            tooltip="SSIM between the AI and sound reconstructions; closer to 1.0 means they align.",
        )

        with st.expander("What Does This Mean?"):
            st.markdown(
                "- **How clear the picture looks (PSNR):** Like judging if a photo is crisp or grainy—higher numbers mean clearer shots.\n"
                "- **How closely the details match (SSIM):** Imagine comparing two LEGO builds; if every brick lines up, SSIM is close to 1.0.\n"
                "- **How much the shapes line up (Overlap):** Think of tracing paper stacked together; more shared highlights mean better overlap."
            )

    overlap_pct = float(ai_overlap_score)
    state["max_overlap_seen"] = max(state.get("max_overlap_seen", 0.0), overlap_pct)
    max_overlap_so_far = float(state["max_overlap_seen"])
    reward_points = float(
        max(
            state.get("difficulty_reward_points", 0.0),
            state.get("latest_lifetime_reward", 0.0),
        )
    )
    if overlap_pct >= 40.0:
        reward_points += float(
            np.interp(overlap_pct, [40.0, 60.0, 100.0], [1.0, 3.0, 5.0])
        )
    state["difficulty_reward_points"] = reward_points
    history = _record_performance_history(
        state,
        overlap_pct,
        float(metrics.ssim),
        float(metrics.psnr),
        float(sound_overlap_score),
        sound_reference_overlap=float(sound_reference_overlap),
    )
    markers: list[int] = list(state.get("_sound_target_markers", []))
    metrics_spec = prepare_metrics_chart(history, markers=markers)
    if metrics_spec:
        try:
            metrics_path = run_paths.charts / "metrics.png"
            export_chart_png(metrics_spec, metrics_path)
        except Exception:  # pragma: no cover - defensive
            chart_files.pop("metrics", None)
            logger.exception("Failed to export metrics chart")
        else:
            chart_files["metrics"] = str(metrics_path)
    else:
        chart_files.pop("metrics", None)
    target_progress, improvement_signal, volatility_signal, reward_signal = _derive_difficulty_metrics(
        history
    )
    generation_difficulty = float(
        np.clip(state.get("latest_generation_difficulty", 0.0), 0.0, 1.0)
    )
    improvement_signal = max(
        improvement_signal,
        float(np.clip(state.get("latest_generation_improvement", 0.0), 0.0, 1.0)),
    )
    reward_signal = max(
        reward_signal,
        float(np.clip(state.get("latest_generation_peak", 0.0) / 5.0, 0.0, 1.0)),
    )
    reseed_progress = min(1.0, state.get("sound_reseed_count", 0) / 10.0)
    blended_target = max(target_progress, reseed_progress, generation_difficulty)
    previous_progress = float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0))
    updated_progress = float(
        np.clip(0.6 * previous_progress + 0.4 * blended_target, 0.0, 1.0)
    )
    state["difficulty_progress"] = updated_progress
    state["difficulty_improvement"] = float(improvement_signal)
    state["difficulty_volatility"] = float(volatility_signal)
    state["difficulty_reward"] = float(reward_signal)
    difficulty_progress = updated_progress

    if history:
        latest = history[-1]
        latest["difficulty_progress"] = difficulty_progress * 100.0
        latest["difficulty_target"] = blended_target * 100.0
        latest["reward_signal"] = reward_signal * 100.0
        latest["reward_points"] = float(state.get("difficulty_reward_points", 0.0))

    st.subheader("Performance signals")
    control_cols = st.columns([4, 1])
    auto_follow = bool(state.get("metrics_autofollow_toggle", True))
    with control_cols[1]:
        if st.button("Snap to latest", key="metrics_snap_latest"):
            state["metrics_autofollow_toggle"] = True
            auto_follow = True
    with control_cols[0]:
        auto_follow = st.toggle(
            "Auto-follow latest",
            value=auto_follow,
            key="metrics_autofollow_toggle",
            help=(
                "When enabled the chart automatically tracks the newest "
                "observations. Hold Shift and use the scroll wheel to pan "
                "left or right."
            ),
        )
    auto_follow = bool(auto_follow)
    metrics_spec = prepare_metrics_chart(
        history,
        markers=markers,
        window=_PERFORMANCE_CHART_WINDOW,
        auto_follow=auto_follow,
    )
    if metrics_spec:
        st.vega_lite_chart(metrics_spec, use_container_width=True)
        st.caption(
            "Hold Shift while scrolling to move horizontally. The view pauses "
            "where you leave it when auto-follow is disabled; click Snap to "
            "latest to rejoin the live stream."
        )
        try:
            metrics_path = run_paths.charts / "metrics.png"
            export_chart_png(metrics_spec, metrics_path)
        except Exception:  # pragma: no cover - defensive
            chart_files.pop("metrics", None)
            logger.exception("Failed to export metrics chart")
        else:
            chart_files["metrics"] = str(metrics_path)
    else:
        chart_files.pop("metrics", None)
        st.caption(
            "Performance chart will appear once multiple observations contain "
            "varying values."
        )

    difficulty_box = st.sidebar.expander("Difficulty signals", expanded=False)
    with difficulty_box:
        st.metric("Adaptive difficulty", f"{difficulty_progress * 100:.0f}%")
        momentum = float(state.get("difficulty_improvement", 0.0)) * 100.0
        variability = float(state.get("difficulty_volatility", 0.0)) * 100.0
        trend_cols = st.columns(2)
        trend_cols[0].metric("Difficulty momentum", f"{momentum:.1f} pts")
        trend_cols[1].metric("Difficulty range", f"{variability:.1f} pts")
        reward_cols = st.columns(3)
        reward_cols[0].metric(
            "High-overlap reward",
            f"{float(state.get('difficulty_reward', 0.0)) * 100:.0f} pts",
            help="Boost awarded when overlap exceeds 40%, encouraging harder scenarios.",
        )
        reward_cols[1].metric(
            "Lifetime reward",
            f"{state.get('difficulty_reward_points', 0.0):.1f} pts",
            help="Cumulative bonus reflecting consistently strong overlap scores.",
        )
        reward_cols[2].metric(
            "Generation bonus",
            f"{state.get('latest_generation_reward', 0.0):.2f}",
            help="Average reward captured across the most recent generation.",
        )

    with progress_tab:
        st.write(
            "The overlap score measures agreement as ``1 - |original - reconstruction|``,"
            " so a perfect match reaches 100% while gaps in the prediction reduce the"
            " percentage."
        )

        st.subheader("Shape guessing AI")
        ai_guess_map: dict[str, ShapeGuess] = {
            guess.color: guess for guess in guess_shapes(ai_colored)
        }
        sound_guess_map: dict[str, ShapeGuess] = {
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

    with quick_tab:
        st.subheader("Live preview")
        overview_row = [
            (colored_original, f"Sound-derived image ({source_label})"),
            (packet_display, "Encoded packet (noise + signal)"),
            (ai_colored, "AI reconstruction (colourised)"),
            (sound_colored, "Sound-only reconstruction (colourised)"),
        ]
        overlay_row = [
            (noise_display, "Predicted noise contribution"),
            (ai_overlap_color, "Colour overlap: AI vs original"),
            (sound_overlap_color, "Colour overlap: Sound vs AI"),
            (sound_reference_overlap_color, "Colour overlap: Sound vs original"),
        ]

        for columns, content in ((st.columns(4), overview_row), (st.columns(4), overlay_row)):
            for col, (image, caption) in zip(columns, content):
                _render_image(col, image, caption)

        st.caption(
            "Red highlights information present only in the generated candidate, blue marks"
            " reference-only structure, and neutral grayscale indicates shared content."
        )

    with tune_tab:
        st.subheader("Difficulty presets")
        mode_names = list(_DIFFICULTY_MODE_PRESETS.keys())
        current_mode = state.get("difficulty_mode", mode_names[0])
        if current_mode not in mode_names:
            current_mode = mode_names[0]
        selected_mode = st.select_slider(
            "Difficulty focus",
            options=mode_names,
            value=current_mode,
            help="Single dial that steers all modelling settings and adaptive tuning.",
            disabled=not state.get("show_advanced_controls", False),
            key="tune_difficulty_mode",
        )
        state["difficulty_mode"] = selected_mode
        if not state.get("show_advanced_controls", False):
            st.caption("Switch Step 2 to **Let me tweak** to adjust difficulty presets here.")

    if show_advanced:
        with st.sidebar.expander("Evolution run controls", expanded=True):
            if hyper_enabled and hyper_profile is not None:
                population_size = int(
                    max(1, hyper_profile.batch_size or population_size)
                )
                generations_to_queue = int(
                    max(
                        1,
                        hyper_profile.queue_generations
                        or hyper_profile.dwell_generations
                        or generations_to_queue,
                    )
                )
                autosave_interval = int(
                    max(1, hyper_profile.autosave_interval or autosave_interval)
                )
                target_subjects = int(
                    max(
                        hyper_profile.target_subjects,
                        population_size * max(hyper_profile.dwell_generations, 1),
                    )
                )
                state["population_size"] = population_size
                state["generations_to_queue"] = generations_to_queue
                state["autosave_interval"] = autosave_interval
                state["evolution_mode"] = "Finite"

                col_top, col_bottom = st.columns(2)
                col_top.metric("Subjects per generation", population_size)
                col_top.metric("Queued generations", generations_to_queue)
                col_bottom.metric("Subjects per sound cycle", target_subjects)
                col_bottom.metric("Autosave interval", autosave_interval)
                if hyper_profile.mean_duration > 0.0:
                    throughput = hyper_profile.throughput
                    timing_text = (
                        f"Avg generation {hyper_profile.mean_duration:.2f}s • {throughput:.1f} subjects/s"
                        if throughput > 0.0
                        else f"Avg generation {hyper_profile.mean_duration:.2f}s"
                    )
                    st.caption(timing_text)
                st.caption(
                    "Hyper performance mode tunes these parameters from runtime throughput and the current difficulty so you can focus on results."
                )
            else:
                with st.sidebar.expander("Evolution cadence", expanded=False):
                    population_size = int(
                        st.number_input(
                            "AI attempts per generation",
                            min_value=1,
                            max_value=32,
                            value=population_size,
                            step=1,
                            key="population_size_input",
                        )
                    )
                    generations_to_queue = int(
                        st.number_input(
                            "Generations to queue",
                            min_value=1,
                            value=generations_to_queue,
                            step=1,
                            key="generations_to_queue_input",
                        )
                    )
                    autosave_interval = int(
                        st.number_input(
                            "Autosave every N generations",
                            min_value=1,
                            value=autosave_interval,
                            step=1,
                            key="autosave_interval_input",
                        )
                    )
                    st.caption(
                        "Finite batches follow these values when you press Run finite batch. "
                        "Continuous runs ignore the queue length and stream one generation at a time."
                    )
                state["population_size"] = population_size
                state["generations_to_queue"] = generations_to_queue
                state["autosave_interval"] = autosave_interval

    if advanced_mode:
        with st.sidebar.expander("Adversarial mode (beta)", expanded=False):
            st.checkbox(
                "Enable generator vs decoder co-evolution",
                key="adversarial_enabled",
                value=bool(state.get("adversarial_enabled", False)),
                help=(
                    "Trains a predictive generator to approximate the decoder's output without passing through"
                    " the channel, while the decoder adapts its denoise level."
                ),
            )
        if hyper_enabled and hyper_profile is not None:
            autosave_interval = int(
                max(1, hyper_profile.autosave_interval or autosave_interval)
            )
        state["population_size"] = population_size
        state["generations_to_queue"] = generations_to_queue
        state["autosave_interval"] = autosave_interval
        state["evolution_mode"] = "Infinite" if state.get("run_infinite", False) else "Finite"
    if state.get("adversarial_enabled", False) and "adversarial" not in state:
        state["adversarial"] = AdversarialManager()

    if reset_button:
        state.pop("evolution_manager", None)
        state.pop("manager", None)
        state.pop("evolution_signature", None)
        state["pending_generations"] = 0
        state["run_infinite"] = False
        state["active_run_id"] = None
        state.pop("adaptive_scene", None)
        state["sound_score_remaining"] = 0.0
        state["sound_target_score"] = 0.0
        state["difficulty"] = 0.0
        state["evolution_trees"] = {}
        state.pop("active_tree_id", None)
        state["active_parent_seeds"] = []
        state["quick_start_infinite_preference"] = False
        st.info("Cleared evolution history.")

    if reload_button:
        state.pop("evolution_manager", None)
        state.pop("manager", None)
        state.pop("evolution_signature", None)
        state["autosave_checked"] = False
        state.pop("adaptive_scene", None)
        state["evolution_trees"] = {}
        state.pop("active_tree_id", None)
        state["active_parent_seeds"] = []
        state["active_run_id"] = None

    tree_label = _default_tree_label(sound_seed, current_resolution, current_sample_rate)
    force_new_tree = bool(state.pop("_spawn_new_tree_requested", False))
    manager = _ensure_manager(
        original=original,
        encoder=encoder,
        decoder=decoder,
        population_size=population_size,
        seed=seed,
        autosave_interval=autosave_interval,
        force_new_tree=force_new_tree,
        tree_label=tree_label,
    )
    state["manager"] = manager
    _update_tree_label(state, tree_label)
    _refresh_active_parents(state, manager)

    if advanced_mode:
        with st.sidebar.expander("Selective breeding", expanded=False):
            parent_entries = sorted(
                manager.parent_lineage,
                key=lambda entry: (entry.origin_generation, -entry.metrics.ssim),
            )
            parent_labels = {
                entry.seed: (
                    f"Gen {entry.origin_generation} • seed {entry.seed} "
                    f"(SSIM {entry.metrics.ssim:.3f}, overlap {entry.overlap_score:.1f}%)"
                )
                for entry in parent_entries
            }
            parent_options = list(parent_labels.keys())
            default_selection = [
                seed for seed in state.get("active_parent_seeds", []) if seed in parent_labels
            ]
            if not default_selection and parent_options:
                default_selection = [parent_options[-1]]
            parent_selection = st.multiselect(
                "Active parent seeds",
                parent_options,
                default=default_selection,
                key="parent_seed_select",
                format_func=lambda seed: parent_labels.get(seed, f"Seed {seed}"),
                help=(
                    "Selected parent seeds remain in every generation and act as anchors for "
                    "new child mutations. Leave the selection empty to use the full lineage."
                ),
            )
            state["active_parent_seeds"] = [int(seed) for seed in parent_selection]
            _sync_active_tree_state(state)

            if parent_entries:
                active_parent_set = {int(seed) for seed in state.get("active_parent_seeds", [])}
                summary_rows = [
                    {
                        "Seed": entry.seed,
                        "Generation": entry.origin_generation,
                        "SSIM": f"{entry.metrics.ssim:.3f}",
                        "Overlap (%)": f"{entry.overlap_score:.1f}",
                        "Appearances": entry.appearances,
                        "Reward": f"{entry.cumulative_reward:.2f}",
                        "Peak reward": f"{entry.peak_reward:.2f}",
                        "Active": "★" if entry.seed in active_parent_set else "",
                    }
                    for entry in parent_entries
                ]
                st.table(summary_rows)
            else:
                st.info("Run at least one generation to populate the parent lineage.")

    if save_button:
        save_path = manager.save(autosave_dir)
        st.success(f"Saved evolution session to {save_path}")

    run_store: RunStore = state["run_store"]

    if run_button:
        infinite_preference = bool(state.get("quick_start_infinite_preference", False))
        if infinite_preference:
            state["pending_generations"] = 0
            state["run_infinite"] = True
            state["evolution_mode"] = "Infinite"
            logger.info("Starting continuous evolution; will run until paused")
        else:
            state["pending_generations"] = generations_to_queue
            state["run_infinite"] = False
            state["evolution_mode"] = "Finite"
            logger.info("Queued %d generations for finite evolution", generations_to_queue)
        _sync_active_tree_state(state)
        if not state.get("active_run_id"):
            new_run_id = run_store.start_run()
            state["active_run_id"] = new_run_id
            logger.info("Started run %s", new_run_id)

    if stop_button:
        state["run_infinite"] = False
        state["pending_generations"] = 0
        state["evolution_mode"] = "Finite"
        logger.info("Requested evolution stop")
        _sync_active_tree_state(state)
        state["active_run_id"] = None

    pending_generations = int(state.get("pending_generations", 0))
    runs_to_execute = 0
    finite_batch = False
    if pending_generations > 0:
        finite_batch = True
        runs_to_execute = min(pending_generations, _MAX_GENERATIONS_PER_TICK)
    elif state.get("run_infinite", False):
        runs_to_execute = 1
    if state.get("run_infinite", False) and runs_to_execute == 0:
        runs_to_execute = 1

    generations_ran = 0
    reseeded = False
    score_consumed = 0.0
    if runs_to_execute:
        for _ in range(runs_to_execute):
            selection = state.get("active_parent_seeds", [])
            parent_selection = selection if selection else None
            generation = manager.run_generation(parent_selection=parent_selection)
            _refresh_active_parents(state, manager, generation)
            state["latest_generation_reward"] = float(generation.reward_summary)
            state["latest_generation_peak"] = float(generation.reward_peak)
            state["latest_generation_difficulty"] = float(generation.difficulty_level)
            state["latest_generation_difficulty_raw"] = float(generation.difficulty_raw)
            state["latest_generation_improvement"] = float(generation.improvement)
            state["difficulty_reward_points"] = float(manager.lifetime_reward)
            state["latest_lifetime_reward"] = float(manager.lifetime_reward)
            reward_signal = float(np.clip(generation.reward_peak / 5.0, 0.0, 1.0))
            state["difficulty_reward"] = max(
                float(state.get("difficulty_reward", 0.0)),
                reward_signal,
            )
            score_consumed += _generation_sound_score(generation)
            generations_ran += 1
            if finite_batch:
                state["pending_generations"] = max(
                    int(state.get("pending_generations", 0)) - 1,
                    0,
                )
        logger.debug(
            "Executed %d generation(s); pending=%d infinite=%s",
            generations_ran,
            int(state.get("pending_generations", 0)),
            state.get("run_infinite", False),
        )
        if hyper_enabled:
            profile = manager.hyper_profile
            state["_hyper_profile_cache"] = profile
            state["population_size"] = int(
                max(1, profile.batch_size or state.get("population_size", population_size))
            )
            state["autosave_interval"] = int(
                max(1, profile.autosave_interval or state.get("autosave_interval", autosave_interval))
            )
            state["generations_to_queue"] = int(
                max(1, profile.queue_generations or state.get("generations_to_queue", generations_to_queue))
            )
            state["sound_target_dwell"] = int(
                max(1, profile.dwell_generations or state.get("sound_target_dwell", target_dwell))
            )
            state["last_sound_target_dwell"] = int(state["sound_target_dwell"])
            state["evolution_mode"] = "Finite"
            population_size = int(state["population_size"])
            autosave_interval = int(state["autosave_interval"])
            generations_to_queue = int(state["generations_to_queue"])
            target_dwell = int(state["sound_target_dwell"])
            sound_budget_settings["target_dwell"] = target_dwell
            hyper_budget_settings = {
                "target_dwell": target_dwell,
                "difficulty_target": float(auto_settings.get("difficulty_target", difficulty_progress)),
            }
            target_score = _ensure_sound_score_budget(
                state,
                hyper_budget_settings,
                difficulty_progress=float(
                    np.clip(state.get("difficulty_progress", difficulty_progress), 0.0, 1.0)
                ),
            )
        _sync_active_tree_state(state)
        capped_before = min(remaining_before, target_dwell)
        remaining_after = max(0, int(capped_before) - generations_ran)
        state["sound_generations_left"] = int(remaining_after)

    trigger_rerun = False
    if generations_ran:
        score_remaining = max(0.0, score_remaining_before - score_consumed)
        state["sound_score_remaining"] = score_remaining

        if remaining_after == 0:
            _mark_sound_target_transition(state)
            new_seed, next_rate, next_resolution = _refresh_sound_scene(
                state,
                float(np.clip(state.get("difficulty_progress", 0.0), 0.0, 1.0)),
                target_dwell,
                improvement=float(state.get("difficulty_improvement", 0.0)),
                volatility=float(state.get("difficulty_volatility", 0.0)),
                max_overlap=float(state.get("max_overlap_seen", max_overlap_so_far)),
            )
            seed = int(state.get("shared_seed", seed))
            sound_seed = int(state.get("active_sound_seed", new_seed))
            current_sample_rate = next_rate
            current_resolution = next_resolution
            state["sound_score_remaining"] = float(
                state.get("sound_target_score", float(target_score))
            )
            score_remaining = float(state["sound_score_remaining"])
            state["sound_generations_left"] = int(target_dwell)
            remaining_after = int(target_dwell)
            reseeded = True
            st.info(
                "Auto-randomised sound scene after exhausting the score budget "
                f"(seed {sound_seed}, {next_rate:,} Hz, {next_resolution}×{next_resolution} px; "
                f"new target ≈ {state.get('sound_target_score', target_score):.1f})."
            )
            if _auto_refresh_pinterest_reference(state):
                st.info(
                    "Fetched a new Pinterest inspiration image after meeting the target "
                    "so the AI keeps adapting to fresh art."
                )

    else:
        state["sound_score_remaining"] = score_remaining_before
        state["sound_generations_left"] = int(min(remaining_before, target_dwell))
        score_consumed = 0.0

    trigger_rerun = _should_schedule_rerun(
        generations_ran=generations_ran,
        reseeded=reseeded,
        run_infinite=bool(state.get("run_infinite", False)),
        pending_generations=int(state.get("pending_generations", 0)),
    )

    if generations_ran and len(manager.generations) % manager.autosave_interval == 0:
        save_path = manager.save(autosave_dir)
        st.success(f"Autosaved evolution session to {save_path}")
        logger.info("Autosaved evolution session to %s", save_path)

    current_target_score = float(state.get("sound_target_score", float(target_score)))
    remaining_score = float(state.get("sound_score_remaining", current_target_score))
    delta_display: str | None = None
    if generations_ran and score_consumed > 0.0:
        delta_display = f"-{score_consumed:.2f}"
    st.sidebar.metric(
        "Sound score remaining",
        f"{remaining_score:.2f} / {current_target_score:.2f}",
        delta_display,
    )

    generation_progress_rows: list[dict[str, Any]] = []
    best_candidate_summary: dict[str, Any] | None = None

    if manager.generations:
        with tune_tab:
            st.subheader("Evolution progress")
            generation_progress_rows = []
            for record in manager.generations:
                components = _aggregate_reward_components(record)
                row: dict[str, Any] = {
                    "Generation": int(record.index),
                    "Best SSIM": _finite_or_none(record.best_candidate.metrics.ssim),
                    "Best PSNR": _finite_or_none(record.best_candidate.metrics.psnr),
                    "Best overlap": _finite_or_none(record.best_candidate.overlap_score),
                    "Avg reward": _finite_or_none(record.reward_summary),
                    "Peak reward": _finite_or_none(record.reward_peak),
                    "Difficulty": _finite_or_none(record.difficulty_level),
                    "Lifetime reward": _finite_or_none(record.cumulative_reward),
                    "reward_total": _finite_or_none(record.reward_summary),
                    "reward_overlap": _finite_or_none(components.get("overlap")),
                    "reward_msssim": _finite_or_none(components.get("msssim")),
                    "reward_dct_corr": _finite_or_none(components.get("dct_corr")),
                    "difficulty_raw": _finite_or_none(record.difficulty_raw),
                    "difficulty_normalized": _finite_or_none(
                        record.difficulty_normalized
                    ),
                }
                if record.checkpoint_tag:
                    row["_checkpoint_tag"] = record.checkpoint_tag
                generation_progress_rows.append(row)
                _append_global_progress_row(state, manager, row)

            global_history: list[dict[str, Any]] = state.get("_global_progress_history", [])
            if global_history:
                sanitized_rows, dropped_values = sanitize_progress_rows(global_history)
                spec, message = prepare_trend_chart(
                    sanitized_rows, had_non_finite=dropped_values
                )
                if spec:
                    st.vega_lite_chart(spec, use_container_width=True)
                    logger.debug(
                        "Rendered trend chart with %d records", len(sanitized_rows)
                    )
                    try:
                        trend_path = run_paths.charts / "trend.png"
                        export_chart_png(spec, trend_path)
                    except Exception:  # pragma: no cover - defensive
                        chart_files.pop("trend", None)
                        logger.exception("Failed to export trend chart")
                    else:
                        chart_files["trend"] = str(trend_path)
                else:
                    chart_files.pop("trend", None)
                if message:
                    st.caption(message)
            else:
                chart_files.pop("trend", None)

            gen_indices = [record.index for record in manager.generations]
            default_gen = gen_indices[-1]
            if len(gen_indices) > 1:
                selected_generation = st.select_slider(
                    "Select generation",
                    options=gen_indices,
                    value=default_gen,
                    key="selected_generation",
                    format_func=lambda idx: f"Generation {idx}",
                )
            else:
                selected_generation = default_gen
                st.caption("Only one generation so far; displaying the latest results.")
            generation = manager.generations[selected_generation]
            best_candidate = generation.best_candidate
            best_candidate_image = _apply_color_template(
                best_candidate.reconstruction, color_template
            )

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
            reward_cols = st.columns(2)
            reward_cols[0].metric(
                "Generation reward", f"{generation.reward_summary:.2f}",
                help="Mean reward achieved across the generation's candidates.",
            )
            reward_cols[1].metric(
                "Difficulty target", f"{generation.difficulty_level * 100:.0f}%",
                help="Adaptive difficulty derived from reward and overlap performance.",
            )

            badge_cols = st.columns(2)
            badge_cols[0].markdown(
                _metric_badge_html(
                    "Global pooled metrics (PSNR/SSIM)",
                    metrics.psnr,
                    metrics.ssim,
                ),
                unsafe_allow_html=True,
            )
            badge_cols[1].markdown(
                _metric_badge_html(
                    "Per-candidate strict metrics (PSNR/SSIM)",
                    best_candidate.metrics.psnr,
                    best_candidate.metrics.ssim,
                ),
                unsafe_allow_html=True,
            )

            st.subheader("Generation gallery")
            cols_per_row = min(4, len(generation.candidates))
            for offset in range(0, len(generation.candidates), cols_per_row):
                row = st.columns(cols_per_row)
                for col, candidate in zip(
                    row, generation.candidates[offset : offset + cols_per_row]
                ):
                    caption = (
                        f"Seed {candidate.seed}\nPSNR {candidate.metrics.psnr:.2f} dB\nSSIM {candidate.metrics.ssim:.3f}"
                    )
                    candidate_image = _apply_color_template(
                        candidate.reconstruction, color_template
                    )
                    _render_image(col, candidate_image, caption)

            st.subheader("Candidate inspector")
            option_labels = [
                f"AI {idx + 1}: Seed {cand.seed} – SSIM {cand.metrics.ssim:.3f}"
                for idx, cand in enumerate(generation.candidates)
            ]
            candidate_indices = list(range(len(generation.candidates)))
            default_candidate = next(
                (
                    i
                    for i, cand in enumerate(generation.candidates)
                    if cand.seed == best_candidate.seed
                ),
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
            inspected_color = colorize_comparison(
                manager.original, inspected.reconstruction
            )

            inspect_cols = st.columns(4)
            _render_image(inspect_cols[0], colored_original, "Evolution reference")
            inspected_reconstruction = _apply_color_template(
                inspected.reconstruction, color_template
            )
            _render_image(
                inspect_cols[1],
                inspected_reconstruction,
                f"Candidate seed {inspected.seed}",
            )
            _render_image(
                inspect_cols[2],
                inspect_overlap_map,
                f"Overlap map ({inspect_overlap_score:.1f}%)",
            )
            _render_image(
                inspect_cols[3],
                inspected_color,
                "Colour overlap vs reference",
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

        with tune_tab:
            st.subheader("Signal codec laboratory")
            run_id = state.get("active_run_id")
            if run_id:
                artifact_dir = run_store.artifacts_directory(run_id)
                st.caption(
                    f"Active run {run_id} — artifacts stored in {artifact_dir}"
                )
            else:
                st.caption(
                    "Start an evolution run to automatically save generated artifacts."
                )

            image_lane, audio_lane = st.columns(2)

            with image_lane:
                st.markdown("#### Image → WAV")
                sample_rate_choice = int(
                    st.slider(
                        "Sample rate (Hz)",
                        min_value=8_000,
                        max_value=96_000,
                        value=int(np.clip(current_sample_rate, 8_000, 96_000)),
                        step=1_000,
                        key="codec_sample_rate",
                    )
                )
                source_options: list[str] = []
                if best_candidate is not None:
                    source_options.append("Evolution best")
                source_options.append("Upload")
                source_choice = st.radio(
                    "Source image",
                    options=source_options,
                    index=0,
                    key="codec_image_source",
                )

                source_image: np.ndarray | None = None
                source_label = ""
                if source_choice == "Evolution best" and best_candidate is not None:
                    source_image = np.asarray(best_candidate_image, dtype=np.float32)
                    source_label = f"seed-{best_candidate.seed}"
                else:
                    uploaded_image = st.file_uploader(
                        "Upload image",
                        type=("png", "jpg", "jpeg", "bmp"),
                        key="codec_image_upload",
                    )
                    if uploaded_image is not None:
                        try:
                            uploaded_image.seek(0)
                            source_image = _load_uploaded_image(uploaded_image)
                            source_label = uploaded_image.name or "upload"
                        except ValueError as exc:
                            st.error(str(exc))

                if source_image is not None:
                    _render_image(
                        st,
                        source_image,
                        f"Source · {source_label}" if source_label else "Source image",
                    )
                    waveform = encode_image_to_waveform(
                        source_image,
                        sample_rate=sample_rate_choice,
                    )
                    wav_bytes = encode_image_to_wav_bytes(
                        source_image,
                        sample_rate=sample_rate_choice,
                    )
                    roundtrip = decode_waveform_to_image(
                        waveform,
                        sample_rate=sample_rate_choice,
                        resolution=source_image.shape[:2],
                    )
                    _render_image(st, roundtrip, "Heuristic round-trip")
                    metrics_roundtrip = compute_metrics(source_image, roundtrip)
                    st.markdown(
                        _metric_badge_html(
                            "Round-trip vs source",
                            metrics_roundtrip.psnr,
                            metrics_roundtrip.ssim,
                        ),
                        unsafe_allow_html=True,
                    )
                    _render_audio(st, wav_bytes)
                    download_name = _sanitize_filename(
                        source_label or "image",
                        f"_{sample_rate_choice}hz.wav",
                    )
                    _render_download_link(
                        st,
                        label="Download waveform",
                        data=wav_bytes,
                        file_name=download_name,
                        mime="audio/wav",
                        key="download_waveform_bytes",
                    )
                    if run_id:
                        if st.button("Save waveform artifact", key="save_waveform_artifact"):
                            saved_path = run_store.save_bytes(run_id, download_name, wav_bytes)
                            st.success(f"Saved waveform artifact to {saved_path}")
                    else:
                        st.caption("Artifacts will be persisted once a run is active.")
                else:
                    st.caption("Provide an image to encode it into audio.")

            with audio_lane:
                st.markdown("#### WAV → Image")
                uploaded_wav = st.file_uploader(
                    "Upload WAV audio",
                    type=("wav", "wave"),
                    key="codec_wav_upload",
                )
                target_resolution = int(
                    st.number_input(
                        "Target resolution",
                        min_value=32,
                        max_value=512,
                        value=int(current_resolution),
                        step=8,
                        key="codec_target_resolution",
                    )
                )
                if uploaded_wav is not None:
                    try:
                        uploaded_wav.seek(0)
                        wav_bytes = uploaded_wav.read()
                        waveform, detected_rate = load_waveform_from_wav(wav_bytes)
                        prediction = predict_image_from_waveform(
                            waveform,
                            sample_rate=detected_rate,
                            resolution=(target_resolution, target_resolution),
                        )
                        _render_audio(st, wav_bytes)
                        st.markdown(
                            f"**Detected sample rate** — {detected_rate:,} Hz, {waveform.size} samples"
                        )
                        _render_image(st, prediction, "Predicted image")
                        reference_original = colored_original
                        if reference_original.shape[:2] != prediction.shape[:2]:
                            reference_original = _resize_image(
                                reference_original, prediction.shape[:2]
                            )
                        original_metrics = compute_metrics(
                            reference_original, prediction
                        )
                        reference_best = best_candidate_image
                        if reference_best.shape[:2] != prediction.shape[:2]:
                            reference_best = _resize_image(
                                reference_best, prediction.shape[:2]
                            )
                        best_metrics = compute_metrics(reference_best, prediction)
                        metric_cols = st.columns(2)
                        metric_cols[0].markdown(
                            _metric_badge_html(
                                "Vs original reference",
                                original_metrics.psnr,
                                original_metrics.ssim,
                            ),
                            unsafe_allow_html=True,
                        )
                        metric_cols[1].markdown(
                            _metric_badge_html(
                                "Vs evolution best",
                                best_metrics.psnr,
                                best_metrics.ssim,
                            ),
                            unsafe_allow_html=True,
                        )
                        png_bytes = _image_to_png_bytes(prediction)
                        download_png = _sanitize_filename(
                            (uploaded_wav.name or "waveform"),
                            "_prediction.png",
                        )
                        _render_download_link(
                            st,
                            label="Download predicted image",
                            data=png_bytes,
                            file_name=download_png,
                            mime="image/png",
                            key="download_predicted_image",
                        )
                        if run_id:
                            if st.button("Save image artifact", key="save_prediction_artifact"):
                                saved = run_store.save_image(run_id, download_png, prediction)
                                st.success(f"Saved image artifact to {saved}")
                        else:
                            st.caption("Artifacts will be persisted once a run is active.")
                    except ValueError as exc:
                        st.error(f"Failed to decode audio: {exc}")
                else:
                    st.caption("Upload a WAV file to reconstruct an image.")
    else:
        with tune_tab:
            st.info("Run at least one generation to visualise evolution progress.")
        with tune_tab:
            st.info(
                "Run at least one evolution cycle to unlock the signal codec workflow."
            )

    available_charts: list[tuple[str, Path]] = []
    for key, path_str in chart_files.items():
        if not isinstance(path_str, str):
            continue
        chart_path = Path(path_str)
        if chart_path.exists():
            available_charts.append((key, chart_path))

    with st.sidebar.expander("Chart exports", expanded=False):
        if available_charts:
            label_map = {"trend": "Trend chart", "metrics": "Metrics chart"}
            for key, chart_path in sorted(available_charts, key=lambda item: item[0]):
                cache_key = f"{key}:{chart_path}"
                cached_entry = download_cache.get(cache_key)
                try:
                    stat_info = chart_path.stat()
                    modified_ns = getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))
                except OSError:  # pragma: no cover - filesystem guard
                    logger.exception("Failed to stat chart %s for download", chart_path)
                    download_cache.pop(cache_key, None)
                    continue
                if not cached_entry or cached_entry.get("mtime") != modified_ns:
                    try:
                        chart_bytes = chart_path.read_bytes()
                    except OSError:  # pragma: no cover - filesystem guard
                        logger.exception("Failed to load chart %s for download", chart_path)
                        download_cache.pop(cache_key, None)
                        continue
                    cached_entry = {"mtime": modified_ns, "bytes": chart_bytes}
                    download_cache[cache_key] = cached_entry
                chart_bytes = cached_entry["bytes"]
                label = label_map.get(key, chart_path.stem.replace("_", " ").title())
                _render_download_link(
                    st,
                    label=f"Download {label}",
                    data=chart_bytes,
                    file_name=chart_path.name,
                    mime="image/png",
                    key=f"download_{key}_chart",
                )
        else:
            st.caption("Charts will appear once evolution has enough history to plot.")

    export_payload = _session_export_payload(
        state=state,
        manager=manager,
        metrics=metrics,
        sound_metrics=sound_metrics,
        sound_reference_metrics=sound_reference_metrics,
        ai_sound_alignment=ai_sound_alignment,
        ai_overlap_score=ai_overlap_score,
        sound_overlap_score=sound_overlap_score,
        sound_reference_overlap=sound_reference_overlap,
        sound_clip=sound_clip,
        base_encoder_sigma=base_encoder_sigma,
        base_decoder_sigma=base_decoder_sigma,
        encoder_sigma=encoder_sigma,
        denoise_sigma=denoise_sigma,
        current_sample_rate=current_sample_rate,
        current_resolution=current_resolution,
        sample_rate_range=sample_rate_range,
        resolution_range=resolution_range,
        seed=int(seed),
        sound_seed=int(sound_seed),
        target_dwell=target_dwell,
        best_candidate_summary=best_candidate_summary,
    )

    export_bytes = _build_export_bundle(export_payload, generation_progress_rows)
    export_b64 = base64.b64encode(export_bytes).decode("ascii")
    st.sidebar.markdown(
        (
            f"<a href=\"data:application/zip;base64,{export_b64}\" "
            "download=\"umbra_session_snapshot.zip\" "
            "class=\"st-emotion-cache-3a2x2s e1nzilvr5\">Download session snapshot</a>"
        ),
        unsafe_allow_html=True,
    )

    # Ensure Keep Improving keeps ticking by scheduling a rerun after work
    if trigger_rerun:
        logger.debug("Scheduling rerun for continued evolution")
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


