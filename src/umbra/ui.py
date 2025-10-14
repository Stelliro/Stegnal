"""Desktop GUI for Project Umbra built with Tkinter."""

from __future__ import annotations

import ctypes
import html
import json
import logging
import math
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageTk

try:  # pragma: no cover - optional dependency during import on minimal envs
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as exc:  # pragma: no cover - import is deferred until runtime
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency for memory accounting
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from .codec import decode_waveform_to_image, encode_image_to_waveform
from .decoding import NoiseStreamDecoder
from .demo_packager import build_demo_executable
from .encoding import NoiseStreamEncoder
from .evolution import EvolutionManager
from .metrics import ReconstructionMetrics, compute_metrics
from .reconstruction import suggest_sample_rate, suggest_transmission_profile
from .visualization import multiplicative_overlap

logger = logging.getLogger(__name__)

_DISPLAY_MAX_EDGE = 720
_GRAPH_WIDTH = 900
_GRAPH_HEIGHT = 320
_HISTORY_LIMIT = 600
_AI_PSNR_BASELINE = 20.0
_AI_PSNR_TARGET = 60.0
_PINTEREST_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
_PINTEREST_DEFAULT_FEEDS = (
    "https://www.pinterest.com/ideas/pictures-of-the-universe/935563701058/",
)
_PINTEREST_IMAGE_PATTERN = re.compile(r"https://i\\.pinimg\\.com/[^'\"\\s>]+")
_PINTEREST_IMAGE_TAG_PATTERN = re.compile(
    r"<img[^>]+src=\"(?P<url>https://i\\.pinimg\\.com/[^\"]+)\"[^>]*alt=\"(?P<title>[^\"]*)\"",
    flags=re.IGNORECASE,
)
_PINTEREST_JSON_SCRIPT_PATTERN = re.compile(
    r"<script[^>]+type=\"application/json\"[^>]*>(?P<payload>.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)


class PinterestDownloadError(RuntimeError):
    """Raised when a Pinterest download fails."""


def _process_memory_usage() -> tuple[int, int] | None:
    """Return the process RSS and total system memory in bytes when available."""

    try:
        if psutil is not None:
            process = psutil.Process(os.getpid())
            rss = int(process.memory_info().rss)
            total = int(psutil.virtual_memory().total)
            if total > 0:
                return rss, total
    except Exception:
        logger.debug("psutil memory query failed", exc_info=True)

    if sys.platform.startswith("win"):
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):  # pragma: no cover - platform specific
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            class MEMORYSTATUSEX(ctypes.Structure):  # pragma: no cover - platform specific
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                rss = int(counters.WorkingSetSize)
            else:
                rss = 0

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = int(status.ullTotalPhys)
            else:
                total = 0
            if rss > 0 and total > 0:
                return rss, total
        except Exception:
            logger.debug("Windows memory query failed", exc_info=True)
        return None

    try:
        with open("/proc/self/statm", encoding="utf8") as handle:
            parts = handle.readline().split()
            if len(parts) >= 2:
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                rss = int(rss_pages * page_size)
                total = int(os.sysconf("SC_PHYS_PAGES") * page_size)
                if rss >= 0 and total > 0:
                    return rss, total
    except Exception:
        logger.debug("/proc memory query failed", exc_info=True)

    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss)
        if rss > 0:
            if sys.platform == "darwin":
                rss_bytes = rss
            else:
                rss_bytes = rss * 1024
            total = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            if total > 0:
                return int(rss_bytes), total
    except Exception:
        logger.debug("resource memory query failed", exc_info=True)

    return None


def _memory_relief_delay(threshold: float = 0.92) -> tuple[float, float]:
    """Return a back-off delay (seconds) and utilisation ratio if memory is tight."""

    stats = _process_memory_usage()
    if not stats:
        return 0.0, 0.0
    rss, total = stats
    if total <= 0:
        return 0.0, 0.0
    ratio = rss / float(total)
    if ratio < threshold:
        return 0.0, ratio
    overshoot = ratio - threshold
    delay = 0.05 + overshoot * 2.0
    delay = max(0.02, min(delay, 0.75))
    return delay, ratio


@dataclass
class UmbraAppState:
    """In-memory state container used by the desktop UI."""

    history: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=_HISTORY_LIMIT))
    composite_scores: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_LIMIT))
    sound_scores: deque[float] = field(default_factory=lambda: deque(maxlen=_HISTORY_LIMIT))

    def record_generation(
        self,
        generation_index: int,
        metrics: ReconstructionMetrics,
        overlap: float,
        *,
        sound_metrics: ReconstructionMetrics | None = None,
        sound_overlap: float | None = None,
        sound_reference_metrics: ReconstructionMetrics | None = None,
        sound_reference_overlap: float | None = None,
    ) -> dict[str, float]:
        """Store metrics for a completed generation and return the entry."""

        overlap_value = _nan_guard(overlap, 0.0)
        psnr_value = _nan_guard(metrics.psnr, _AI_PSNR_BASELINE)
        ssim_value = _nan_guard(metrics.ssim, 0.0)

        ai_score = _compute_composite_score(overlap_value, psnr_value, ssim_value)
        # Default to a zero composite score so generations without a successful
        # sound reconstruction never receive credit from the sound-first
        # scoreboard.
        composite_score = 0.0
        entry: dict[str, float] = {
            "generation": float(generation_index),
            "overlap": overlap_value,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "ai_overlap": overlap_value,
            "ai_psnr": psnr_value,
            "ai_ssim": ssim_value,
            "ai_score": ai_score,
        }
        if sound_metrics is not None and sound_overlap is not None:
            sound_overlap_value = _nan_guard(sound_overlap, 0.0)
            sound_psnr_value = _nan_guard(sound_metrics.psnr, _AI_PSNR_BASELINE)
            sound_ssim_value = _nan_guard(sound_metrics.ssim, 0.0)
            sound_score = _compute_composite_score(
                sound_overlap_value,
                sound_psnr_value,
                sound_ssim_value,
            )
            entry.update(
                {
                    "sound_psnr": sound_psnr_value,
                    "sound_ssim": sound_ssim_value,
                    "sound_overlap": sound_overlap_value,
                    "sound_score": sound_score,
                }
            )
            if math.isfinite(sound_score):
                self.sound_scores.append(sound_score)
            composite_score = float(sound_score)
        if sound_reference_metrics is not None and sound_reference_overlap is not None:
            entry.update(
                {
                    "sound_reference_psnr": _nan_guard(
                        sound_reference_metrics.psnr, _AI_PSNR_BASELINE
                    ),
                    "sound_reference_ssim": _nan_guard(sound_reference_metrics.ssim, 0.0),
                    "sound_reference_overlap": _nan_guard(sound_reference_overlap, 0.0),
                }
            )
        entry["composite_score"] = composite_score
        self.history.append(entry)
        self.composite_scores.append(float(composite_score))
        return entry

    def as_rows(self) -> list[dict[str, float]]:
        """Return a copy of the history rows for analytics."""

        return list(self.history)


def _nan_guard(value: float, fallback: float) -> float:
    return float(np.nan_to_num(value, nan=fallback, posinf=fallback, neginf=fallback))


def _compute_composite_score(overlap_pct: float, psnr: float, ssim: float) -> float:
    """Combine overlap, PSNR, and SSIM into a single performance score."""

    overlap_value = _nan_guard(overlap_pct, 0.0)
    psnr_value = _nan_guard(psnr, _AI_PSNR_BASELINE)
    ssim_value = _nan_guard(ssim, 0.0)

    overlap_norm = float(np.clip(overlap_value / 100.0, 0.0, 1.0))
    psnr_span = max(_AI_PSNR_TARGET - _AI_PSNR_BASELINE, 1e-6)
    psnr_norm = float(np.clip((psnr_value - _AI_PSNR_BASELINE) / psnr_span, 0.0, 1.0))
    ssim_norm = float(np.clip(ssim_value, 0.0, 1.0))

    composite = float(np.clip(0.4 * overlap_norm + 0.3 * psnr_norm + 0.3 * ssim_norm, 0.0, 1.0))
    return composite * 100.0


def _clamp_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot display empty image array")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Expected image shaped [H, W, 3] or [H, W]")
    return np.clip(arr, 0.0, 1.0)


def _to_photo_image(array: np.ndarray, *, max_edge: int = _DISPLAY_MAX_EDGE) -> ImageTk.PhotoImage:
    clamped = _clamp_image(array)
    data = (clamped * 255.0).astype(np.uint8)
    image = Image.fromarray(data, mode="RGB")
    if max_edge and max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return ImageTk.PhotoImage(image)


def _download_bytes(
    url: str,
    *,
    timeout: float = 10.0,
    referer: str | None = None,
    opener: Callable[[urllib.request.Request], Any] | None = None,
) -> bytes:
    headers = {
        "User-Agent": _PINTEREST_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9," "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer:
        headers["Referer"] = referer

    request = urllib.request.Request(url, headers=headers)
    opener_fn = opener or urllib.request.urlopen
    try:
        with opener_fn(request) as response:
            return response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - defensive networking guard
        raise PinterestDownloadError(f"Unable to download {url}: {exc}") from exc


def _normalize_pinterest_source(_source: str | None) -> str:
    return _PINTEREST_DEFAULT_FEEDS[0]


def _collect_pinterest_nodes(payload: Any) -> Iterable[Mapping[str, Any]]:
    stack = [payload]
    collected: list[Mapping[str, Any]] = []
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            collected.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return collected


def _normalize_pinterest_url(url: str) -> str:
    cleaned = html.unescape(url or "")
    match = re.search(r"https://i\\.pinimg\\.com/[^\s\"'>)]+", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(0)
    parsed = urllib.parse.urlparse(cleaned)
    normalized = parsed._replace(query="", fragment="")
    return normalized.geturl()


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

    if not candidates:
        for url in _PINTEREST_IMAGE_PATTERN.findall(text):
            candidates.append((_normalize_pinterest_url(url), "Pinterest inspiration"))

    return candidates


def _parse_pinterest_feed(feed_data: bytes | str) -> list[tuple[str, str]]:
    text = feed_data.decode("utf-8", errors="replace") if isinstance(feed_data, bytes) else feed_data
    candidates = _parse_pinterest_feed_xml(text)
    if not candidates:
        candidates = _parse_pinterest_html(text)
    if not candidates:
        raise ValueError("Unable to parse Pinterest feed XML")

    deduped: OrderedDict[str, str] = OrderedDict()
    for url, label in candidates:
        normalized = _normalize_pinterest_url(url)
        if normalized not in deduped:
            deduped[normalized] = label
    return list(deduped.items())


def fetch_pinterest_inspiration(*, timeout: float = 10.0) -> tuple[np.ndarray, str]:
    """Download a random image from the curated Pinterest board."""

    feed_url = _normalize_pinterest_source(None)
    feed_bytes = _download_bytes(feed_url, timeout=timeout)
    try:
        candidates = _parse_pinterest_feed(feed_bytes)
    except ValueError as exc:  # pragma: no cover - defensive
        raise PinterestDownloadError(str(exc)) from exc

    image_url, title = random.choice(candidates)
    referer = feed_url if feed_url.startswith("http") else "https://www.pinterest.com/"

    def _attempt(url: str) -> bytes:
        return _download_bytes(url, timeout=timeout, referer=referer)

    try:
        image_bytes = _attempt(image_url)
    except PinterestDownloadError as exc:
        fallback_urls = [
            image_url.replace("/736x/", replacement)
            for replacement in ("/564x/", "/474x/", "/236x/")
        ]
        for candidate in fallback_urls:
            candidate = candidate.strip()
            if not candidate or candidate == image_url:
                continue
            try:
                image_bytes = _attempt(candidate)
                break
            except PinterestDownloadError:
                continue
        else:
            raise exc

    with Image.open(BytesIO(image_bytes)) as downloaded:
        array = np.asarray(downloaded.convert("RGB"), dtype=np.float32) / 255.0

    label = (title or "Pinterest inspiration").strip() or "Pinterest inspiration"
    return np.clip(array, 0.0, 1.0), f"{label} (Pinterest)"


class UmbraDesktopApp:
    """Lightweight Tkinter front-end for Project Umbra."""

    def __init__(self, root: tk.Tk) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover - defensive
            raise RuntimeError("Tkinter is not available") from _IMPORT_ERROR

        self.root = root
        self.root.title("Project Umbra")
        self._configure_fullscreen()
        self.encoder = NoiseStreamEncoder()
        self.decoder = NoiseStreamDecoder()
        self.manager: EvolutionManager | None = None
        self.reference_image: np.ndarray | None = None
        self.reference_label = ""
        self.reconstruction: np.ndarray | None = None
        self.sound_reconstruction: np.ndarray | None = None
        self.state = UmbraAppState()
        self._status_var = tk.StringVar(value="Select a reference image to begin.")
        self._run_mode_var = tk.StringVar(value="finite")
        self._score_threshold = tk.DoubleVar(value=88.0)
        self._primary_score_var = tk.StringVar(value="Sound↔AI composite score: –")
        self._baseline_score_var = tk.StringVar(value="AI baseline score: –")
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._last_refresh = 0.0
        self._latest_sound_payload: dict[str, Any] | None = None
        self._latest_generation_entry: dict[str, float] | None = None

        self._reference_label_widget: tk.Label | None = None
        self._reference_image_widget: tk.Label | None = None
        self._reference_photo: ImageTk.PhotoImage | None = None
        self._recon_label_widget: tk.Label | None = None
        self._recon_image_widget: tk.Label | None = None
        self._recon_photo: ImageTk.PhotoImage | None = None
        self._sound_label_widget: tk.Label | None = None
        self._sound_image_widget: tk.Label | None = None
        self._sound_photo: ImageTk.PhotoImage | None = None
        self._graph_canvas: tk.Canvas | None = None

        self._build_layout()
        self._use_demo_image()
        self.root.after(200, self._poll_queue)

    # ------------------------------------------------------------------ UI setup
    def _configure_fullscreen(self) -> None:
        try:
            self.root.state("zoomed")  # type: ignore[attr-defined]
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                self.root.attributes("-fullscreen", False)

    def _maybe_pause_for_memory(self) -> None:
        """Sleep briefly when system memory is under pressure."""

        delay, ratio = _memory_relief_delay()
        if delay > 0.0:
            logger.debug(
                "Memory pressure %.1f%% detected; pausing worker for %.3f s",
                ratio * 100.0,
                delay,
            )
            time.sleep(delay)
        else:
            time.sleep(0.0)

    def _build_layout(self) -> None:
        control_frame = tk.Frame(self.root, bg="#101010")
        control_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Button(control_frame, text="Load image…", command=self._choose_image).pack(
            side=tk.LEFT, padx=6, pady=6
        )
        tk.Button(control_frame, text="Pinterest inspiration", command=self._fetch_pinterest_async).pack(
            side=tk.LEFT, padx=6, pady=6
        )
        tk.Button(control_frame, text="Demo gradient", command=self._use_demo_image).pack(
            side=tk.LEFT, padx=6, pady=6
        )
        tk.Button(
            control_frame,
            text="Build demo exe model here",
            command=self._build_demo_executable,
        ).pack(side=tk.LEFT, padx=6, pady=6)

        tk.Checkbutton(
            control_frame,
            text="Keep running until paused",
            variable=self._run_mode_var,
            onvalue="infinite",
            offvalue="finite",
            command=self._toggle_run_mode,
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(control_frame, text="Auto-refresh at sound score ≥", fg="white", bg="#101010").pack(
            side=tk.LEFT, padx=(20, 4)
        )
        threshold_spin = tk.Spinbox(
            control_frame,
            from_=50,
            to=100,
            increment=1,
            textvariable=self._score_threshold,
            width=5,
        )
        threshold_spin.pack(side=tk.LEFT, padx=4)

        tk.Button(control_frame, text="Start evolution", command=self.start_evolution).pack(
            side=tk.LEFT, padx=10, pady=6
        )
        tk.Button(control_frame, text="Pause", command=self.stop_evolution).pack(
            side=tk.LEFT, padx=6, pady=6
        )

        tk.Label(control_frame, textvariable=self._primary_score_var, fg="#8fdc6d", bg="#101010").pack(
            side=tk.RIGHT, padx=12
        )
        tk.Label(control_frame, textvariable=self._baseline_score_var, fg="#f4d35e", bg="#101010").pack(
            side=tk.RIGHT, padx=12
        )
        tk.Label(control_frame, textvariable=self._status_var, fg="white", bg="#101010").pack(
            side=tk.RIGHT, padx=12
        )

        preview_frame = tk.Frame(self.root, bg="#181818")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(preview_frame, bg="#181818")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._reference_label_widget = tk.Label(left_frame, text="Reference", fg="white", bg="#181818")
        self._reference_label_widget.pack()
        self._reference_image_widget = tk.Label(left_frame, bg="#101010")
        self._reference_image_widget.pack(padx=6, pady=6)

        right_frame = tk.Frame(preview_frame, bg="#181818")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._recon_label_widget = tk.Label(right_frame, text="AI reconstruction", fg="white", bg="#181818")
        self._recon_label_widget.pack()
        self._recon_image_widget = tk.Label(right_frame, bg="#101010")
        self._recon_image_widget.pack(padx=6, pady=6)

        sound_frame = tk.Frame(preview_frame, bg="#181818")
        sound_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._sound_label_widget = tk.Label(
            sound_frame,
            text="Sound-only reconstruction",
            fg="white",
            bg="#181818",
        )
        self._sound_label_widget.pack()
        self._sound_image_widget = tk.Label(sound_frame, bg="#101010")
        self._sound_image_widget.pack(padx=6, pady=6)

        graph_frame = tk.Frame(self.root, bg="#101010")
        graph_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
        self._graph_canvas = tk.Canvas(
            graph_frame,
            width=_GRAPH_WIDTH,
            height=_GRAPH_HEIGHT,
            bg="#0e0e0e",
            highlightthickness=0,
        )
        self._graph_canvas.pack(fill=tk.BOTH, expand=True)
        self._graph_canvas.bind("<Configure>", lambda _event: self._draw_graph())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ Controls
    def _toggle_run_mode(self) -> None:
        mode = self._run_mode_var.get()
        if mode not in {"infinite", "finite"}:
            self._run_mode_var.set("finite")

    def _choose_image(self) -> None:
        if filedialog is None:
            messagebox.showerror("Unavailable", "File selection dialog is not available in this environment.")
            return
        path = filedialog.askopenfilename(title="Select reference image")
        if not path:
            return
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load image: {exc}")
            return
        self._set_reference(array, Path(path).name)
        self._status_var.set(f"Loaded reference {path}")

    def _use_demo_image(self) -> None:
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        gradient = np.outer(x, np.ones_like(x))
        demo = np.stack([gradient, gradient[::-1], np.sqrt(gradient)], axis=-1)
        self._set_reference(demo, "Demo gradient")
        self._status_var.set("Using demo gradient scene.")

    def _set_reference(self, image: np.ndarray, label: str) -> None:
        self.reference_image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        self.reference_label = label
        if self._reference_label_widget is not None:
            self._reference_label_widget.config(text=f"Reference · {label}")
        self._update_reference_preview()
        if self._running:
            self.stop_evolution()
            if self._worker is not None and self._worker.is_alive():
                self._worker.join(timeout=1.0)
        self.manager = None
        self._reset_manager()

    def _fetch_pinterest_async(self) -> None:
        self._status_var.set("Fetching Pinterest inspiration…")
        thread = threading.Thread(target=self._download_pinterest_image, daemon=True)
        thread.start()

    def _download_pinterest_image(self) -> None:
        try:
            image, label = fetch_pinterest_inspiration(timeout=12.0)
        except Exception as exc:
            logger.warning("Pinterest download failed: %s", exc)
            self._queue.put(("status", f"Pinterest fetch failed: {exc}"))
            return
        self._queue.put(("pinterest", image, label))

    # ------------------------------------------------------------------ Evolution loop
    def start_evolution(self) -> None:
        if self._running:
            return
        if self.reference_image is None:
            self._status_var.set("Select a reference image first.")
            return
        if self.manager is None:
            self.manager = self._create_manager()
        self._running = True
        self._status_var.set("Evolution running…")
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop_evolution(self) -> None:
        self._running = False
        self._status_var.set("Evolution paused. Press start to resume.")

    def _reset_manager(self) -> None:
        if self.reference_image is None:
            return
        self.manager = self._create_manager()
        self.state = UmbraAppState()
        self._draw_graph()

    def _create_manager(self) -> EvolutionManager:
        assert self.reference_image is not None
        base_seed = int(time.time()) & 0x7FFFFFFF
        return EvolutionManager(
            self.reference_image,
            self.encoder,
            self.decoder,
            population_size=6,
            base_seed=base_seed,
            autosave_interval=6,
        )

    def _run_loop(self) -> None:
        assert self.manager is not None
        while self._running:
            try:
                generation = self.manager.run_generation()
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.exception("Evolution failed", exc_info=exc)
                self._queue.put(("status", f"Evolution error: {exc}"))
                self._running = False
                break

            if not generation.candidates:
                self._queue.put(("status", "No candidates evaluated this generation."))
                self._maybe_pause_for_memory()
                continue

            best = generation.best_candidate
            reconstruction = np.asarray(best.reconstruction, dtype=np.float32)
            sound_payload: dict[str, Any] = {}
            sound_image: np.ndarray | None = None
            try:
                sample_rate = suggest_sample_rate(reconstruction)
                segments, marker_duration = suggest_transmission_profile(reconstruction)
                waveform = encode_image_to_waveform(
                    reconstruction,
                    sample_rate=sample_rate,
                    segments=segments,
                    marker_duration=marker_duration,
                )
                sound_image = decode_waveform_to_image(
                    waveform,
                    sample_rate=sample_rate,
                    resolution=reconstruction.shape[:2],
                    segments=segments,
                    marker_duration=marker_duration,
                )
                base_reference = (
                    self.reference_image if self.reference_image is not None else reconstruction
                )
                ref_image = np.asarray(base_reference, dtype=np.float32)
                ref_image = np.clip(ref_image, 0.0, 1.0)
                recon_clipped = np.clip(reconstruction, 0.0, 1.0)
                sound_clipped = np.clip(np.asarray(sound_image, dtype=np.float32), 0.0, 1.0)
                sound_vs_ai = compute_metrics(recon_clipped, sound_clipped)
                _, sound_overlap = multiplicative_overlap(recon_clipped, sound_clipped)
                sound_vs_reference = compute_metrics(ref_image, sound_clipped)
                _, sound_reference_overlap = multiplicative_overlap(ref_image, sound_clipped)
                sound_payload = {
                    "sound_metrics": sound_vs_ai,
                    "sound_overlap": float(sound_overlap),
                    "sound_reference_metrics": sound_vs_reference,
                    "sound_reference_overlap": float(sound_reference_overlap),
                    "sample_rate": int(sample_rate),
                    "segments": int(segments),
                    "marker_duration": float(marker_duration),
                }
            except Exception as exc:  # pragma: no cover - audio pipeline fallback
                logger.debug("Failed to derive sound reconstruction: %s", exc)

            entry = self.state.record_generation(
                generation.index,
                best.metrics,
                best.overlap_score,
                sound_metrics=sound_payload.get("sound_metrics"),
                sound_overlap=sound_payload.get("sound_overlap"),
                sound_reference_metrics=sound_payload.get("sound_reference_metrics"),
                sound_reference_overlap=sound_payload.get("sound_reference_overlap"),
            )
            self._latest_sound_payload = dict(sound_payload) if sound_payload else None
            self._latest_generation_entry = dict(entry)
            self._queue.put(
                (
                    "generation",
                    reconstruction,
                    best.metrics,
                    entry,
                    sound_image,
                    sound_payload,
                )
            )

            if self.reference_image is not None and self._run_mode_var.get() == "infinite":
                threshold = float(self._score_threshold.get())
                if entry["composite_score"] >= threshold and (time.time() - self._last_refresh) > 30.0:
                    self._queue.put(("refresh_pinterest", None, None))
                    self._last_refresh = time.time()

            self._maybe_pause_for_memory()

        self._worker = None

    # ------------------------------------------------------------------ Event processing
    def _poll_queue(self) -> None:
        processed = False
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            processed = True
            kind = message[0]
            if kind == "generation":
                (
                    _,
                    reconstruction,
                    _metrics,
                    entry,
                    sound_image,
                    sound_payload,
                ) = message
                self._update_reconstruction(np.asarray(reconstruction, dtype=np.float32))
                self._update_sound_reconstruction(
                    None if sound_image is None else np.asarray(sound_image, dtype=np.float32)
                )
                status_parts = [
                    f"Generation {entry.get('generation', 0):.0f}",
                ]
                sound_score = entry.get("sound_score")
                if sound_score is not None:
                    status_parts.append(
                        "Sound↔AI overlap {ov:.2f}%".format(
                            ov=entry.get("sound_overlap", 0.0)
                        )
                    )
                    if "sound_psnr" in entry:
                        status_parts.append(
                            "Sound↔AI PSNR {ps:.2f} dB".format(
                                ps=entry.get("sound_psnr", 0.0)
                            )
                        )
                    status_parts.append(
                        "Sound↔AI SSIM {ss:.3f}".format(
                            ss=entry.get("sound_ssim", 0.0)
                        )
                    )
                    status_parts.append(f"Sound↔AI score {sound_score:.2f}")
                else:
                    status_parts.append("Sound↔AI score unavailable")
                status_parts.append(
                    "AI↔Reference overlap {ov:.2f}%".format(
                        ov=entry.get("overlap", 0.0)
                    )
                )
                status_parts.append(
                    "AI↔Reference PSNR {ps:.2f} dB".format(ps=entry.get("psnr", 0.0))
                )
                status_parts.append(
                    "AI↔Reference SSIM {ss:.3f}".format(ss=entry.get("ssim", 0.0))
                )
                self._status_var.set(" · ".join(status_parts))
                composite_score = entry.get("composite_score")
                if "sound_score" in entry and composite_score is not None:
                    self._primary_score_var.set(
                        f"Sound↔AI composite score: {composite_score:.2f}"
                    )
                elif composite_score is not None:
                    self._primary_score_var.set(
                        f"Sound↔AI composite score: {composite_score:.2f} (sound reconstruction unavailable)"
                    )
                else:
                    self._primary_score_var.set("Sound↔AI composite score: –")

                ai_baseline = entry.get("ai_score")
                if ai_baseline is not None:
                    self._baseline_score_var.set(f"AI baseline score: {ai_baseline:.2f}")
                else:
                    self._baseline_score_var.set("AI baseline score: –")
                self._draw_graph()
            elif kind == "status":
                _, text = message
                self._status_var.set(str(text))
            elif kind == "pinterest":
                _, image, label = message
                self._set_reference(image, label)
                self._status_var.set(f"Loaded Pinterest inspiration: {label}")
            elif kind == "refresh_pinterest":
                self._fetch_pinterest_async()

        if processed:
            self._draw_graph()
        self.root.after(200, self._poll_queue)

    def _build_demo_executable(self) -> None:
        if self.reconstruction is None and self.reference_image is None:
            self._status_var.set("Generate a candidate before building the demo executable.")
            if messagebox is not None:
                messagebox.showinfo(
                    "Demo builder",
                    "Run at least one generation or select an image before packaging the demo.",
                )
            return

        image = self.reconstruction
        if image is None and self.reference_image is not None:
            image = self.reference_image
        if image is None:
            self._status_var.set("No image available for demo packaging.")
            return

        payload = self._latest_sound_payload or {}
        sample_rate = int(payload.get("sample_rate", 48_000))
        segments = int(payload.get("segments", 1))
        marker_duration = float(payload.get("marker_duration", 0.05))
        label = self.reference_label or "Best candidate"

        metadata: dict[str, Any] = {}
        if self._latest_generation_entry is not None:
            entry = self._latest_generation_entry
            for key in ("sound_score", "ai_score", "composite_score"):
                if key in entry:
                    metadata[key] = float(entry[key])

        try:
            destination = build_demo_executable(
                np.asarray(image, dtype=np.float32),
                sample_rate=sample_rate,
                segments=max(1, segments),
                marker_duration=max(0.001, marker_duration),
                label=label,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - packaging is user initiated
            logger.exception("Failed to build demo executable")
            self._status_var.set(f"Demo build failed: {exc}")
            if messagebox is not None:
                messagebox.showerror("Demo builder", f"Failed to build demo executable: {exc}")
            return

        self._status_var.set(f"Demo executable saved to {destination}")
        if messagebox is not None:
            messagebox.showinfo(
                "Demo builder",
                f"A demo executable has been written to:\n{destination}",
            )

    # ------------------------------------------------------------------ Rendering helpers
    def _update_reference_preview(self) -> None:
        if self.reference_image is None or self._reference_image_widget is None:
            return
        try:
            photo = _to_photo_image(self.reference_image)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to render reference preview: %s", exc)
            return
        self._reference_photo = photo
        self._reference_image_widget.config(image=self._reference_photo)

    def _update_reconstruction(self, reconstruction: np.ndarray) -> None:
        self.reconstruction = np.clip(reconstruction, 0.0, 1.0)
        if self._recon_image_widget is None:
            return
        try:
            photo = _to_photo_image(self.reconstruction)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to render reconstruction preview: %s", exc)
            return
        self._recon_photo = photo
        self._recon_image_widget.config(image=self._recon_photo)

    def _update_sound_reconstruction(self, sound_image: np.ndarray | None) -> None:
        if sound_image is None:
            return
        self.sound_reconstruction = np.clip(sound_image, 0.0, 1.0)
        if self._sound_image_widget is None:
            return
        try:
            photo = _to_photo_image(self.sound_reconstruction)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to render sound reconstruction preview: %s", exc)
            return
        self._sound_photo = photo
        self._sound_image_widget.config(image=self._sound_photo)

    def _draw_graph(self) -> None:
        if self._graph_canvas is None:
            return
        width = int(self._graph_canvas.winfo_width() or _GRAPH_WIDTH)
        height = int(self._graph_canvas.winfo_height() or _GRAPH_HEIGHT)
        margin = 40
        self._graph_canvas.delete("all")
        self._graph_canvas.create_rectangle(0, 0, width, height, fill="#0e0e0e", outline="")
        rows = self.state.as_rows()
        if len(rows) < 2:
            self._graph_canvas.create_text(
                width // 2,
                height // 2,
                fill="#666",
                text="Run a few generations to view the sound score trend.",
            )
            return

        composite_rows = [
            (row["generation"], row["composite_score"]) for row in rows if "composite_score" in row
        ]
        if len(composite_rows) < 2:
            self._graph_canvas.create_text(
                width // 2,
                height // 2,
                fill="#666",
                text="Waiting for sound reconstructions…",
            )
            return

        x_values = [generation for generation, _ in composite_rows]
        values = [score for _, score in composite_rows]
        baseline_rows = [(row["generation"], row["ai_score"]) for row in rows if "ai_score" in row]
        x_min, x_max = min(x_values), max(x_values)
        y_min = min(values)
        y_max = max(values)
        if baseline_rows:
            baseline_vals = [val for _, val in baseline_rows]
            y_min = min(y_min, min(baseline_vals))
            y_max = max(y_max, max(baseline_vals))
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        def _scale_x(val: float) -> float:
            return margin + (val - x_min) / (x_max - x_min) * (width - 2 * margin)

        def _scale_y(val: float) -> float:
            return height - margin - (val - y_min) / (y_max - y_min) * (height - 2 * margin)

        points: list[float] = []
        for generation, score in composite_rows:
            points.extend([_scale_x(generation), _scale_y(score)])

        self._graph_canvas.create_line(points, fill="#f4d35e", width=3, smooth=True)
        if len(baseline_rows) >= 2:
            baseline_points: list[float] = []
            for generation, score in baseline_rows:
                baseline_points.extend([_scale_x(generation), _scale_y(score)])
            self._graph_canvas.create_line(baseline_points, fill="#58a6ff", width=2, dash=(4, 3))
        self._graph_canvas.create_line(margin, height - margin, width - margin, height - margin, fill="#333")
        self._graph_canvas.create_line(margin, margin, margin, height - margin, fill="#333")
        self._graph_canvas.create_text(
            margin,
            margin - 10,
            text="Sound↔AI composite score",
            fill="#f4d35e",
            anchor=tk.W,
        )
        if baseline_rows:
            self._graph_canvas.create_text(
                margin,
                margin + 14,
                text="AI baseline score",
                fill="#58a6ff",
                anchor=tk.W,
            )
        self._graph_canvas.create_text(
            width - margin,
            height - margin + 20,
            text="Generation",
            fill="#888",
        )
        self._graph_canvas.create_text(
            width - margin,
            margin,
            text=f"Latest {values[-1]:.2f}",
            fill="#f4d35e",
            anchor=tk.E,
        )
        if baseline_rows:
            latest_baseline = baseline_rows[-1][1]
            self._graph_canvas.create_text(
                width - margin,
                margin + 16,
                text=f"AI {latest_baseline:.2f}",
                fill="#58a6ff",
                anchor=tk.E,
            )

    # ------------------------------------------------------------------ Shutdown
    def _on_close(self) -> None:
        self._running = False
        self.root.after(200, self.root.destroy)


def main() -> None:
    """Launch the Tkinter desktop application."""

    if tk is None or _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tkinter is unavailable on this system; install Tk support to run the desktop UI."
        ) from _IMPORT_ERROR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s: %(message)s")
    root = tk.Tk()
    UmbraDesktopApp(root)
    root.mainloop()


__all__ = [
    "UmbraDesktopApp",
    "UmbraAppState",
    "fetch_pinterest_inspiration",
    "_compute_composite_score",
    "_normalize_pinterest_url",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    main()
