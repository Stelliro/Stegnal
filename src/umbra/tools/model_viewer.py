"""Desktop helper for inspecting saved Umbra models and testing conversions."""

from __future__ import annotations

import json
import tkinter as tk
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image

from umbra.reconstruction import (
    image_to_waveform,
    reconstruct_from_waveform,
    waveform_to_wav_bytes,
)
from umbra.run_helpers import runs_root
from umbra.sound import load_waveform_from_wav


@dataclass
class ScoreBreakdown:
    """Composite score for a saved model."""

    total: float | None
    components: dict[str, float]


def _candidate_stat_files(path: Path) -> list[Path]:
    """Return likely statistic files for ``path``."""

    if path.is_file():
        return [path]

    preferred = [
        "summary.json",
        "stats.json",
        "metrics.json",
        "model.json",
        "run_summary.json",
    ]
    candidates: list[Path] = []
    for name in preferred:
        candidate = path / name
        if candidate.exists():
            candidates.append(candidate)
    candidates.extend(sorted(path.glob("*.json")))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in seen and candidate.exists():
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def load_model_stats(path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Attempt to load the most relevant statistics file for ``path``."""

    for candidate in _candidate_stat_files(path):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data, candidate
    return None, None


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_total_score(stats: Mapping[str, Any]) -> ScoreBreakdown:
    """Derive a 0–100 total score for the provided ``stats`` mapping."""

    components: dict[str, float] = {}
    total_weight = 0.0
    weighted = 0.0

    def add_component(name: str, value: Any, *, reference: float, weight: float) -> None:
        nonlocal total_weight, weighted
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(numeric) or reference <= 0:
            return
        normalized = _clip(numeric / reference, 0.0, 1.0)
        components[name] = normalized * 100.0
        weighted += normalized * weight
        total_weight += weight

    metrics = stats.get("metrics", {}) if isinstance(stats, Mapping) else {}
    if isinstance(metrics, Mapping):
        ai_metrics = metrics.get("ai_vs_reference", {})
        if isinstance(ai_metrics, Mapping):
            add_component("AI PSNR", ai_metrics.get("psnr"), reference=20.0, weight=25.0)
            add_component("AI SSIM", ai_metrics.get("ssim"), reference=1.0, weight=20.0)
        overlap = metrics.get("overlap", {})
        if isinstance(overlap, Mapping):
            add_component(
                "AI overlap",
                overlap.get("ai_vs_reference"),
                reference=100.0,
                weight=15.0,
            )
        pooled = metrics.get("global_pooled", {})
        if isinstance(pooled, Mapping):
            add_component(
                "Global PSNR",
                pooled.get("psnr"),
                reference=20.0,
                weight=10.0,
            )
        reward = metrics.get("reward_total")
        if reward is not None:
            add_component("Reward total", reward, reference=500.0, weight=5.0)

    manager = stats.get("manager") if isinstance(stats, Mapping) else None
    if isinstance(manager, Mapping):
        add_component(
            "Total score",
            manager.get("total_score"),
            reference=600.0,
            weight=15.0,
        )
        add_component(
            "Latest total score",
            manager.get("latest_total_score"),
            reference=100.0,
            weight=10.0,
        )
        best = manager.get("best_candidate")
        if isinstance(best, Mapping):
            add_component(
                "Best overlap",
                best.get("overlap"),
                reference=100.0,
                weight=5.0,
            )

    lifetime = stats.get("lifetime_reward")
    if lifetime is not None:
        add_component("Lifetime reward", lifetime, reference=500.0, weight=5.0)

    if total_weight == 0:
        return ScoreBreakdown(total=None, components=components)

    total = (weighted / total_weight) * 100.0
    return ScoreBreakdown(total=total, components=components)


def _format_components(components: Mapping[str, float]) -> str:
    if not components:
        return "No scoring components detected."
    lines = ["Component breakdown:"]
    for name, value in sorted(components.items()):
        lines.append(f"  • {name}: {value:.1f} / 100")
    return "\n".join(lines)


def _summarise_stats(stats: Mapping[str, Any]) -> str:
    sections: list[str] = []
    metrics = stats.get("metrics")
    if isinstance(metrics, Mapping):
        ai = metrics.get("ai_vs_reference")
        if isinstance(ai, Mapping):
            sections.append(
                "AI vs reference: PSNR={:.2f} dB, SSIM={:.3f}".format(
                    float(ai.get("psnr", 0.0)),
                    float(ai.get("ssim", 0.0)),
                )
            )
        sound_ai = metrics.get("sound_vs_ai")
        if isinstance(sound_ai, Mapping):
            sections.append(
                "Sound vs AI: PSNR={:.2f} dB, SSIM={:.3f}".format(
                    float(sound_ai.get("psnr", 0.0)),
                    float(sound_ai.get("ssim", 0.0)),
                )
            )
        sound_ref = metrics.get("sound_vs_reference")
        if isinstance(sound_ref, Mapping):
            sections.append(
                "Sound vs reference: PSNR={:.2f} dB, SSIM={:.3f}".format(
                    float(sound_ref.get("psnr", 0.0)),
                    float(sound_ref.get("ssim", 0.0)),
                )
            )
        overlap = metrics.get("overlap")
        if isinstance(overlap, Mapping):
            ai_overlap = float(overlap.get("ai_vs_reference", 0.0))
            sound_ai_overlap = overlap.get("sound_vs_ai")
            sound_ref_overlap = overlap.get("sound_vs_reference")
            parts = [f"AI={ai_overlap:.1f}%"]
            if sound_ai_overlap is not None:
                parts.append(f"Sound↔AI={float(sound_ai_overlap):.1f}%")
            if sound_ref_overlap is not None:
                parts.append(f"Sound↔Ref={float(sound_ref_overlap):.1f}%")
            sections.append("Overlap: " + ", ".join(parts))
    manager = stats.get("manager")
    if isinstance(manager, Mapping):
        sections.append(
            "Manager: generations={}, population={}, total score={:.2f}".format(
                int(manager.get("generation_count", 0)),
                int(manager.get("population_size", 0)),
                float(manager.get("total_score", 0.0)),
            )
        )
    if not sections:
        return "No detailed statistics found."
    return "\n".join(sections)


class ModelViewerApp:
    """Tkinter front-end for selecting models and testing conversions."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Umbra Model Viewer")
        self.root.geometry("900x540")

        self.score_var = tk.StringVar(value="Select a model to inspect its score.")
        self.detail_var = tk.StringVar(value="")

        main = ttk.Frame(root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.LabelFrame(main, text="Saved models")
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.model_list = tk.Listbox(list_frame, height=20, width=28)
        self.model_list.pack(side=tk.TOP, fill=tk.Y, expand=True, padx=4, pady=4)
        self.model_list.bind("<<ListboxSelect>>", self._on_model_selected)

        button_row = ttk.Frame(list_frame)
        button_row.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(button_row, text="Refresh", command=self.refresh_models).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(button_row, text="Browse…", command=self.browse_model).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        info_frame = ttk.Frame(main)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        score_frame = ttk.LabelFrame(info_frame, text="Model overview")
        score_frame.pack(fill=tk.X)
        ttk.Label(score_frame, textvariable=self.score_var).pack(
            anchor=tk.W, padx=8, pady=8
        )

        detail_frame = ttk.LabelFrame(info_frame, text="Details")
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.detail_text = tk.Text(detail_frame, height=12, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.detail_text.configure(state=tk.DISABLED)

        conversion_frame = ttk.LabelFrame(info_frame, text="Conversion tools")
        conversion_frame.pack(fill=tk.X, pady=12)

        res_frame = ttk.Frame(conversion_frame)
        res_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(res_frame, text="Image resolution (px)").pack(side=tk.LEFT)
        self.resolution_var = tk.IntVar(value=192)
        ttk.Spinbox(res_frame, from_=32, to=512, textvariable=self.resolution_var, width=6).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(
            conversion_frame,
            text="Image → WAV",
            command=self.convert_image_to_wav,
        ).pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(
            conversion_frame,
            text="WAV → Image",
            command=self.convert_wav_to_image,
        ).pack(fill=tk.X, padx=8, pady=(0, 8))

        notes_frame = ttk.LabelFrame(info_frame, text="Notes")
        notes_frame.pack(fill=tk.X)
        ttk.Label(
            notes_frame,
            text=(
                "Future roadmap: investigate MP4 → WAV → MP4 pipelines for audiovisual "
                "experiments."
            ),
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=8)

        self.refresh_models()

    def refresh_models(self) -> None:
        self.model_list.delete(0, tk.END)
        root_dir = runs_root()
        if not root_dir.exists():
            self.score_var.set(f"No runs directory found at {root_dir.resolve()}")
            return
        runs: Sequence[Path] = sorted(
            path for path in root_dir.iterdir() if path.is_dir()
        )
        for run in runs:
            self.model_list.insert(tk.END, run.name)
        if not runs:
            self.score_var.set("No saved runs discovered yet.")

    def browse_model(self) -> None:
        initial_dir = runs_root()
        selected = filedialog.askopenfilename(
            title="Select model stats", initialdir=initial_dir, filetypes=[("JSON", "*.json")]
        )
        if selected:
            self.display_model(Path(selected))

    def _on_model_selected(self, event: tk.Event[tk.Listbox]) -> None:
        selection = event.widget.curselection()
        if not selection:
            return
        index = selection[0]
        model_name = event.widget.get(index)
        self.display_model(runs_root() / model_name)

    def display_model(self, path: Path) -> None:
        stats, source = load_model_stats(path)
        if not stats or not source:
            messagebox.showwarning(
                "Model stats",
                "Unable to locate a readable statistics file for the selected model.",
            )
            return
        breakdown = compute_total_score(stats)
        if breakdown.total is None:
            self.score_var.set("Total score unavailable for this model.")
        else:
            self.score_var.set(f"Composite score: {breakdown.total:.1f} / 100")
        summary_lines = [f"Source: {source}"]
        summary_lines.append(_summarise_stats(stats))
        summary_lines.append("")
        summary_lines.append(_format_components(breakdown.components))
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "\n".join(summary_lines))
        self.detail_text.configure(state=tk.DISABLED)

    def convert_image_to_wav(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")],
        )
        if not image_path:
            return
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Image conversion", f"Failed to open image: {exc}")
            return
        array = np.asarray(image, dtype=np.float32) / 255.0
        sample_rate = 44100
        try:
            waveform = image_to_waveform(array, sample_rate=sample_rate)
            wav_bytes = waveform_to_wav_bytes(waveform, sample_rate)
        except Exception as exc:  # pragma: no cover - defensive
            messagebox.showerror("Image conversion", f"Conversion failed: {exc}")
            return
        output_path = filedialog.asksaveasfilename(
            title="Save WAV file",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not output_path:
            return
        try:
            with open(output_path, "wb") as handle:
                handle.write(wav_bytes)
        except OSError as exc:
            messagebox.showerror("Image conversion", f"Failed to save WAV file: {exc}")
            return
        messagebox.showinfo(
            "Image conversion",
            f"Saved waveform to {output_path}",
        )

    def convert_wav_to_image(self) -> None:
        wav_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not wav_path:
            return
        try:
            with open(wav_path, "rb") as handle:
                wav_bytes = handle.read()
        except OSError as exc:
            messagebox.showerror("Waveform conversion", f"Unable to read WAV: {exc}")
            return
        try:
            waveform, sample_rate = load_waveform_from_wav(wav_bytes)
            resolution = self.resolution_var.get()
            image_array = reconstruct_from_waveform(
                waveform,
                resolution=(resolution, resolution),
                sample_rate=sample_rate,
            )
        except Exception as exc:  # pragma: no cover - defensive
            messagebox.showerror("Waveform conversion", f"Conversion failed: {exc}")
            return
        image = Image.fromarray((image_array * 255.0).clip(0, 255).astype(np.uint8))
        output_path = filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")],
        )
        if not output_path:
            return
        try:
            image.save(output_path)
        except OSError as exc:
            messagebox.showerror("Waveform conversion", f"Failed to save image: {exc}")
            return
        messagebox.showinfo(
            "Waveform conversion",
            f"Saved reconstructed image to {output_path}",
        )


def main() -> None:
    root = tk.Tk()
    ModelViewerApp(root)
    try:
        root.mainloop()
    finally:
        root.destroy()


if __name__ == "__main__":  # pragma: no cover - manual use only
    main()
