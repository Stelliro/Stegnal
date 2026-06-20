# ui.py - FIXED VERSION
import json
import logging
import os
import queue
import secrets
import string
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlparse, urlunparse

import numpy as np
from PIL import Image, ImageTk

# Internal Imports
from .audio_mixer import AudioEngine
from .config import load_config
from .evolution import EvolutionManager
from .metrics import composite_score as _composite_score
from .payload import DataPayloadCodec

logger = logging.getLogger("Umbra")

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class UmbraDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PROJECT UMBRA // NEURAL AIR GAP")
        self.root.geometry("1600x1000")
        self.config = load_config()
        self.log_queue = queue.Queue()
        
        self.audio = AudioEngine()
        self.payload_codec = DataPayloadCodec(redundancy=9)
        
        self.simulation_mode_var = tk.BooleanVar(value=True)
        # Default auto-difficulty to True so it balances itself
        self.auto_difficulty_var = tk.BooleanVar(value=True)
        # CPU / GPU compute placement for candidate evaluation
        self.compute_mode_var = tk.StringVar(value="cpu")
        # Let the search tune color/gamma genes (off = purist; usually best)
        self.unlock_genes_var = tk.BooleanVar(value=False)
        # Reward model trained from each generation; persisted via Save/Load.
        self.model = None
        
        self.state = type('obj', (object,), {'running':False, 'difficulty':0.1, 'reference_image':None, 'generation_history':deque(), 'evolution_manager':None})()
        
        self._style = ttk.Style()
        self._apply_dark_mode()
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.pack(expand=1, fill="both")
        self.tab_evo = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_evo, text="EVOLUTION")
        
        self._build_evolution_ui(self.tab_evo)
        self._setup_logging()
        self.root.after(100, self._poll_log_queue)

    def _apply_dark_mode(self):
        self.root.configure(bg="#1e1e1e")
        self._style.theme_use("clam")
        self._style.configure(".", background="#1e1e1e", foreground="#e0e0e0", font=("Consolas", 10))
        self._style.configure("TButton", background="#333", foreground="#00ff41")

    def _setup_logging(self):
        if not any(isinstance(h, QueueHandler) for h in logging.getLogger().handlers):
            h = QueueHandler(self.log_queue)
            h.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
            logging.getLogger().addHandler(h)
            logging.getLogger('evolution').addHandler(h)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            if hasattr(self, 'console_log'):
                self.console_log.configure(state='normal')
                self.console_log.insert(tk.END, msg + "\n")
                self.console_log.see(tk.END)
                self.console_log.configure(state='disabled')
        self.root.after(100, self._poll_log_queue)

    def _build_evolution_ui(self, parent):
        ctrl = ttk.Frame(parent, padding=10)
        ctrl.pack(fill=tk.X)
        ttk.Button(ctrl, text="[ LOAD TARGET ]", command=self._load_reference).pack(side=tk.LEFT, padx=5)
        self.btn_start = ttk.Button(ctrl, text="INITIATE", command=self._start_evolution)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(ctrl, text="ABORT", command=self._stop_evolution)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Difficulty Slider
        self.diff_scale = ttk.Scale(ctrl, from_=0.01, to=1.0, orient=tk.HORIZONTAL, command=lambda v: setattr(self.state, 'difficulty', float(v)))
        self.diff_scale.set(0.1)
        self.diff_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        ttk.Checkbutton(ctrl, text="AUTO-DIFFICULTY", variable=self.auto_difficulty_var).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl, text="COMPUTE:").pack(side=tk.LEFT, padx=(10, 2))
        self.cb_compute = ttk.Combobox(
            ctrl, values=["cpu", "gpu", "hybrid"], state="readonly",
            width=7, textvariable=self.compute_mode_var,
        )
        self.cb_compute.pack(side=tk.LEFT, padx=2)
        self.cb_compute.bind("<<ComboboxSelected>>", self._on_compute_mode_change)

        ttk.Checkbutton(ctrl, text="UNLOCK GENES", variable=self.unlock_genes_var,
                        command=self._on_unlock_change).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="EXPORT JSON", command=self._export_data).pack(side=tk.RIGHT)
        ttk.Button(ctrl, text="LOAD MODEL", command=self._load_model).pack(side=tk.RIGHT, padx=2)
        ttk.Button(ctrl, text="SAVE MODEL", command=self._save_model).pack(side=tk.RIGHT, padx=2)

        self.audio_panel = ttk.LabelFrame(parent, text="INTERFERENCE LAB", padding=10)
        self.audio_panel.pack(fill=tk.X, padx=10)
        f_aud = ttk.Frame(self.audio_panel)
        f_aud.pack(fill=tk.X)
        self.cb_speaker = ttk.Combobox(f_aud, values=self.audio.get_devices('output'), state="readonly", width=30)
        self.cb_speaker.pack(side=tk.LEFT, padx=5)
        self.cb_mic = ttk.Combobox(f_aud, values=self.audio.get_devices('input'), state="readonly", width=30)
        self.cb_mic.pack(side=tk.LEFT, padx=5)
        if self.cb_speaker['values']:
            self.cb_speaker.current(0)
        if self.cb_mic['values']:
            self.cb_mic.current(0)
        ttk.Checkbutton(f_aud, text="SILENT SIM", variable=self.simulation_mode_var).pack(side=tk.LEFT, padx=10)
        self.btn_capture = ttk.Button(f_aud, text="CAPTURE x10 → TRAIN", command=self._capture_batch)
        self.btn_capture.pack(side=tk.LEFT, padx=10)
        self.lbl_status = ttk.Label(f_aud, text="READY", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT)

        img_f = ttk.Frame(parent)
        img_f.pack(expand=True, fill="both", padx=10)
        self.reference_canvas = self._mk_canvas(img_f, "SOURCE", 0)
        self.recon_canvas = self._mk_canvas(img_f, "RECONSTRUCTION", 1)
        self.air_canvas = self._mk_canvas(img_f, "AIR GAP RESULT", 2)
        self.hof_canvas = self._mk_canvas(img_f, "BEST GENE", 3)

        self.graph_canvas = tk.Canvas(parent, height=150, bg="#111", highlightthickness=0)
        self.graph_canvas.pack(fill=tk.X, padx=10, pady=5)
        self.metrics_label = ttk.Label(parent, text="READY.", foreground="#00ff41")
        self.metrics_label.pack(padx=10, anchor="w")
        self.console_log = scrolledtext.ScrolledText(parent, height=6, bg="#000", fg="#ccc", font=("Consolas", 9), state='disabled')
        self.console_log.pack(fill=tk.X, padx=10, pady=5)

    def _mk_canvas(self, p, title, col):
        f = ttk.Frame(p)
        f.grid(row=0, column=col, sticky="nsew", padx=2)
        p.columnconfigure(col, weight=1)
        ttk.Label(f, text=title, anchor="center").pack()
        c = tk.Canvas(f, bg="#000", highlightthickness=1, highlightbackground="#444")
        c.pack(expand=True, fill="both")
        return c

    def _load_reference(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            with Image.open(path) as img:
                # --- FIX 1: FORCE RESIZE TO 256x256 ---
                # This prevents the "cannot reshape" error
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                
                self.state.reference_image = arr
                self._display_image(self.reference_canvas, arr)
                self.state.generation_history.clear()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _display_image(self, canvas, array):
        if array is None or array.size == 0:
            return
        try:
            img_u8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(img_u8)
            canvas.update_idletasks()
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw < 10:
                return
            img = img.resize((cw, ch), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=photo)
            canvas.image = photo
        except Exception:
            pass

    def _start_evolution(self):
        if self.state.reference_image is None: 
            messagebox.showwarning("Missing Input", "Please load a target image first.")
            return

        if not self.state.evolution_manager:
            idx_out = int(self.cb_speaker.get().split(":")[0]) if self.cb_speaker.get() else 0
            idx_in = int(self.cb_mic.get().split(":")[0]) if self.cb_mic.get() else 0
            
            self.state.evolution_manager = EvolutionManager(
                self.state.reference_image,
                audio_out_idx=idx_out,
                audio_in_idx=idx_in,
                simulation_mode=self.simulation_mode_var.get(),
                compute_mode=self.compute_mode_var.get(),
                unlock_genes=self.unlock_genes_var.get(),
            )
            self.state.generation_history.clear()
            
        self.state.running = True
        if not hasattr(self, 'evo_thread') or not self.evo_thread.is_alive():
            self.evo_thread = threading.Thread(target=self._evolution_loop, daemon=True)
            self.evo_thread.start()

    def _evolution_loop(self):
        while self.state.running:
            try:
                if self.state.evolution_manager and self.state.evolution_manager.interference_cache:
                    # --- FIX 2: SAFE VISUALIZATION PAD ---
                    raw_wave = self.state.evolution_manager.interference_cache[0]
                    target_len = 256 * 256 * 3
                    
                    if raw_wave.size < target_len:
                        display_wave = np.pad(raw_wave, (0, target_len - raw_wave.size), constant_values=0)
                    else:
                        display_wave = raw_wave[:target_len]
                        
                    p = (display_wave.reshape(256, 256, 3) + 1.0) / 2.0
                    self.root.after(0, lambda i=p: self._display_image(self.air_canvas, i))
                
                # --- AUTO DIFFICULTY (SSIM-gated) ---
                # Only make the channel harder once the run is recognizable
                # (best SSIM >= threshold); otherwise ease off. Harder is earned.
                if self.auto_difficulty_var.get() and len(self.state.generation_history) > 0:
                    best = self.state.generation_history[-1].best_candidate
                    best_ssim = float(getattr(best.metrics, 'ssim', 0.0) or 0.0)
                    self.state.difficulty = self.state.evolution_manager.recommend_difficulty(
                        self.state.difficulty, best_ssim
                    )
                    self.root.after(0, lambda: self.diff_scale.set(self.state.difficulty))

                rec = self.state.evolution_manager.evolve_generation(self.state.difficulty)
                self.state.generation_history.append(rec)
                try:
                    self._ensure_model().train_from_generation(
                        self.state.evolution_manager,
                        source="simulation" if self.simulation_mode_var.get() else "acoustic",
                    )
                except Exception:
                    logger.debug("model train_from_generation failed", exc_info=True)
                self.root.after(0, self._update_ui_post_gen, rec)
                time.sleep(0.01)
            except Exception: 
                traceback.print_exc()
                time.sleep(1.0)

    def _update_ui_post_gen(self, rec):
        best = rec.best_candidate
        hunt = f"ST: {int(best.genes.start_sample)}"
        ncc_val = getattr(best.metrics, 'ssim', 0)
        status = f"GEN: {rec.generation} | REWARD: {best.reward:.4f} | {hunt} | NCC: {ncc_val:.4f}"
        self.metrics_label.config(text=status)
        self._display_image(self.recon_canvas, best.reconstruction)
        self._display_image(self.hof_canvas, best.reconstruction)
        self._update_graph()

    def _update_graph(self):
        self.graph_canvas.delete("all")
        hist = list(self.state.generation_history)[-100:]
        if not hist:
            return
        w, h = self.graph_canvas.winfo_width(), self.graph_canvas.winfo_height()
        data = [getattr(r.best_candidate.metrics, 'ssim', 0) for r in hist]
        if len(data) < 2:
            return
        
        pts = []
        for i, v in enumerate(data): 
            x = (i / (len(data) - 1)) * w
            y = h - (np.clip(v, 0, 1) * h)
            pts.extend([x, y])
            
        self.graph_canvas.create_line(pts, fill="#8be9fd", width=2)

    def _stop_evolution(self):
        self.state.running = False

    def _on_compute_mode_change(self, _event=None):
        """Apply the CPU/GPU compute mode live to a running manager."""
        mode = self.compute_mode_var.get()
        mgr = getattr(self.state, "evolution_manager", None)
        if mgr is not None:
            mgr.update_settings(compute_mode=mode)
        logger.info(f"Compute mode set to {mode.upper()}")

    def _on_unlock_change(self):
        """Toggle color/gamma gene unlocking on a running manager."""
        val = self.unlock_genes_var.get()
        mgr = getattr(self.state, "evolution_manager", None)
        if mgr is not None:
            mgr.update_settings(unlock_genes=val)
        logger.info(f"Gene unlock {'ON' if val else 'OFF (purist)'}")

    def _ensure_model(self):
        """Lazily create the reward model that accumulates across the session."""
        if self.model is None:
            from .checkpoint import UmbraModel
            self.model = UmbraModel(name="umbra-session")
        return self.model

    def _save_model(self):
        from tkinter import filedialog, messagebox
        model = self._ensure_model()
        path = filedialog.asksaveasfilename(
            defaultextension=".umbra.json", filetypes=[("Umbra model", "*.umbra.json")],
            initialfile=f"umbra_model_{int(time.time())}.umbra.json",
        )
        if not path:
            return
        try:
            model.save(path)
            self.lbl_status.config(text=f"Saved model ({model.generations_trained} steps)", foreground="green")
            logger.info(model.summary())
        except Exception as exc:
            messagebox.showerror("Save Model", str(exc))

    def _load_model(self):
        from tkinter import filedialog, messagebox
        from .checkpoint import UmbraModel
        path = filedialog.askopenfilename(filetypes=[("Umbra model", "*.umbra.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self.model = UmbraModel.load(path)
            self.lbl_status.config(text=f"Loaded: {self.model.summary()}", foreground="green")
            logger.info(self.model.summary())
        except Exception as exc:
            messagebox.showerror("Load Model", str(exc))

    def _capture_batch(self):
        """Record the current target's noise through speaker->mic, then train."""
        from tkinter import messagebox
        if self.state.reference_image is None:
            messagebox.showwarning("Capture", "Load a target image first.")
            return
        if self.simulation_mode_var.get():
            messagebox.showwarning("Capture", "Uncheck SILENT SIM to capture real audio.")
            return
        threading.Thread(target=self._capture_batch_worker, daemon=True).start()

    def _capture_batch_worker(self, repeats: int = 10):
        try:
            from .capture import record_batch, score_recordings
            from .encoding import NoiseStreamEncoder
            import tempfile

            idx_out = int(self.cb_speaker.get().split(":")[0]) if self.cb_speaker.get() else 0
            idx_in = int(self.cb_mic.get().split(":")[0]) if self.cb_mic.get() else 0
            self.root.after(0, lambda: self.lbl_status.config(text="Capturing...", foreground="orange"))

            seed = secrets.randbelow(2**31)
            ref = self.state.reference_image
            packet = NoiseStreamEncoder(sigma=float(self.state.difficulty)).encode(ref, seed=seed)
            out_dir = Path(tempfile.gettempdir()) / f"umbra_capture_{int(time.time())}"
            manifest = record_batch(self.audio, packet.encoded, out_dir=out_dir,
                                    repeats=repeats, idx_out=idx_out, idx_in=idx_in, label="air")
            paths = sorted(out_dir.glob("air_*.wav"))
            if not paths:
                self.root.after(0, lambda: self.lbl_status.config(text="Capture failed (no audio)", foreground="red"))
                return
            feats, rewards = score_recordings(paths, ref, seed=seed)
            self._ensure_model().train(feats, rewards, source="acoustic",
                                       recordings=[p.name for p in paths],
                                       notes=f"speaker->mic x{manifest['successful']}")
            msg = f"Captured {manifest['successful']}/{repeats}, trained model ({self.model.total_samples} samples)"
            self.root.after(0, lambda: self.lbl_status.config(text=msg, foreground="green"))
            logger.info(msg)
        except Exception as exc:
            logger.error(f"Capture batch failed: {exc}", exc_info=True)
            self.root.after(0, lambda e=exc: self.lbl_status.config(text=f"Capture error: {e}", foreground="red"))

    def _export_data(self):
        if not self.state.evolution_manager:
            return

        p = filedialog.asksaveasfilename(
            defaultextension=".json", 
            filetypes=[("JSON Data", "*.json")],
            initialfile=f"umbra_run_{int(time.time())}.json"
        )
        if not p:
            return
        
        success = self.state.evolution_manager.export_history_json(p)
        
        if success:
            count = len(self.state.evolution_manager.history)
            self.console_log.configure(state='normal')
            self.console_log.insert(tk.END, f"Exported {count} gens to {os.path.basename(p)}\n")
            self.console_log.configure(state='disabled')
        else:
            messagebox.showerror("Export Failed", "Could not write history file.")

    def _on_close(self):
        self.state.running = False
        self.root.destroy()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_pinterest_url(url: str) -> str:
    """Strip query parameters and fragment from a Pinterest image URL."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _generate_unique_model_path(directory: Path, generation_index: int) -> Path:
    """Generate a unique JSON file path inside *directory*."""
    alphabet = string.ascii_lowercase
    while True:
        tag = "".join(secrets.choice(alphabet) for _ in range(8))
        candidate = directory / f"{tag}_{generation_index}.json"
        if not candidate.exists():
            return candidate


# ---------------------------------------------------------------------------
# App state tracker for generation history
# ---------------------------------------------------------------------------

_HISTORY_MAXLEN = 500


class UmbraAppState:
    """Tracks generation metrics across an evolution run."""

    def __init__(self, maxlen: int = _HISTORY_MAXLEN) -> None:
        self.history: deque[dict] = deque(maxlen=maxlen)
        self.composite_scores: deque[float] = deque(maxlen=maxlen)
        self.sound_scores: deque[float] = deque(maxlen=maxlen)
        self.readability_scores: deque[float] = deque(maxlen=maxlen)

    def record_generation(
        self,
        generation: int,
        metrics,
        overlap: float,
        *,
        sound_metrics=None,
        sound_overlap: float = 0.0,
        sound_reference_metrics=None,
        sound_reference_overlap: float = 0.0,
        sound_reference_partial: float = 0.0,
        sound_alignment_partial: float = 0.0,
        sound_score: float = 0.0,
        sound_readability_score: float = 0.0,
        sound_alignment_score: float = 0.0,
        team_score: float = 0.0,
        ai_score_value: float | None = None,
        frame_time_ms: float = 0.0,
        execution_backend: str = "cpu",
    ) -> dict:
        ai_score = ai_score_value if ai_score_value is not None else _composite_score(overlap, metrics.psnr, metrics.ssim)

        sr_metrics = sound_reference_metrics or metrics

        entry: dict = {
            "generation": generation,
            "overlap": overlap,
            "psnr": metrics.psnr,
            "ssim": metrics.ssim,
            "ai_score": ai_score,
            "sound_psnr": sr_metrics.psnr,
            "sound_ssim": sr_metrics.ssim,
            "sound_overlap": sound_reference_overlap,
            "sound_score": sound_score,
            "sound_readability_score": sound_readability_score,
            "sound_alignment_score": sound_alignment_score,
            "sound_reference_partial": sound_reference_partial * 100.0,
            "sound_alignment_partial": sound_alignment_partial * 100.0,
            "team_score": team_score,
            "composite_score": team_score,
            "frame_time_ms": frame_time_ms,
            "frame_fps": 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0,
            "execution_backend": execution_backend,
        }

        self.history.append(entry)
        self.composite_scores.append(entry["composite_score"])
        self.sound_scores.append(entry["sound_score"])
        self.readability_scores.append(entry["sound_readability_score"])
        return entry


# ---------------------------------------------------------------------------
# Pinterest dataset management
# ---------------------------------------------------------------------------

@dataclass
class PinterestDatasetEntry:
    identifier: str
    url: str
    label: str
    theme: str
    filename: str
    size_bytes: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


class _Acquisition:
    """Result of acquiring a single image from a dataset."""

    def __init__(
        self,
        dataset_id: str,
        use_index: int,
        remaining_uses: int,
        declared_edge: int,
        actual_edge: int,
        image: Image.Image | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.use_index = use_index
        self.remaining_uses = remaining_uses
        self.declared_edge = declared_edge
        self.actual_edge = actual_edge
        self.image = image


class PinterestDatasetManager:
    """Lightweight dataset rotation manager."""

    def __init__(
        self,
        root: Path,
        feed_sources: dict,
        size_sequence: tuple[int, ...] = (256,),
        max_preview_pixels: int = 256 * 256,
        pool_size: int = 5,
        cycles_per_image: int = 10,
        min_edge: int = 64,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._feed_sources = feed_sources
        self._size_sequence = size_sequence
        self._max_preview_pixels = max_preview_pixels
        self._pool_size = pool_size
        self._cycles_per_image = cycles_per_image
        self._min_edge = min_edge

        self._state: dict = {}
        self._used_urls: set[str] = set()
        self._archive: list[dict] = []
        self._use_counter = 0
        self._size_index = 0

    def _save_state(self) -> None:
        state_path = self._root / "state.json"
        with open(state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    def acquire_image(self) -> _Acquisition:
        """Get the next image, cycling through uses and archiving when exhausted."""
        dataset_id = self._state.get("dataset_id")
        if dataset_id is None:
            raise RuntimeError("No active dataset")

        entries = self._state.get("entries", [])
        rotation = self._state.get("rotation", {})
        queue_ids = rotation.get("queue", [])
        index = rotation.get("index", 0)

        if not queue_ids:
            raise RuntimeError("Empty rotation queue")

        entry_id = queue_ids[index % len(queue_ids)]
        entry_data = next((e for e in entries if e["identifier"] == entry_id), None)
        if entry_data is None:
            raise RuntimeError(f"Entry {entry_id} not found")

        self._use_counter += 1
        remaining = max(0, self._cycles_per_image - self._use_counter)

        declared_edge = self._size_sequence[self._size_index % len(self._size_sequence)]
        self._size_index += 1

        dataset_dir = self._root / dataset_id
        img_path = dataset_dir / entry_data["filename"]
        actual_edge = declared_edge
        img = None
        if img_path.exists():
            img = Image.open(img_path)
            actual_edge = min(max(img.width, img.height), declared_edge)

        acq = _Acquisition(
            dataset_id=dataset_id,
            use_index=self._use_counter,
            remaining_uses=remaining,
            declared_edge=declared_edge,
            actual_edge=actual_edge,
            image=img,
        )

        if self._use_counter >= self._cycles_per_image:
            self._archive.append({"dataset_id": dataset_id})
            self._state.pop("dataset_id", None)
            self._use_counter = 0

        return acq


def main():
    root = tk.Tk()
    UmbraDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()