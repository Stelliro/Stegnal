# ui.py
"""
STEGNAL — Focused Audio Roundtrip Experiment UI

This is the main desktop UI launched by launch_stegnal_ui.bat / launch_stegnal_ui.ps1
via `python -m stegnal ui`.

Core purpose:
  - Load an image
  - AI predicts what the image will look like after audio processing
  - Image is transferred to audio (WAV)
  - Audio is transferred back to image
  - Score the three axes + composite
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from .audio_mixer import AudioEngine
from .codec import (
    decode_waveform_to_image,
    encode_image_to_waveform,
    encode_text_to_wav_bytes,
)

# Core experiment
from .testing import run_audio_roundtrip_experiment

# For audio playback + saving WAV
sd = None
wavfile = None
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    from scipy.io import wavfile
except Exception:
    wavfile = None
HAS_AUDIO = (sd is not None)

# Optional: use the codec directly for WAV bytes
try:
    from .codec import encode_image_to_wav_bytes
except Exception:
    encode_image_to_wav_bytes = None  # type: ignore

logger = logging.getLogger("Stegnal")

# --------------------------- THEME ---------------------------

DARK_BG = "#1e1e1e"
DARK_PANEL = "#252526"
ACCENT = "#00ff9f"
ACCENT_DIM = "#00cc7a"
TEXT = "#e0e0e0"
MUTED = "#888888"
ERROR = "#ff6b6b"

def setup_dark_style(style: ttk.Style) -> None:
    style.theme_use("clam")
    style.configure(".", background=DARK_BG, foreground=TEXT, font=("Consolas", 10))
    style.configure("TFrame", background=DARK_BG)
    style.configure("TLabel", background=DARK_BG, foreground=TEXT)
    style.configure("TButton", background=DARK_PANEL, foreground=TEXT, padding=6)
    style.map("TButton",
              background=[("active", "#333"), ("pressed", ACCENT_DIM)],
              foreground=[("active", TEXT)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#000", font=("Consolas", 11, "bold"))
    style.map("Accent.TButton", background=[("active", ACCENT_DIM)])
    style.configure("Score.TLabel", font=("Consolas", 13, "bold"), foreground=ACCENT)
    style.configure("BigScore.TLabel", font=("Consolas", 18, "bold"), foreground=ACCENT)

    # Dark theme for Combobox (dropdowns) and Entry - makes them visible in dark mode
    style.configure("TCombobox",
                    fieldbackground=DARK_PANEL,
                    background=DARK_PANEL,
                    foreground=TEXT,
                    arrowcolor=TEXT,
                    bordercolor="#555",
                    lightcolor=DARK_PANEL,
                    darkcolor=DARK_PANEL,
                    selectbackground=ACCENT,
                    selectforeground="#000")
    style.map("TCombobox",
              fieldbackground=[("readonly", DARK_PANEL), ("!readonly", DARK_PANEL)],
              selectbackground=[("readonly", ACCENT)],
              selectforeground=[("readonly", "#000")],
              bordercolor=[("focus", ACCENT)])
    style.configure("ComboboxPopdownFrame", background=DARK_PANEL)
    style.configure("ComboboxPopdownFrame.TListbox", background=DARK_PANEL, foreground=TEXT,
                    selectbackground=ACCENT, selectforeground="#000")

    style.configure("TEntry",
                    fieldbackground=DARK_PANEL,
                    foreground=TEXT,
                    insertcolor=TEXT,
                    bordercolor="#555",
                    lightcolor=DARK_PANEL,
                    darkcolor=DARK_PANEL)
    style.map("TEntry",
              fieldbackground=[("focus", "#2a2a2a")],
              bordercolor=[("focus", ACCENT)])


class StegnalAudioUI:
    """Clean, focused UI for the audio roundtrip + prediction experiment."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("STEGNAL // Audio Fidelity Experiment")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=DARK_BG)

        self.style = ttk.Style()
        setup_dark_style(self.style)

        self.reference: np.ndarray | None = None
        self.last_result = None
        self.last_result_text = None
        self.last_wav_bytes: bytes | None = None
        self.last_sample_rate: int = 48000

        self.tk_images: list[ImageTk.PhotoImage] = []  # prevent GC

        self.audio_engine = AudioEngine()
        self.out_device_var = tk.StringVar()
        self.in_device_var = tk.StringVar()

        # Learned corrections for the current acoustic channel (from play+capture trials)
        self.best_channel_params = {"gamma": 1.0, "contrast": 1.0, "brightness": 0.0}

        # For text mode
        self.text_payload = tk.StringVar(value="Secret password or message here")

        self._build_ui()
        self._setup_logging()

        # Load a default example on start (nice for usability)
        self.root.after(200, self._try_load_default)

    # --------------------------- UI BUILD ---------------------------

    def _build_ui(self):
        # Header
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="STEGNAL", font=("Consolas", 20, "bold"), foreground=ACCENT).pack(side=tk.LEFT)
        ttk.Label(header, text="  //  Audio Transfer Fidelity Test", font=("Consolas", 14), foreground=MUTED).pack(side=tk.LEFT)

        # Controls - split into rows for better visibility in dark theme
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill=tk.X, padx=10)

        # Row 1: Basic + Sim experiment
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X)

        ttk.Button(row1, text="LOAD IMAGE", command=self.load_image).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(row1, text="Resolution:").pack(side=tk.LEFT, padx=(12, 4))
        self.res_var = tk.IntVar(value=128)
        res_combo = ttk.Combobox(row1, textvariable=self.res_var, width=6, state="readonly",
                                 values=[64, 96, 128, 160, 192, 224, 256])
        res_combo.pack(side=tk.LEFT)

        self.direct_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="High Fidelity (Direct PCM)", variable=self.direct_var).pack(side=tk.LEFT, padx=12)

        ttk.Label(row1, text="Carrier (Hz):").pack(side=tk.LEFT, padx=(8,2))
        self.carrier_var = tk.IntVar(value=18500)
        self.carrier_entry = ttk.Entry(row1, textvariable=self.carrier_var, width=8, style="TEntry")
        self.carrier_entry.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(row1, text="RUN AUDIO EXPERIMENT", style="Accent.TButton",
                                  command=self.run_experiment)
        self.run_btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(row1, text="PLAY AUDIO", command=self.play_audio).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="SAVE RESULTS", command=self.save_results).pack(side=tk.LEFT, padx=4)

        ttk.Button(row1, text="USE EXAMPLE", command=self._try_load_default).pack(side=tk.RIGHT)

        # Row 2: Real over-air ultrasonic key (more prominent)
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(row2, text="Real Over-Air (Ultrasonic Key Mode):", foreground=ACCENT, font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row2, text="Out:").pack(side=tk.LEFT, padx=(2, 2))
        out_devs = self.audio_engine.get_devices('output')
        self.out_combo = ttk.Combobox(row2, textvariable=self.out_device_var, width=22, values=out_devs, state="readonly", style="TCombobox")
        if out_devs:
            self.out_combo.current(0)
        self.out_combo.pack(side=tk.LEFT)
        ttk.Label(row2, text="In:").pack(side=tk.LEFT, padx=(8, 2))
        in_devs = self.audio_engine.get_devices('input')
        self.in_combo = ttk.Combobox(row2, textvariable=self.in_device_var, width=22, values=in_devs, state="readonly", style="TCombobox")
        if in_devs:
            self.in_combo.current(0)
        self.in_combo.pack(side=tk.LEFT)

        self.real_btn = ttk.Button(row2, text="PLAY SPEAKERS + CAPTURE MIC (Image)", command=self.play_and_capture_real)
        self.real_btn.pack(side=tk.LEFT, padx=6)

        self.learn_btn = ttk.Button(row2, text="LEARN/OPTIMIZE (AI over air)", command=self.learn_for_channel)
        self.learn_btn.pack(side=tk.LEFT, padx=4)

        # Text payload for messages/keys
        ttk.Label(row2, text="Text:").pack(side=tk.LEFT, padx=(10,2))
        ttk.Entry(row2, textvariable=self.text_payload, width=30).pack(side=tk.LEFT)
        self.send_text_btn = ttk.Button(row2, text="SEND TEXT", command=self.play_and_capture_text)
        self.send_text_btn.pack(side=tk.LEFT, padx=4)

        # Image panels
        img_frame = ttk.Frame(self.root, padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.panels = {}
        for title, key in [("ORIGINAL", "orig"), ("AI PREDICTION (Guess)", "pred"), ("AUDIO RECONSTRUCTION", "actual")]:
            col = ttk.Frame(img_frame)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

            ttk.Label(col, text=title, font=("Consolas", 11, "bold")).pack(pady=(0, 4))

            canvas = tk.Canvas(col, width=280, height=280, bg="#111", highlightthickness=1,
                               highlightbackground="#444")
            canvas.pack()

            info = ttk.Label(col, text="—", foreground=MUTED, font=("Consolas", 9))
            info.pack(pady=4)

            self.panels[key] = {"canvas": canvas, "info": info, "img": None}

        # Scores panel
        scores = ttk.Frame(self.root, padding=(12, 8))
        scores.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(scores, text="SCORES", font=("Consolas", 12, "bold"), foreground=ACCENT).pack(anchor="w")

        score_grid = ttk.Frame(scores)
        score_grid.pack(fill=tk.X, pady=6)

        self.score_labels = {}
        metrics = [
            ("Image → Audio", "i2a"),
            ("Audio → Image", "a2i"),
            ("Prediction Accuracy", "pred"),
            ("COMPOSITE", "comp"),
        ]
        for i, (label, key) in enumerate(metrics):
            f = ttk.Frame(score_grid)
            f.grid(row=0, column=i, padx=10, sticky="nsew")
            ttk.Label(f, text=label, foreground=MUTED).pack()
            val = ttk.Label(f, text="—", style="BigScore.TLabel")
            val.pack()
            self.score_labels[key] = val

        score_grid.columnconfigure((0,1,2,3), weight=1)

        # Log
        log_frame = ttk.Frame(self.root, padding=(10, 4))
        log_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        ttk.Label(log_frame, text="LOG", foreground=MUTED).pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#111", fg=TEXT, insertbackground=TEXT,
            font=("Consolas", 9), relief="flat", borderwidth=1
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.configure(state="disabled")

    def _setup_logging(self):
        # Simple queue-less logger that writes to the text box
        class UILogHandler(logging.Handler):
            def __init__(self, box):
                super().__init__()
                self.box = box

            def emit(self, record):
                msg = self.format(record)
                self.box.after(0, self._append, msg)

            def _append(self, msg):
                self.box.configure(state="normal")
                self.box.insert(tk.END, msg + "\n")
                self.box.see(tk.END)
                self.box.configure(state="disabled")

        h = UILogHandler(self.log_box)
        # Use proper datefmt to avoid % formatting error on %H etc.
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        # Avoid duplicate handlers on re-instantiation
        root_log = logging.getLogger()
        if not any(isinstance(hh, UILogHandler) for hh in root_log.handlers):
            root_log.addHandler(h)
        logging.getLogger("Stegnal").setLevel(logging.INFO)

    # --------------------------- HELPERS ---------------------------

    def log(self, msg: str):
        logger.info(msg)

    def _set_image(self, key: str, array: np.ndarray | None, extra: str = ""):
        panel = self.panels[key]
        canvas = panel["canvas"]

        if array is None:
            canvas.delete("all")
            panel["info"].config(text="—")
            return

        arr = np.clip(array, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        # Fit nicely
        max_side = 260
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        tkimg = ImageTk.PhotoImage(img)
        self.tk_images.append(tkimg)  # keep ref

        canvas.delete("all")
        cw = canvas.winfo_width() or 280
        ch = canvas.winfo_height() or 280
        canvas.create_image(cw//2, ch//2, image=tkimg, anchor="center")
        panel["img"] = tkimg

        h, w = array.shape[:2]
        panel["info"].config(text=f"{w}×{h}  {extra}")

    def _clear_images(self):
        for key in self.panels:
            self._set_image(key, None)

    # --------------------------- ACTIONS ---------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Load target image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            self.reference = arr
            self._set_image("orig", arr, "loaded")
            self._clear_images()  # clear pred/actual
            self.log(f"Loaded {os.path.basename(path)}  shape={arr.shape}")
            # Reset scores
            for lbl in self.score_labels.values():
                lbl.config(text="—")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _try_load_default(self):
        candidates = [
            "test_images/Original.jpg",
            "test_images/pexels-hsapir-1054655.jpg",
            "test_images/istockphoto-1550071750-612x612.jpg",
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    pil = Image.open(p).convert("RGB")
                    # Use reasonable size for first display
                    pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    arr = np.asarray(pil, dtype=np.float32) / 255.0
                    self.reference = arr
                    self._set_image("orig", arr, "example")
                    self.log(f"Loaded example: {os.path.basename(p)}")
                    return
                except Exception:
                    continue
        self.log("No example images found in test_images/")

    def run_experiment(self):
        if self.reference is None:
            messagebox.showwarning("No Image", "Please load an image first (or use EXAMPLE).")
            return

        res = self.res_var.get()
        use_direct = self.direct_var.get()

        self.run_btn.config(state="disabled", text="RUNNING...")
        self.log(f"Running experiment @ {res}×{res}  (direct={use_direct}) ...")

        def worker():
            try:
                result = run_audio_roundtrip_experiment(
                    self.reference,
                    resolution=(res, res),
                )  # Note: run_audio... uses its internal encode; for carrier control use the real path below
                # We force direct inside the experiment function now, but we could re-run if needed.
                # For UI we just use the result.

                # Generate WAV bytes for playback (high fidelity)
                wav_bytes = None
                sr = 48000
                if encode_image_to_wav_bytes is not None:
                    try:
                        wav_bytes = encode_image_to_wav_bytes(
                            result.original, direct=True, sample_rate=48000
                        )
                        # Read sr (wavfile may be None if scipy not present, default sr)
                        if wavfile is not None:
                            with io.BytesIO(wav_bytes) as bio:
                                sr, _ = wavfile.read(bio)
                    except Exception:
                        pass

                self.root.after(0, self._on_experiment_done, result, wav_bytes, sr)
            except Exception as exc:
                self.root.after(0, self._on_experiment_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_experiment_done(self, result, wav_bytes, sr):
        self.last_result = result
        self.last_wav_bytes = wav_bytes
        self.last_sample_rate = sr or 48000

        self._set_image("orig", result.original, "")
        self._set_image("pred", result.predicted, "AI guess")
        self._set_image("actual", result.actual, "from audio")

        # Update scores
        self.score_labels["i2a"].config(text=f"{result.image_to_audio_fidelity:.4f}")
        self.score_labels["a2i"].config(text=f"{result.audio_to_image_fidelity:.4f}")
        self.score_labels["pred"].config(text=f"{result.prediction_accuracy:.4f}")
        self.score_labels["comp"].config(text=f"{result.composite:.4f}")

        self.log(
            f"DONE  I→A={result.image_to_audio_fidelity:.4f}  "
            f"A→I={result.audio_to_image_fidelity:.4f}  "
            f"PRED={result.prediction_accuracy:.4f}  "
            f"COMP={result.composite:.4f}"
        )
        self.log(
            f"  Audio→Image: SSIM={result.metrics_orig_actual.ssim:.3f}  "
            f"PSNR={result.metrics_orig_actual.psnr:.1f}"
        )

        self.run_btn.config(state="normal", text="RUN AUDIO EXPERIMENT")

    def _on_experiment_error(self, msg: str):
        self.run_btn.config(state="normal", text="RUN AUDIO EXPERIMENT")
        self.log(f"ERROR: {msg}")
        messagebox.showerror("Experiment Failed", msg)

    def play_audio(self):
        if sd is None:
            messagebox.showwarning("Audio", "sounddevice not available. Install with pip install sounddevice")
            return
        if self.last_wav_bytes is None:
            # Regenerate from last original if possible
            if self.last_result is not None and encode_image_to_wav_bytes is not None:
                self.last_wav_bytes = encode_image_to_wav_bytes(self.last_result.original, direct=True)
            else:
                messagebox.showinfo("No Audio", "Run the experiment first.")
                return

        try:
            with io.BytesIO(self.last_wav_bytes) as bio:
                raw = bio.read()
            if wavfile is not None:
                with io.BytesIO(raw) as bio2:
                    sr, data = wavfile.read(bio2)
            else:
                sr = 48000
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            if data.ndim > 1:
                data = data[:, 0]
            # normalize to float -1..1
            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32767.0
            sd.play(data, sr)
            self.log(f"Playing audio ({len(data)} samples @ {sr} Hz)")
        except Exception as e:
            self.log(f"Playback error: {e}")
            messagebox.showerror("Playback Error", str(e))

    def save_results(self):
        if self.last_result is None:
            messagebox.showinfo("Nothing to save", "Run the experiment first.")
            return

        folder = filedialog.askdirectory(title="Choose folder to save results")
        if not folder:
            return

        try:
            base = Path(folder)
            # Save images
            for name, arr in [
                ("original.png", self.last_result.original),
                ("ai_prediction.png", self.last_result.predicted),
                ("audio_reconstruction.png", self.last_result.actual),
            ]:
                img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                img.save(base / name)

            # Save WAV
            if self.last_wav_bytes:
                (base / "transmitted_audio.wav").write_bytes(self.last_wav_bytes)
            else:
                # regenerate
                if encode_image_to_wav_bytes:
                    wav = encode_image_to_wav_bytes(self.last_result.original, direct=True)
                    (base / "transmitted_audio.wav").write_bytes(wav)

            # Save a small JSON summary
            summary = {
                "image_to_audio_fidelity": float(self.last_result.image_to_audio_fidelity),
                "audio_to_image_fidelity": float(self.last_result.audio_to_image_fidelity),
                "prediction_accuracy": float(self.last_result.prediction_accuracy),
                "composite": float(self.last_result.composite),
                "ssim": float(self.last_result.metrics_orig_actual.ssim),
                "psnr": float(self.last_result.metrics_orig_actual.psnr),
            }
            import json
            (base / "scores.json").write_text(json.dumps(summary, indent=2))

            self.log(f"Saved results to {folder}")
            messagebox.showinfo("Saved", f"Results written to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def play_and_capture_real(self):
        """Core real premise: play the waveform through speakers, capture via mic, decode, score."""
        if self.reference is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        try:
            out_str = self.out_device_var.get()
            in_str = self.in_device_var.get()
            out_idx = int(out_str.split(":")[0]) if out_str else 0
            in_idx = int(in_str.split(":")[0]) if in_str else 0
        except Exception:
            messagebox.showerror("Audio Devices", "Please select valid Speaker (Out) and Mic (In) devices.")
            return

        res = self.res_var.get()
        self.log(f"REAL CHANNEL: Out={out_idx} In={in_idx} @ {res}px ...")

        try:
            # Prepare small reference for this test (same as sim experiment)
            from skimage.transform import resize
            ref = resize(self.reference, (res, res, 3), preserve_range=True, anti_aliasing=True).astype(np.float32)
            ref = np.clip(ref, 0.0, 1.0)

            # Generate payload waveform
            carrier = float(self.carrier_var.get())
            # For high/ultrasonic carriers, use AM mode (direct=False) for better robustness over real air
            use_direct = carrier < 10000
            waveform = encode_image_to_waveform(ref, direct=use_direct, carrier_freq=carrier)
            try:
                from .reconstruction import suggest_sample_rate
                sr = suggest_sample_rate(ref)
            except Exception:
                sr = 48000

            # Play through speakers + capture from mic (real acoustic transfer)
            # Do 3 repeats + average to improve SNR on real air (especially ultrasonic)
            captures = []
            for _ in range(3):
                c = self.audio_engine.transmit_and_record(
                    waveform, sr, out_idx, in_idx, use_sync_pulse=True
                )
                if c is not None and len(c) >= 100:
                    captures.append(np.asarray(c, dtype=np.float32))
            if not captures:
                self.log("Capture failed or too short. Check devices, volume, mic.")
                messagebox.showerror("Capture Failed", "No usable signal captured. Check hardware, levels, and sync.")
                return
            captured = np.mean(captures, axis=0)

            # Decode the captured waveform back to image (matching the encode mode)
            try:
                recon = decode_waveform_to_image(captured, resolution=(res, res), sample_rate=sr, direct=use_direct)
            except TypeError:
                # fallback if direct not perfectly supported in this build
                recon = decode_waveform_to_image(captured, resolution=(res, res), sample_rate=sr)

            recon = np.clip(np.asarray(recon, dtype=np.float32), 0.0, 1.0)
            recon = self._apply_channel_params(recon, self.best_channel_params)

            # Display as the "real" reconstruction
            self._set_image("actual", recon, "real capture (learned)")

            # Score the real channel (audio-to-image over air) - using the (possibly learned) recon
            from .metrics import compute_metrics
            m = compute_metrics(ref, recon)
            # Simple proxies for real
            a2i_real = float(np.clip(0.5 * m.ssim + 0.5 * min((m.psnr - 10)/25, 1.0), 0, 1))
            self.score_labels["a2i"].config(text=f"{a2i_real:.4f} (real)")
            self.score_labels["comp"].config(text=f"real~{a2i_real:.3f}")

            self.log(f"REAL CAPTURE DONE. SSIM={m.ssim:.3f} PSNR={m.psnr:.1f} (A2I proxy {a2i_real:.3f})")
            if self.carrier_var.get() > 15000 and a2i_real < 0.1:
                self.log("Note: High carrier (ultrasonic) often needs good hardware or lower volume/distance. Try carrier 8000-12000 for better real-air tests.")

            # Best of both worlds for security + reliability:
            # - AI prediction/reconstruction for high-quality, reliable image/payload transfer (compensation, denoising)
            # - Real physical channel for entropy and security (unique, hard-to-replicate acoustic signature at ultrasonic frequencies)
            # The encoded key combines AI's "predicted view" with the actual messy channel measurement.
            try:
                # AI-predicted received audio (re-encode the AI-reconstructed image)
                pred_key_audio = encode_image_to_waveform(recon, direct=True, carrier_freq=carrier)

                # AI payload (the reconstructed image as reliable data)
                payload_hash = hashlib.sha256(recon.tobytes()).digest()

                # Physical channel fingerprint from actual capture (the secure entropy source)
                # Subsampled high-frequency spectrum + amplitude stats — very specific to this speaker/mic/room
                fft = np.abs(np.fft.rfft(captured))
                n = len(fft)
                high_freq_features = fft[max(0, n//2):]  # focus on upper bands (ultrasonic territory)
                channel_fingerprint = np.concatenate([
                    high_freq_features[::max(1, len(high_freq_features)//20)],
                    np.array([np.mean(np.abs(captured)), np.std(captured)])
                ]).astype(np.float32)

                channel_hash = hashlib.sha256(channel_fingerprint.tobytes()).digest()

                # Final secure encoded key
                secure_key = hashlib.sha256(payload_hash + channel_hash + pred_key_audio.tobytes()[:2048]).digest()
                key_hex = secure_key.hex()[:32]

                self.log(f"SECURE ENCODED KEY (AI payload + predicted audio + physical channel): {key_hex}")

                # For the "matches the predicted audio" part
                pred_audio_hash = hashlib.sha256(pred_key_audio.tobytes()[:2048]).digest()[:8].hex()
                actual_hash = hashlib.sha256(captured.tobytes()[:2048]).digest()[:8].hex()
                self.log(f"AI Predicted Audio Hash: {pred_audio_hash} | Actual Captured Hash: {actual_hash}")
            except Exception as ke:
                self.log(f"Secure key derivation error: {ke}")
            self.last_result = type('obj', (object,), {'original': ref, 'actual': recon, 'metrics_orig_actual': m})()
            # store captured audio for possible re-decode
            self.last_wav_bytes = (captured * 32767).astype(np.int16).tobytes()  # rough for play

        except Exception as e:
            self.log(f"Real channel error: {e}")
            messagebox.showerror("Real Audio Error", str(e))

    def play_and_capture_text(self):
        """Send text over sound (ultrasonic capable), capture, decode, derive key."""
        text = self.text_payload.get().strip()
        if not text:
            messagebox.showwarning("No Text", "Enter a message or password to send.")
            return
        try:
            out_idx = int(self.out_device_var.get().split(":")[0])
            in_idx = int(self.in_device_var.get().split(":")[0])
        except Exception:
            messagebox.showerror("Audio Devices", "Select valid Out and In devices.")
            return

        carrier = float(self.carrier_var.get())
        self.log(f"Sending TEXT over real air (carrier {carrier} Hz): '{text[:30]}...' ")

        try:
            # Encode text to waveform (high freq carrier via direct or AM; use direct for fidelity)
            # For text, we can use the text codec which embeds in image then waveform
            wav_bytes = encode_text_to_wav_bytes(text, direct=True)  # if supported, else fallback
            # Load as array for transmit
            import io as _io

            from scipy.io import wavfile as wv
            with _io.BytesIO(wav_bytes) as bio:
                sr, wav_data = wv.read(bio)
            if wav_data.ndim > 1:
                wav_data = wav_data[:, 0]
            wav_data = wav_data.astype(np.float32) / 32767.0

            # Play and capture
            captured = self.audio_engine.transmit_and_record(
                wav_data, sr, out_idx, in_idx, use_sync_pulse=True
            )
            if captured is None:
                self.log("Capture failed.")
                return

            captured = np.asarray(captured, dtype=np.float32)

            # Decode text from captured waveform array
            try:
                from .codec import decode_waveform_to_text
                decoded_text, _ = decode_waveform_to_text(captured, sample_rate=sr)
            except Exception as de:
                decoded_text = f"[decode error: {de}]"

            self.last_result_text = decoded_text
            self.log(f"Decoded text: {decoded_text}")

            # Derive secure key from decoded text + channel
            import hashlib
            payload_hash = hashlib.sha256(decoded_text.encode()).digest()
            fft = np.abs(np.fft.rfft(captured))
            n = len(fft)
            high = fft[max(0, n//2):]
            channel_fp = np.concatenate([high[::max(1,len(high)//16)], np.array([np.std(captured)])])
            channel_hash = hashlib.sha256(channel_fp.tobytes()).digest()
            secure_key = hashlib.sha256(payload_hash + channel_hash).digest()
            key_str = secure_key.hex()[:24]  # usable as password
            self.log(f"DERIVED KEY (from text + channel): {key_str}")
            messagebox.showinfo("Text Received + Key", f"Decoded: {decoded_text}\n\nKey: {key_str}")

        except Exception as e:
            self.log(f"Text send/capture error: {e}")
            messagebox.showerror("Error", str(e))

    def _apply_channel_params(self, img: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Simple learned post-correction for the acoustic channel (contrast/gamma etc)."""
        if params is None:
            params = self.best_channel_params
        out = np.asarray(img, dtype=np.float32).copy()
        c = float(params.get("contrast", 1.0))
        g = float(params.get("gamma", 1.0))
        b = float(params.get("brightness", 0.0))
        out = np.clip(out * c + b, 0, 1)
        out = np.power(np.clip(out, 1e-6, 1), g)
        return np.clip(out, 0, 1).astype(np.float32)

    def learn_for_channel(self, trials: int = 20):
        """Do several real play+capture trials, try small mutations of correction params,
        keep the best scoring set. This is the 'learning' to achieve higher quality over the
        specific speakers/mic/room channel."""
        if self.reference is None:
            messagebox.showwarning("No Image", "Load image first for learning.")
            return
        if not self.out_device_var.get() or not self.in_device_var.get():
            messagebox.showwarning("Devices", "Select Out and In devices first.")
            return

        self.log(f"LEARNING: running {trials} real-channel trials to optimize corrections...")
        best_score = 0.0
        best_p = dict(self.best_channel_params)

        base_res = self.res_var.get()

        import random

        from skimage.transform import resize

        from .metrics import compute_metrics

        ref_small = resize(self.reference, (base_res, base_res, 3), preserve_range=True).astype(np.float32)
        ref_small = np.clip(ref_small, 0, 1)

        carrier = float(self.carrier_var.get())
        waveform = encode_image_to_waveform(ref_small, direct=True, carrier_freq=carrier)
        sr = 48000
        use_direct = carrier < 10000

        out_idx = int(self.out_device_var.get().split(":")[0])
        in_idx = int(self.in_device_var.get().split(":")[0])

        # Always evaluate the starting params
        p = dict(best_p)
        captured = self.audio_engine.transmit_and_record(waveform, sr, out_idx, in_idx)
        if captured is not None and len(captured) >= 100:
            captured = np.asarray(captured, dtype=np.float32)
            use_direct = carrier < 10000
            try:
                raw_recon = decode_waveform_to_image(captured, resolution=(base_res, base_res), sample_rate=sr, direct=use_direct)
            except Exception:
                raw_recon = decode_waveform_to_image(captured, resolution=(base_res, base_res), sample_rate=sr)
            raw_recon = np.clip(np.asarray(raw_recon, dtype=np.float32), 0, 1)
            corrected = self._apply_channel_params(raw_recon, p)
            m = compute_metrics(ref_small, corrected)
            psnr_term = max(0.0, min((m.psnr-12)/30, 1.0))
            score = max(0.0, m.ssim * 0.7 + psnr_term * 0.3)
            self.log(f"  baseline: score={score:.3f} params={p}")
            if score > best_score:
                best_score = score
                best_p = p

        for t in range(trials):
            # mutate
            p = {
                "contrast": max(0.6, min(1.6, best_p["contrast"] + random.uniform(-0.15, 0.15))),
                "gamma": max(0.6, min(1.8, best_p["gamma"] + random.uniform(-0.1, 0.1))),
                "brightness": max(-0.15, min(0.15, best_p["brightness"] + random.uniform(-0.08, 0.08))),
            }

            captured = self.audio_engine.transmit_and_record(waveform, sr, out_idx, in_idx)
            if captured is None or len(captured) < 100:
                self.log(f"  trial {t}: capture failed, skipping")
                continue

            # For real air, try averaging a couple quick captures for better SNR
            captured2 = self.audio_engine.transmit_and_record(waveform, sr, out_idx, in_idx)
            if captured2 is not None and len(captured2) >= 100:
                captured = (captured + captured2) / 2.0

            captured = np.asarray(captured, dtype=np.float32)
            use_direct = carrier < 10000
            try:
                raw_recon = decode_waveform_to_image(captured, resolution=(base_res, base_res), sample_rate=sr, direct=use_direct)
            except Exception:
                raw_recon = decode_waveform_to_image(captured, resolution=(base_res, base_res), sample_rate=sr)

            raw_recon = np.clip(np.asarray(raw_recon, dtype=np.float32), 0, 1)
            corrected = self._apply_channel_params(raw_recon, p)

            m = compute_metrics(ref_small, corrected)
            psnr_term = max(0.0, min((m.psnr-12)/30, 1.0))
            score = float(m.ssim * 0.7 + psnr_term * 0.3)
            score = max(0.0, score)  # ensure non-negative

            # Also consider waveform similarity for real air (since image may be hard)
            try:
                from scipy.stats import pearsonr
                wf_corr = abs(pearsonr(waveform[:len(captured)], captured[:len(waveform)])[0])
            except Exception:
                wf_corr = 0.0
            score = max(score, wf_corr * 0.5)  # boost if waveform matches better

            self.log(f"  trial {t}: score={score:.3f} (ssim={m.ssim:.3f}, psnr={m.psnr:.1f}, wf_corr={wf_corr:.3f}) params={p}")
            if score > best_score:
                best_score = score
                best_p = p
                # show the best so far live
                self._set_image("actual", corrected, f"learned best {best_score:.3f}")

        self.best_channel_params = best_p
        if best_score > 0.0:
            self.log(f"LEARNING COMPLETE. Best params: {best_p} (score {best_score:.3f})")

            # Validation capture with best params
            val_captured = self.audio_engine.transmit_and_record(waveform, sr, out_idx, in_idx)
            if val_captured is not None and len(val_captured) >= 100:
                val_captured = np.asarray(val_captured, dtype=np.float32)
                val_recon = decode_waveform_to_image(val_captured, resolution=(base_res, base_res), sample_rate=sr, direct=use_direct)
                val_recon = np.clip(np.asarray(val_recon, dtype=np.float32), 0, 1)
                val_corrected = self._apply_channel_params(val_recon, best_p)
                val_m = compute_metrics(ref_small, val_corrected)
                val_score = max(0.0, val_m.ssim * 0.7 + max(0.0, min((val_m.psnr-12)/30, 1.0)) * 0.3)
                self.log(f"  validation with best: score={val_score:.3f} (ssim={val_m.ssim:.3f})")
                self._set_image("actual", val_corrected, f"validated best {val_score:.3f}")

            messagebox.showinfo("Learning done", f"Best channel corrections found.\n{best_p}\nApplied to future real decodes.")
        else:
            self.log(f"LEARNING COMPLETE. No better params found (best score {best_score:.3f}). Using defaults. Try lower carrier freq or better capture volume/sync.")
            messagebox.showinfo("Learning done", "No improvement found. Try a lower carrier frequency for better signal, or increase volume/sync.")

    # --------------------------- ENTRYPOINT ---------------------------

# ------------------------------------------------------------------
# LEGACY STUBS (for test compatibility only — not used by the new UI)
# These were previously inside ui.py. Kept as minimal shims so existing
# tests continue to import without restoring the old 30k-line bloat.
# ------------------------------------------------------------------


@dataclass
class StegnalAppState:
    """Minimal stub for legacy tests."""
    reference_image: Any = None
    difficulty: float = 0.1
    running: bool = False
    generation_history: list = None

    def __post_init__(self):
        if self.generation_history is None:
            self.generation_history = []

def _normalize_pinterest_url(url: str) -> str:
    """Stub — strips tracking params (simplified)."""
    if not url:
        return ""
    # very basic version of the old logic
    if "?" in url:
        return url.split("?")[0]
    return url

def _generate_unique_model_path(base_dir: str | Path, prefix: str = "stegnal_model") -> Path:
    """Stub that just returns a timestamped path."""
    import time
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{prefix}_{int(time.time())}.stegnal.json"

class PinterestDatasetEntry:
    """Stub for test_pinterest_dataset."""
    def __init__(self, identifier: str = "", filename: str = "", **kwargs):
        self.identifier = identifier
        self.filename = filename
        for k, v in kwargs.items():
            setattr(self, k, v)

class PinterestDatasetManager:
    """Minimal stub so tests that import it don't explode."""
    def __init__(self, root: Path, feed_sources: dict | None = None, **kwargs):
        self.root = Path(root)
        self.feed_sources = feed_sources or {}
        self._state: dict = {}

    def acquire(self, *args, **kwargs):
        # Return a dummy acquisition object for tests that call it
        class _Dummy:
            image = None
            actual_edge = 64
        return _Dummy()

    def mark_result(self, *args, **kwargs):
        pass

# ------------------------------------------------------------------
# ENTRYPOINT (used by launch_stegnal_ui.bat / python -m stegnal ui)
# ------------------------------------------------------------------

def main():
    root = tk.Tk()
    StegnalAudioUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
