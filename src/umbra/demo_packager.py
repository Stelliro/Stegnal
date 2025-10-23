"""Utilities for bundling shareable Umbra demo archives and executables."""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
import textwrap
import zipapp
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
from PIL import Image

from .codec import encode_image_to_wav_bytes

PACKAGE_NAME = "umbra_demo.pyz"

_DEMO_MAIN = textwrap.dedent(
    """
    \"\"\"Command-line entry point for the Umbra codec demo.\"\"\"

    from __future__ import annotations

    import argparse
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from umbra.codec import (
        decode_wav_bytes_to_image,
        encode_image_to_wav_bytes,
    )


    def _load_image(path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.clip(array, 0.0, 1.0)


    def _save_image(array: np.ndarray, path: Path) -> None:
        clipped = np.clip(array, 0.0, 1.0)
        image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
        image.save(path)


    def _encode(image_path: Path, wav_path: Path, sample_rate: int) -> None:
        array = _load_image(image_path)
        wav_bytes = encode_image_to_wav_bytes(array, sample_rate=sample_rate)
        wav_path.write_bytes(wav_bytes)


    def _decode(wav_path: Path, image_path: Path, resolution: tuple[int, int]) -> None:
        wav_bytes = wav_path.read_bytes()
        image, _ = decode_wav_bytes_to_image(wav_bytes, resolution=resolution)
        _save_image(image, image_path)


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description="Umbra codec demo")
        subparsers = parser.add_subparsers(dest="command", required=True)

        encode_parser = subparsers.add_parser("encode", help="Encode an image to WAV")
        encode_parser.add_argument("image", type=Path, help="Path to the source image")
        encode_parser.add_argument(
            "wav",
            type=Path,
            help="Destination WAV file",
        )
        encode_parser.add_argument(
            "--sample-rate",
            type=int,
            default=48000,
            help="Sample rate used during encoding (default: 48000)",
        )

        decode_parser = subparsers.add_parser("decode", help="Decode a WAV back to an image")
        decode_parser.add_argument("wav", type=Path, help="Path to the WAV file")
        decode_parser.add_argument(
            "image",
            type=Path,
            help="Destination path for the reconstructed image",
        )
        decode_parser.add_argument(
            "--rows",
            type=int,
            required=True,
            help="Height of the encoded image",
        )
        decode_parser.add_argument(
            "--cols",
            type=int,
            required=True,
            help="Width of the encoded image",
        )

        args = parser.parse_args(argv)

        if args.command == "encode":
            _encode(args.image, args.wav, args.sample_rate)
            return 0

        if args.command == "decode":
            resolution = (args.rows, args.cols)
            _decode(args.wav, args.image, resolution)
            return 0

        parser.error("Unknown command")
        return 1


    if __name__ == "__main__":  # pragma: no cover - entry point
        sys.exit(main())
    """
)


def build_demo_package(source_root: Path | None = None) -> tuple[str, bytes]:
    """Create a portable ``.pyz`` archive bundling the Umbra codec demo."""

    module_root = Path(source_root or Path(__file__).resolve().parent)
    if not module_root.exists():
        raise FileNotFoundError(f"Unable to locate Umbra sources at {module_root}")

    with tempfile.TemporaryDirectory(prefix="umbra_demo_") as tmp_dir:
        staging_root = Path(tmp_dir)
        app_root = staging_root / "app"
        app_root.mkdir(parents=True, exist_ok=True)

        target_module_dir = app_root / "umbra"
        shutil.copytree(module_root, target_module_dir, dirs_exist_ok=True)

        main_path = app_root / "__main__.py"
        main_path.write_text(_DEMO_MAIN)

        output_path = staging_root / PACKAGE_NAME
        zipapp.create_archive(
            app_root,
            target=output_path,
            interpreter="/usr/bin/env python3",
        )

        return PACKAGE_NAME, output_path.read_bytes()


_DEMO_GUI_TEMPLATE = Template(
    textwrap.dedent(
        '''
        """Standalone Umbra demo generated from the desktop application."""

        from __future__ import annotations

        import base64
        import io
        import tkinter as tk
        from pathlib import Path
        from tkinter import filedialog, messagebox, ttk

        import numpy as np
        from PIL import Image, ImageTk

        from umbra.codec import (
            decode_wav_bytes_to_image,
            encode_image_to_wav_bytes,
            encode_text_to_image,
            encode_text_to_waveform,
            save_waveform_as_wav,
        )

        _SAMPLE_IMAGE_B64 = "$IMAGE_B64"
        _SAMPLE_WAV_B64 = "$WAV_B64"
        _SAMPLE_METADATA = $METADATA


        def _sample_image_array() -> np.ndarray:
            data = base64.b64decode(_SAMPLE_IMAGE_B64.encode("ascii"))
            with Image.open(io.BytesIO(data)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            return np.clip(array, 0.0, 1.0)


        def _sample_wav_bytes() -> bytes:
            return base64.b64decode(_SAMPLE_WAV_B64.encode("ascii"))


        def _array_to_photo(array: np.ndarray) -> ImageTk.PhotoImage:
            clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
            image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
            return ImageTk.PhotoImage(image)


        def _resize_for_preview(array: np.ndarray, max_edge: int = 480) -> np.ndarray:
            clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
            rows, cols = clipped.shape[:2]
            longest = max(rows, cols, 1)
            scale = min(1.0, float(max_edge) / float(longest))
            if scale >= 1.0:
                return clipped
            new_size = (max(1, int(cols * scale)), max(1, int(rows * scale)))
            image = Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")
            resized = image.resize(new_size, Image.BILINEAR)
            return np.asarray(resized, dtype=np.float32) / 255.0


        class DemoApp:
            """Compact Tk interface for sharing Umbra reconstructions."""

            def __init__(self, root: tk.Tk) -> None:
                self.root = root
                self.root.title("Umbra Demo")
                self.root.geometry("840x640")

                self.metadata = dict(_SAMPLE_METADATA)
                self.sample_array = _sample_image_array()
                self.preview_array = self.sample_array
                self.preview_photo = _array_to_photo(_resize_for_preview(self.sample_array))
                self.status_var = tk.StringVar(value="Previewing bundled reconstruction.")
                self.text_metadata = None
                self.text_payload: str | None = None

                self.sample_rate = int(self.metadata.get("sample_rate", 48000))
                self.segments = int(self.metadata.get("segments", 1))
                self.marker = float(self.metadata.get("marker_duration", 0.05))
                self.rows = int(self.metadata.get("rows", self.sample_array.shape[0]))
                self.cols = int(self.metadata.get("cols", self.sample_array.shape[1]))

                self._build_layout()

            # ---------------------------------------------------------- layout
            def _build_layout(self) -> None:
                main = ttk.Frame(self.root, padding=16)
                main.pack(fill=tk.BOTH, expand=True)

                header = ttk.Label(main, text="Project Umbra demo", font=("Segoe UI", 16, "bold"))
                header.pack(anchor=tk.W)
                ttk.Label(
                    main,
                    text="Preview the shared reconstruction or try encoding your own images and text.",
                    wraplength=560,
                ).pack(anchor=tk.W, pady=(4, 16))

                content = ttk.Frame(main)
                content.pack(fill=tk.BOTH, expand=True)

                preview_frame = ttk.LabelFrame(content, text="Preview")
                preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

                self.preview_label = ttk.Label(preview_frame, image=self.preview_photo)
                self.preview_label.pack(padx=8, pady=8)

                info_lines = [
                    f"Sample rate: {self.sample_rate} Hz",
                    f"Segments: {self.segments}",
                    f"Marker tone: {self.marker:.3f} s",
                    f"Resolution: {self.rows}×{self.cols}",
                ]
                label_text = self.metadata.get("label")
                if label_text:
                    info_lines.insert(0, f"Label: {label_text}")
                ttk.Label(preview_frame, text="\n".join(info_lines), justify=tk.LEFT).pack(padx=8, pady=(0, 12))

                sample_buttons = ttk.Frame(preview_frame)
                sample_buttons.pack(fill=tk.X, padx=8, pady=(0, 12))
                ttk.Button(sample_buttons, text="Save sample image…", command=self.save_sample_image).pack(
                    fill=tk.X, pady=2
                )
                ttk.Button(sample_buttons, text="Save sample WAV…", command=self.save_sample_wav).pack(
                    fill=tk.X, pady=2
                )
                ttk.Button(sample_buttons, text="Reset preview", command=self.reset_preview).pack(
                    fill=tk.X, pady=2
                )

                actions = ttk.LabelFrame(content, text="Try it yourself")
                actions.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                ttk.Button(
                    actions,
                    text="Encode an image to WAV…",
                    command=self.encode_image_to_wav,
                ).pack(fill=tk.X, padx=12, pady=(12, 4))
                ttk.Button(
                    actions,
                    text="Decode a WAV to image…",
                    command=self.decode_wav_to_image,
                ).pack(fill=tk.X, padx=12, pady=4)

                text_frame = ttk.LabelFrame(actions, text="Text to colour static")
                text_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 4))
                self.text_widget = tk.Text(text_frame, height=5, wrap=tk.WORD)
                self.text_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                self.text_widget.insert(
                    "1.0",
                    "Paste text (even long passages) and Umbra will embed it into colourful static.",
                )
                text_buttons = ttk.Frame(text_frame)
                text_buttons.pack(fill=tk.X, padx=4, pady=(0, 4))
                ttk.Button(text_buttons, text="Preview static", command=self.preview_text_static).pack(
                    side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
                )
                ttk.Button(text_buttons, text="Save text as WAV…", command=self.save_text_as_wav).pack(
                    side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
                )

                ttk.Label(actions, textvariable=self.status_var, wraplength=320).pack(
                    fill=tk.X, padx=12, pady=(8, 0)
                )

            # ----------------------------------------------------------- helpers
            def _update_preview(self, array: np.ndarray) -> None:
                self.preview_array = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
                resized = _resize_for_preview(self.preview_array)
                self.preview_photo = _array_to_photo(resized)
                self.preview_label.configure(image=self.preview_photo)

            def _text_value(self) -> str:
                if self.text_widget is None:
                    return ""
                return self.text_widget.get("1.0", tk.END).strip()

            # ---------------------------------------------------------- callbacks
            def reset_preview(self) -> None:
                self._update_preview(self.sample_array)
                self.status_var.set("Preview reset to bundled reconstruction.")

            def save_sample_image(self) -> None:
                path = filedialog.asksaveasfilename(
                    title="Save sample image",
                    defaultextension=".png",
                    filetypes=[("PNG", "*.png")],
                )
                if not path:
                    return
                try:
                    image = Image.fromarray((self.sample_array * 255.0).astype(np.uint8), mode="RGB")
                    image.save(Path(path))
                    self.status_var.set(f"Saved sample image to {path}")
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Save image", f"Failed to save image: {exc}")
                    self.status_var.set(f"Save failed: {exc}")

            def save_sample_wav(self) -> None:
                path = filedialog.asksaveasfilename(
                    title="Save sample WAV",
                    defaultextension=".wav",
                    filetypes=[("WAV", "*.wav")],
                )
                if not path:
                    return
                try:
                    Path(path).write_bytes(_sample_wav_bytes())
                    self.status_var.set(f"Saved sample WAV to {path}")
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Save WAV", f"Failed to save WAV: {exc}")
                    self.status_var.set(f"Save failed: {exc}")

            def encode_image_to_wav(self) -> None:
                image_path = filedialog.askopenfilename(
                    title="Select image",
                    filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")],
                )
                if not image_path:
                    return
                try:
                    with Image.open(image_path) as img:
                        array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                    wav_bytes = encode_image_to_wav_bytes(
                        array,
                        sample_rate=self.sample_rate,
                        segments=max(1, self.segments),
                        marker_duration=self.marker,
                    )
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Encode image", f"Failed to encode image: {exc}")
                    self.status_var.set(f"Encode failed: {exc}")
                    return

                save_path = filedialog.asksaveasfilename(
                    title="Save encoded WAV",
                    defaultextension=".wav",
                    filetypes=[("WAV", "*.wav")],
                )
                if not save_path:
                    return
                try:
                    Path(save_path).write_bytes(wav_bytes)
                    self.status_var.set(f"Encoded WAV saved to {save_path}")
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Save WAV", f"Failed to save WAV: {exc}")
                    self.status_var.set(f"Save failed: {exc}")

            def decode_wav_to_image(self) -> None:
                wav_path = filedialog.askopenfilename(
                    title="Select WAV",
                    filetypes=[("WAV", "*.wav")],
                )
                if not wav_path:
                    return
                try:
                    wav_bytes = Path(wav_path).read_bytes()
                    image, detected = decode_wav_bytes_to_image(
                        wav_bytes,
                        resolution=(self.rows, self.cols),
                        sample_rate=self.sample_rate,
                        segments=max(1, self.segments),
                        marker_duration=self.marker,
                    )
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Decode WAV", f"Failed to decode WAV: {exc}")
                    self.status_var.set(f"Decode failed: {exc}")
                    return

                save_path = filedialog.asksaveasfilename(
                    title="Save reconstructed image",
                    defaultextension=".png",
                    filetypes=[("PNG", "*.png")],
                )
                if save_path:
                    try:
                        preview = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
                        preview.save(Path(save_path))
                    except Exception as exc:  # pragma: no cover - GUI safety
                        messagebox.showerror("Save image", f"Failed to save image: {exc}")
                        self.status_var.set(f"Save failed: {exc}")
                        return

                self._update_preview(image)
                self.status_var.set(f"Decoded WAV at {detected} Hz")

            def preview_text_static(self) -> None:
                text = self._text_value()
                if not text:
                    self.status_var.set("Enter text to encode before previewing.")
                    return
                try:
                    image, metadata = encode_text_to_image(text)
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Text encoding", f"Failed to encode text: {exc}")
                    self.status_var.set(f"Text encoding failed: {exc}")
                    return
                self.text_metadata = metadata
                self.text_payload = text
                self._update_preview(image)
                self.status_var.set(f"Previewing colour static for {len(text)} characters.")

            def save_text_as_wav(self) -> None:
                text = self._text_value()
                if not text:
                    self.status_var.set("Enter text to encode before saving a WAV.")
                    return
                try:
                    waveform, metadata = encode_text_to_waveform(text)
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Text to WAV", f"Failed to encode text: {exc}")
                    self.status_var.set(f"Text encode failed: {exc}")
                    return
                path = filedialog.asksaveasfilename(
                    title="Save text WAV",
                    defaultextension=".wav",
                    filetypes=[("WAV", "*.wav")],
                )
                if not path:
                    return
                try:
                    save_waveform_as_wav(
                        waveform,
                        sample_rate=int(metadata.sample_rate or self.sample_rate),
                        path=Path(path),
                    )
                    self.status_var.set(f"Saved text signal to {path}")
                except Exception as exc:  # pragma: no cover - GUI safety
                    messagebox.showerror("Save WAV", f"Failed to save text WAV: {exc}")
                    self.status_var.set(f"Text WAV save failed: {exc}")


        def main() -> None:
            root = tk.Tk()
            DemoApp(root)
            root.mainloop()


        if __name__ == "__main__":  # pragma: no cover - GUI entry point
            main()
        '''
    )
)


def _array_to_png_bytes(image: np.ndarray) -> bytes:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    png_image = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    png_image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_demo_executable(
    image: np.ndarray,
    *,
    sample_rate: int,
    segments: int,
    marker_duration: float,
    label: str,
    metadata: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Package a desktop demo executable seeded with the provided candidate image.

    Parameters
    ----------
    image:
        RGB array in ``[0, 1]`` used as the reference reconstruction.
    sample_rate:
        Sample rate used for the bundled WAV preview.
    segments:
        Number of transmission segments captured from the evolution run.
    marker_duration:
        Duration, in seconds, of the segment marker tone.
    label:
        Human-readable label describing the reconstruction.
    metadata:
        Optional extra metadata to embed for display within the demo.
    output_dir:
        Destination directory for the packaged executable. Defaults to ``dist``.
    """

    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    png_bytes = _array_to_png_bytes(array)
    wav_bytes = encode_image_to_wav_bytes(
        array,
        sample_rate=int(sample_rate),
        segments=int(segments),
        marker_duration=float(marker_duration),
    )

    rows, cols = array.shape[:2]
    meta: dict[str, Any] = {
        "label": str(label),
        "sample_rate": int(sample_rate),
        "segments": int(segments),
        "marker_duration": float(marker_duration),
        "rows": int(rows),
        "cols": int(cols),
    }
    if metadata:
        meta.update({key: value for key, value in metadata.items() if key not in meta})

    script = _DEMO_GUI_TEMPLATE.substitute(
        IMAGE_B64=base64.b64encode(png_bytes).decode("ascii"),
        WAV_B64=base64.b64encode(wav_bytes).decode("ascii"),
        METADATA=repr(meta),
    )

    module_root = Path(__file__).resolve().parent
    if not module_root.exists():
        raise FileNotFoundError(f"Unable to locate Umbra sources at {module_root}")

    output_directory = Path(output_dir or Path.cwd() / "dist")
    output_directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="umbra_demo_exe_") as tmp_dir:
        staging_root = Path(tmp_dir)
        app_root = staging_root / "app"
        app_root.mkdir(parents=True, exist_ok=True)

        shutil.copytree(module_root, app_root / "umbra", dirs_exist_ok=True)
        (app_root / "__main__.py").write_text(script, encoding="utf-8")

        pyz_path = staging_root / "umbra_demo.pyz"
        zipapp.create_archive(app_root, target=pyz_path, interpreter="/usr/bin/env python3")

        exe_path = output_directory / "umbra_demo.exe"
        shutil.copyfile(pyz_path, exe_path)
        return exe_path


__all__ = ["build_demo_package", "build_demo_executable"]
