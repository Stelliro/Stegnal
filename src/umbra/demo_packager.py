# demo_packager.py

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
        decode_parser.add_argument("image", type=Path, help="Destination image file")
        decode_parser.add_argument(
            "--resolution",
            type=int,
            nargs=2,
            default=(256, 256),
            help="Expected resolution (height width; default 256 256)",
        )

        args = parser.parse_args(argv or sys.argv[1:])
        if args.command == "encode":
            _encode(args.image, args.wav, args.sample_rate)
        elif args.command == "decode":
            _decode(args.wav, args.image, tuple(args.resolution))
        return 0


    if __name__ == "__main__":
        sys.exit(main())
    """
)


_DEMO_GUI_TEMPLATE = Template(
    textwrap.dedent(
        """
        \"\"\"Standalone Tkinter GUI demo for the Umbra codec.\"\"\"

        import base64
        import io
        import sys
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk

        import numpy as np

        from umbra.codec import decode_wav_bytes_to_image

        IMAGE_B64 = "$IMAGE_B64"
        WAV_B64 = "$WAV_B64"
        METADATA = $METADATA

        class UmbraDemoApp:
            def __init__(self, root: tk.Tk) -> None:
                self.root = root
                self.root.title("Umbra Demo")
                self.root.geometry("800x600")

                self.label = tk.Label(root, text=METADATA["label"])
                self.label.pack(pady=10)

                self.image_canvas = tk.Canvas(root, bg="gray")
                self.image_canvas.pack(fill=tk.BOTH, expand=True)

                self.decode_button = tk.Button(root, text="Decode WAV", command=self._decode)
                self.decode_button.pack(pady=10)

                self._display_image()

            def _display_image(self) -> None:
                img_data = base64.b64decode(IMAGE_B64)
                img = Image.open(io.BytesIO(img_data))
                photo = ImageTk.PhotoImage(img)
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.image_canvas.image = photo

            def _decode(self) -> None:
                wav_data = base64.b64decode(WAV_B64)
                image, _ = decode_wav_bytes_to_image(
                    wav_data,
                    resolution=METADATA["resolution"],
                    sample_rate=METADATA["sample_rate"],
                    segments=METADATA["segments"],
                    marker_duration=METADATA["marker_duration"],
                )
                img = Image.fromarray((image * 255).astype(np.uint8))
                photo = ImageTk.PhotoImage(img)
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.image_canvas.image = photo

        if __name__ == "__main__":
            root = tk.Tk()
            UmbraDemoApp(root)
            root.mainloop()
        """
    )
)


def _array_to_png_bytes(array: np.ndarray) -> bytes:
    img = Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def build_demo_package(
    image: np.ndarray,
    *,
    sample_rate: int = 48000,
    segments: int = 1,
    marker_duration: float = 0.05,
    label: str = "Umbra Demo",
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Bundle a reconstruction into a shareable Python archive."""

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
        "resolution": (int(rows), int(cols)),
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

    with tempfile.TemporaryDirectory(prefix="umbra_demo_") as tmp_dir:
        staging_root = Path(tmp_dir)
        app_root = staging_root / "app"
        app_root.mkdir(parents=True, exist_ok=True)

        shutil.copytree(module_root, app_root / "umbra", dirs_exist_ok=True)
        (app_root / "__main__.py").write_text(script, encoding="utf-8")

        pyz_path = output_directory / PACKAGE_NAME
        zipapp.create_archive(app_root, target=pyz_path, interpreter="/usr/bin/env python3")
        return pyz_path


def build_demo_executable(
    image: np.ndarray,
    *,
    sample_rate: int = 48000,
    segments: int = 1,
    marker_duration: float = 0.05,
    label: str = "Umbra Demo",
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Bundle a reconstruction into a standalone executable."""

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