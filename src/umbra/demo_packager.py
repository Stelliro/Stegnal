"""Utilities for bundling a shareable Umbra demo archive."""

from __future__ import annotations

import shutil
import tempfile
import textwrap
import zipapp
from pathlib import Path

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


__all__ = ["build_demo_package"]
