# cli.py

"""Command-line interface for Project Umbra's test build."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .metrics import compute_metrics
from .pipeline import run_pipeline
from .testing import run_smoke_test

logger = logging.getLogger(__name__)


def _add_common_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True, help="Shared seed used for permutation")


def command_encode(args: argparse.Namespace) -> None:
    if args.sigma <= 0:
        raise ValueError("Sigma must be positive")
    encoder = NoiseStreamEncoder(sigma=args.sigma)
    try:
        packet = encoder.encode_from_path(args.image, args.seed)
    except Exception as exc:
        logger.error(f"Failed to encode image: {exc}")
        raise
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    packet.to_file(args.output)
    print(f"Packet saved to {args.output}")


def command_decode(args: argparse.Namespace) -> None:
    if args.denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")
    packet = NoisePacket.from_file(args.packet)
    decoder = NoiseStreamDecoder(denoise_sigma=args.denoise_sigma)
    decoder.decode_to_image(packet, args.seed, args.output)
    print(f"Reconstruction saved to {args.output}")


def command_pipeline(args: argparse.Namespace) -> None:
    if args.sigma <= 0:
        raise ValueError("Sigma must be positive")
    if args.denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")
    result = run_pipeline(
        image_path=args.image,
        seed=args.seed,
        sigma=args.sigma,
        packet_path=args.packet,
        reconstruction_path=args.reconstruction,
        denoise_sigma=args.denoise_sigma,
    )
    metrics = result.metrics.as_dict()
    print("Pipeline complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.3f}")


def command_evaluate(args: argparse.Namespace) -> None:
    encoder = NoiseStreamEncoder()
    try:
        reference = encoder.load_image(args.reference)
        candidate = encoder.load_image(args.candidate)
    except Exception as exc:
        logger.error(f"Failed to load images for evaluation: {exc}")
        raise
    metrics = compute_metrics(reference, candidate)
    print("Evaluation metrics:")
    print(f"  PSNR: {metrics.psnr:.3f}")
    print(f"  SSIM: {metrics.ssim:.3f}")


def command_smoke_test(args: argparse.Namespace) -> None:
    if args.sigma <= 0:
        raise ValueError("Sigma must be positive")
    if args.denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")
    metrics = run_smoke_test(
        seed=args.seed,
        size=args.size,
        sigma=args.sigma,
        denoise_sigma=args.denoise_sigma,
    )
    print("Smoke test complete. Reconstruction metrics:")
    print(f"  PSNR: {metrics.psnr:.3f}")
    print(f"  SSIM: {metrics.ssim:.3f}")


def command_ui(_args: argparse.Namespace) -> None:
    try:
        from .ui import main as launch_ui
    except ImportError:
        logger.error("UI dependencies not installed; install with 'pip install .[ui]'")
        raise
    launch_ui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project Umbra toy pipeline CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode an image into a noise packet")
    encode_parser.add_argument("--image", required=True, help="Path to the source image")
    encode_parser.add_argument("--output", required=True, help="Path to write the packet (npz)")
    encode_parser.add_argument("--sigma", type=float, default=0.2, help="Standard deviation of injected noise")
    _add_common_seed_argument(encode_parser)
    encode_parser.set_defaults(func=command_encode)

    decode_parser = subparsers.add_parser("decode", help="Decode a packet into an image")
    decode_parser.add_argument("--packet", required=True, help="Path to the encoded packet")
    decode_parser.add_argument("--output", required=True, help="Path for the reconstructed image")
    decode_parser.add_argument(
        "--denoise-sigma",
        type=float,
        default=1.0,
        help="Gaussian denoiser sigma (set to 0 to disable)",
    )
    _add_common_seed_argument(decode_parser)
    decode_parser.set_defaults(func=command_decode)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run encode+decode and report metrics")
    pipeline_parser.add_argument("--image", required=True, help="Path to the source image")
    pipeline_parser.add_argument("--packet", required=True, help="Where to save the intermediate packet")
    pipeline_parser.add_argument("--reconstruction", required=True, help="Where to save the reconstructed image")
    pipeline_parser.add_argument("--sigma", type=float, default=0.2, help="Standard deviation of injected noise")
    pipeline_parser.add_argument(
        "--denoise-sigma",
        type=float,
        default=1.0,
        help="Gaussian denoiser sigma (set to 0 to disable)",
    )
    _add_common_seed_argument(pipeline_parser)
    pipeline_parser.set_defaults(func=command_pipeline)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare two images")
    evaluate_parser.add_argument("--reference", required=True, help="Reference image path")
    evaluate_parser.add_argument("--candidate", required=True, help="Candidate image path")
    evaluate_parser.set_defaults(func=command_evaluate)

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run an encode/decode cycle on a synthetic gradient to validate the pipeline",
    )
    smoke_parser.add_argument("--seed", type=int, default=1234, help="Seed controlling the noise")
    smoke_parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Size of the generated gradient image (pixels)",
    )
    smoke_parser.add_argument(
        "--sigma",
        type=float,
        default=0.25,
        help="Encoder noise standard deviation",
    )
    smoke_parser.add_argument(
        "--denoise-sigma",
        type=float,
        default=0.9,
        help="Decoder Gaussian denoise sigma",
    )
    smoke_parser.set_defaults(func=command_smoke_test)

    ui_parser = subparsers.add_parser("ui", help="Launch the desktop visual explorer")
    ui_parser.set_defaults(func=command_ui)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    func: Callable[[argparse.Namespace], None] = args.func
    func(args)


if __name__ == "__main__":
    main()