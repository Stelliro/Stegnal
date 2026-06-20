# __init__.py

"""Project Umbra package."""

from .codec import (
    decode_wav_bytes_to_image,
    decode_waveform_to_image,
    encode_image_to_wav_bytes,
    encode_image_to_waveform,
)
from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .evolution import Candidate, EvolutionLimitReached, EvolutionManager, Gene, GenerationRecord
from .logging_utils import configure_logging
from .metrics import ReconstructionMetrics, compute_metrics
from .neural import NeuralRewardModel
from .pipeline import PipelineResult, replay_packet, run_pipeline
from .predictor import predict_image_from_waveform
from .reconstruction import (
    GeneratedShape,
    ReconstructionResult,
    blend_predictions,
    create_variations,
    generate_shape_collage,
    image_to_waveform,
    predict_missing_pixels,
    reconstruct_from_waveform,
    run_reconstruction_cycle,
    waveform_to_wav_bytes,
)
from .visualization import colorize_comparison, multiplicative_overlap, normalize_for_display

__all__ = [
    "NoisePacket",
    "NoiseStreamEncoder",
    "NoiseStreamDecoder",
    "encode_image_to_waveform",
    "encode_image_to_wav_bytes",
    "decode_waveform_to_image",
    "decode_wav_bytes_to_image",
    "EvolutionManager",
    "EvolutionLimitReached",
    "Candidate",
    "Gene",
    "GenerationRecord",
    "compute_metrics",
    "ReconstructionMetrics",
    "run_pipeline",
    "replay_packet",
    "PipelineResult",
    "multiplicative_overlap",
    "colorize_comparison",
    "normalize_for_display",
    "configure_logging",
    "NeuralRewardModel",
    "GeneratedShape",
    "ReconstructionResult",
    "blend_predictions",
    "create_variations",
    "generate_shape_collage",
    "image_to_waveform",
    "predict_missing_pixels",
    "predict_image_from_waveform",
    "reconstruct_from_waveform",
    "run_reconstruction_cycle",
    "waveform_to_wav_bytes",
]