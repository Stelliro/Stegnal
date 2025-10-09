"""Project Umbra toy pipeline package."""

from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .evolution import EvolutionManager, ParentLineage
from .logging_utils import configure_logging
from .metrics import ReconstructionMetrics, compute_metrics
from .neural import NeuralRewardModel
from .pipeline import PipelineResult, replay_packet, run_pipeline
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
    "EvolutionManager",
    "ParentLineage",
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
    "reconstruct_from_waveform",
    "run_reconstruction_cycle",
    "waveform_to_wav_bytes",
]
