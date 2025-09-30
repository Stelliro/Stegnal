"""Project Umbra toy pipeline package."""

from .encoding import NoisePacket, NoiseStreamEncoder
from .decoding import NoiseStreamDecoder
from .evolution import EvolutionManager
from .metrics import compute_metrics, ReconstructionMetrics
from .pipeline import run_pipeline, replay_packet, PipelineResult
from .visualization import (
    multiplicative_overlap,
    colorize_comparison,
    normalize_for_display,
)

__all__ = [
    "NoisePacket",
    "NoiseStreamEncoder",
    "NoiseStreamDecoder",
    "EvolutionManager",
    "compute_metrics",
    "ReconstructionMetrics",
    "run_pipeline",
    "replay_packet",
    "PipelineResult",
    "multiplicative_overlap",
    "colorize_comparison",
    "normalize_for_display",
]
