"""Project Umbra toy pipeline package."""

from .decoding import NoiseStreamDecoder
from .encoding import NoisePacket, NoiseStreamEncoder
from .evolution import EvolutionManager
from .metrics import ReconstructionMetrics, compute_metrics
from .pipeline import PipelineResult, replay_packet, run_pipeline
from .visualization import colorize_comparison, multiplicative_overlap, normalize_for_display

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
