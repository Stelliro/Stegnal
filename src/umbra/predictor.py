# predictor.py

"""Helpers for predicting images from audio waveforms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .reconstruction import reconstruct_from_waveform

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover - torch optional dependency
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def predict_image_from_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    resolution: tuple[int, int],
    model: Any | None = None,
    device: str | None = None,
    segments: int = 1,
    marker_duration: float = 0.05,
) -> np.ndarray:
    """Return an RGB image predicted from ``waveform``.

    ``model`` may be a callable accepting a ``torch.Tensor`` input. When ``model``
    is ``None`` or PyTorch is unavailable, the deterministic heuristic decoder is
    used instead.
    """

    if waveform.size == 0:
        return np.zeros(resolution + (3,), dtype=np.float32)

    base_prediction = reconstruct_from_waveform(
        waveform,
        resolution=resolution,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )

    if model is None or torch is None:
        if model is not None and torch is None:
            logger.info("PyTorch not available; using heuristic audio decoder")
        return base_prediction.astype(np.float32)

    try:
        tensor = torch.as_tensor(np.asarray(waveform, dtype=np.float32)).view(1, 1, -1)
        if device is not None:
            tensor = tensor.to(device)
        elif torch.cuda.is_available():  # pragma: no cover - GPU
            tensor = tensor.to("cuda")

        extra_args: dict[str, Any] = {
            "sample_rate": sample_rate,
            "resolution": resolution,
        }

        with torch.no_grad():
            try:
                output = model(tensor, **extra_args)
            except TypeError:
                output = model(tensor)

        if hasattr(output, "detach"):
            output_array = output.detach().cpu().numpy()
        else:
            output_array = np.asarray(output, dtype=np.float32)

        array = np.asarray(output_array, dtype=np.float32)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("Predictor output must have channel dimension of size 3")

        array = array[..., :3]
        array = np.clip(array, 0.0, 1.0)
        if array.shape[:2] != resolution:
            from scipy.ndimage import zoom
            zoom_factors = (resolution[0] / array.shape[0], resolution[1] / array.shape[1], 1)
            array = zoom(array, zoom_factors, order=3)
        return array.astype(np.float32)
    except Exception:  # pragma: no cover - graceful fallback
        logger.exception("Torch predictor failed; falling back to heuristic decoder")
        return base_prediction.astype(np.float32)


__all__ = ["predict_image_from_waveform"]