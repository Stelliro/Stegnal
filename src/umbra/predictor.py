# predictor.py

"""Helpers for predicting images from audio waveforms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .reconstruction import (
    reconstruct_from_waveform,
    suggest_sample_rate,
    suggest_transmission_profile,
)

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


def predict_post_audio_image(
    image: np.ndarray,
    *,
    model: Any | None = None,
    blur_sigma: float = 0.8,
    norm_mode: str = "per_segment",
) -> np.ndarray:
    """Predict what ``image`` will approximately look like after audio roundtrip.

    This is the "AI guess" step: estimate the output of image->audio->image
    *before* performing the actual transfer. The guess is compared to the
    actual audio-decoded image for the prediction score.

    Heuristic channel model (grayscale AM fax):
      - luminance (mean over RGB)
      - per-band or global max-normalization (mimics demod + rescale)
      - mild gaussian blur (band-limit effect of carrier)
      - repeat to 3 channels (color is lost in current carrier)

    When a ``model`` (torch callable) is supplied it is used instead.
    The heuristic is intentionally a close-but-imperfect approximation so that
    the "how well did the agent predict" score is meaningful.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    arr = np.clip(arr, 0.0, 1.0)

    if model is not None and torch is not None:
        try:
            t = torch.as_tensor(arr.transpose(2, 0, 1)[None])  # 1x3xHxW
            if torch.cuda.is_available():
                t = t.to("cuda")
            with torch.no_grad():
                out = model(t)
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 4:
                out = out[0]
            if out.shape[0] == 3:
                out = out.transpose(1, 2, 0)
            out = np.clip(out, 0.0, 1.0)
            if out.shape[:2] != arr.shape[:2]:
                from scipy.ndimage import zoom
                z = (arr.shape[0] / out.shape[0], arr.shape[1] / out.shape[1], 1)
                out = zoom(out, z, order=3)
            return out.astype(np.float32)
        except Exception:
            logger.exception("Torch post-audio predictor failed; using heuristic")

    # --- Heuristic predictor (default) ---
    # Agent does not have *exact* internal knowledge of segment count / norm points;
    # use a simplified global + mild local model + extra blur to represent imperfect prediction.
    # Improved predictor: now assumes the improved color-capable channel
    # Agent tries to guess per-channel structure + some loss
    h, w, _ = arr.shape
    pred = arr.copy()  # start with original (optimistic agent that knows structure)
    # Apply realistic degradations the channel still has: per-ch max norm-ish + blur
    for ch in range(3):
        ch_data = pred[..., ch]
        mx = float(np.max(ch_data)) or 1.0
        ch_data = ch_data / mx
        pred[..., ch] = ch_data

    # Channel crosstalk / loss simulation (agent not perfect)
    pred = pred * 0.92 + 0.04

    eff_blur = float(blur_sigma) * 0.9
    if eff_blur > 0:
        try:
            from scipy.ndimage import gaussian_filter
            for ch in range(3):
                pred[..., ch] = gaussian_filter(pred[..., ch], sigma=eff_blur)
        except Exception:
            pass

    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    return pred


__all__ = ["predict_image_from_waveform", "predict_post_audio_image"]