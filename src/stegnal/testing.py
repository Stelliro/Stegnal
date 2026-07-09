# testing.py

"""Test utilities for validating the Stegnal pipeline without the UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stegnal.codec import (
    decode_wav_bytes_to_image,
    decode_waveform_to_image,
    encode_image_to_wav_bytes,
    encode_image_to_waveform,
)
from stegnal.decoding import NoiseStreamDecoder
from stegnal.encoding import NoiseStreamEncoder
from stegnal.metrics import ReconstructionMetrics, compute_metrics
from stegnal.predictor import predict_post_audio_image
from stegnal.reconstruction import suggest_sample_rate, suggest_transmission_profile

logger = logging.getLogger(__name__)


def run_smoke_test(
    *,
    seed: int = 1234,
    size: int = 128,
    sigma: float = 0.25,
    denoise_sigma: float = 0.9,
) -> ReconstructionMetrics:
    """Execute a minimal encode/decode cycle on a synthetic pattern.

    The function generates a simple gradient image, runs it through the
    stochastic encoder/decoder pair, and returns the reconstruction metrics.
    It can be used from tests or the CLI to ensure the core pipeline works
    without launching the Streamlit dashboard.
    """

    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if denoise_sigma < 0:
        raise ValueError("Denoise sigma must be non-negative")

    size = int(max(8, size))
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gradient = np.outer(coords, coords)
    gradient_rgb = np.stack([gradient, gradient, gradient], axis=-1)

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma)

    packet = encoder.encode(gradient_rgb, seed)
    reconstruction = decoder.decode(packet, seed)
    if reconstruction.ndim == 2:
        reconstruction_rgb = np.stack([reconstruction] * 3, axis=-1)
    elif reconstruction.ndim == 3 and reconstruction.shape[2] == 1:
        reconstruction_rgb = np.repeat(reconstruction, 3, axis=-1)
    else:
        reconstruction_rgb = reconstruction
    reconstruction_rgb = np.clip(reconstruction_rgb, 0.0, 1.0)
    gradient_rgb = np.clip(gradient_rgb, 0.0, 1.0)
    packet_metrics = compute_metrics(gradient_rgb, reconstruction_rgb)

    sample_rate = suggest_sample_rate(reconstruction_rgb)
    segments, marker_duration = suggest_transmission_profile(reconstruction_rgb)
    waveform = encode_image_to_waveform(
        reconstruction_rgb,
        sample_rate=sample_rate,
        segments=segments,
        marker_duration=marker_duration,
    )
    waveform_image, _ = decode_waveform_to_image(
        waveform,
        sample_rate=sample_rate,
        resolution=reconstruction_rgb.shape[:2],
        segments=segments,
        marker_duration=marker_duration,
        return_metadata=True,
    )
    waveform_gray = np.clip(np.asarray(waveform_image, dtype=np.float32)[..., 0], 0.0, 1.0)
    wav_metrics = compute_metrics(gradient, waveform_gray)

    def _normalize_psnr(value: float) -> float:
        return float(np.clip((value - 20.0) / 40.0, 0.0, 1.0))

    def _normalize_ssim(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    packet_score = 0.3 * _normalize_psnr(packet_metrics.psnr) + 0.3 * _normalize_ssim(
        packet_metrics.ssim
    )
    wav_score = 0.3 * _normalize_psnr(wav_metrics.psnr) + 0.3 * _normalize_ssim(
        wav_metrics.ssim
    )
    if abs(packet_score - wav_score) > 0.05:
        logger.warning("Significant discrepancy between packet and WAV scores: %.3f vs %.3f", packet_score, wav_score)

    return packet_metrics


@dataclass
class AudioRoundtripResult:
    """Result of a full image -> audio -> image experiment with AI prediction."""

    original: np.ndarray
    predicted: np.ndarray          # AI guess of post-audio appearance
    actual: np.ndarray             # output of actual audio roundtrip
    waveform: np.ndarray
    sample_rate: int
    # Core scores per user request
    image_to_audio_fidelity: float  # proxy for "how well image got turned into audio"
    audio_to_image_fidelity: float  # how well audio reconstructed to image (vs orig)
    prediction_accuracy: float      # how well guessed matched the actual audio output
    composite: float                # overall experiment score [0-1]
    metrics_orig_actual: ReconstructionMetrics
    metrics_pred_actual: ReconstructionMetrics


def run_audio_roundtrip_experiment(
    image: np.ndarray | str | Path | Image.Image,
    *,
    resolution: tuple[int, int] | None = None,
    predictor_model: Any | None = None,
    segments: int | None = None,
    marker_duration: float | None = None,
) -> AudioRoundtripResult:
    """Run the core Stegnal audio experiment.

    Flow:
      1. AI (predictor) guesses what the image will look like after audio processing.
      2. Image is transferred to audio (WAV waveform).
      3. Audio is transferred back to image.
      4. Scores computed:
         - image_to_audio_fidelity (energy preservation proxy via pre/post gray corr)
         - audio_to_image_fidelity (actual vs original)
         - prediction_accuracy (guess vs actual audio output)
    """
    # Load / normalize input
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)

    if resolution is not None:
        from skimage.transform import resize
        arr = resize(arr, resolution + (3,), preserve_range=True, anti_aliasing=True).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
    else:
        # For the audio fax toy channel, large images get truncated by design (duration cap).
        # Auto-downscale to a practical "transmission" size so scores reflect the channel
        # rather than data loss from artificial limits. User can pass explicit resolution.
        h, w = arr.shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_h = max(8, int(round(h * scale)))
            new_w = max(8, int(round(w * scale)))
            from skimage.transform import resize
            arr = resize(arr, (new_h, new_w, 3), preserve_range=True, anti_aliasing=True).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            logger.info("Auto-downscaled large input to %dx%d for audio experiment", new_h, new_w)

    orig = arr.copy()

    # 1. AI GUESS (before doing the transfer)
    predicted = predict_post_audio_image(orig, model=predictor_model)

    # 2+3. ACTUAL TRANSFER - use direct=True for high audio->image fidelity
    wav_bytes = encode_image_to_wav_bytes(
        orig,
        segments=segments,
        marker_duration=marker_duration,
        direct=True,
    )
    dec_result = decode_wav_bytes_to_image(wav_bytes, resolution=orig.shape[:2], return_metadata=True, direct=True)
    if isinstance(dec_result, tuple):
        actual, _meta = dec_result
    else:
        actual = dec_result
    actual = np.clip(np.asarray(actual, dtype=np.float32), 0.0, 1.0)

    # Ensure shapes match
    if actual.shape != orig.shape:
        from skimage.transform import resize
        actual = resize(actual, orig.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)

    # 4. SCORING
    m_orig_actual = compute_metrics(orig, actual)
    m_pred_actual = compute_metrics(predicted, actual)

    # image_to_audio_fidelity: use correlation between original luminance and the demodulated
    # (we can derive a proxy by re-encoding the actual and comparing internal, or simple:
    #   since the carrier amplitude ~ luminance, we can measure how "consistent" the recovered
    #   is with input structure using partial alignment + fft/edge from existing metrics.
    gray_orig = np.mean(orig, axis=2)
    gray_actual = np.mean(actual, axis=2)
    # Pearson-like correlation (structure preservation in audio domain)
    def _safe_corr(a, b):
        a = a.ravel().astype(np.float32)
        b = b.ravel().astype(np.float32)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        return float(np.clip(np.mean(a * b), -1.0, 1.0))
    encode_corr = _safe_corr(gray_orig, gray_actual)
    image_to_audio_fid = float(np.clip((encode_corr + 1.0) / 2.0, 0.0, 1.0))  # map [-1,1] -> [0,1]

    # audio_to_image_fidelity: how good was the reconstruction (use a bounded composite)
    # normalize psnr/ssim into [0,1] contribution
    psnr_n = float(np.clip((m_orig_actual.psnr - 12.0) / 28.0, 0.0, 1.0))
    audio_to_image_fid = float(np.clip(0.6 * m_orig_actual.ssim + 0.4 * psnr_n, 0.0, 1.0))

    # prediction_accuracy: similarity of AI guess to what actually came out of audio
    psnr_p = float(np.clip((m_pred_actual.psnr - 12.0) / 28.0, 0.0, 1.0))
    prediction_acc = float(np.clip(0.65 * m_pred_actual.ssim + 0.35 * psnr_p, 0.0, 1.0))

    # Overall composite (equal weight on the three pillars described by user)
    composite = float(np.clip(
        (image_to_audio_fid + audio_to_image_fid + prediction_acc) / 3.0,
        0.0, 1.0
    ))

    # For waveform we return a representative; user can re-encode if needed.
    # Here we return the internal waveform used.
    waveform = encode_image_to_waveform(orig)  # lightweight; not bytes

    return AudioRoundtripResult(
        original=orig.astype(np.float32),
        predicted=predicted.astype(np.float32),
        actual=actual.astype(np.float32),
        waveform=waveform.astype(np.float32),
        sample_rate=int(suggest_sample_rate(orig)),
        image_to_audio_fidelity=image_to_audio_fid,
        audio_to_image_fidelity=audio_to_image_fid,
        prediction_accuracy=prediction_acc,
        composite=composite,
        metrics_orig_actual=m_orig_actual,
        metrics_pred_actual=m_pred_actual,
    )


__all__ = ["run_smoke_test", "run_audio_roundtrip_experiment", "AudioRoundtripResult"]