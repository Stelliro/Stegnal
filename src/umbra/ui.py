"""Streamlit-based visual explorer for Project Umbra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from skimage import data

from umbra.decoding import NoiseStreamDecoder
from umbra.encoding import NoiseStreamEncoder
from umbra.metrics import compute_metrics
from umbra.visualization import multiplicative_overlap, normalize_for_display, to_uint8_image


@dataclass(frozen=True)
class SampleImage:
    name: str
    description: str
    loader: Callable[[], np.ndarray]

    def load(self) -> np.ndarray:
        array = np.asarray(self.loader(), dtype=np.float32)
        if array.ndim == 3:
            array = array.mean(axis=2)
        max_val = float(array.max())
        if max_val == 0:
            return np.zeros_like(array)
        if max_val > 1.0:
            array /= 255.0
        return np.clip(array, 0.0, 1.0)


SAMPLES: Dict[str, SampleImage] = {
    "Camera": SampleImage(
        name="Camera",
        description="Classic cameraman grayscale test image",
        loader=lambda: data.camera(),
    ),
    "Checkerboard": SampleImage(
        name="Checkerboard",
        description="High-frequency checkerboard for stress testing",
        loader=lambda: data.checkerboard(),
    ),
    "Coins": SampleImage(
        name="Coins",
        description="Coins with varied textures",
        loader=lambda: data.coins(),
    ),
}


def _load_uploaded_image(uploaded_file) -> Tuple[np.ndarray, str]:
    image = Image.open(uploaded_file).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array, getattr(uploaded_file, "name", "Uploaded image")


def run() -> None:
    """Entry-point for the Streamlit application."""
    st.set_page_config(page_title="Project Umbra Visual Explorer", layout="wide")
    st.title("Project Umbra Visual Explorer")
    st.markdown(
        """
        Use this dashboard to inspect how the Project Umbra toy pipeline encodes
        images into apparent noise and reconstructs them. Adjust the parameters to
        explore how the stochastic encoder and decoder behave, and inspect the
        overlap score that multiplies the generated and detected imagery.
        """
    )

    st.sidebar.header("Input & Parameters")
    seed = st.sidebar.number_input("Shared seed", min_value=0, value=1234, step=1)
    sigma = st.sidebar.slider("Encoder noise σ", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    denoise_sigma = st.sidebar.slider(
        "Decoder denoise σ",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Gaussian blur strength applied after decoding",
    )

    st.sidebar.subheader("Source image")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a grayscale-friendly image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded_file is not None:
        original, uploaded_label = _load_uploaded_image(uploaded_file)
        source_label = f"Uploaded: {uploaded_label}"
    else:
        sample_name = st.sidebar.selectbox(
            "Choose a built-in sample",
            options=list(SAMPLES.keys()),
            format_func=lambda key: f"{SAMPLES[key].name} — {SAMPLES[key].description}",
        )
        original = SAMPLES[sample_name].load()
        source_label = f"Sample: {SAMPLES[sample_name].name}"

    encoder = NoiseStreamEncoder(sigma=sigma)
    decoder = NoiseStreamDecoder(denoise_sigma=denoise_sigma if denoise_sigma > 0 else None)

    packet = encoder.encode(original, int(seed))
    reconstructed = decoder.decode(packet, int(seed))

    noise_map = packet.encoded.reshape(original.shape)
    noise_display = normalize_for_display(noise_map)
    overlap_map, overlap_score = multiplicative_overlap(original, reconstructed)

    metrics = compute_metrics(original, reconstructed)

    st.subheader("Reconstruction quality")
    metric_cols = st.columns(3)
    metric_cols[0].metric("PSNR", f"{metrics.psnr:.2f} dB")
    metric_cols[1].metric("SSIM", f"{metrics.ssim:.3f}")
    metric_cols[2].metric("Multiplicative overlap", f"{overlap_score:.1f}%")

    st.write(
        "The overlap score multiplies the normalized original and reconstructed pixels,"
        " providing a quick proxy for how much of the signal is mutually present."
    )

    st.subheader("Visual comparisons")
    captions = [
        f"Original ({source_label})",
        "Encoded packet (normalized noise)",
        "AI reconstruction",
        "Multiplicative overlap",
    ]
    images = [
        to_uint8_image(original),
        to_uint8_image(noise_display),
        to_uint8_image(reconstructed),
        to_uint8_image(overlap_map),
    ]

    img_cols = st.columns(4)
    for col, caption, image in zip(img_cols, captions, images):
        col.image(image, caption=caption, width="stretch", clamp=True)

    st.markdown(
        """
        ### Next steps
        * Iterate on encoder/decoder designs and plug in learning-based components.
        * Compare overlap metrics across different seeds and hyperparameters.
        * Capture packets from real channels and replay them here for offline study.
        """
    )


def main() -> None:  # pragma: no cover - delegated to Streamlit runtime
    run()


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
