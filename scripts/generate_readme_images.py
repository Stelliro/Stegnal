"""Generate README visuals and capture the Stegnal UI on the secondary monitor."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Secondary monitor is left of primary: -1920,0 1920x1080
SECONDARY_ORIGIN = (-1920, 0)
SECONDARY_SIZE = (1920, 1080)
UI_SIZE = (1400, 900)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\CascadiaMono.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, fill, outline=None, radius=16, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _center_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def _arrow(draw, start, end, color="#00ff9f", width=3):
    draw.line([start, end], fill=color, width=width)
    # simple arrowhead
    x0, y0 = start
    x1, y1 = end
    import math

    angle = math.atan2(y1 - y0, x1 - x0)
    length = 12
    for da in (2.6, -2.6):
        ax = x1 - length * math.cos(angle + da)
        ay = y1 - length * math.sin(angle + da)
        draw.line([(x1, y1), (ax, ay)], fill=color, width=width)


def generate_pipeline_diagram() -> Path:
    """How Stegnal works — horizontal pipeline with exact labels."""
    w, h = 1600, 520
    bg = "#121212"
    panel = "#1e1e1e"
    accent = "#00ff9f"
    muted = "#9a9a9a"
    text = "#e8e8e8"
    card = "#252526"

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    title_f = _font(28, bold=True)
    step_f = _font(16, bold=True)
    body_f = _font(14)
    small_f = _font(12)

    draw.text((48, 28), "STEGNAL  //  How it works", font=title_f, fill=accent)
    draw.text(
        (48, 68),
        "Image or text becomes sound, travels through air, is captured, reconstructed, and scored.",
        font=body_f,
        fill=muted,
    )

    steps = [
        ("1. Payload", "Load image\nor text"),
        ("2. Encode", "Waveform\n(carrier / PCM)"),
        ("3. Transmit", "Speakers\n→ air"),
        ("4. Capture", "Microphone\nrecords"),
        ("5. Reconstruct", "AI predict +\ndecode"),
        ("6. Score / Key", "Fidelity +\nchannel key"),
    ]

    margin_x = 48
    gap = 18
    card_w = (w - 2 * margin_x - gap * (len(steps) - 1)) // len(steps)
    card_h = 220
    top = 140

    centers = []
    for i, (title, body) in enumerate(steps):
        x0 = margin_x + i * (card_w + gap)
        y0 = top
        x1 = x0 + card_w
        y1 = y0 + card_h
        _draw_rounded_rect(draw, (x0, y0, x1, y1), fill=card, outline="#333", radius=18)
        # accent top bar
        draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=4, fill=accent)
        draw.text((x0 + 16, y0 + 28), title, font=step_f, fill=accent)
        # multi-line body
        by = y0 + 70
        for line in body.split("\n"):
            draw.text((x0 + 16, by), line, font=body_f, fill=text)
            by += 22
        centers.append(((x0 + x1) // 2, y1))

        if i < len(steps) - 1:
            ax0 = x1 + 2
            ax1 = x1 + gap - 2
            mid_y = y0 + card_h // 2
            _arrow(draw, (ax0, mid_y), (ax1, mid_y), color=accent, width=3)

    # bottom note
    note_y = top + card_h + 40
    _draw_rounded_rect(
        draw,
        (margin_x, note_y, w - margin_x, note_y + 70),
        fill=panel,
        outline="#333",
        radius=14,
    )
    draw.text(
        (margin_x + 20, note_y + 16),
        "Physical channel = real speakers + room + mic (or simulated). Ultrasonic carriers need capable hardware.",
        font=body_f,
        fill=text,
    )
    draw.text(
        (margin_x + 20, note_y + 40),
        "Noncommercial research prototype — PolyForm Noncommercial 1.0.0",
        font=small_f,
        fill=muted,
    )

    path = OUT / "pipeline.png"
    img.save(path, "PNG", optimize=True)
    return path


def generate_roundtrip_demo() -> Path:
    """Reference vs simulated reconstruction collage using the real codec path."""
    sys.path.insert(0, str(ROOT / "src"))
    from stegnal.codec import decode_waveform_to_image, encode_image_to_waveform
    from stegnal.metrics import compute_metrics

    # Distinct synthetic reference (easy to see visually)
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    ref = np.zeros((n, n, 3), dtype=np.float32)
    ref[..., 0] = xx / (n - 1)
    ref[..., 1] = yy / (n - 1)
    ref[..., 2] = 0.35 + 0.45 * np.sin(xx / 8.0) * np.cos(yy / 10.0)
    # add a solid accent square
    ref[30:70, 40:90] = np.array([0.05, 0.95, 0.55], dtype=np.float32)
    ref = np.clip(ref, 0, 1)

    sr = 48000
    waveform = encode_image_to_waveform(ref, sample_rate=sr, direct=True)
    # mild channel-like noise for realism (still recoverable)
    rng = np.random.default_rng(7)
    noisy = waveform + rng.normal(0, 0.01, size=waveform.shape).astype(np.float32)
    recon = decode_waveform_to_image(noisy, resolution=(n, n), sample_rate=sr, direct=True)
    recon = np.clip(np.asarray(recon, dtype=np.float32), 0, 1)
    m = compute_metrics(ref, recon)

    def to_u8(arr: np.ndarray) -> Image.Image:
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")

    # waveform strip
    wf = waveform[: min(len(waveform), sr // 4)]  # ~0.25s
    wf_img = Image.new("RGB", (n * 3 + 80, 80), "#1a1a1a")
    wdraw = ImageDraw.Draw(wf_img)
    mid = 40
    xs = np.linspace(0, wf_img.width - 1, num=min(len(wf), wf_img.width))
    pts = []
    step = max(1, len(wf) // wf_img.width)
    samples = wf[::step][: wf_img.width]
    for i, s in enumerate(samples):
        y = mid - int(float(s) * 32)
        pts.append((i, y))
    if len(pts) > 1:
        wdraw.line(pts, fill="#00ff9f", width=1)
    wdraw.text((8, 4), "Encoded audio (snippet)", font=_font(12), fill="#888")

    tile = 280
    ref_i = to_u8(ref).resize((tile, tile), Image.Resampling.NEAREST)
    rec_i = to_u8(recon).resize((tile, tile), Image.Resampling.NEAREST)
    # prediction stand-in: slightly softened ref (heuristic UI concept)
    pred = np.clip(0.85 * ref + 0.1, 0, 1)
    pred_i = to_u8(pred).resize((tile, tile), Image.Resampling.NEAREST)

    pad = 24
    header_h = 90
    label_h = 36
    footer_h = 100
    width = pad * 2 + tile * 3 + pad * 2
    height = header_h + tile + label_h + 20 + wf_img.height + footer_h
    canvas = Image.new("RGB", (width, height), "#121212")
    draw = ImageDraw.Draw(canvas)
    title_f = _font(26, bold=True)
    body_f = _font(14)
    small_f = _font(12)

    draw.text((pad, 24), "STEGNAL  //  Roundtrip example", font=title_f, fill="#00ff9f")
    draw.text(
        (pad, 58),
        f"Direct PCM encode → mild noise → decode   |   SSIM={m.ssim:.3f}   PSNR={m.psnr:.1f} dB",
        font=body_f,
        fill="#9a9a9a",
    )

    labels = ["Reference", "AI prediction (concept)", "Reconstructed"]
    images = [ref_i, pred_i, rec_i]
    for i, (im, lab) in enumerate(zip(images, labels)):
        x = pad + i * (tile + pad)
        y = header_h
        canvas.paste(im, (x, y))
        draw.rectangle((x - 1, y - 1, x + tile, y + tile), outline="#333")
        draw.text((x, y + tile + 8), lab, font=body_f, fill="#e0e0e0")

    wf_y = header_h + tile + label_h + 16
    canvas.paste(wf_img.resize((width - 2 * pad, 80)), (pad, wf_y))
    draw.text(
        (pad, wf_y + 90),
        "Simulated path for illustration. Real experiments use speakers + microphone (see UI).",
        font=small_f,
        fill="#888",
    )

    path = OUT / "roundtrip-example.png"
    canvas.save(path, "PNG", optimize=True)
    return path


def generate_key_concept() -> Path:
    """Small diagram: reconstruction + channel fingerprint → key material."""
    w, h = 1200, 420
    img = Image.new("RGB", (w, h), "#121212")
    draw = ImageDraw.Draw(img)
    title_f = _font(24, bold=True)
    step_f = _font(15, bold=True)
    body_f = _font(13)
    accent = "#00ff9f"
    text = "#e8e8e8"
    muted = "#9a9a9a"
    card = "#252526"

    draw.text((40, 28), "STEGNAL  //  Channel-bound key idea", font=title_f, fill=accent)
    draw.text(
        (40, 64),
        "Key material mixes recoverable payload (AI reconstruction) with physical path fingerprint.",
        font=body_f,
        fill=muted,
    )

    boxes = [
        (60, 130, 340, 300, "AI reconstruction", "What the decoder\nrecovers from audio"),
        (430, 130, 710, 300, "Channel fingerprint", "How this room /\nhardware distorted it"),
        (860, 130, 1140, 300, "Derived key bits", "Combined material\n(setup-specific)"),
    ]
    for x0, y0, x1, y1, title, body in boxes:
        _draw_rounded_rect(draw, (x0, y0, x1, y1), fill=card, outline="#333", radius=16)
        draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=4, fill=accent)
        draw.text((x0 + 18, y0 + 28), title, font=step_f, fill=accent)
        by = y0 + 70
        for line in body.split("\n"):
            draw.text((x0 + 18, by), line, font=body_f, fill=text)
            by += 22

    _arrow(draw, (340, 215), (430, 215))
    # plus sign between first two conceptually - already arrow from first to second... better: both feed into key
    _arrow(draw, (710, 215), (860, 215))
    # also arrow from first box down-around? Keep simple: fingerprint arrow from middle

    draw.text(
        (60, 340),
        "Not traditional encryption — experimental / research only. Same gear + room may recover similar material.",
        font=body_f,
        fill=muted,
    )

    path = OUT / "key-concept.png"
    img.save(path, "PNG", optimize=True)
    return path


def _demo_reference(n: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    ref = np.zeros((n, n, 3), dtype=np.float32)
    ref[..., 0] = xx / (n - 1)
    ref[..., 1] = yy / (n - 1)
    ref[..., 2] = 0.35 + 0.45 * np.sin(xx / 8.0) * np.cos(yy / 10.0)
    ref[30:70, 40:90] = np.array([0.05, 0.95, 0.55], dtype=np.float32)
    return np.clip(ref, 0, 1)


def capture_ui_on_secondary() -> Path:
    """Launch Stegnal UI on the second monitor, run a quick sim, then screenshot."""
    import tkinter as tk

    sys.path.insert(0, str(ROOT / "src"))
    from stegnal.ui import StegnalAudioUI

    x, y = SECONDARY_ORIGIN
    win_x = x + 40
    win_y = y + 40
    path = OUT / "ui-screenshot.png"
    done = {"ok": False}

    root = tk.Tk()
    root.title("STEGNAL // Audio Fidelity Experiment")
    root.geometry(f"{UI_SIZE[0]}x{UI_SIZE[1]}+{win_x}+{win_y}")
    root.lift()
    root.attributes("-topmost", True)
    app = StegnalAudioUI(root)

    def _load_demo():
        arr = _demo_reference()
        app.reference = arr
        app._set_image("orig", arr, "demo")
        app.log("Loaded demo reference for README screenshot")
        # kick simulated experiment so panels + scores fill in
        app.run_experiment()
        root.after(200, _wait_for_result)

    def _wait_for_result(tries: int = 0):
        # run_experiment is threaded; poll until scores / images appear
        if app.last_result is not None or tries > 80:
            root.after(500, _do_grab)
            return
        root.after(250, lambda: _wait_for_result(tries + 1))

    def _do_grab():
        root.update_idletasks()
        root.update()
        try:
            rx = root.winfo_rootx()
            ry = root.winfo_rooty()
            rw = root.winfo_width()
            rh = root.winfo_height()
            bbox = (rx, ry, rx + rw, ry + rh)
            shot = ImageGrab.grab(bbox=bbox, all_screens=True)
        except Exception:
            sx, sy = SECONDARY_ORIGIN
            sw, sh = SECONDARY_SIZE
            shot = ImageGrab.grab(bbox=(sx, sy, sx + sw, sy + sh), all_screens=True)
        shot.save(path, "PNG", optimize=True)
        done["ok"] = True
        root.attributes("-topmost", False)
        root.destroy()

    # after UI builds + default-load attempt finishes
    root.after(600, _load_demo)
    root.mainloop()
    if not path.exists():
        raise RuntimeError("UI screenshot was not written")
    return path


def main() -> int:
    print("Generating pipeline diagram…")
    p1 = generate_pipeline_diagram()
    print("  wrote", p1)

    print("Generating roundtrip example…")
    p2 = generate_roundtrip_demo()
    print("  wrote", p2)

    print("Generating key concept…")
    p3 = generate_key_concept()
    print("  wrote", p3)

    print("Capturing UI on secondary monitor…")
    try:
        p4 = capture_ui_on_secondary()
        print("  wrote", p4)
    except Exception as exc:
        print("  UI capture failed:", exc, file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
