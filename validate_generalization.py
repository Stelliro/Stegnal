import os
import numpy as np
from PIL import Image
from umbra.encoding import NoiseStreamEncoder
from umbra.decoding import NoiseStreamDecoder
from umbra.metrics import compute_metrics

# --- THE GOD GENE (From Gen 57) ---
BEST_GENE = {
    "seed": 2763120586,
    "sigma": 0.01,
    "denoise_sigma": 0.44955,
    "inpainter_steps": 13,
    "guidance_scale": 3.078
}

def load_image(path):
    with Image.open(path) as img:
        # Resize to 256x256 to match the training dimensions
        img = img.convert("RGB").resize((256, 256))
        return np.asarray(img, dtype=np.float32) / 255.0

def main():
    folder = "test_images"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Please put images in the '{folder}' directory and run again.")
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"--- TESTING GENE ON {len(files)} NEW IMAGES ---")
    print(f"Gene Config: {BEST_GENE}")
    print("-" * 60)

    encoder = NoiseStreamEncoder(sigma=BEST_GENE["sigma"])
    decoder = NoiseStreamDecoder(denoise_sigma=BEST_GENE["denoise_sigma"])

    avg_psnr = 0
    
    for f in files:
        img_path = os.path.join(folder, f)
        original = load_image(img_path)
        
        # 1. Encode
        # FIXED: Changed from encode_from_array to encode
        try:
            packet = encoder.encode(original, seed=BEST_GENE["seed"])
        except AttributeError:
            # Fallback if the class requires a specific method name
            print(f"Error: Encoder does not accept array input directly.")
            break
        
        # 2. Decode
        recon = decoder.decode(
            packet, 
            seed=BEST_GENE["seed"]
        )
        
        # 3. Score
        metrics = compute_metrics(original, recon)
        print(f"IMAGE: {f:<20} | PSNR: {metrics.psnr:.2f} dB | SSIM: {metrics.ssim:.3f}")
        avg_psnr += metrics.psnr

    if files:
        print("-" * 60)
        final_avg = avg_psnr / len(files)
        print(f"AVERAGE PERFORMANCE: {final_avg:.2f} dB")
        
        if final_avg > 30.0:
            print("RESULT: 🟢 UNIVERSAL ENCODING CONFIRMED (Visually Lossless)")
        elif final_avg > 20.0:
            print("RESULT: 🟡 GENERALIZATION OKAY (Recognizable but noisy)")
        else:
            print("RESULT: 🔴 OVERFITTING DETECTED (Gene only works for source image)")

if __name__ == "__main__":
    main()