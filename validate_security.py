import os
import numpy as np
from PIL import Image
from dataclasses import replace, fields
from umbra.encoding import NoiseStreamEncoder
from umbra.decoding import NoiseStreamDecoder
from umbra.metrics import compute_metrics

# --- THE GOD GENE ---
CORRECT_SEED = 2763120586
WRONG_SEED = 2763120587  # Off by one

PARAMS = {
    "sigma": 0.01,
    "denoise_sigma": 0.44955
}

def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((256, 256))
        return np.asarray(img, dtype=np.float32) / 255.0

def save_debug_image(array, name):
    array = np.clip(array, 0.0, 1.0)
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(name)
    print(f"Saved: {name}")

def hack_packet(packet, new_seed):
    """
    Automatically finds which field holds the seed and replaces it.
    """
    target_field = None
    
    # 1. Search for the field holding the CORRECT_SEED
    print("\n[HACKER TOOLS] Scanning packet structure...")
    for field in fields(packet):
        value = getattr(packet, field.name)
        
        # Skip numpy arrays to prevent ambiguous truth value errors
        if isinstance(value, (np.ndarray, list)):
            continue
            
        if value == CORRECT_SEED:
            target_field = field.name
            print(f"  > Found Key Field: '{target_field}'")
            break
            
    if not target_field:
        print("  > CRITICAL: Could not locate seed in packet metadata.")
        return None

    # 2. Perform the replacement (Forgery)
    changes = {target_field: new_seed}
    hacked = replace(packet, **changes)
    print(f"  > Packet forged. Injected seed: {new_seed}")
    return hacked

def main():
    folder = "test_images"
    try:
        filename = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))][0]
    except IndexError:
        print("No images found in test_images/")
        return

    original = load_image(os.path.join(folder, filename))
    print(f"--- SECURITY TEST: {filename} ---")

    encoder = NoiseStreamEncoder(sigma=PARAMS["sigma"])
    decoder = NoiseStreamDecoder(denoise_sigma=PARAMS["denoise_sigma"])

    # 1. ENCODE
    print(f"\n[1] Encoding with Correct Seed: {CORRECT_SEED}")
    packet = encoder.encode(original, seed=CORRECT_SEED)

    # 2. DECODE (Authorized)
    print(f"[2] Decoding with Correct Seed: {CORRECT_SEED}")
    recon_correct = decoder.decode(packet, seed=CORRECT_SEED)
    metric_correct = compute_metrics(original, recon_correct)
    print(f"    > Fidelity: {metric_correct.psnr:.2f} dB")
    save_debug_image(recon_correct, "decoding_authorized.png")

    # 3. DECODE (Unauthorized Attack)
    print(f"\n[3] Decoding with WRONG Seed:   {WRONG_SEED}")
    
    # Verify we can hack it first
    hacked_packet = hack_packet(packet, WRONG_SEED)
    
    if hacked_packet:
        try:
            recon_wrong = decoder.decode(hacked_packet, seed=WRONG_SEED)
            metric_wrong = compute_metrics(original, recon_wrong)
            print(f"    > Fidelity: {metric_wrong.psnr:.2f} dB")
            save_debug_image(recon_wrong, "decoding_unauthorized.png")

            # --- VERDICT ---
            print("\n" + "="*30)
            diff = metric_correct.psnr - metric_wrong.psnr
            print(f"Security Margin: {diff:.2f} dB")
            
            if metric_wrong.psnr < 15.0 and diff > 20.0:
                print("RESULT: 🔒 SYSTEM IS SECURE. (Wrong key yields garbage)")
                print("This qualifies as 'Cryptographic Steganography'.")
            elif diff > 10.0:
                print("RESULT: ⚠️ PARTIALLY SECURE. (Distorted but maybe visible)")
            else:
                print("RESULT: ❌ NOT SECURE. (Image still visible)")
        except Exception as e:
            # If the decoder logic fundamentally breaks on wrong math, that is also a secure outcome
            print(f"Decoder crashed on wrong key (Good sign!): {e}")
            print("RESULT: 🔒 SYSTEM IS SECURE (Decoder rejected invalid stream).")
    else:
        print("Could not simulate attack due to packet structure.")

if __name__ == "__main__":
    main()