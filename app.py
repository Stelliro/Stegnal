import dataclasses
import os
import threading
import time
import traceback
from datetime import datetime
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import sounddevice as sd
from PIL import Image
from scipy.io import wavfile

from umbra.audio import audio_to_image_data, image_data_to_audio
from umbra.audio_mixer import EnvironmentMonitor
from umbra.decoding import NoiseStreamDecoder

# --- UMBRA IMPORTS ---
from umbra.encoding import NoisePacket, NoiseStreamEncoder

# --- THE GOD GENE (UNIVERSAL KEY) ---
GOD_GENE = {
    "seed": 2763120586,
    "sigma": 0.01,
    "denoise_sigma": 0.44955
}

# --- THEME CONFIG ---
COLOR_BG = "#1a1a1a"
COLOR_PANEL = "#2b2b2b"
COLOR_ACCENT = "#1f6aa5" # Tech Blue
COLOR_ACCENT_HOVER = "#144870"
COLOR_SUCCESS = "#2ea043" # Matrix Green
COLOR_SUCCESS_HOVER = "#237a33"
COLOR_WARN = "#d35400"
FONT_UI = ("Roboto Medium", 13)
FONT_HEADER = ("Roboto Medium", 20)
FONT_MONO = ("Consolas", 12)

class UmbraTerminal(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 1. WINDOW SETUP ---
        self.title("PROJECT UMBRA // TERMINAL")
        self.geometry("1280x850")
        ctk.set_appearance_mode("dark")
        
        # --- 2. AUDIO BACKEND INIT ---
        # We initialize this immediately to fetch the device list
        self.env_monitor = EnvironmentMonitor()
        self.audio_devices = self.env_monitor.get_devices()
        self.dynamic_sigma_mode = False

        # --- 2b. RUNTIME STATE ---
        # Encoder/decoder working buffers and the auto-detected packet schema.
        self.current_image = None
        self.current_audio = None
        self.sample_rate = None
        self.loaded_packet_data = None
        self.enc_image_ref = None
        self.dec_image_ref = None
        self.data_field_name = None
        self.seed_field_name = None

        # --- 3. GRID LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 4. SIDEBAR (CONTROLS) ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color=COLOR_PANEL)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Logo & Version
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text="PROJECT UMBRA", font=FONT_HEADER, text_color=COLOR_ACCENT)
        self.lbl_logo.pack(pady=(30, 10), padx=20)
        
        self.lbl_version = ctk.CTkLabel(self.sidebar, text="v0.9.2 // CLASSIFIED", font=FONT_MONO, text_color="gray")
        self.lbl_version.pack(pady=(0, 20), padx=20)

        # Load Button
        self.btn_load = ctk.CTkButton(self.sidebar, text="LOAD TARGET IMAGE", font=FONT_UI, 
                                      fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, 
                                      command=self.load_image)
        self.btn_load.pack(pady=10, padx=20, fill="x")

        # --- [START] NEW AUDIO CONTROLS ---
        # Separator Line
        self.div_audio = ctk.CTkFrame(self.sidebar, height=2, fg_color="#444")
        self.div_audio.pack(pady=20, padx=20, fill="x")

        # Header
        self.lbl_audio = ctk.CTkLabel(self.sidebar, text="AUDIO INPUTS", font=("Roboto", 14, "bold"), text_color="#ccc")
        self.lbl_audio.pack(pady=(0, 10), padx=20, anchor="w")

        # Input 1: Music / Virtual Cable
        self.lbl_src1 = ctk.CTkLabel(self.sidebar, text="Music Source (Cable):", font=("Roboto", 11), text_color="gray")
        self.lbl_src1.pack(pady=(5, 0), padx=20, anchor="w")
        self.combo_music = ctk.CTkOptionMenu(self.sidebar, values=self.audio_devices)
        self.combo_music.pack(pady=5, padx=20, fill="x")

        # Input 2: Microphone
        self.lbl_src2 = ctk.CTkLabel(self.sidebar, text="Microphone:", font=("Roboto", 11), text_color="gray")
        self.lbl_src2.pack(pady=(5, 0), padx=20, anchor="w")
        self.combo_mic = ctk.CTkOptionMenu(self.sidebar, values=self.audio_devices)
        self.combo_mic.pack(pady=5, padx=20, fill="x")

        # Link Button
        self.btn_link_audio = ctk.CTkButton(
            self.sidebar, 
            text="LINK ROOM AUDIO", 
            font=FONT_UI,
            fg_color=COLOR_WARN, 
            hover_color="#d35400",
            command=self.toggle_audio_environment
        )
        self.btn_link_audio.pack(pady=15, padx=20, fill="x")

        # Live Noise Readout
        self.lbl_live_sigma = ctk.CTkLabel(self.sidebar, text="Noise: 0.0%", font=FONT_MONO, text_color="#efefef")
        self.lbl_live_sigma.pack(pady=5, padx=20)
        # --- [END] NEW AUDIO CONTROLS ---

        # --- 5. MAIN AREA ---
        self.main_area = ctk.CTkFrame(self, fg_color=COLOR_BG)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.tabview = ctk.CTkTabview(self.main_area, fg_color=COLOR_PANEL)
        self.tabview.pack(fill="both", expand=True)
        
        self.tab_encode = self.tabview.add("ENCODER")
        self.tab_decode = self.tabview.add("DECODER")
        self.tab_logs = self.tabview.add("SYSTEM LOGS")
        
        self.setup_encoder_tab()
        self.setup_decoder_tab()
        self.setup_logs_tab()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def setup_encoder_tab(self):
        """Build the ENCODER tab: image preview + carrier playback/export."""
        tab = self.tab_encode
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tab, text="ENCODE IMAGE → CARRIER",
            font=FONT_HEADER, text_color=COLOR_ACCENT,
        ).pack(pady=(20, 10))

        self.lbl_img_preview = ctk.CTkLabel(
            tab,
            text="NO TARGET LOADED\n\nUse 'LOAD TARGET IMAGE' in the sidebar",
            width=256, height=256, fg_color=COLOR_BG,
            font=FONT_MONO, text_color="gray",
        )
        self.lbl_img_preview.pack(pady=15)

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(pady=10)

        self.btn_play_noise = ctk.CTkButton(
            btn_row, text="▶  PLAY CARRIER", font=FONT_UI, state="disabled",
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.play_noise,
        )
        self.btn_play_noise.grid(row=0, column=0, padx=10)

        self.btn_save_wav = ctk.CTkButton(
            btn_row, text="EXPORT .WAV", font=FONT_UI, state="disabled",
            fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
            command=self.save_wav,
        )
        self.btn_save_wav.grid(row=0, column=1, padx=10)

    def setup_decoder_tab(self):
        """Build the DECODER tab: load a carrier .wav and reconstruct it."""
        tab = self.tab_decode
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tab, text="CARRIER → RECONSTRUCT",
            font=FONT_HEADER, text_color=COLOR_ACCENT,
        ).pack(pady=(20, 10))

        ctk.CTkButton(
            tab, text="LOAD CARRIER (.WAV)", font=FONT_UI,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self.load_wav,
        ).pack(pady=10)

        self.lbl_dec_preview = ctk.CTkLabel(
            tab, text="AWAITING SIGNAL", width=256, height=256,
            fg_color=COLOR_BG, font=FONT_MONO, text_color="gray",
        )
        self.lbl_dec_preview.pack(pady=15)

        self.btn_decode = ctk.CTkButton(
            tab, text="DECRYPT WITH UNIVERSAL KEY", font=FONT_UI, state="disabled",
            fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER,
            command=self.decode_signal,
        )
        self.btn_decode.pack(pady=10)

    def setup_logs_tab(self):
        """Build the SYSTEM LOGS tab: scrolling terminal output."""
        tab = self.tab_logs
        self.log_box = ctk.CTkTextbox(tab, font=FONT_MONO, activate_scrollbars=True)
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_box.configure(state="disabled")
        self.log("System initialized. Universal Key loaded.", "BOOT")

    def _init_ui(self):
        # Main Grid: Header (0), Content (1), Log (2)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) 
        self.grid_rowconfigure(2, weight=0) # Log doesn't expand infinitely

        # --- 1. HEADER ---
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        self.lbl_title = ctk.CTkLabel(self.header_frame, text="PROJECT UMBRA", font=("Roboto", 24, "bold"), text_color="white")
        self.lbl_title.pack(side="left")
        
        self.lbl_subtitle = ctk.CTkLabel(self.header_frame, text=" // NEURAL STEGANOGRAPHY LINK", font=("Roboto", 24), text_color="gray")
        self.lbl_subtitle.pack(side="left")

        # --- 2. MAIN CONTENT AREA ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (ENCODE) ---
        self.frame_enc = ctk.CTkFrame(self.content_frame, fg_color=COLOR_PANEL, corner_radius=10)
        self.frame_enc.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.frame_enc, text="TRANSMITTER (ENCODE)", font=FONT_HEADER).pack(pady=(20, 10))
        
        self.btn_load_img = ctk.CTkButton(self.frame_enc, text="LOAD SOURCE IMAGE", font=FONT_UI, command=self.load_image, height=40)
        self.btn_load_img.pack(pady=10, padx=20, fill="x")

        # Image Container (Fixed size aspect)
        self.preview_enc_frame = ctk.CTkFrame(self.frame_enc, fg_color="black", width=256, height=256)
        self.preview_enc_frame.pack(pady=10)
        self.preview_enc_frame.pack_propagate(False) # Don't shrink
        
        self.lbl_img_preview = ctk.CTkLabel(self.preview_enc_frame, text="NO SIGNAL", font=FONT_MONO, text_color="gray")
        self.lbl_img_preview.place(relx=0.5, rely=0.5, anchor="center")

        # Action Buttons
        self.btn_play_noise = ctk.CTkButton(self.frame_enc, text="▶ PREVIEW CARRIER (AUDIO)", command=self.play_noise, 
                                          state="disabled", fg_color=COLOR_WARN, font=FONT_UI)
        self.btn_play_noise.pack(pady=(20, 5), padx=20, fill="x")

        self.btn_save_wav = ctk.CTkButton(self.frame_enc, text="⬇ EXPORT CARRIER (.WAV)", command=self.save_wav, 
                                        state="disabled", fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=FONT_UI)
        self.btn_save_wav.pack(pady=5, padx=20, fill="x")

        # --- RIGHT PANEL (DECODE) ---
        self.frame_dec = ctk.CTkFrame(self.content_frame, fg_color=COLOR_PANEL, corner_radius=10)
        self.frame_dec.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(self.frame_dec, text="RECEIVER (DECODE)", font=FONT_HEADER).pack(pady=(20, 10))

        self.btn_load_wav = ctk.CTkButton(self.frame_dec, text="LOAD CARRIER SIGNAL (.WAV)", font=FONT_UI, command=self.load_wav, height=40)
        self.btn_load_wav.pack(pady=10, padx=20, fill="x")

        # Image Container
        self.preview_dec_frame = ctk.CTkFrame(self.frame_dec, fg_color="black", width=256, height=256)
        self.preview_dec_frame.pack(pady=10)
        self.preview_dec_frame.pack_propagate(False)

        self.lbl_dec_preview = ctk.CTkLabel(self.preview_dec_frame, text="AWAITING INPUT", font=FONT_MONO, text_color="gray")
        self.lbl_dec_preview.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_decode = ctk.CTkButton(self.frame_dec, text="🔓 DECRYPT & RECONSTRUCT", command=self.decode_signal, 
                                      state="disabled", fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER, font=FONT_UI, height=40)
        self.btn_decode.pack(pady=(20, 20), padx=20, fill="x")

        # --- 3. TERMINAL LOG ---
        self.log_box = ctk.CTkTextbox(self, height=120, font=FONT_MONO, activate_scrollbars=True)
        self.log_box.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.log_box.configure(state="disabled")

    def toggle_audio_environment(self):
        """Link/Unlink the audio monitor."""
        if not self.env_monitor.running:
            try:
                # 1. Get Device IDs from Dropdowns
                # Format is "Index: Name", so we split by ":" and take the first part
                id_music = int(self.combo_music.get().split(":")[0])
                id_mic = int(self.combo_mic.get().split(":")[0])
                
                # 2. Start Monitor
                if self.env_monitor.start(id_music, id_mic):
                    self.dynamic_sigma_mode = True
                    self.btn_link_audio.configure(text="UNLINK AUDIO", fg_color=COLOR_SUCCESS)
                    self.log(f"Linked Audio. Music: {id_music}, Mic: {id_mic}", "SYSTEM")
                    
                    # 3. Start Background Thread
                    threading.Thread(target=self._monitor_loop, daemon=True).start()
            except Exception as e:
                self.log(f"Audio Link Error: {e}", "ERROR")
        else:
            self.env_monitor.stop()
            self.dynamic_sigma_mode = False
            self.btn_link_audio.configure(text="LINK ROOM AUDIO", fg_color=COLOR_WARN)
            self.log("Unlinked Audio Environment.", "SYSTEM")

    def _monitor_loop(self):
        """Update Sigma based on Room Noise."""
        while self.env_monitor.running:
            noise = self.env_monitor.current_interference_level
            
            # Update UI Label
            self.lbl_live_sigma.configure(text=f"Noise: {noise*100:.1f}%")
            
            # Map Noise (0.0 - 1.0) to Sigma (0.01 - 0.8)
            new_sigma = 0.01 + (noise * 0.8)
            GOD_GENE["sigma"] = new_sigma
            
            time.sleep(0.1)


    def log(self, message, source="INFO"):
        """Appends text to the terminal log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] [{source}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.update()

    def _detect_packet_structure(self):
        if self.data_field_name and self.seed_field_name:
            return
        
        # Heuristic detection
        likely_data = ['encoded', 'data', 'payload', 'noise']
        likely_seed = ['permutation_seed', 'seed', 'key']
        
        fields = dataclasses.fields(NoisePacket)
        f_names = [f.name for f in fields]
        
        self.data_field_name = next((n for n in likely_data if n in f_names), 'data')
        self.seed_field_name = next((n for n in likely_seed if n in f_names), 'seed')
        
        self.log(f"Packet Protocol Detected: Payload='{self.data_field_name}' | Key='{self.seed_field_name}'", "KERNEL")

    # --- ENCODER LOGIC ---
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if not path:
            return

        try:
            self.log(f"Loading source: {os.path.basename(path)}...", "ENCODER")
            pil_img = Image.open(path).convert("RGB").resize((256, 256))
            self.current_image = np.asarray(pil_img, dtype=np.float32) / 255.0
            
            # Display
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(256, 256))
            self.lbl_img_preview.configure(image=ctk_img, text="")

            self.enc_image_ref = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(256, 256))
            self.lbl_img_preview.configure(image=self.enc_image_ref, text="")
            
            self.log("Encoding latent image to noise stream...", "ENCODER")
            self._detect_packet_structure()
            
            encoder = NoiseStreamEncoder(sigma=GOD_GENE["sigma"])
            
            # Construct Encode Call
            kwargs = {self.seed_field_name: GOD_GENE["seed"]} if self.seed_field_name != "seed" else {"seed": GOD_GENE["seed"]}
            try:
                packet = encoder.encode(self.current_image, **kwargs)
            except TypeError:
                packet = encoder.encode(self.current_image, GOD_GENE["seed"])

            raw_data = getattr(packet, self.data_field_name)
            if raw_data is None:
                raise ValueError("Encoding failed: Empty payload.")

            self.current_audio, self.sample_rate = image_data_to_audio(raw_data)
            
            self.btn_play_noise.configure(state="normal")
            self.btn_save_wav.configure(state="normal")
            self.log(f"Encoding Complete. Carrier Size: {len(self.current_audio)} samples.", "SUCCESS")
            
        except Exception as e:
            traceback.print_exc()
            self.log(f"Critical Error: {e}", "ERROR")
            messagebox.showerror("Error", str(e))

    def play_noise(self):
        if self.current_audio is not None:
            self.log("Playing carrier signal...", "AUDIO")
            sd.play(self.current_audio, self.sample_rate)

    def save_wav(self):
        if self.current_audio is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if path:
            wavfile.write(path, self.sample_rate, self.current_audio)
            self.log(f"Carrier signal exported to {os.path.basename(path)}", "IO")

    # --- DECODER LOGIC ---
    def load_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Audio", "*.wav")])
        if not path:
            return
        
        try:
            self.log(f"Analyzing signal: {os.path.basename(path)}...", "DECODER")
            self.loaded_packet_data = audio_to_image_data(path)
            self.btn_decode.configure(state="normal")
            
            # Reset Preview
            self.lbl_dec_preview.configure(image=None, text="SIGNAL LOCKED\nREADY TO DECRYPT")
            self.log("Signal integrity verified. Ready to decode.", "DECODER")
        except Exception as e:
            self.log(f"Load Error: {e}", "ERROR")
            messagebox.showerror("Error", str(e))

    def decode_signal(self):
        if self.loaded_packet_data is None:
            return
        self._detect_packet_structure()

        try:
            self.log("Attempting decryption with Universal Key...", "NEURAL")
            
            # Reconstruct Packet
            packet_args = {
                self.data_field_name: self.loaded_packet_data,
                self.seed_field_name: GOD_GENE["seed"]
            }
            
            # Fill defaults
            for f in dataclasses.fields(NoisePacket):
                if f.name not in packet_args:
                    if f.name in ['image_shape', 'shape']:
                        packet_args[f.name] = (256, 256, 3)
                    elif f.name == 'metadata':
                        packet_args[f.name] = {}
                    else:
                        packet_args[f.name] = None

            packet = NoisePacket(**packet_args)
            decoder = NoiseStreamDecoder(denoise_sigma=GOD_GENE["denoise_sigma"])
            
            kwargs = {self.seed_field_name: GOD_GENE["seed"]} if self.seed_field_name != "seed" else {"seed": GOD_GENE["seed"]}
            
            try:
                recon = decoder.decode(packet, **kwargs)
            except TypeError:
                recon = decoder.decode(packet, GOD_GENE["seed"])

            if recon is None:
                raise ValueError("Reconstruction yielded null result.")

            # Clip and Display
            img_uint8 = np.clip(recon * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(256, 256))
            self.lbl_dec_preview.configure(image=ctk_img, text="")
            self.dec_image_ref = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(256, 256))
            self.lbl_dec_preview.configure(image=self.dec_image_ref, text="")
            self.log("Decryption Successful. Image reconstructed.", "SUCCESS")
            
        except Exception as e:
            traceback.print_exc()
            self.log(f"Decryption Failed: {e}", "FATAL")
            messagebox.showerror("Decode Error", f"Failed to reconstruct packet.\n{str(e)}")

if __name__ == "__main__":
    app = UmbraTerminal()
    app.mainloop()