import logging
import os
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter

logger = logging.getLogger("Umbra")

def highpass_filter(data, cutoff=300, fs=48000, order=5):
    """Kills low-end room rumble and fan noise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

class InterferenceSynth:
    def __init__(self):
        self.running = False
        self.stream = None
        self.severity = 0.0
        
    def start(self, device_index):
        if self.running:
            return
        self.running = True
        def callback(outdata, frames, time, status):
            if self.severity > 0:
                noise = np.random.uniform(-0.5, 0.5, frames) * self.severity
                outdata[:] = noise.reshape(-1, 1)
            else:
                outdata[:] = 0.0
        try:
            dev_info = sd.query_devices(device_index, 'output')
            native_rate = int(dev_info['default_samplerate'])
            self.stream = sd.OutputStream(device=device_index, channels=1, callback=callback, samplerate=native_rate)
            self.stream.start()
        except Exception as e:
            logger.error(f"Synth failed: {e}")
            self.running = False

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False

    def set_severity(self, val):
        self.severity = float(val)

class AudioEngine:
    def __init__(self):
        self.master_volume = 0.5
        self.input_gain = 1.0 
        self.interrupt_flag = False
        self.current_interference_level = 0.0
        self.latency_ms = 1000.0
        self.cache_dir = "cache/initial_scans"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.synth = InterferenceSynth()
        self.monitor_stream = None
        self.monitor_running = False

    def get_devices(self, kind='input'):
        try:
            devs = sd.query_devices()
            valid = []
            for i, d in enumerate(devs):
                if kind == 'input' and d['max_input_channels'] > 0:
                    valid.append(f"{i}: {d['name']}")
                elif kind == 'output' and d['max_output_channels'] > 0:
                    valid.append(f"{i}: {d['name']}")
            return valid
        except Exception:
            return []

    def set_master_volume(self, val):
        self.master_volume = np.clip(float(val), 0.0, 1.0)

    def set_input_sensitivity(self, val):
        self.input_gain = float(val) * 5.0

    def set_latency(self, val_ms):
        self.latency_ms = float(val_ms)

    def start_monitoring(self, device_index):
        if self.monitor_running:
            return
        self.monitor_running = True
        def monitor_callback(indata, frames, time, status):
            vol = np.linalg.norm(indata) * 10
            self.current_interference_level = np.clip(vol, 0.0, 1.0)
        try:
            self.monitor_stream = sd.InputStream(device=device_index, channels=1, callback=monitor_callback)
            self.monitor_stream.start()
        except Exception:
            self.monitor_running = False

    def stop_monitoring(self):
        if self.monitor_stream:
            self.monitor_stream.stop()
            self.monitor_stream.close()
            self.monitor_stream = None
        self.monitor_running = False

    def run_hardware_diagnostic(self, idx_out, idx_in):
        logger.info("--- HARDWARE DIAGNOSTIC START ---")
        fs = 48000
        t = np.linspace(0, 0.5, int(fs*0.5), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.5
        rec = self.transmit_and_record(tone, fs, idx_out, idx_in, use_sync_pulse=True)
        if rec is not None and np.max(np.abs(rec)) > 0.01:
            return True, "Link Verified"
        return False, "No Signal Detected"

    def transmit_and_record(self, wav_data, target_sample_rate, idx_out, idx_in, use_sync_pulse=True):
        passes = 1 if use_sync_pulse else 3
        silence = np.zeros(int(target_sample_rate * 0.2)) 
        combined_payload = wav_data
        for _ in range(passes - 1):
            combined_payload = np.concatenate([combined_payload, silence, wav_data])

        raw_rec = self._internal_io(combined_payload, target_sample_rate, idx_out, idx_in, use_sync_pulse)
        if raw_rec is None:
            return None

        clean_rec = highpass_filter(raw_rec, cutoff=300, fs=target_sample_rate)
        if passes > 1:
            step = len(wav_data) + len(silence)
            chunks = [clean_rec[i*step : i*step + len(wav_data)] for i in range(passes)]
            min_len = min(len(c) for c in chunks)
            return np.mean([c[:min_len] for c in chunks], axis=0)
        return clean_rec

    def _internal_io(self, payload, fs, idx_out, idx_in, sync):
        self.interrupt_flag = False
        peak = np.max(np.abs(payload))
        final_tx = payload / (peak + 1e-9) if peak > 0 else payload
        if sync:
            t = np.linspace(0, 0.2, int(fs * 0.2), False)
            beep = np.sin(2 * np.pi * 1000 * t) * 0.8
            final_tx = np.concatenate([beep, np.zeros(int(fs*0.1)), final_tx])

        recorded, finished, pos = [], threading.Event(), 0
        def in_cb(indata, f, t, s):
            recorded.append(indata.copy() * self.input_gain)

        def out_cb(outdata, f, t, s):
            nonlocal pos
            rem = len(final_tx) - pos
            if rem <= 0:
                outdata[:] = 0
                finished.set()
                raise sd.CallbackStop
            size = min(rem, f)
            outdata[:size] = (final_tx[pos:pos+size] * self.master_volume).reshape(-1, 1)
            pos += size

        try:
            with sd.InputStream(device=idx_in, samplerate=fs, channels=1, callback=in_cb):
                sd.sleep(100)
                with sd.OutputStream(device=idx_out, samplerate=fs, channels=1, callback=out_cb):
                    finished.wait(timeout=(len(final_tx)/fs) + 5.0)
                    sd.sleep(500 + int(self.latency_ms))
            full = np.concatenate(recorded).flatten()
            if sync:
                peaks = np.where(np.abs(full) > 0.2)[0]
                if not len(peaks):
                    return None
                start = peaks[0] + int(fs * 0.3) - 500
                return full[start : start + len(payload)]
            return full[:len(payload)]
        except Exception:
            return None

    def abort_transmission(self):
        self.interrupt_flag = True


def list_audio_devices(kind='input'):
    """Return ``"<index>: <name>"`` strings for devices of the given kind."""
    try:
        devs = sd.query_devices()
        valid = []
        for i, d in enumerate(devs):
            if kind == 'input' and d['max_input_channels'] > 0:
                valid.append(f"{i}: {d['name']}")
            elif kind == 'output' and d['max_output_channels'] > 0:
                valid.append(f"{i}: {d['name']}")
        return valid
    except Exception:
        return []


class EnvironmentMonitor:
    """Room-audio monitor for the standalone ``app.py`` "Terminal" UI.

    Opens an input stream on the chosen microphone and exposes a normalized
    interference level (0.0-1.0) derived from the live signal. The music
    device index is retained for diagnostics/future mixing, but the noise
    estimate is driven by the microphone capture.
    """

    def __init__(self):
        self.running = False
        self.current_interference_level = 0.0
        self.stream = None
        self.music_device = None
        self.mic_device = None

    def get_devices(self, kind='input'):
        return list_audio_devices(kind)

    def start(self, music_device_index, mic_device_index):
        """Begin monitoring the microphone. Returns ``True`` on success."""
        if self.running:
            return False
        self.music_device = music_device_index
        self.mic_device = mic_device_index

        def monitor_callback(indata, frames, time, status):
            vol = np.linalg.norm(indata) * 10
            self.current_interference_level = float(np.clip(vol, 0.0, 1.0))

        try:
            self.stream = sd.InputStream(
                device=mic_device_index, channels=1, callback=monitor_callback
            )
            self.stream.start()
            self.running = True
            return True
        except Exception as e:
            logger.error(f"EnvironmentMonitor failed to start: {e}")
            self.running = False
            return False

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.running = False
        self.current_interference_level = 0.0