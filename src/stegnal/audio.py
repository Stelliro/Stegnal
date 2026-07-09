
import numpy as np
from scipy.io import wavfile

# FIXED CONSTANT: 
# Images are 0.0-1.0. Noise adds roughly +/- 3.0 sigma. 
# A factor of 5.0 ensures we fit in the audio range without clipping or losing scale.
AUDIO_SCALE_FACTOR = 5.0

def image_data_to_audio(data_array: np.ndarray, sample_rate: int = 48000) -> tuple[np.ndarray, int]:
    """
    Converts data to int16 audio using a FIXED scale factor.
    This preserves relative amplitude for the decoder.
    """
    flat_data = data_array.flatten()
    
    # Scale down to fit into -1.0 ... 1.0
    normalized = flat_data / AUDIO_SCALE_FACTOR
    
    # Clip to prevent wrapping distortion (hard limit)
    normalized = np.clip(normalized, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (normalized * 32767).astype(np.int16)
    
    return audio_int16, sample_rate

def audio_to_image_data(audio_path: str, target_shape=(256, 256, 3)) -> np.ndarray:
    """
    Reads WAV and restores the original amplitude.
    """
    sample_rate, audio_int16 = wavfile.read(audio_path)
    
    # Normalize back to float -1.0 to 1.0
    audio_float = audio_int16.astype(np.float32) / 32767.0
    
    # RESTORE AMPLITUDE (The Critical Fix)
    data = audio_float * AUDIO_SCALE_FACTOR
    
    required_size = np.prod(target_shape)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        
    current_size = data.size
    if current_size < required_size:
        padded = np.zeros(required_size, dtype=np.float32)
        padded[:current_size] = data
        data = padded
    else:
        data = data[:required_size]
        
    return data.reshape(target_shape)