import logging
import struct
import zlib

import numpy as np

logger = logging.getLogger(__name__)


class DataPayloadCodec:
    """
    Robust codec with 'Distributed Scrambling'.
    Spreads data bits across the entire image to prevent 'top-loading' (squished look).
    """
    def __init__(self, redundancy=9):
        self.redundancy = redundancy 
        self.magic = b"UMBR"
        # Fixed seed ensures the scrambler is deterministic (Decoder can reverse it)
        self.scramble_seed = 42 

    def _get_permutation(self, capacity):
        """Generates a fixed random shuffle pattern for the pixels."""
        rng = np.random.RandomState(self.scramble_seed)
        return rng.permutation(capacity)

    def encode_file(self, file_bytes, shape=(512, 512)):
        """Wraps data, adds redundancy, and SCRAMBLES it across the frame."""
        # 1. Compress
        try:
            compressed = zlib.compress(file_bytes)
        except Exception:
            compressed = file_bytes
            
        # 2. Header
        length_header = struct.pack(">I", len(compressed))
        full_payload = self.magic + length_header + compressed
        
        # 3. Bit Expansion
        bits = []
        for byte in full_payload:
            for i in range(8):
                bit = (byte >> (7-i)) & 1
                bits.extend([bit] * self.redundancy)
        
        capacity = shape[0] * shape[1]
        if len(bits) > capacity:
            raise ValueError(f"File too big! Need {len(bits)} pixels, have {capacity}.")
            
        # 4. Padding (Zero fill)
        padding = [0] * (capacity - len(bits))
        all_bits = np.array(bits + padding, dtype=np.float32)
        
        # 5. SCRAMBLE (Distribute bits to fix 'squished' look)
        perm = self._get_permutation(capacity)
        scrambled_bits = np.zeros_like(all_bits)
        scrambled_bits[perm] = all_bits
        
        # 6. Reshape to Image
        img_gray = scrambled_bits.reshape(shape)
        img_rgb = np.stack([img_gray] * 3, axis=-1)
        
        return img_rgb

    def decode_image(self, img_arr):
        """Unscrambles the noisy image and extracts data."""
        logger.debug("Decoding payload from image")
        
        # 1. Flatten & Grayscale
        if img_arr.ndim == 3:
            flat = img_arr.mean(axis=2).flatten()
        else:
            flat = img_arr.flatten()
            
        # 2. Threshold
        raw_bits = (flat > 0.5).astype(int)
        
        # 3. UNSCRAMBLE (Reverse the shuffle)
        capacity = len(raw_bits)
        perm = self._get_permutation(capacity)

        unscrambled_bits = raw_bits[perm]
        
        # 4. Vote (Demodulate)
        limit = len(unscrambled_bits) - (len(unscrambled_bits) % self.redundancy)
        chunks = unscrambled_bits[:limit].reshape(-1, self.redundancy)
        sums = chunks.sum(axis=1)
        demodulated_bits = (sums > (self.redundancy / 2)).astype(int)
        
        # 5. Bytes
        pad_len = (8 - (len(demodulated_bits) % 8)) % 8
        if pad_len > 0:
            demodulated_bits = np.concatenate([demodulated_bits, np.zeros(pad_len, dtype=int)])
            
        byte_chunks = demodulated_bits.reshape(-1, 8)
        powers = 1 << np.arange(7, -1, -1)
        stream_bytes = byte_chunks.dot(powers).astype(np.uint8).tobytes()
        
        # 6. Search for Magic
        start_index = stream_bytes.find(self.magic)
        if start_index == -1:
            logger.error("Sync marker 'UMBR' not found in decoded stream")
            return None

        try:
            cursor = start_index + 4
            if cursor + 4 > len(stream_bytes):
                logger.error("Stream truncated: cannot read length header")
                return None
            (data_len,) = struct.unpack(">I", stream_bytes[cursor:cursor+4])
            cursor += 4
            if data_len > len(stream_bytes) - cursor:
                logger.error(
                    "Declared payload length %d exceeds available data %d",
                    data_len,
                    len(stream_bytes) - cursor,
                )
                return None
            compressed_data = stream_bytes[cursor : cursor+data_len]
            return zlib.decompress(compressed_data)
        except Exception as exc:
            logger.error("Payload extraction failed: %s", exc)
            return None