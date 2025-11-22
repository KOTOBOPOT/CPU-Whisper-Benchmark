"""
Persistent C++ process model wrapper for Whisper using simple text protocol.
Hybrid approach: Python creates mel-spectrogram (transformers), C++ does inference (ONNX).
"""
import io
import logging
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class WhisperCppPersistentModel:
    """Wrapper for C++ Whisper model using persistent process with simple text protocol."""
    
    def __init__(self, model_path: str, binary_path: str, num_threads: int = 4):
        self.model_path = Path(model_path)
        self.binary_path = Path(binary_path)
        self.num_threads = num_threads
        self.process = None
        
        # Initialize processor for mel-spectrogram extraction and tokenizer for decoding
        logger.info("Loading processor for mel-spectrogram extraction")
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base")
        self.tokenizer = self.processor.tokenizer
        
        # Validate paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")
        
        self._start_process()
    
    def _start_process(self):
        """Start the C++ process in interactive mode."""
        cmd = [
            str(self.binary_path),
            "--model-path", str(self.model_path),
            "--threads", str(self.num_threads),
            "--interactive"
        ]
        
        logger.info(f"Starting C++ process: {' '.join(cmd)}")
        
        try:
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
                stderr=None,  # Let stderr go to Docker logs
            text=True,
            bufsize=1  # Line buffered
        )
        
            # Wait for READY signal (may need to skip stderr lines)
            max_attempts = 10
            ready_found = False
            for _ in range(max_attempts):
                line = self.process.stdout.readline().strip()
                logger.debug(f"Got line from C++: {line}")
                if line == "READY":
                    ready_found = True
                    break
                
            if not ready_found:
                raise RuntimeError("C++ process did not send READY signal")
            
            logger.info("C++ process is ready")
            
            # Warm-up: first inference includes ONNX Runtime graph optimization
            # Run a dummy request to exclude this from actual measurements
            self._warmup()
                    
            except Exception as e:
            logger.error(f"Failed to start C++ process: {e}")
            if self.process:
                self.process.kill()
            raise
    
    def _warmup(self):
        """Warm-up the model with a dummy request to trigger ONNX Runtime optimizations."""
        try:
            logger.info("Warming up model with dummy request...")
            
            # Create 1 second of silence (16kHz)
            dummy_audio = np.zeros(16000, dtype=np.float32)
            
            # Convert to bytes (WAV format)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)
                # Convert float32 [-1, 1] to int16
                audio_int16 = (dummy_audio * 32767).astype(np.int16)
                wav.writeframes(audio_int16.tobytes())
            
            dummy_bytes = wav_buffer.getvalue()
            
            # Run transcription (ignore result)
            _ = self.transcribe(audio_bytes=dummy_bytes, filename="warmup.wav")
            logger.info("Model warm-up complete")
            
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")
    
    def transcribe(self, audio_bytes: bytes = None, audio_path: Union[str, Path] = None, filename: str = None) -> str:
        """
        Transcribe audio using persistent C++ process.
        Hybrid: Python creates mel-spectrogram (transformers), C++ does inference (ONNX).
        
        Args:
            audio_bytes: Audio file contents as bytes
            audio_path: Path to audio file (alternative to audio_bytes)
            filename: Original filename (unused, kept for API compatibility)
            
        Returns:
            Transcribed text as string
        """
        temp_mel_file = None
        
        try:
            # Load audio using librosa
            if audio_bytes:
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            elif audio_path:
                audio_array, _ = librosa.load(audio_path, sr=16000)
            else:
                raise ValueError("Either audio_bytes or audio_path must be provided")
            
            # Create mel-spectrogram using transformers processor
            logger.debug("Creating mel-spectrogram with Python (transformers)")
            inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000)
            mel_spec = inputs.input_features.numpy()[0]  # Shape: [80, 3000]
            
            # Save mel-spectrogram to temp file (binary format)
            temp_mel_file = Path(tempfile.gettempdir()) / "mel_spec.bin"
            with open(temp_mel_file, 'wb') as f:
                # Write shape (2 ints) + data (floats)
                f.write(struct.pack('ii', mel_spec.shape[0], mel_spec.shape[1]))
                f.write(mel_spec.astype(np.float32).tobytes())
            
            logger.debug(f"Mel-spec shape: {mel_spec.shape}, mean: {mel_spec.mean():.4f}")
            
            # Send command to C++ process
            command = f"MEL {temp_mel_file}\n"
            logger.debug(f"Sending to C++: {command.strip()}")
            
            self.process.stdin.write(command)
            self.process.stdin.flush()
            
            # Read result
            result_line = self.process.stdout.readline().strip()
            
            if result_line.startswith("ERROR:"):
                error_msg = result_line[6:].strip()
                raise RuntimeError(f"C++ process error: {error_msg}")
            
            # Parse token IDs from C++ response
            if result_line.startswith("TOKENS:"):
                token_str = result_line[7:]  # Remove "TOKENS:" prefix
                if token_str:
                    token_ids = [int(t) for t in token_str.split(',')]
                    return self.tokenizer.decode(token_ids, skip_special_tokens=True)
                return ""
            
            # Should not reach here
            logger.warning(f"Unexpected C++ response format: {result_line}")
            return ""
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self._restart_process()
            raise
        finally:
            # Clean up temp mel-spectrogram file
            if temp_mel_file and temp_mel_file.exists():
            try:
                    temp_mel_file.unlink()
            except:
                pass
    
    def _restart_process(self):
        """Restart the C++ process if it crashes."""
        logger.warning("Restarting C++ process...")
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except:
                pass
        self._start_process()
    
    def __del__(self):
        """Cleanup: terminate the C++ process."""
        if self.process:
            try:
                # Send QUIT command
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            finally:
                self.process = None
