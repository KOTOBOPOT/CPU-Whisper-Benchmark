"""Whisper model implementation using CTranslate2 with float32 precision."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import librosa
import ctranslate2
from transformers import AutoProcessor


class WhisperModel:
    """Whisper model using CTranslate2 with float32 (baseline) precision."""

    def __init__(self, weights_path: Optional[Path]) -> None:
        self.weights_path = weights_path
        
        # Validate weights path
        if weights_path is None or not weights_path.exists():
            error_msg = (
                f"CTranslate2 model weights not found at {weights_path}. "
                "Please convert the model using weights_utils/whisper-ctranslate2/convert_whisper.py"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info("Loading CTranslate2 model from %s", weights_path)
        logging.info("Compute type: float32 (baseline, no quantization)")
        
        try:
            # Load CTranslate2 model with float32 precision
            self.model = ctranslate2.models.Whisper(
                str(weights_path),
                device="cpu",
                compute_type="float32"
            )
            
            # Load processor/tokenizer from original model
            # CTranslate2 models don't include preprocessor_config.json
            logging.info("Loading processor from openai/whisper-large-v3")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            
            logging.info("Model loaded successfully on CPU with float32 precision")
            
        except Exception as e:
            logging.error("Failed to load CTranslate2 model: %s", e)
            raise

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe audio using CTranslate2 Whisper model.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename for logging
            
        Returns:
            Transcribed text as string
        """
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Load audio with librosa (Whisper expects 16kHz sample rate)
            audio_array, sample_rate = librosa.load(audio_buffer, sr=16000)
            
            # Extract features using the processor (return as numpy, not PyTorch tensors)
            inputs = self.processor(audio_array, return_tensors="np", sampling_rate=16000)
            features = inputs.input_features  # Shape: [batch_size, n_mels, n_frames]
            
            # Convert features to CTranslate2 format
            features_ct2 = ctranslate2.StorageView.from_array(features)
            
            # Run inference with CTranslate2
            # Whisper.generate requires prompts with special tokens
            # Token IDs for whisper-large-v3: 50258=<|startoftranscript|>, 50263=<|ru|>, 50360=<|transcribe|>
            prompts = [[50258, 50263, 50360]]  # Russian transcription
            
            results = self.model.generate(
                features_ct2,
                prompts,
                beam_size=1,  # Greedy decoding for speed
                return_scores=False,
                return_no_speech_prob=False
            )
            
            # Get the first (and only) result
            if not results or not results[0] or not results[0].sequences:
                logging.warning("No transcription generated for %s", filename)
                return ""
            
            # Extract token IDs from the first sequence
            token_ids = results[0].sequences_ids[0]
            
            # Decode using the tokenizer
            transcription = self.processor.decode(token_ids, skip_special_tokens=True)
            
            logging.info("Successfully transcribed %s (length: %d chars)", filename, len(transcription))
            return transcription.strip()
            
        except Exception as e:
            error_msg = f"Error transcribing {filename}: {str(e)}"
            logging.error(error_msg)
            return f"[Transcription failed: {str(e)}]"

