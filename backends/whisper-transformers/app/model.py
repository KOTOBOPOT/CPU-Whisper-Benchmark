"""Whisper model implementation using HuggingFace transformers."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import torch
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration


class WhisperModel:
    """Whisper model using HuggingFace transformers with openai/whisper-small."""

    def __init__(self, weights_path: Optional[Path]) -> None:
        self.weights_path = weights_path
        
        # Use custom weights path if provided, otherwise use default model
        if weights_path is not None and weights_path.exists():
            logging.info("Loading model from weights at %s", weights_path)
            model_name = str(weights_path)
        else:
            logging.info("Loading default openai/whisper-small model")
            model_name = "openai/whisper-small"
        
        # Initialize processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logging.info("Model moved to GPU")
            else:
                logging.info("Model running on CPU")
                
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe audio using Whisper model."""
        try:
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Load audio with librosa (whisper expects 16kHz sample rate)
            audio_array, sample_rate = librosa.load(audio_buffer, sr=16000)
            
            # Process audio
            inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000)
            input_features = inputs.input_features
            
            # Move to same device as model
            if torch.cuda.is_available():
                input_features = input_features.cuda()
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(inputs=input_features)
            
            # Decode transcription
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logging.info("Successfully transcribed %s", filename)
            return transcription.strip()
            
        except Exception as e:
            error_msg = f"Error transcribing {filename}: {str(e)}"
            logging.error(error_msg)
            return f"[Transcription failed: {str(e)}]"
