"""Whisper model implementation using PyTorch with Dynamic Quantization."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import torch
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration


class WhisperModel:
    """Whisper model using PyTorch with Dynamic Quantization (int8)."""

    def __init__(self, weights_path: Optional[Path]) -> None:
        self.weights_path = weights_path
        
        # Determine model name
        if weights_path is not None:
            weights_str = str(weights_path)
            # Check if it's a HuggingFace model name (contains '/') or local path
            if '/' in weights_str and not Path(weights_str).exists():
                # It's a HuggingFace model name (e.g., "openai/whisper-large-v3")
                model_name = weights_str
                logging.info("Loading HuggingFace model: %s", model_name)
            elif Path(weights_str).exists():
                # It's a local path
                model_name = weights_str
                logging.info("Loading model from local path: %s", model_name)
            else:
                # Assume it's a HuggingFace model name
                model_name = weights_str
                logging.info("Loading HuggingFace model: %s", model_name)
        else:
            # Используем Whisper Large v3 по умолчанию
            model_name = "openai/whisper-large-v3"
            logging.info("Loading default model: %s", model_name)
        
        # Initialize processor and model
        try:
            logging.info("Loading processor and model...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Apply Dynamic Quantization
            # Квантизируем только Linear слои в int8, остальное остается float32
            logging.info("Applying Dynamic Quantization (int8) to Linear layers...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # Квантизируем только Linear слои
                dtype=torch.qint8
            )
            
            # Ensure model is on CPU (quantization работает только на CPU)
            if torch.cuda.is_available():
                logging.warning("Dynamic quantization works on CPU only. Model will run on CPU.")
            self.model = self.model.cpu()
            
            logging.info("Model loaded successfully with Dynamic Quantization (int8)")
            logging.info("Quantized layers: Linear (int8), other layers: float32")
                
        except Exception as e:
            logging.error("Failed to load model: %s", e)
            raise

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe audio using PyTorch Whisper model with Dynamic Quantization.
        
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
            
            # Process audio
            inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000)
            input_features = inputs.input_features
            
            # Ensure input is on CPU (quantized model работает только на CPU)
            input_features = input_features.cpu()
            
            # Generate transcription with Russian language
            # Используем forced_decoder_ids для русского языка
            # Token IDs для whisper-large-v3: 50258=<|startoftranscript|>, 50263=<|ru|>, 50360=<|transcribe|>
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="ru",
                task="transcribe"
            )
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=1,  # Greedy decoding для скорости (как в CTranslate2)
                    # Не указываем max_new_tokens - Whisper сам остановится по специальным токенам
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            logging.info("Successfully transcribed %s (length: %d chars)", filename, len(transcription))
            return transcription.strip()
            
        except Exception as e:
            error_msg = f"Error transcribing {filename}: {str(e)}"
            logging.error(error_msg)
            return f"[Transcription failed: {str(e)}]"
