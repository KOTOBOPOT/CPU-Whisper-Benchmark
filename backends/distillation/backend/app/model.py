"""Whisper distilled model implementation for transcription."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import io

import torch
import librosa
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)


class WhisperModel:
    """Whisper model for speech-to-text transcription."""

    def __init__(
        self,
        checkpoint_path: Optional[Path],
        device: str = "cpu",
        dtype: str = "float32",
        language: str = "ru",
        task: str = "transcribe",
    ) -> None:
        """
        Initialize Whisper model.

        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on (cpu, cuda:0, etc.)
            dtype: Data type for inference (float32, float16, bfloat16)
            language: Language code for transcription (default: ru)
            task: Task type (transcribe or translate)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.language = language
        self.task = task
        
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for Whisper model")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Determine dtype
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA not available! Falling back to CPU")
            self.device = "cpu"
            if dtype == "bfloat16":
                logger.warning("bfloat16 not recommended for CPU, using float32")
                dtype = "float32"
        
        self.dtype = getattr(torch, dtype)
        
        logger.info(f"Loading Whisper processor from {checkpoint_path}...")
        self.processor = WhisperProcessor.from_pretrained(str(checkpoint_path))
        
        logger.info(f"Loading Whisper model from {checkpoint_path}...")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Model loaded: {self.model.__class__.__name__}")
        logger.info(f"Model size: {num_params:.1f}M parameters")
        logger.info(f"Device: {self.device}, dtype: {dtype}")

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for logging)

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Processing audio file: {filename} ({len(audio_bytes)} bytes)")
            
            # Load audio from bytes
            audio = self._load_audio_from_bytes(audio_bytes)
            
            # Prepare inputs
            inputs = self.processor.feature_extractor(
                audio,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            input_features = inputs.input_features.to(self.device, dtype=self.dtype)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Configure generation parameters
            gen_kwargs = {
                "max_length": 448,
                "num_beams": 1,  # Greedy decoding for speed
                "return_timestamps": False,
            }
            
            # Add language parameters for multilingual models
            if hasattr(self.model.generation_config, "is_multilingual") and \
               self.model.generation_config.is_multilingual:
                gen_kwargs["language"] = self.language
                gen_kwargs["task"] = self.task
            
            logger.info("Generating transcription...")
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            
            # Decode result
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            logger.info(f"Transcription complete: {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            raise

    def _load_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Load audio from bytes and resample to target sampling rate.

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            Audio data as numpy array
        """
        target_sr = self.processor.feature_extractor.sampling_rate
        
        # Load audio using librosa from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
        
        duration = len(audio) / sr
        logger.info(f"Audio loaded: duration={duration:.2f}s, sampling_rate={sr}Hz")
        
        return audio.astype(np.float32)

