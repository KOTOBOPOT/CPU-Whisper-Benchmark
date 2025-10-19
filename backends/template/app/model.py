"""Mock Whisper model stub to keep API wiring consistent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class WhisperModel:
    """Very small stub that pretends to perform speech-to-text."""

    def __init__(self, weights_path: Optional[Path]) -> None:
        self.weights_path = weights_path
        if weights_path is None:
            logging.info("Starting mock model without weights.")
        else:
            logging.info("Starting mock model with weights at %s", weights_path)
            if not weights_path.exists():
                logging.warning("Weights path %s does not exist.", weights_path)

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Produce a deterministic pseudo transcription for the demo API."""
        size_in_kb = len(audio_bytes) // 1024
        filename_stub = Path(filename).stem or "audio"
        if self.weights_path:
            return (
                f"[mock transcript for {filename_stub} | size={size_in_kb}KB | "
                f"weights={self.weights_path}]"
            )
        return f"[mock transcript for {filename_stub} | size={size_in_kb}KB]"
