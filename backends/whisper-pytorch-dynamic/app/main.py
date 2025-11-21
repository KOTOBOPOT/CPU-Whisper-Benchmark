"""FastAPI backend for PyTorch Whisper with Dynamic Quantization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .model import WhisperModel
from utils.fastapi_app import create_app


def _load_weights_path() -> Optional[Path]:
    """Load model path from environment variable.
    
    Can be either:
    - HuggingFace model name (e.g., "openai/whisper-large-v3")
    - Local path to model directory
    """
    env_value = os.getenv("WHISPER_MODEL_WEIGHTS_PATH")
    if not env_value:
        # Default to Whisper Large v3
        return None
    
    # Check if it's a HuggingFace model name (contains '/')
    if '/' in env_value and not Path(env_value).exists():
        # It's a HuggingFace model name, return as string (not Path)
        return Path(env_value)  # Will be converted to string in model.py
    
    # It's a local path
    return Path(env_value).expanduser().resolve()


model = WhisperModel(weights_path=_load_weights_path())
app = create_app(model)
