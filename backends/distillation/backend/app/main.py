"""FastAPI backend for Whisper distilled model implementing the Whisper API contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .model import WhisperModel
from utils.fastapi_app import create_app


def _load_checkpoint_path() -> Optional[Path]:
    """Load checkpoint path from environment variable."""
    env_value = os.getenv("WHISPER_MODEL_CHECKPOINT_PATH")
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


def _get_device() -> str:
    """Get device from environment variable or default to cpu."""
    return os.getenv("WHISPER_DEVICE", "cpu")


def _get_dtype() -> str:
    """Get data type from environment variable or default to float32."""
    return os.getenv("WHISPER_DTYPE", "float32")


def _get_language() -> str:
    """Get language from environment variable or default to ru."""
    return os.getenv("WHISPER_LANGUAGE", "ru")


def _get_task() -> str:
    """Get task from environment variable or default to transcribe."""
    return os.getenv("WHISPER_TASK", "transcribe")


# Initialize model
model = WhisperModel(
    checkpoint_path=_load_checkpoint_path(),
    device=_get_device(),
    dtype=_get_dtype(),
    language=_get_language(),
    task=_get_task(),
)

# Create FastAPI app
app = create_app(model)

