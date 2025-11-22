"""FastAPI backend for C++ ONNX Runtime Whisper implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .model import WhisperCppPersistentModel
from utils.fastapi_app import create_app


def _load_model_path() -> Path:
    """Load model path from environment variable."""
    model_path = os.getenv("WHISPER_MODEL_PATH", "/app/models/whisper-base.onnx")
    return Path(model_path).expanduser().resolve()


def _load_binary_path() -> Path:
    """Load C++ binary path from environment variable."""
    binary_path = os.getenv("WHISPER_CPP_BINARY_PATH", "/app/whisper_benchmark")
    return Path(binary_path).expanduser().resolve()


def _load_num_threads() -> int:
    """Load number of threads from environment variable."""
    return int(os.getenv("WHISPER_NUM_THREADS", "4"))


# Initialize the persistent model
model = WhisperCppPersistentModel(
    model_path=str(_load_model_path()),
    binary_path=str(_load_binary_path()),
    num_threads=_load_num_threads()
)

# Create FastAPI app using the standard utility
app = create_app(model)