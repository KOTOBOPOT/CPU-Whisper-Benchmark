"""FastAPI backend for Whisper.cpp with GGML quantization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .model import WhisperModel
from utils.fastapi_app import create_app


def _load_weights_path() -> Optional[Path]:
    env_value = os.getenv("WHISPER_MODEL_WEIGHTS_PATH")
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


def _load_whisper_cpp_bin() -> Optional[Path]:
    env_value = os.getenv("WHISPER_CPP_BIN")
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


model = WhisperModel(
    weights_path=_load_weights_path(),
    whisper_cpp_bin=_load_whisper_cpp_bin()
)
app = create_app(model)
