"""FastAPI backend for CTranslate2 Whisper with int16 quantization."""

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


model = WhisperModel(weights_path=_load_weights_path())
app = create_app(model)

