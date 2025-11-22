"""Mock FastAPI backend implementing the Whisper API contract from docs/model_api.md."""

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


def _load_model_name() -> Optional[str]:
    return os.getenv("WHISPER_MODEL_NAME")


model = WhisperModel(weights_path=_load_weights_path(), model_name=_load_model_name())
app = create_app(model)
