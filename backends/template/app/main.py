"""Mock FastAPI backend implementing the Whisper API contract from docs/model_api.md."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette import status

from .model import WhisperModel

SUPPORTED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/opus",
}

app = FastAPI(title="Mock Whisper Backend", version="0.1.0")


def _load_weights_path() -> Optional[Path]:
    env_value = os.getenv("WHISPER_MODEL_WEIGHTS_PATH")
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


@app.on_event("startup")
async def startup_event() -> None:
    global model
    model = WhisperModel(weights_path=_load_weights_path())


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)) -> dict[str, str]:
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio MIME type.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file received.",
        )

    transcript = model.transcribe(audio_bytes=audio_bytes, filename=file.filename)
    return {"text": transcript}
