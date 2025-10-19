"""Reusable FastAPI application factory for the Whisper mock backend."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette import status

SUPPORTED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/opus",
}


class TranscriptionModel(Protocol):
    """Protocol describing the interface required by the FastAPI app."""

    def transcribe(self, *, audio_bytes: bytes, filename: str) -> str:
        ...


def create_app(model: TranscriptionModel) -> FastAPI:
    """Create a FastAPI app instance configured for Whisper transcription."""

    app = FastAPI(title="Mock Whisper Backend", version="0.1.0")

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

    return app
