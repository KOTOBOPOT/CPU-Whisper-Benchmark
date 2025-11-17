# syntax=docker/dockerfile:1
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backends/distillation/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backends/distillation/backend/app ./app
COPY utils ./utils

ENV PYTHONPATH=/app

ENV WHISPER_BACKEND_PORT=8001

EXPOSE ${WHISPER_BACKEND_PORT}

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${WHISPER_BACKEND_PORT}"]

