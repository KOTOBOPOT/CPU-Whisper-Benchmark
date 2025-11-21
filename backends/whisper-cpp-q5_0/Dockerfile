# syntax=docker/dockerfile:1
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for librosa, audio processing, and building whisper.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backends/whisper-cpp-q5_0/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Build whisper.cpp
# Копируем whisper.cpp в контейнер (исключаем build директорию)
COPY whisper.cpp /tmp/whisper.cpp

# Компилируем whisper.cpp (очищаем build если есть)
RUN cd /tmp/whisper.cpp && \
    rm -rf build && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/whisper-cli /usr/local/bin/whisper-cli && \
    chmod +x /usr/local/bin/whisper-cli && \
    # Копируем библиотеки whisper
    cp build/src/libwhisper.so* /usr/local/lib/ && \
    # Копируем все библиотеки GGML (libggml, libggml-base, libggml-cpu)
    cp build/ggml/src/*.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/whisper.cpp

# Copy application
COPY backends/whisper-cpp-q5_0/app ./app
COPY utils ./utils

ENV PYTHONPATH=/app
ENV WHISPER_BACKEND_PORT=8000
ENV WHISPER_WORKERS=4
ENV WHISPER_CPP_BIN=/usr/local/bin/whisper-cli
ENV WHISPER_CPP_THREADS=12

EXPOSE ${WHISPER_BACKEND_PORT}

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${WHISPER_BACKEND_PORT} --workers ${WHISPER_WORKERS}"]
