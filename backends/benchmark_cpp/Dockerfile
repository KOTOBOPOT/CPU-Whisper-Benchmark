# syntax=docker/dockerfile:1
# Multi-stage build for C++ Whisper backend service
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    wget \
    libsndfile1-dev \
    libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime (using 1.17.0 for API compatibility)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.0.tgz && \
    mv onnxruntime-linux-x64-1.17.0 /opt/onnxruntime && \
    ls -la /opt/onnxruntime/include/ && \
    rm onnxruntime-linux-x64-1.17.0.tgz

# Set ONNX Runtime path for build
ENV ONNX_RUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=$ONNX_RUNTIME_DIR/lib:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /build
COPY backends/benchmark_cpp/CMakeLists.txt ./
COPY backends/benchmark_cpp/include ./include
COPY backends/benchmark_cpp/src ./src

# Build the C++ binary
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    libjsoncpp25 \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (runtime only)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.0.tgz && \
    mv onnxruntime-linux-x64-1.17.0 /opt/onnxruntime && \
    rm onnxruntime-linux-x64-1.17.0.tgz

# Set ONNX Runtime path
ENV ONNX_RUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=$ONNX_RUNTIME_DIR/lib:$LD_LIBRARY_PATH

# Create app directory
WORKDIR /app

# Install Python dependencies FIRST (before copying ANY code or binaries)
# This layer will be cached as long as requirements.txt doesn't change
# Using BuildKit cache mount to speed up pip installs across builds
COPY backends/benchmark_cpp/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the built binary from builder stage AFTER pip install
# Changes to C++ code won't invalidate pip cache
COPY --from=builder /build/build/whisper_benchmark /app/whisper_benchmark

# Copy Python application files
COPY backends/benchmark_cpp/app ./app
COPY utils ./utils

# Create directory for models
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV WHISPER_MODEL_PATH=/app/models/whisper-base.onnx
ENV WHISPER_CPP_BINARY_PATH=/app/whisper_benchmark
ENV WHISPER_NUM_THREADS=4
ENV WHISPER_BACKEND_PORT=8005

# Expose port
EXPOSE ${WHISPER_BACKEND_PORT}

# Run the FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${WHISPER_BACKEND_PORT}"]