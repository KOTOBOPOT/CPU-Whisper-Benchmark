# C++ ONNX Runtime Whisper Backend

A high-performance Whisper ASR backend implementation using C++ and ONNX Runtime, integrated with the whisper-transformers benchmark infrastructure.

## Overview

This backend provides a C++ implementation of Whisper ASR using ONNX Runtime, wrapped in a FastAPI service for compatibility with the benchmark runner. It uses a persistent C++ process for optimal performance, eliminating the overhead of repeated model loading.

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Benchmark Runner   │────▶│  FastAPI Service │────▶│  C++ Process    │
│                     │ HTTP │  (Python)        │ IPC │  (Persistent)   │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
                                      │                        │
                                      ▼                        ▼
                                 Audio Files              ONNX Model
```

## Features

- **High Performance**: Native C++ implementation with ONNX Runtime
- **Persistent Process**: Model loaded once, reused for all requests
- **JSON IPC**: Efficient communication between Python and C++ 
- **Thread-Safe**: Supports concurrent requests with request ID tracking
- **Docker Integration**: Multi-stage build for easy deployment

## Quick Start

### Using Docker (Recommended)

1. Build and start the service:
```bash
docker-compose up --build
```

2. The service will be available at `http://localhost:8005`

### Manual Setup

1. Build the C++ binary:
```bash
./build.sh
```

2. Convert Whisper model to ONNX:
```bash
python convert_to_onnx.py --model-size base --output-dir models/
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the service:
```bash
./start.sh
```

## Integration with Benchmark Runner

The backend is fully integrated with the benchmark infrastructure:

```bash
# Set environment variables
export WHISPER_BACKEND_NAME=benchmark_cpp
export WHISPER_BACKEND_PORT=8005

# Run benchmarks
cd ../..
./benchmark/start.sh
```

## Configuration

Environment variables:
- `WHISPER_MODEL_PATH`: Path to ONNX model (default: `/app/models/whisper-base.onnx`)
- `WHISPER_CPP_BINARY_PATH`: Path to C++ binary (default: `/app/whisper_benchmark`)
- `WHISPER_NUM_THREADS`: Number of threads for ONNX Runtime (default: `4`)
- `WHISPER_BACKEND_PORT`: Service port (default: `8005`)

## Performance

The persistent mode eliminates:
- Model loading time (~500ms per request)
- ONNX Runtime initialization (~100ms per request)
- Process creation overhead (~50ms per request)

Total savings: **~650ms per request** compared to subprocess mode.

## API Endpoints

- `GET /health`: Health check
- `POST /process_audio`: Transcribe audio file (multipart/form-data)

## Development

### Project Structure
```
benchmark_cpp/
├── app/                    # FastAPI service
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   └── model.py           # Persistent C++ wrapper
├── src/                   # C++ source code
│   ├── main.cpp
│   ├── audio_utils.cpp
│   ├── metrics.cpp
│   └── onnx_model.cpp
├── include/               # C++ headers
├── models/                # ONNX models directory
├── docker-compose.yml     # Docker service config
├── Dockerfile            # Multi-stage build
├── requirements.txt      # Python dependencies
└── start.sh             # Service startup script
```

### Building from Source

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y cmake g++ libsndfile1-dev

# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Build
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime
make -j$(nproc)
```

## Testing

Test the service endpoint:
```bash
curl -X POST -F "audio=@test.wav" http://localhost:8005/process_audio
```

## Troubleshooting

1. **Model not found**: Ensure ONNX model exists at the path specified in `WHISPER_MODEL_PATH`
2. **Binary not found**: Check that C++ binary is built and located at `WHISPER_CPP_BINARY_PATH`
3. **Port conflict**: Change `WHISPER_BACKEND_PORT` if 8005 is already in use

## License

This project follows the same license as the parent whisper-transformers repository.
