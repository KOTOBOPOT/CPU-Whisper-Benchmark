# Models Directory

This directory should contain the ONNX format Whisper model files.

## Getting a Model

### Option 1: Convert from Hugging Face

Use the provided conversion script:

```bash
cd ..
python convert_to_onnx.py --model openai/whisper-base --output models/model.onnx
```

Available models:
- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large`
- `openai/whisper-large-v2`
- `openai/whisper-large-v3`

### Option 2: Download Pre-converted

If you have a pre-converted ONNX model, place it in this directory and name it `model.onnx`.

## Model Format

The model should be in ONNX format with the following characteristics:
- Input: Audio features (mel spectrogram)
- Output: Token IDs
- Compatible with ONNX Runtime 1.16.0+

## File Structure

```
models/
├── README.md (this file)
└── model.onnx (your Whisper ONNX model)