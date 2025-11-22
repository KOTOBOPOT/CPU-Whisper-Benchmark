"""
Скрипт для конвертации Whisper модели в ONNX формат используя optimum
"""
import argparse
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor


def convert_whisper_to_onnx_optimum(model_id: str, output_path: str):
    """
    Конвертирует Whisper модель в ONNX используя optimum (рекомендуемый способ).
    
    Args:
        model_id: HuggingFace model ID (e.g., "openai/whisper-base")
        output_path: Путь для сохранения ONNX модели (директория)
    """
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading and converting model: {model_id}")
    print(f"Output directory: {output_path}")
    
    # Load and export model
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        export=True,
        provider="CPUExecutionProvider"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Save to disk
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    print(f"\n✓ Model successfully exported to: {output_path}")
    
    # Show created files
    print(f"\nCreated files:")
    for file in output_path.glob("*"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")


def convert_whisper_simple(model_id: str, output_file: str):
    """
    Простая конвертация encoder в ONNX (для базовых применений).
    
    Args:
        model_id: HuggingFace model ID  
        output_file: Путь для сохранения ONNX файла
    """
    print(f"Loading model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Generate dummy input
    sample_rate = 16000
    duration = 30
    num_samples = sample_rate * duration
    dummy_audio = torch.randn(num_samples)
    inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
    
    print(f"Input shape: {inputs['input_features'].shape}")
    print(f"Exporting encoder to ONNX: {output_file}")
    
    # Export encoder only (simple approach)
    torch.onnx.export(
        model.model.encoder,
        inputs['input_features'],
        output_file,
        input_names=['input_features'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_features': {0: 'batch', 2: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"✓ Encoder exported to: {output_file}")
    model_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✓ Model size: {model_size:.2f} MB")
    print("\nNOTE: This exports only the encoder. For full transcription,")
    print("use --use-optimum flag for complete model export.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: Full model export with optimum
  python convert_to_onnx.py --model-id openai/whisper-base --output models/ --use-optimum
  
  # Simple: Encoder only (for basic use)
  python convert_to_onnx.py --model-id openai/whisper-base --output models/whisper-base.onnx
        """
    )
    parser.add_argument(
        "--model-id",
        default="openai/whisper-base",
        help="HuggingFace model ID (default: openai/whisper-base)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (directory for --use-optimum, file otherwise)"
    )
    parser.add_argument(
        "--use-optimum",
        action="store_true",
        help="Use optimum library for full model export (recommended)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.use_optimum:
            convert_whisper_to_onnx_optimum(args.model_id, args.output)
        else:
            convert_whisper_simple(args.model_id, args.output)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
