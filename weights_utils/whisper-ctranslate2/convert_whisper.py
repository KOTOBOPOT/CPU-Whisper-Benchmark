#!/usr/bin/env python3
"""Convert Whisper model to CTranslate2 format with different quantization types.

This script downloads Whisper models from HuggingFace and converts them to
CTranslate2 format with float32, int8_float32, and int16 quantization for CPU inference.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_conversion(model_name: str, quantization: str, output_dir: Path, force: bool = False) -> None:
    """Run ct2-transformers-converter to convert the model.
    
    Args:
        model_name: HuggingFace model name (e.g., 'openai/whisper-tiny')
        quantization: Quantization type (float32, int8_float32, int16)
        output_dir: Output directory for converted model
        force: If True, overwrite existing directory
    """
    cmd = [
        "ct2-transformers-converter",
        "--model", model_name,
        "--quantization", quantization,
        "--output_dir", str(output_dir),
        "--copy_files", "tokenizer.json",
    ]
    
    # Add --force flag if directory exists or force is requested
    if force or output_dir.exists():
        if output_dir.exists():
            logger.warning(f"Output directory already exists: {output_dir}")
            logger.info("Using --force to overwrite existing directory")
        cmd.append("--force")
    
    logger.info(f"Converting {model_name} with {quantization} quantization...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Conversion successful! Output saved to: {output_dir}")
        if result.stdout:
            logger.debug(f"Stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed with exit code {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        raise
    except FileNotFoundError:
        logger.error("ct2-transformers-converter not found. Please install ctranslate2:")
        logger.error("  pip install ctranslate2")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper model to CTranslate2 format with quantization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace model name (default: openai/whisper-large-v3)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory for converted models"
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["float32", "int8_float32", "int16"],
        choices=["float32", "int8", "int8_float32", "int8_float16", "int16", "float16"],
        help="Quantization types to apply (default: float32 int8_float32 int16)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output directories"
    )
    
    args = parser.parse_args()
    
    # Create base output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir.absolute()}")
    
    # Extract model base name for directory naming
    model_base = args.model.split("/")[-1].replace("whisper-", "")
    
    # Convert for each quantization type
    for quant in args.quantizations:
        # Determine output subdirectory name
        if quant == "int8_float32":
            subdir_name = f"whisper-{model_base}-ct2-int8"
        elif quant == "int16":
            subdir_name = f"whisper-{model_base}-ct2-int16"
        elif quant == "float32":
            subdir_name = f"whisper-{model_base}-ct2-float32"
        else:
            subdir_name = f"whisper-{model_base}-ct2-{quant}"
        
        output_subdir = args.output_dir / subdir_name
        
        try:
            run_conversion(args.model, quant, output_subdir, force=args.force)
        except Exception as e:
            logger.error(f"Failed to convert with {quant} quantization: {e}")
            sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("All conversions completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Converted models are saved in: {args.output_dir.absolute()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Update .env file: LOCAL_WHISPER_WEIGHTS_DIR=<path_to_output_dir>")
    logger.info("2. Start backends with appropriate WHISPER_MODEL_WEIGHTS_PATH")
    logger.info(f"   Example for int8:")
    logger.info(f"   WHISPER_MODEL_WEIGHTS_PATH=/app/weights/whisper-{model_base}-ct2-int8 \\")
    logger.info("     ./backends/whisper-ctranslate2-int8/start.sh")


if __name__ == "__main__":
    main()

