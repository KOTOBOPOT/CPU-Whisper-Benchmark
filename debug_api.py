"""Small helper script to send an audio file to the Whisper backend for debugging."""

from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests


def _get_backend_port() -> str:
    if (env_port := os.getenv("WHISPER_BACKEND_PORT")):
        return env_port

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            key, sep, value = line.partition("=")
            if sep and key.strip() == "WHISPER_BACKEND_PORT":
                return value.strip().strip("\"'")

    return "7590"


DEFAULT_PORT = _get_backend_port()
DEFAULT_URL = f"http://0.0.0.0:{DEFAULT_PORT}/process_audio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send audio file to Whisper backend debug API")
    parser.add_argument("audio_path", type=Path, help="Path to the audio file to send")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Endpoint URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    if not args.audio_path.is_file():
        raise SystemExit(f"Audio file not found: {args.audio_path}")

    mime_type, _ = mimetypes.guess_type(str(args.audio_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    with args.audio_path.open("rb") as audio_file:
        files = {"file": (args.audio_path.name, audio_file, mime_type)}
        try:
            response = requests.post(args.url, files=files, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:  # type: ignore[assignment]
            raise SystemExit(f"Request failed: {exc}") from exc

    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise SystemExit(f"Failed to decode JSON: {exc}\nRaw response: {response.text}") from exc

    print(payload)


if __name__ == "__main__":
    main()
