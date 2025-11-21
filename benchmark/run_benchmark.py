#!/usr/bin/env python3
"""Whisper benchmark runner.

Sequentially sends audio files from the selected dataset to a running backend
and collects latency and accuracy metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from tqdm import tqdm

try:
    import requests
except ImportError as exc:  # pragma: no cover - early, deterministic failure
    raise SystemExit(
        "The benchmark requires the 'requests' package. Install it with 'pip install requests'."
    ) from exc


ANNOTATION_FILENAME = "annotation.csv"
FILES_SUBDIR = "files"
DEFAULT_ENDPOINT = "/process_audio"


@dataclass
class SampleResult:
    filename: str
    reference: str
    hypothesis: str
    latency_ms: float
    status: str
    error: Optional[str] = None


@dataclass
class RunningStats:
    total_latency_ms: float = 0.0
    processed: int = 0
    failed: int = 0
    wer_distance: int = 0
    wer_ref_words: int = 0
    cer_distance: int = 0
    cer_ref_chars: int = 0
    total_wall_time_ms: float = 0.0  # Wall-clock time for throughput calculation

    def update_success(
        self,
        latency_ms: float,
        wer_distance: int,
        ref_words: int,
        cer_distance: int,
        ref_chars: int,
    ) -> None:
        self.total_latency_ms += latency_ms
        self.processed += 1
        self.wer_distance += wer_distance
        self.wer_ref_words += ref_words
        self.cer_distance += cer_distance
        self.cer_ref_chars += ref_chars

    def update_failure(self) -> None:
        self.failed += 1

    def average_latency(self) -> Optional[float]:
        if self.processed == 0:
            return None
        return self.total_latency_ms / self.processed

    def throughput(self) -> Optional[float]:
        """Calculate throughput in requests per second."""
        if self.total_wall_time_ms == 0:
            return None
        return (self.processed + self.failed) / (self.total_wall_time_ms / 1000.0)

    def wer(self) -> Optional[float]:
        if self.wer_ref_words == 0:
            return None
        return self.wer_distance / self.wer_ref_words

    def cer(self) -> Optional[float]:
        if self.cer_ref_chars == 0:
            return None
        return self.cer_distance / self.cer_ref_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper benchmark against deployed backend.")
    parser.add_argument("--bench-name", required=True, help="Dataset/benchmark identifier (BENCH_NAME).")
    parser.add_argument(
        "--backend-host",
        default="http://127.0.0.1",
        help="URL or hostname of the backend (scheme optional).",
    )
    parser.add_argument("--backend-port", type=int, required=True, help="Backend port (WHISPER_BACKEND_PORT).")
    parser.add_argument("--backend-name", default="unknown", help="Backend name used in results path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where benchmark artefacts will be stored.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Root directory with benchmark datasets.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Relative path of transcription endpoint (default: /transcribe).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Timeout for a single request in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between requests (for rate limiting).",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra HTTP header to add to each request (can be passed multiple times).",
    )
    parser.add_argument(
        "--payload-key",
        default="file",
        help="Multipart field name expected by the backend for audio payload.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Key in backend JSON response that contains transcription text.",
    )
    parser.add_argument(
        "--allow-blank-text",
        action="store_true",
        help="If set, blank hypotheses are treated as valid (WER counts them).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for concurrent requests (default: 1 = sequential).",
    )
    return parser.parse_args()


def build_base_url(host: str, port: int) -> str:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    netloc = parsed.netloc or parsed.path
    if not netloc:
        raise ValueError(f"Could not parse backend host '{host}'.")
    # If port already specified, keep it.
    if ":" not in netloc.split("@")[ -1 ]:
        netloc = f"{netloc}:{port}"
    return urlunparse((parsed.scheme or "http", netloc, "", "", "", ""))


def ensure_dataset_paths(data_root: Path, bench_name: str) -> Tuple[Path, Path]:
    bench_dir = data_root / bench_name
    if not bench_dir.is_dir():
        raise FileNotFoundError(f"Benchmark dataset '{bench_name}' not found at {bench_dir}")
    annotation = bench_dir / ANNOTATION_FILENAME
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation file missing: {annotation}")
    files_dir = bench_dir / FILES_SUBDIR
    if not files_dir.is_dir():
        raise FileNotFoundError(f"Audio files directory missing: {files_dir}")
    return annotation, files_dir


def parse_headers(raw_headers: Iterable[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for item in raw_headers:
        if "=" not in item:
            raise ValueError(f"Invalid header format '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def load_annotation(annotation_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "filename" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(
                f"Annotation file must contain 'filename' and 'text' columns. Columns found: {reader.fieldnames}"
            )
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename:
                continue
            rows.append((filename, text))
    return rows


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def edit_distance(reference: List[str], hypothesis: List[str]) -> int:
    # Classic dynamic programming (Levenshtein distance).
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ref_token = reference[i - 1]
        for j in range(1, n + 1):
            hyp_token = hypothesis[j - 1]
            substitution_cost = 0 if ref_token == hyp_token else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + substitution_cost,  # substitution
            )
    return dp[m][n]


def compute_distances(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()
    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)
    return (
        edit_distance(ref_words, hyp_words),
        len(ref_words),
        edit_distance(ref_chars, hyp_chars),
        len(ref_chars),
    )


def extract_text(payload: object, key: str) -> str:
    if isinstance(payload, dict):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
        nested = payload.get("result") if isinstance(payload.get("result"), dict) else None
        if nested and key in nested and isinstance(nested[key], str):
            return nested[key]
    raise ValueError(f"Could not extract transcription text with key '{key}'. Response: {payload}")


def process_single_request(
    filename: str,
    reference: str,
    files_dir: Path,
    url: str,
    headers: Dict[str, str],
    payload_key: str,
    text_key: str,
    allow_blank_text: bool,
    request_timeout: float,
    session: requests.Session,
) -> SampleResult:
    """Process a single audio file request.
    
    Returns:
        SampleResult with transcription and metrics
    """
    audio_path = files_dir / filename
    if not audio_path.is_file():
        return SampleResult(
            filename=filename,
            reference=reference,
            hypothesis="",
            latency_ms=0.0,
            status="missing_file",
            error=f"Audio file not found: {audio_path}",
        )

    try:
        with audio_path.open("rb") as file_handle:
            mime_type, _ = mimetypes.guess_type(str(audio_path))
            if mime_type is None:
                mime_type = "application/octet-stream"
            files = {payload_key: (filename, file_handle, mime_type)}
            start_ts = time.perf_counter()
            response = session.post(url, files=files, headers=headers, timeout=request_timeout)
            latency_ms = (time.perf_counter() - start_ts) * 1000
            response.raise_for_status()
            payload = response.json()
            hypothesis = extract_text(payload, text_key)
            if not allow_blank_text and not hypothesis:
                raise ValueError("Backend returned empty transcription text.")
            wer_dist, ref_words, cer_dist, ref_chars = compute_distances(reference, hypothesis)
            return SampleResult(
                filename=filename,
                reference=reference,
                hypothesis=hypothesis,
                latency_ms=latency_ms,
                status="ok",
            )
    except Exception as error:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start_ts) if 'start_ts' in locals() else 0.0
        return SampleResult(
            filename=filename,
            reference=reference,
            hypothesis="",
            latency_ms=latency_ms * 1000,
            status="error",
            error=str(error),
        )


def run(args: argparse.Namespace) -> None:
    annotation_path, files_dir = ensure_dataset_paths(args.data_root, args.bench_name)
    items = load_annotation(annotation_path)
    if args.max_samples is not None:
        items = items[: max(args.max_samples, 0)]
    if not items:
        raise ValueError("No samples to process. Check annotation file or max-samples filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = build_base_url(args.backend_host, args.backend_port)
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    headers = parse_headers(args.header)

    session = requests.Session()
    stats = RunningStats()
    predictions: List[SampleResult] = []

    # Start wall-clock timer for throughput calculation
    wall_start = time.perf_counter()

    if args.workers == 1:
        # Sequential processing (original behavior)
        for filename, reference in tqdm(items, desc="Processing"):
            result = process_single_request(
                filename, reference, files_dir, url, headers,
                args.payload_key, args.text_key, args.allow_blank_text,
                args.request_timeout, session
            )
            predictions.append(result)
            
            if result.status == "ok":
                wer_dist, ref_words, cer_dist, ref_chars = compute_distances(result.reference, result.hypothesis)
                stats.update_success(result.latency_ms, wer_dist, ref_words, cer_dist, ref_chars)
            else:
                stats.update_failure()
            
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        # Parallel processing with ThreadPoolExecutor
        print(f"Processing with {args.workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    process_single_request,
                    filename, reference, files_dir, url, headers,
                    args.payload_key, args.text_key, args.allow_blank_text,
                    args.request_timeout, session
                ): (filename, reference)
                for filename, reference in items
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_item), total=len(items), desc="Processing"):
                result = future.result()
                predictions.append(result)
                
                if result.status == "ok":
                    wer_dist, ref_words, cer_dist, ref_chars = compute_distances(result.reference, result.hypothesis)
                    stats.update_success(result.latency_ms, wer_dist, ref_words, cer_dist, ref_chars)
                else:
                    stats.update_failure()

    # Calculate total wall-clock time
    wall_end = time.perf_counter()
    stats.total_wall_time_ms = (wall_end - wall_start) * 1000

    write_outputs(args, predictions, stats, len(items))


def write_outputs(
    args: argparse.Namespace,
    predictions: List[SampleResult],
    stats: RunningStats,
    total_requested: int,
) -> None:
    predictions_path = args.output_dir / "predictions.csv"
    metrics_path = args.output_dir / "metrics.json"
    summary_path = args.output_dir / "summary.txt"

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["filename", "reference", "hypothesis", "latency_ms", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in predictions:
            writer.writerow(
                {
                    "filename": item.filename,
                    "reference": item.reference,
                    "hypothesis": item.hypothesis,
                    "latency_ms": f"{item.latency_ms:.2f}",
                    "status": item.status,
                    "error": item.error or "",
                }
            )

    metrics_payload = {
        "bench_name": args.bench_name,
        "backend_name": args.backend_name,
        "requested_samples": total_requested,
        "processed_samples": stats.processed,
        "failed_samples": stats.failed,
        "average_latency_ms": stats.average_latency(),
        "throughput_rps": stats.throughput(),
        "total_wall_time_ms": stats.total_wall_time_ms,
        "wer": stats.wer(),
        "cer": stats.cer(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)

    lines = [
        f"Benchmark: {args.bench_name}",
        f"Backend: {args.backend_name}",
        f"Endpoint: {args.endpoint}",
        f"Samples requested: {total_requested}",
        f"Samples processed: {stats.processed}",
        f"Samples failed: {stats.failed}",
        f"Average latency (ms): {stats.average_latency():.2f}" if stats.average_latency() is not None else "Average latency (ms): n/a",
        f"Throughput (req/s): {stats.throughput():.2f}" if stats.throughput() is not None else "Throughput (req/s): n/a",
        f"Total wall time (s): {stats.total_wall_time_ms / 1000:.2f}",
        f"WER: {stats.wer():.4f}" if stats.wer() is not None else "WER: n/a",
        f"CER: {stats.cer():.4f}" if stats.cer() is not None else "CER: n/a",
    ]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - present actionable message
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)
