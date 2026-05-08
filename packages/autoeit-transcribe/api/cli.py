"""Primary command-line entrypoint for Task I transcription."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.runtime import request_from_cli_args
from services.transcription_service import run_transcription_batch
from src.asr.pipeline import EXPECTED_SENTENCE_COUNT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoEIT Task I transcription pipeline")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--prompt-xlsx", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SENTENCE_COUNT)
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="es")
    parser.add_argument("--sheet", default=None, help="Process a single participant sheet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = request_from_cli_args(args)
    result = run_transcription_batch(request)
    print(f"Processed {result.processed_sheets} participant sheet(s).")
    print(f"Output workbook: {result.output_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
