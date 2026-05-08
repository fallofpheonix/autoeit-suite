from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchTranscriptionRequest:
    audio_dir: Path
    prompt_xlsx: Path
    output_xlsx: Path
    expected_count: int
    model_size: str
    device: str
    compute_type: str
    language: str
    only_sheet: str | None = None


@dataclass(frozen=True)
class BatchTranscriptionResult:
    processed_sheets: int
    output_xlsx: Path
