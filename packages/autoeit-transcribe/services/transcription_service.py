from __future__ import annotations

from core.entities import BatchTranscriptionRequest, BatchTranscriptionResult
from src.asr.pipeline import run_batch


def run_transcription_batch(request: BatchTranscriptionRequest) -> BatchTranscriptionResult:
    transcripts = run_batch(
        audio_dir=request.audio_dir,
        prompt_workbook=request.prompt_xlsx,
        output_workbook=request.output_xlsx,
        expected_count=request.expected_count,
        model_size=request.model_size,
        device=request.device,
        compute_type=request.compute_type,
        language=request.language,
        only_sheet=request.only_sheet,
    )
    return BatchTranscriptionResult(
        processed_sheets=len(transcripts),
        output_xlsx=request.output_xlsx,
    )
