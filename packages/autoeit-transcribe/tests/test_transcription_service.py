from __future__ import annotations

from pathlib import Path

from core.entities import BatchTranscriptionRequest
from services.transcription_service import run_transcription_batch


def test_run_transcription_batch_single_sheet(tmp_path: Path) -> None:
    output_path = tmp_path / "single_sheet_output.xlsx"
    request = BatchTranscriptionRequest(
        audio_dir=Path("data/raw"),
        prompt_xlsx=Path("data/metadata/AutoEIT Sample Audio for Transcribing.xlsx"),
        output_xlsx=output_path,
        expected_count=30,
        model_size="models/faster-whisper-tiny",
        device="cpu",
        compute_type="int8",
        language="es",
        only_sheet="38010-2A",
    )

    result = run_transcription_batch(request)

    assert result.processed_sheets == 1
    assert result.output_xlsx == output_path
    assert output_path.exists()
