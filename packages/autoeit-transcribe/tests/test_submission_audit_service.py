from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import SubmissionValidationError
from services.submission_audit_service import ensure_submission_bundle_is_consistent


def test_submission_audit_rejects_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(SubmissionValidationError):
        ensure_submission_bundle_is_consistent(tmp_path)


def test_submission_audit_accepts_minimum_bundle(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    (submission_dir / "output").mkdir(parents=True)
    (submission_dir / "notebooks").mkdir(parents=True)
    (submission_dir / "docs").mkdir(parents=True)

    (submission_dir / "README.md").write_text("ok", encoding="utf-8")
    (submission_dir / "run_submission.sh").write_text("#!/usr/bin/env bash", encoding="utf-8")
    (submission_dir / "docs" / "README.md").write_text("ok", encoding="utf-8")
    (submission_dir / "notebooks" / "task1_transcription.ipynb").write_text("{}", encoding="utf-8")
    (submission_dir / "output" / "AutoEIT_Task1_Transcriptions_submission.xlsx").write_text(
        "placeholder",
        encoding="utf-8",
    )

    ensure_submission_bundle_is_consistent(tmp_path)
