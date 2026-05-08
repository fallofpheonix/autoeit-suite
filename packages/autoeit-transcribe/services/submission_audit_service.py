from __future__ import annotations

from pathlib import Path

from core.errors import SubmissionValidationError


def ensure_submission_bundle_is_consistent(project_root: Path) -> None:
    submission_root = project_root / "submission"
    if not submission_root.exists():
        raise SubmissionValidationError("Missing submission folder.")

    must_exist = [
        submission_root / "README.md",
        submission_root / "run_submission.sh",
        submission_root / "output" / "AutoEIT_Task1_Transcriptions_submission.xlsx",
        submission_root / "notebooks" / "task1_transcription.ipynb",
        submission_root / "docs" / "README.md",
    ]
    missing = [path for path in must_exist if not path.exists()]
    if missing:
        raise SubmissionValidationError(
            "Submission bundle is incomplete: " + ", ".join(str(path) for path in missing)
        )
