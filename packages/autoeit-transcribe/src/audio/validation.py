"""Audio file validation utilities."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma"}
)


def validate_audio_file(path: str | Path) -> Path:
    """Validate that *path* points to a supported audio file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{p.suffix}'.  "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p
