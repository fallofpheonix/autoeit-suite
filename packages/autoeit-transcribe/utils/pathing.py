from __future__ import annotations

from pathlib import Path


def repo_root_from_file(anchor_file: str | Path) -> Path:
    return Path(anchor_file).resolve().parent
