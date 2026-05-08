"""Workbook IO utilities for reading and writing Excel files."""

from __future__ import annotations

from pathlib import Path

import openpyxl


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory of *path* exists and return a :class:`Path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def last_populated_header_column(worksheet: openpyxl.worksheet.worksheet.Worksheet) -> int:
    """Return the 1-based index of the last non-empty cell in row 1.

    If row 1 is completely empty, returns 1.
    """
    last = 1
    for cell in worksheet[1]:
        if cell.value is not None:
            last = cell.column
    return last
