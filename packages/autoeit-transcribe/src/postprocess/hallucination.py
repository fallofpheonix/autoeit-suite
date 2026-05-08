"""Hallucination detection and filtering for ASR output.

Whisper (and faster-whisper) occasionally emits boilerplate phrases that are
not present in the learner audio — so-called *hallucinations*.  These patterns
are specific to the Spanish-language model and the EIT context.
"""

from __future__ import annotations

import re

# Compiled patterns that match common Whisper hallucination phrases.
# All matches are performed on the stripped, case-folded text.
HALLUCINATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^gracias por ver.*$", re.IGNORECASE),
    re.compile(r"^subtitulado por.*$", re.IGNORECASE),
    re.compile(r"^hasta la pr[oó]xima.*$", re.IGNORECASE),
    re.compile(r"^suscr[ií]bete.*$", re.IGNORECASE),
    re.compile(r"^www\..*$", re.IGNORECASE),
    re.compile(r"^\[música\]$", re.IGNORECASE),
    re.compile(r"^\[aplausos\]$", re.IGNORECASE),
    re.compile(r"^\[risas\]$", re.IGNORECASE),
    re.compile(r"^subtítulos.*$", re.IGNORECASE),
]


def is_hallucination(text: str) -> bool:
    """Return *True* if *text* matches a known ASR hallucination pattern."""
    stripped = text.strip()
    return any(pattern.fullmatch(stripped) for pattern in HALLUCINATION_PATTERNS)


def filter_hallucinations(text: str) -> str:
    """Return *text* unchanged unless it is a hallucination, in which case return ``''``."""
    return "" if is_hallucination(text) else text
