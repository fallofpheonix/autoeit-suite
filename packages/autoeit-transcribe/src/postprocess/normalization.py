"""Text normalization utilities.

Normalization is applied to ASR output before comparison/export.
Learner errors, hesitations, and repetitions are intentionally preserved;
only machine-generated formatting artifacts are cleaned up.
"""

from __future__ import annotations

import re
import unicodedata

# Characters to strip from normalized text (punctuation, except apostrophes).
_STRIP_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_transcription_text(text: str) -> str:
    """Return a lightly normalized version of *text* suitable for downstream comparison.

    Transformations applied (in order):
    1. Unicode NFC normalization (combines diacritics).
    2. Lowercase conversion.
    3. Strip punctuation (except apostrophes used in contractions).
    4. Collapse runs of whitespace.

    Intentionally *not* applied:
    - Removal of disfluencies (uh, um, er…).
    - Correction of grammar or spelling errors.
    - Removal of repetitions.
    """
    if not text:
        return ""
    # NFC so that accented Spanish characters are handled consistently.
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _STRIP_PUNCTUATION.sub("", text)
    return collapse_whitespace(text)
