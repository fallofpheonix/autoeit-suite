"""Deterministic alignment of ASR segments to a fixed number of prompts.

The aligner maps an arbitrary number of ASR-produced segments onto exactly
*expected_count* (default 30) slots using two strategies:

Merge strategy (too many segments):
    Greedily merge the pair of adjacent segments separated by the smallest
    silence gap until the count reaches *expected_count*.

Split strategy (too few segments):
    Greedily split the longest segment (by token count) at its midpoint until
    the count reaches *expected_count* or no further splits are possible.

Both strategies are deterministic given the same input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EXPECTED_PROMPT_COUNT: int = 30

# Minimum number of tokens a segment must have for it to be eligible for
# splitting during the split phase.  Segments shorter than this are never
# split because the resulting halves would be too short to be meaningful.
MIN_TOKENS_FOR_SPLIT: int = 6


class AlignmentError(ValueError):
    """Raised when segments cannot be aligned to *expected_count* items."""


@dataclass(frozen=True)
class Segment:
    """A single ASR segment with timing information."""

    start: float
    end: float
    text: str


def _merge_key(left: Segment, right: Segment, index: int) -> tuple[float, int, int]:
    """Sorting key that favours merging segments with the smallest gap."""
    gap = max(0.0, right.start - left.end)
    token_count = len(left.text.split()) + len(right.text.split())
    return (gap, token_count, index)


def align_to_prompts(
    segments: Iterable[Segment],
    *,
    expected_count: int = EXPECTED_PROMPT_COUNT,
    cleanup_fn=None,
) -> list[str]:
    """Map *segments* onto exactly *expected_count* text strings.

    Parameters
    ----------
    segments:
        Iterable of :class:`Segment` objects produced by the ASR engine.
    expected_count:
        Number of prompts to align to.  The pipeline enforces exactly 30.
    cleanup_fn:
        Optional callable ``(str) -> str`` applied to each text value before
        processing.  Defaults to ``str.strip``.

    Returns
    -------
    list[str]
        Exactly *expected_count* transcription strings, one per prompt slot.

    Raises
    ------
    AlignmentError
        If the segments cannot be aligned even after applying split/merge.
    """
    if cleanup_fn is None:
        cleanup_fn = str.strip

    groups = [s for s in segments if cleanup_fn(s.text)]

    # ── Split phase: too few segments ─────────────────────────────────────────
    while len(groups) < expected_count:
        candidates = []
        for idx, seg in enumerate(groups):
            tokens = seg.text.split()
            if len(tokens) < MIN_TOKENS_FOR_SPLIT:
                continue
            mid = len(tokens) // 2
            left_text = cleanup_fn(" ".join(tokens[:mid]))
            right_text = cleanup_fn(" ".join(tokens[mid:]))
            # If either side becomes empty after cleanup, skip this segment as
            # a split candidate but continue considering other segments.
            if not left_text or not right_text:
                continue
            candidates.append(
                (
                    idx,
                    seg,
                    left_text,
                    right_text,
                    seg.end - seg.start,
                    len(tokens),
                )
            )
        if not candidates:
            break
        (
            split_idx,
            seg,
            left_text,
            right_text,
            _duration,
            _token_count,
        ) = max(
            candidates,
            key=lambda item: (item[4], item[5]),
        )
        midpoint_time = seg.start + (seg.end - seg.start) / 2.0
        groups[split_idx : split_idx + 1] = [
            Segment(start=seg.start, end=midpoint_time, text=left_text),
            Segment(start=midpoint_time, end=seg.end, text=right_text),
        ]

    if len(groups) < expected_count:
        raise AlignmentError(
            f"ASR produced {len(groups)} non-empty segment(s) after fallback splitting; "
            f"expected {expected_count}."
        )

    # ── Merge phase: too many segments ────────────────────────────────────────
    while len(groups) > expected_count:
        merge_idx = min(
            range(len(groups) - 1),
            key=lambda i: _merge_key(groups[i], groups[i + 1], i),
        )
        left = groups[merge_idx]
        right = groups[merge_idx + 1]
        merged_text = cleanup_fn(f"{left.text} {right.text}")
        groups[merge_idx : merge_idx + 2] = [
            Segment(start=left.start, end=right.end, text=merged_text)
        ]

    return [cleanup_fn(g.text) for g in groups]
