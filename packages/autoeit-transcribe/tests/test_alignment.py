"""Tests for segment-to-prompt alignment."""

from __future__ import annotations

import pytest

from src.align.alignment import align_to_prompts, AlignmentError, Segment, EXPECTED_PROMPT_COUNT


def _make_segments(texts: list[str]) -> list[Segment]:
    """Create evenly spaced Segment objects from a list of texts."""
    return [
        Segment(start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


class TestAlignToPrompts:
    EXPECTED = EXPECTED_PROMPT_COUNT  # 30

    def test_exact_count_passes_through(self):
        segs = _make_segments([f"sentence {i}" for i in range(self.EXPECTED)])
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert len(result) == self.EXPECTED

    def test_texts_preserved_for_exact_count(self):
        texts = [f"word{i}" for i in range(self.EXPECTED)]
        segs = _make_segments(texts)
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert result == texts

    def test_too_many_segments_merged(self):
        # 35 segments → should be merged down to 30
        segs = _make_segments([f"token {i}" for i in range(35)])
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert len(result) == self.EXPECTED

    def test_too_few_segments_split(self):
        # 28 segments, each with at least 6 tokens → should split up to 30
        texts = [" ".join(f"word{j}" for j in range(8)) for _ in range(28)]
        segs = _make_segments(texts)
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert len(result) == self.EXPECTED

    def test_unsplittable_too_few_raises(self):
        # 5 short segments with few tokens — cannot be split to 30
        segs = _make_segments(["a", "b", "c", "d", "e"])
        with pytest.raises(AlignmentError):
            align_to_prompts(segs, expected_count=self.EXPECTED)

    def test_empty_texts_filtered(self):
        # Segments with empty text should be ignored.
        segs = _make_segments([""] * 5 + [f"token {i}" for i in range(self.EXPECTED)])
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert len(result) == self.EXPECTED

    def test_returns_list_of_strings(self):
        segs = _make_segments([f"word {i}" for i in range(self.EXPECTED)])
        result = align_to_prompts(segs, expected_count=self.EXPECTED)
        assert all(isinstance(s, str) for s in result)

    def test_custom_expected_count(self):
        segs = _make_segments(["hello world foo"] * 10)
        result = align_to_prompts(segs, expected_count=10)
        assert len(result) == 10
