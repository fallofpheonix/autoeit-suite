"""Tests for WER/CER evaluation metrics."""

from __future__ import annotations

import pytest

from src.eval.metrics import compute_wer, compute_cer, evaluate, report_disagreements


class TestComputeWer:
    def test_identical_strings_wer_zero(self):
        assert compute_wer("hola mundo", "hola mundo") == pytest.approx(0.0)

    def test_completely_different_wer_one(self):
        wer = compute_wer("hola mundo", "foo bar")
        assert wer == pytest.approx(1.0)

    def test_one_substitution(self):
        # "hola mundo" → "hola mucho": 1 sub out of 2 words = 0.5
        wer = compute_wer("hola mundo", "hola mucho")
        assert wer == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert compute_wer("HOLA MUNDO", "hola mundo") == pytest.approx(0.0)


class TestComputeCer:
    def test_identical_strings_cer_zero(self):
        assert compute_cer("abc", "abc") == pytest.approx(0.0)

    def test_completely_different_high_cer(self):
        cer = compute_cer("abc", "xyz")
        assert cer > 0.0

    def test_case_insensitive(self):
        assert compute_cer("ABC", "abc") == pytest.approx(0.0)


class TestEvaluate:
    def test_returns_eval_result(self):
        result = evaluate("hola mundo", "hola mundo")
        assert result.wer == pytest.approx(0.0)
        assert result.cer == pytest.approx(0.0)
        assert result.reference == "hola mundo"
        assert result.hypothesis == "hola mundo"

    def test_imperfect_transcription(self):
        result = evaluate("yo como manzanas", "yo tomo manzanas")
        assert result.wer > 0.0


class TestReportDisagreements:
    def test_no_disagreements_when_perfect(self):
        refs = ["hola mundo"] * 5
        hyps = ["hola mundo"] * 5
        result = report_disagreements(refs, hyps)
        assert result == []

    def test_flags_high_wer_items(self):
        refs = ["correct sentence here", "another good one"]
        hyps = ["completely wrong output", "another good one"]
        disagreements = report_disagreements(refs, hyps, wer_threshold=0.5)
        # First item should be flagged.
        assert len(disagreements) >= 1
        assert disagreements[0][0] == 1  # 1-based index

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            report_disagreements(["a", "b"], ["c"])

    def test_custom_threshold(self):
        refs = ["hola mundo"]
        hyps = ["hola mundo"]
        # With threshold=0.0, any non-zero WER would be flagged; WER=0 here.
        result = report_disagreements(refs, hyps, wer_threshold=0.0)
        assert result == []
