"""Tests for transcription normalization utilities."""

from __future__ import annotations

import pytest

from src.postprocess.normalization import collapse_whitespace, normalize_transcription_text
from src.postprocess.hallucination import is_hallucination, filter_hallucinations


class TestCollapseWhitespace:
    def test_multiple_spaces_collapsed(self):
        assert collapse_whitespace("hello   world") == "hello world"

    def test_leading_trailing_stripped(self):
        assert collapse_whitespace("  hello  ") == "hello"

    def test_newlines_collapsed(self):
        assert collapse_whitespace("hello\nworld") == "hello world"

    def test_tabs_collapsed(self):
        assert collapse_whitespace("a\t\tb") == "a b"

    def test_empty_string(self):
        assert collapse_whitespace("") == ""

    def test_only_whitespace(self):
        assert collapse_whitespace("   ") == ""


class TestNormalizeTranscriptionText:
    def test_lowercased(self):
        assert normalize_transcription_text("HOLA MUNDO") == "hola mundo"

    def test_punctuation_removed(self):
        assert normalize_transcription_text("¡Hola, mundo!") == "hola mundo"

    def test_accents_preserved(self):
        # Accented characters are valid learner output.
        result = normalize_transcription_text("mañana")
        assert "mañana" in result

    def test_empty_string(self):
        assert normalize_transcription_text("") == ""

    def test_whitespace_collapsed(self):
        result = normalize_transcription_text("  hola   mundo  ")
        assert result == "hola mundo"

    def test_learner_errors_preserved(self):
        # Misspellings must not be corrected.
        raw = "yo hablas español"
        result = normalize_transcription_text(raw)
        assert "hablas" in result

    def test_apostrophe_preserved(self):
        # Apostrophes in Spanish contractions should be kept.
        result = normalize_transcription_text("d'acuerdo")
        assert "'" in result


class TestIsHallucination:
    def test_known_hallucination_gracias(self):
        assert is_hallucination("Gracias por ver este video")

    def test_known_hallucination_subtitulado(self):
        assert is_hallucination("Subtitulado por el equipo de doblaje")

    def test_hasta_proxima(self):
        assert is_hallucination("Hasta la próxima")

    def test_normal_text_not_hallucination(self):
        assert not is_hallucination("El niño come una manzana")

    def test_empty_string_not_hallucination(self):
        assert not is_hallucination("")

    def test_case_insensitive(self):
        assert is_hallucination("GRACIAS POR VER ESTE VIDEO")


class TestFilterHallucinations:
    def test_hallucination_returns_empty(self):
        assert filter_hallucinations("Gracias por ver este video") == ""

    def test_normal_text_unchanged(self):
        text = "El gato está en el tejado"
        assert filter_hallucinations(text) == text
