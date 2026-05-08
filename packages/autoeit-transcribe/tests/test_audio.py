"""Tests for audio segmentation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.preprocessing import load_audio, normalize_loudness, TARGET_SAMPLE_RATE
from src.audio.segmentation import detect_silence_gaps, AudioSegment
from src.audio.validation import validate_audio_file, SUPPORTED_EXTENSIONS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_waveform(
    sr: int,
    *,
    speech_regions: list[tuple[float, float]],
    total_duration: float,
    amplitude: float = 0.5,
    silence_amplitude: float = 0.001,
) -> np.ndarray:
    """Construct a synthetic waveform with speech and silence regions."""
    total_samples = int(sr * total_duration)
    waveform = np.full(total_samples, silence_amplitude, dtype=np.float32)
    for start_sec, end_sec in speech_regions:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        waveform[start_sample:end_sample] = amplitude
    return waveform


# ── normalize_loudness ────────────────────────────────────────────────────────

class TestNormalizeLoudness:
    def test_peaks_at_one(self):
        data = np.array([0.2, -0.4, 0.1], dtype=np.float32)
        result = normalize_loudness(data)
        assert np.isclose(np.abs(result).max(), 1.0)

    def test_silent_signal_unchanged(self):
        data = np.zeros(100, dtype=np.float32)
        result = normalize_loudness(data)
        assert np.array_equal(result, data)

    def test_preserves_sign(self):
        data = np.array([-0.5, 0.5], dtype=np.float32)
        result = normalize_loudness(data)
        assert result[0] < 0
        assert result[1] > 0


# ── detect_silence_gaps ───────────────────────────────────────────────────────

class TestDetectSilenceGaps:
    SR = 16_000

    def test_single_speech_region(self):
        waveform = _make_waveform(
            self.SR,
            speech_regions=[(1.0, 2.0)],
            total_duration=3.0,
        )
        gaps = detect_silence_gaps(waveform, self.SR)
        assert len(gaps) == 1
        start, end = gaps[0]
        assert start < 1.5
        assert end > 1.5

    def test_two_speech_regions_separated_by_silence(self):
        waveform = _make_waveform(
            self.SR,
            speech_regions=[(0.5, 1.5), (3.0, 4.0)],
            total_duration=5.0,
        )
        gaps = detect_silence_gaps(waveform, self.SR, min_silence_ms=400.0)
        assert len(gaps) == 2

    def test_silent_waveform_returns_empty(self):
        waveform = np.zeros(self.SR * 2, dtype=np.float32)
        gaps = detect_silence_gaps(waveform, self.SR)
        assert gaps == []

    def test_segment_timestamps_are_ordered(self):
        waveform = _make_waveform(
            self.SR,
            speech_regions=[(0.3, 0.8), (1.5, 2.0), (3.0, 3.5)],
            total_duration=4.0,
        )
        gaps = detect_silence_gaps(waveform, self.SR, min_silence_ms=400.0)
        starts = [g[0] for g in gaps]
        assert starts == sorted(starts)


# ── validate_audio_file ───────────────────────────────────────────────────────

class TestValidateAudioFile:
    def test_existing_mp3(self, tmp_path):
        # Write synthetic (non-audio) bytes — this tests file existence and
        # extension validation only, not audio decodability.
        f = tmp_path / "test.mp3"
        f.write_bytes(b"\x00" * 10)
        result = validate_audio_file(f)
        assert result == f

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_audio_file(tmp_path / "missing.wav")

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "audio.xyz"
        f.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="Unsupported"):
            validate_audio_file(f)

    def test_all_supported_extensions_accepted(self, tmp_path):
        # Synthetic files — tests extension validation only.
        for ext in SUPPORTED_EXTENSIONS:
            f = tmp_path / f"file{ext}"
            f.write_bytes(b"\x00")
            validate_audio_file(f)  # should not raise
