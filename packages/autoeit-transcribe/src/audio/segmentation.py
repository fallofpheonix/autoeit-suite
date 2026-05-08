"""Voice-activity-detection (VAD) based audio segmentation.

Splits a continuous EIT recording into individual learner response segments by
detecting silence gaps.  The algorithm works entirely in the time domain so it
does not require an external VAD model at segmentation time (faster-whisper's
own VAD is applied during transcription).

The primary strategy is energy-based silence detection:
    1.  Compute short-time RMS energy over the waveform.
    2.  Threshold below a configurable fraction of the mean energy to find
        silence regions.
    3.  Collect voiced segments between silence regions.
    4.  Merge segments that are too short into their neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import load_audio, TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class AudioSegment:
    """Time-stamped segment of a waveform."""

    start_sec: float
    end_sec: float
    waveform: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _frame_rms(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute per-frame RMS energy."""
    # Guard against empty or too-short waveforms and invalid parameters.
    # For these cases, return an empty RMS array so callers can handle the
    # absence of frames gracefully.
    if (
        waveform is None
        or frame_length <= 0
        or hop_length <= 0
        or len(waveform) == 0
        or len(waveform) < frame_length
    ):
        return np.empty(0, dtype=np.float32)

    n_frames = 1 + (len(waveform) - frame_length) // hop_length
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + frame_length]
        rms[i] = float(np.sqrt(np.sum(np.square(frame)) / len(frame)))
    return rms


def detect_voiced_regions(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    silence_threshold_fraction: float = 0.05,
    min_silence_ms: float = 300.0,
    min_speech_ms: float = 200.0,
) -> list[tuple[float, float]]:
    """Return a list of (start_sec, end_sec) pairs for *voiced* regions.

    Parameters
    ----------
    waveform:
        Mono float32 PCM audio.
    sample_rate:
        Sample rate of *waveform*.
    frame_ms:
        Analysis frame length in milliseconds.
    hop_ms:
        Frame hop size in milliseconds.
    silence_threshold_fraction:
        Frames with RMS below ``silence_threshold_fraction × mean_rms`` are
        labelled as silence.
    min_silence_ms:
        Minimum duration for a silence gap to be recognised as a boundary.
    min_speech_ms:
        Minimum duration of a voiced region; shorter regions are merged into
        their neighbour.
    """
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)
    min_silence_frames = int(min_silence_ms / hop_ms)
    min_speech_frames = int(min_speech_ms / hop_ms)

    rms = _frame_rms(waveform, frame_length, hop_length)
    mean_rms = float(rms.mean()) or 1e-9
    threshold = silence_threshold_fraction * mean_rms
    is_voiced = rms > threshold

    # Collect contiguous voiced / silent runs.
    segments: list[tuple[float, float]] = []
    in_speech = False
    speech_start = 0
    silence_run = 0

    for i, voiced in enumerate(is_voiced):
        if voiced:
            if not in_speech:
                speech_start = i
                in_speech = True
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    end_frame = i - silence_run + 1
                    if (end_frame - speech_start) >= min_speech_frames:
                        start_sec = speech_start * hop_length / sample_rate
                        end_sec = end_frame * hop_length / sample_rate
                        segments.append((start_sec, end_sec))
                    in_speech = False
                    silence_run = 0

    if in_speech:
        end_frame = len(is_voiced)
        if (end_frame - speech_start) >= min_speech_frames:
            start_sec = speech_start * hop_length / sample_rate
            end_sec = len(waveform) / sample_rate
            segments.append((start_sec, end_sec))

    return segments


def detect_silence_gaps(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    silence_threshold_fraction: float = 0.05,
    min_silence_ms: float = 300.0,
    min_speech_ms: float = 200.0,
) -> list[tuple[float, float]]:
    """Deprecated alias for :func:`detect_voiced_regions`.

    Despite the name, this function returns *voiced* (speech) regions,
    not silence gaps. It is kept for backwards compatibility and simply
    forwards all arguments to :func:`detect_voiced_regions`.
    """
    return detect_voiced_regions(
        waveform=waveform,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        silence_threshold_fraction=silence_threshold_fraction,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
    )


def segment_audio_file(
    path: str | Path,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    silence_threshold_fraction: float = 0.05,
    min_silence_ms: float = 300.0,
    min_speech_ms: float = 200.0,
) -> list[AudioSegment]:
    """Load *path* and return a list of voiced :class:`AudioSegment` objects."""
    waveform = load_audio(path, target_sr=target_sr)
    gaps = detect_silence_gaps(
        waveform,
        sample_rate=target_sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        silence_threshold_fraction=silence_threshold_fraction,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
    )
    segments: list[AudioSegment] = []
    for start_sec, end_sec in gaps:
        start_sample = int(start_sec * target_sr)
        end_sample = int(end_sec * target_sr)
        segments.append(
            AudioSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                waveform=waveform[start_sample:end_sample],
                sample_rate=target_sr,
            )
        )
    return segments
