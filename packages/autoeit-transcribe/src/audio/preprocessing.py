"""Audio preprocessing: loading, resampling, and loudness normalization.

Uses PyAV (bundled with faster-whisper) for broad codec support so that
.mp3, .m4a, .ogg, etc. can all be read without requiring FFmpeg on PATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE: int = 16_000


def load_audio(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and return a mono float32 waveform resampled to *target_sr*.

    Parameters
    ----------
    path:
        Path to the audio file (any format supported by PyAV).
    target_sr:
        Desired output sample rate in Hz.  Defaults to 16 000 Hz as required
        by Whisper.

    Returns
    -------
    numpy.ndarray
        1-D float32 array in the range [-1, 1].
    """
    import av  # bundled with faster-whisper

    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
        frames: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray()[0])
    if not frames:
        raise ValueError(f"No audio data decoded from {path!r}.")
    waveform = np.concatenate(frames, axis=0).astype(np.float32)
    return waveform


def normalize_loudness(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize *waveform* to [-1, 1].

    If the signal is silent (all zeros), returns it unchanged.
    """
    peak = np.abs(waveform).max()
    if peak == 0.0:
        return waveform
    return waveform / peak
