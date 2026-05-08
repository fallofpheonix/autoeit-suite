"""faster-whisper model wrapper for AutoEIT ASR.

Provides a thin, configurable interface around :class:`faster_whisper.WhisperModel`
with settings tuned for non-native Spanish learner speech.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_HF_XET_DISABLE_KEY = "HF_HUB_DISABLE_XET"


@dataclass
class WhisperConfig:
    """Configuration for the faster-whisper ASR model."""

    model_size: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "es"
    beam_size: int = 5
    vad_filter: bool = True
    # Local path to a pre-downloaded model; overrides hub download when set.
    local_model_dir: Path | None = None
    # Additional keyword arguments forwarded to model.transcribe().
    transcribe_kwargs: dict = field(default_factory=dict)


def build_model(config: WhisperConfig):
    """Instantiate and return a :class:`faster_whisper.WhisperModel`.

    Sets ``HF_HUB_DISABLE_XET=1`` to prevent stalling Hugging Face Xet
    downloads in environments where it is unavailable.
    """
    os.environ.setdefault(_HF_XET_DISABLE_KEY, "1")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required.  Install it via: pip install faster-whisper"
        ) from exc

    model_source: str | Path = (
        config.local_model_dir if config.local_model_dir is not None else config.model_size
    )
    return WhisperModel(
        str(model_source),
        device=config.device,
        compute_type=config.compute_type,
    )


def transcribe_audio(
    model,
    audio_path: str | Path,
    *,
    config: WhisperConfig,
) -> Iterable:
    """Transcribe *audio_path* and return the raw segment iterator from faster-whisper.

    The caller is responsible for consuming the iterator before the model is
    used again, as faster-whisper returns a lazy generator.
    """
    kwargs = {
        "language": config.language,
        "beam_size": config.beam_size,
        "vad_filter": config.vad_filter,
        **config.transcribe_kwargs,
    }
    segments, _info = model.transcribe(str(audio_path), **kwargs)
    return segments
