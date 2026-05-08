"""WER and CER evaluation metrics for AutoEIT transcription.

Uses the *jiwer* library for robust, standards-compliant WER/CER calculation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    """Container for evaluation results between a hypothesis and a reference."""

    wer: float
    cer: float
    reference: str
    hypothesis: str


def _normalize_text(text: str) -> str:
    """Lower-case, strip, and collapse internal whitespace to single spaces."""
    # Using .split() without arguments splits on all whitespace and collapses
    # consecutive whitespace characters; joining with a single space yields a
    # normalized, formatting-agnostic representation.
    return " ".join(text.lower().split())


def compute_wer(reference: str, hypothesis: str) -> float:
    """Return the Word Error Rate (WER) between *reference* and *hypothesis*.

    Both strings are compared after lower-casing and whitespace normalization
    (collapsing all internal whitespace to single spaces).
    """
    try:
        from jiwer import wer as jiwer_wer
    except ImportError as exc:
        raise RuntimeError(
            "jiwer is required for WER computation.  Install it via: pip install jiwer"
        ) from exc
    return float(jiwer_wer(_normalize_text(reference), _normalize_text(hypothesis)))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Return the Character Error Rate (CER) between *reference* and *hypothesis*.

    Both strings are compared after lower-casing and whitespace normalization
    (collapsing all internal whitespace to single spaces).
    """
    try:
        from jiwer import cer as jiwer_cer
    except ImportError as exc:
        raise RuntimeError(
            "jiwer is required for CER computation.  Install it via: pip install jiwer"
        ) from exc
    return float(jiwer_cer(_normalize_text(reference), _normalize_text(hypothesis)))


def evaluate(reference: str, hypothesis: str) -> EvalResult:
    """Compute both WER and CER and return an :class:`EvalResult`."""
    return EvalResult(
        wer=compute_wer(reference, hypothesis),
        cer=compute_cer(reference, hypothesis),
        reference=reference,
        hypothesis=hypothesis,
    )


def report_disagreements(
    references: list[str],
    hypotheses: list[str],
    *,
    wer_threshold: float = 0.5,
) -> list[tuple[int, EvalResult]]:
    """Return (1-based index, EvalResult) pairs where WER exceeds *wer_threshold*.

    Parameters
    ----------
    references:
        List of gold-standard transcriptions.
    hypotheses:
        List of ASR-produced transcriptions.
    wer_threshold:
        Items with WER above this value are flagged as disagreements.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} reference(s) vs "
            f"{len(hypotheses)} hypothesis(es)."
        )
    disagreements: list[tuple[int, EvalResult]] = []
    for idx, (ref, hyp) in enumerate(zip(references, hypotheses), start=1):
        result = evaluate(ref, hyp)
        if result.wer > wer_threshold:
            disagreements.append((idx, result))
    return disagreements
