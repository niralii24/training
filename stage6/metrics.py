"""
stage6/metrics.py
-----------------
Core forced-alignment quality metrics extracted from WhisperX output.

All functions receive `aligned_segments` – the ``"segments"`` list returned
by ``WhisperXAligner.align()``.  Each segment is a dict with, at minimum:

    {
        "start": float,
        "end":   float,
        "text":  str,
        "words": [
            {"word": str, "start": float|None, "end": float|None,
             "score": float|None},
            …
        ],
        "chars": [                       # present when return_char_alignments=True
            {"char": str, "start": float|None, "end": float|None,
             "score": float|None},
            …
        ],
    }
"""

from __future__ import annotations
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# 1. Word alignment ratio
# ---------------------------------------------------------------------------

def compute_word_alignment_ratio(aligned_segments: list[dict]) -> float:
    """
    Fraction of words that received valid start *and* end timestamps.

    A word that WhisperX could not align to any acoustic frame is returned
    with ``start=None`` / ``end=None`` and is counted as *unaligned*.

    Returns
    -------
    float in [0, 1].  1.0 = every word aligned successfully.
    """
    total = 0
    aligned = 0
    for seg in aligned_segments:
        for w in seg.get("words", []):
            total += 1
            if w.get("start") is not None and w.get("end") is not None:
                aligned += 1
    return float(aligned / total) if total else 0.0


# ---------------------------------------------------------------------------
# 2. Timing deviation
# ---------------------------------------------------------------------------

def compute_timing_deviation(
    original_segments: list[dict],
    aligned_segments: list[dict],
) -> dict[str, float]:
    """
    Measure how far the aligned word cluster lies from the Whisper segment
    boundary predictions.

    For each segment we compute:
      - ``orig_center``  = (seg_start + seg_end) / 2   from Whisper
      - ``aln_center``   = (first_word_start + last_word_end) / 2  from WhisperX

    Then record |orig_center − aln_center| as the *deviation*.

    Returns
    -------
    dict with keys ``mean``, ``std``, ``max``, ``p90`` (all in seconds).
    """
    deviations: list[float] = []

    for orig, aln in zip(original_segments, aligned_segments):
        words = [w for w in aln.get("words", [])
                 if w.get("start") is not None and w.get("end") is not None]
        if not words:
            continue

        orig_center = (orig.get("start", 0.0) + orig.get("end", 0.0)) / 2.0
        aln_center  = (words[0]["start"] + words[-1]["end"]) / 2.0
        deviations.append(abs(aln_center - orig_center))

    if not deviations:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "p90": 0.0}

    arr = np.array(deviations)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "max":  float(np.max(arr)),
        "p90":  float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# 3. Unaligned segment ratio
# ---------------------------------------------------------------------------

def compute_unaligned_segment_ratio(aligned_segments: list[dict]) -> float:
    """
    Fraction of segments where *no* word received alignment timestamps.

    This captures segments that WhisperX completely failed to pin to the
    audio, often indicating a hallucinated or out-of-vocabulary sequence.

    Returns
    -------
    float in [0, 1].
    """
    if not aligned_segments:
        return 0.0

    unaligned = sum(
        1 for seg in aligned_segments
        if not any(w.get("start") is not None for w in seg.get("words", []))
    )
    return float(unaligned / len(aligned_segments))


# ---------------------------------------------------------------------------
# 4. Average alignment confidence (word level)
# ---------------------------------------------------------------------------

def compute_avg_alignment_confidence(aligned_segments: list[dict]) -> float:
    """
    Mean of WhisperX per-word ``score`` values across all segments.

    Scores are in [0, 1] where 1 = perfect ctc-alignment confidence.

    Returns
    -------
    float in [0, 1].  0.0 if no scores are available.
    """
    scores: list[float] = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            s = w.get("score")
            if s is not None:
                scores.append(float(s))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# 5. Phoneme-level alignment confidence (character proxy)
# ---------------------------------------------------------------------------

def compute_phoneme_confidence(
    aligned_segments: list[dict],
) -> dict[str, Any] | None:
    """
    Aggregate per-character alignment scores as a phoneme-level proxy.

    WhisperX can return character-level alignments (``return_char_alignments=True``).
    Character alignments are the closest approximation to phoneme confidence
    available without a dedicated G2P / phoneme recogniser.

    Returns
    -------
    dict with keys:
        mean, std, min, max, p10, p25, p50, p75, p90
    or ``None`` if no character alignments are present (model fallback).
    """
    char_scores: list[float] = []
    for seg in aligned_segments:
        for ch in seg.get("chars", []):
            s = ch.get("score")
            if s is not None:
                char_scores.append(float(s))

    if not char_scores:
        return None          # alignment model didn't emit char-level scores

    arr = np.array(char_scores)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "p10":  float(np.percentile(arr, 10)),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p90":  float(np.percentile(arr, 90)),
        "n_chars": len(char_scores),
    }


# ---------------------------------------------------------------------------
# 6. Per-segment word confidence breakdown (bonus helper)
# ---------------------------------------------------------------------------

def compute_per_segment_confidence(
    aligned_segments: list[dict],
) -> list[dict[str, Any]]:
    """
    Return per-segment confidence summary – useful for detailed diagnostics.
    """
    result = []
    for seg in aligned_segments:
        scores = [w["score"] for w in seg.get("words", [])
                  if w.get("score") is not None]
        result.append({
            "start":   seg.get("start"),
            "end":     seg.get("end"),
            "text":    seg.get("text", ""),
            "n_words": len(seg.get("words", [])),
            "n_scored": len(scores),
            "mean_score": float(np.mean(scores)) if scores else None,
            "min_score":  float(np.min(scores))  if scores else None,
        })
    return result
