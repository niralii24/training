"""
stage6/stage6_runner.py
-----------------------
Stage 6 – Forced Alignment Scoring (WhisperX)
=============================================

Entry point consumed by main_pipeline.py.

Inputs
------
audio_path    : path to the cleaned WAV produced by stage 1
stage5_output : dict returned by run_stage5()
language      : ISO-639-1 language code from stage 2 (e.g. "ar", "en")
device        : "cuda" | "cpu"

Returns
-------
dict with the following keys:

Core metrics
  word_alignment_ratio      float [0,1]   – fraction of words with timestamps
  timing_deviation          dict          – mean/std/max/p90 (seconds)
  unaligned_segment_ratio   float [0,1]   – fraction of fully-unaligned segs
  avg_alignment_confidence  float [0,1]   – mean WhisperX word score
  phoneme_confidence        dict | None   – character-level score distribution

Anomaly detection
  hallucinated_segments     list[dict]    – segments flagged as hallucinations
  skipped_regions           list[dict]    – audio gaps with no word coverage
  overlapping_misalignments list[dict]    – word pairs with overlapping windows

Composite
  alignment_quality_score   float [0,1]   – single AQS summary
  per_segment_confidence    list[dict]    – per-segment confidence breakdown

Raw
  aligned_segments          list[dict]    – raw WhisperX segment output
"""

from __future__ import annotations

import logging
import librosa
import numpy as np

from .aligner      import WhisperXAligner
from .metrics      import (
    compute_word_alignment_ratio,
    compute_timing_deviation,
    compute_unaligned_segment_ratio,
    compute_avg_alignment_confidence,
    compute_phoneme_confidence,
    compute_per_segment_confidence,
)
from .hallucination import (
    detect_hallucinated_segments,
    detect_skipped_regions,
    detect_overlapping_misalignments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage6(
    audio_path:    str,
    stage5_output: dict,
    language:      str,
    device:        str = "cuda",
    skip_gap_sec:  float = 2.0,
) -> dict:
    """
    Main entry point for Stage 6 – Forced Alignment Scoring.

    Parameters
    ----------
    audio_path    : Cleaned WAV file path (output of stage 1).
    stage5_output : Output dict of ``run_stage5()``.
    language      : ISO-639-1 code (``"ar"``, ``"en"``, …).
    device        : ``"cuda"`` or ``"cpu"``.
    skip_gap_sec  : Minimum silence gap (seconds) to flag as a skipped region.

    Returns
    -------
    dict – see module docstring for full schema.
    """

    # ── 0. Pull the Whisper segments from stage 5 ─────────────────────────
    details = stage5_output.get("details", [])
    if not details:
        raise ValueError(
            "stage5_output['details'] is empty – make sure stage 5 ran successfully."
        )

    raw_segments: list[dict] = details[0].get("segments", [])
    if not raw_segments:
        # Minimal fallback: treat the whole file as one segment
        audio_duration = _audio_duration(audio_path)
        raw_segments = [{
            "start": 0.0,
            "end":   audio_duration,
            "text":  stage5_output.get("reference_transcript", ""),
        }]
        logger.warning(
            "Stage 5 returned no segments; created a single fallback segment "
            "spanning the full audio (%.1f s).", audio_duration
        )

    # ── 1. Forced alignment via WhisperX ──────────────────────────────────
    # WhisperX requires a valid ISO-639-1 code; guard against "unknown" /
    # empty values from stage 2 by falling back to English, which still
    # gives useful word-level timing even for non-English audio.
    _lang = language if (language and language not in ("unknown", "und", "xx")) else "en"
    logger.info("Stage 6 ▶ loading WhisperX aligner (lang=%r, device=%s) …",
                _lang, device)
    aligner = WhisperXAligner(language=_lang, device=device)

    logger.info("Stage 6 ▶ aligning %d segments …", len(raw_segments))
    aligned_result   = aligner.align(audio_path, raw_segments)
    aligned_segments = aligned_result.get("segments", [])
    logger.info("Stage 6 ▶ alignment complete (%d segments returned).",
                len(aligned_segments))

    # ── 2. Audio duration (needed for skipped-region detection) ───────────
    audio_dur = _audio_duration(audio_path)

    # ── 3. Core metrics ───────────────────────────────────────────────────
    word_ratio      = compute_word_alignment_ratio(aligned_segments)
    timing_dev      = compute_timing_deviation(raw_segments, aligned_segments)
    unaligned_ratio = compute_unaligned_segment_ratio(aligned_segments)
    avg_conf        = compute_avg_alignment_confidence(aligned_segments)
    phoneme_conf    = compute_phoneme_confidence(aligned_segments)
    per_seg_conf    = compute_per_segment_confidence(aligned_segments)

    logger.info(
        "Stage 6 metrics │ word_ratio=%.3f │ timing_dev_mean=%.3fs │ "
        "unaligned_ratio=%.3f │ avg_conf=%.3f",
        word_ratio, timing_dev["mean"], unaligned_ratio, avg_conf,
    )

    # ── 4. Anomaly detection ──────────────────────────────────────────────
    hallucinated = detect_hallucinated_segments(
        raw_segments, aligned_segments, audio_dur
    )
    skipped  = detect_skipped_regions(aligned_segments, audio_dur,
                                      min_gap=skip_gap_sec)
    overlaps = detect_overlapping_misalignments(aligned_segments)

    logger.info(
        "Stage 6 anomalies │ hallucinated_segs=%d │ skipped_regions=%d │ "
        "overlapping_words=%d",
        len(hallucinated), len(skipped), len(overlaps),
    )

    # ── 5. Composite Alignment Quality Score (AQS) ────────────────────────
    aqs = _compute_aqs(
        word_ratio      = word_ratio,
        avg_conf        = avg_conf,
        unaligned_ratio = unaligned_ratio,
        timing_dev_mean = timing_dev["mean"],
        n_hallucinated  = len(hallucinated),
        skip_fraction   = sum(r["duration"] for r in skipped) / (audio_dur + 1e-8),
        n_overlaps      = len(overlaps),
    )
    logger.info("Stage 6 ▶ AQS = %.4f", aqs)

    # ── 6. Pack and return ────────────────────────────────────────────────
    return {
        # Core metrics
        "word_alignment_ratio":     word_ratio,
        "timing_deviation":         timing_dev,
        "unaligned_segment_ratio":  unaligned_ratio,
        "avg_alignment_confidence": avg_conf,
        "phoneme_confidence":       phoneme_conf,

        # Anomalies
        "hallucinated_segments":     hallucinated,
        "skipped_regions":           skipped,
        "overlapping_misalignments": overlaps,

        # Composite
        "alignment_quality_score": aqs,
        "per_segment_confidence":  per_seg_conf,

        # Raw output (for downstream stages or inspection)
        "aligned_segments": aligned_segments,
        "audio_duration":   audio_dur,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds using librosa (no FFmpeg required)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return float(len(y) / sr)


def _compute_aqs(
    *,
    word_ratio:      float,
    avg_conf:        float,
    unaligned_ratio: float,
    timing_dev_mean: float,
    n_hallucinated:  int,
    skip_fraction:   float,
    n_overlaps:      int,
) -> float:
    """
    Alignment Quality Score (AQS) – a single [0, 1] summary of alignment
    health, suitable for use as a feature in downstream scoring.

    Component weights
    -----------------
    0.35 × word_alignment_ratio          (main accuracy signal)
    0.25 × avg_alignment_confidence      (model certainty)
    0.20 × (1 − unaligned_segment_ratio) (segment-level coverage)
    0.20 × timing_score                  (closeness to Whisper times)

    Penalties (capped)
    ------------------
    −0.05 per hallucinated segment  (max −0.30)
    −0.50 × skip_fraction           (proportional to skipped audio)
    −0.01 per overlapping word pair (max −0.10)
    """
    # Timing score: 1.0 at 0 s deviation, 0.0 at ≥ 3 s deviation
    timing_score = float(np.clip(1.0 - timing_dev_mean / 3.0, 0.0, 1.0))

    base = (
        0.35 * word_ratio
        + 0.25 * avg_conf
        + 0.20 * (1.0 - unaligned_ratio)
        + 0.20 * timing_score
    )

    hallucination_penalty = min(0.30, n_hallucinated * 0.05)
    skip_penalty          = min(0.30, skip_fraction * 0.50)
    overlap_penalty       = min(0.10, n_overlaps * 0.01)

    aqs = base - hallucination_penalty - skip_penalty - overlap_penalty
    return float(np.clip(aqs, 0.0, 1.0))
