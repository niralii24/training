"""
stage6/pause_alignment.py
--------------------------
Punctuation–pause alignment scoring.

Problem
-------
Two transcripts that differ only in punctuation placement (؟ ، . etc.) produce
nearly identical acoustic alignment scores because WhisperX aligns *words*,
not punctuation.  Yet punctuation carries real acoustic information: a question
mark (؟) or comma (،) in Arabic predicts a brief silence/pause in the audio.

Solution: two complementary signals measured directly from WhisperX output and
from audio energy, combined into a single ``score`` in [0, 1].

Signal A – Inter-word gap discrimination (WhisperX timing-based)
    Words whose surface form ends with a pause-punctuation character should
    be followed by a larger inter-word gap than words without punctuation.
    ``gap_ratio = mean_punct_gap / mean_non_punct_gap``
    A gap_ratio > 2 means punctuation does predict acoustic pauses.

Signal B – Bi-directional audio-silence match (librosa energy-based)
    We detect raw silence intervals from the audio's RMS energy, then
    measure precision (fraction of silences that have a nearby punctuation
    boundary) and recall (fraction of punctuation boundaries that have a
    nearby silence).  F1 of precision × recall is Signal B.

The two signals are blended, with more weight given to Signal B when enough
audio pauses are detected.

Supported pause characters
--------------------------
Arabic:  ؟ (U+061F)  ، (U+060C)  ؛ (U+061B)
Latin:   .  !  ?  :

Usage
-----
from stage6.pause_alignment import compute_punctuation_pause_score

result = compute_punctuation_pause_score(
    aligned_segments = aligned_result["segments"],
    audio_path       = "stage1_clean.wav",
)
score = result["score"]   # float in [0, 1]
"""
from __future__ import annotations

import logging
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Punctuation characters that mark a prose pause / sentence boundary
PAUSE_PUNCT: frozenset[str] = frozenset("؟،؛.!?:")


# ---------------------------------------------------------------------------
# Audio silence detection
# ---------------------------------------------------------------------------

def detect_audio_pauses(
    audio_path:   str,
    sr_target:    int   = 16_000,
    frame_length: int   = 512,
    hop_length:   int   = 128,
    threshold_db: float = -42.0,
    min_dur:      float = 0.12,
) -> list[dict]:
    """
    Detect silence / pause intervals using RMS energy thresholding.

    Parameters
    ----------
    audio_path   : Path to the (already-cleaned) WAV file.
    sr_target    : Re-sample to this rate before analysis (16 kHz default).
    frame_length : FFT frame length in samples.
    hop_length   : Hop size in samples.
    threshold_db : Frames below this dBFS level are considered silent.
    min_dur      : Minimum pause duration (seconds) to keep.

    Returns
    -------
    List of dicts ``{"start": float, "end": float, "duration": float}``.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    except Exception as exc:
        logger.warning("pause_alignment: could not load %s: %s", audio_path, exc)
        return []

    rms    = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    pauses: list[dict] = []
    in_pause = False
    t_start  = 0.0

    for t, db in zip(times, rms_db):
        t = float(t)
        if db < threshold_db:
            if not in_pause:
                in_pause = True
                t_start  = t
        else:
            if in_pause:
                in_pause = False
                dur = t - t_start
                if dur >= min_dur:
                    pauses.append({"start": t_start, "end": t, "duration": dur})

    # Handle trailing silence
    if in_pause:
        dur = float(times[-1]) - t_start
        if dur >= min_dur:
            pauses.append({"start": t_start, "end": float(times[-1]), "duration": dur})

    logger.debug("pause_alignment: detected %d audio pauses in %s", len(pauses), audio_path)
    return pauses


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_punctuation_pause_score(
    aligned_segments: list[dict],
    audio_path:       str,
    tolerance_sec:    float = 0.40,
) -> dict:
    """
    Score how well the transcript's punctuation marks predict audio pauses.

    Parameters
    ----------
    aligned_segments : ``"segments"`` list from ``WhisperXAligner.align()``.
    audio_path       : Cleaned WAV used for the alignment (to detect silences).
    tolerance_sec    : Max distance (seconds) at which a punctuation boundary
                       is considered to "match" an audio pause.

    Returns
    -------
    dict with keys:
        score                – float [0, 1] combined quality score
        gap_ratio            – float  mean_punct_gap / mean_non_punct_gap
        punct_coverage       – float  fraction of punct positions with gap ≥ tolerance
        f1                   – float  precision/recall match with audio silences
        precision            – float | None
        recall               – float | None
        n_punct_boundaries   – int   number of punctuation word-boundaries found
        n_audio_pauses       – int   number of detected audio silences
        details              – dict  raw intermediate values for diagnostics
    """
    # ── Signal A: inter-word gap discrimination ───────────────────────────
    punct_gaps:  list[float] = []   # gaps AFTER words ending with punctuation
    non_punct_gaps: list[float] = []
    punct_times: list[float] = []   # absolute audio end-times of punct words

    for seg in aligned_segments:
        words = [
            w for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        for i, w in enumerate(words):
            wtext       = (w.get("word") or "").strip()
            ends_punct  = bool(wtext) and wtext[-1] in PAUSE_PUNCT
            w_end       = float(w["end"])

            if i < len(words) - 1:
                gap = max(0.0, float(words[i + 1]["start"]) - w_end)
                if ends_punct:
                    punct_gaps.append(gap)
                    punct_times.append(w_end)
                else:
                    non_punct_gaps.append(gap)
            elif ends_punct:
                # Last word in segment – record time for Signal B only
                punct_times.append(w_end)

    mean_punct_gap     = float(np.mean(punct_gaps))     if punct_gaps     else 0.0
    mean_non_punct_gap = float(np.mean(non_punct_gaps)) if non_punct_gaps else 0.01
    gap_ratio          = mean_punct_gap / (mean_non_punct_gap + 1e-6)

    # Fraction of punct word-boundaries where the following gap is meaningful
    punct_coverage = (
        sum(1 for g in punct_gaps if g >= tolerance_sec) / len(punct_gaps)
        if punct_gaps else 0.0
    )

    # gap_ratio score: 0.0 at ratio≤1 (no discrimination), 1.0 at ratio≥4
    ratio_score = float(np.clip((gap_ratio - 1.0) / 3.0, 0.0, 1.0))

    # Signal A combined value
    signal_a = 0.55 * ratio_score + 0.45 * punct_coverage

    # ── Signal B: audio silence ↔ punctuation match ───────────────────────
    audio_pauses = detect_audio_pauses(audio_path)
    n_pauses     = len(audio_pauses)
    n_punct      = len(punct_times)

    precision: Optional[float] = None
    recall:    Optional[float] = None
    f1:        float           = signal_a   # default when signal B unavailable

    if n_pauses >= 1 and n_punct >= 1:
        # How many audio pauses are "covered" by a nearby punct boundary?
        matched_pauses = sum(
            1 for p in audio_pauses
            if any(abs(pt - p["start"]) <= tolerance_sec for pt in punct_times)
        )
        precision = matched_pauses / n_pauses

        # How many punct boundaries have a nearby audio pause?
        matched_punct = sum(
            1 for pt in punct_times
            if any(abs(pt - p["start"]) <= tolerance_sec for p in audio_pauses)
        )
        recall = matched_punct / n_punct

        denom = (precision + recall)
        f1    = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0

    # ── Blend signals ─────────────────────────────────────────────────────
    # Favour Signal B when we have enough audio evidence, otherwise rely on A
    w_b     = 0.65 if (n_pauses >= 3 and n_punct >= 2) else 0.30
    w_a     = 1.0 - w_b
    combined = w_a * signal_a + w_b * f1

    logger.debug(
        "punct_pause │ gap_ratio=%.2f  coverage=%.2f  signal_a=%.3f  "
        "f1=%.3f  →  score=%.3f  (n_punct=%d  n_audio_pauses=%d)",
        gap_ratio, punct_coverage, signal_a, f1, combined, n_punct, n_pauses,
    )

    return {
        "score":              float(np.clip(combined, 0.0, 1.0)),
        "gap_ratio":          float(gap_ratio),
        "punct_coverage":     float(punct_coverage),
        "f1":                 float(f1),
        "precision":          float(precision) if precision is not None else None,
        "recall":             float(recall)    if recall    is not None else None,
        "n_punct_boundaries": n_punct,
        "n_audio_pauses":     n_pauses,
        "details": {
            "mean_punct_gap_sec":     mean_punct_gap,
            "mean_non_punct_gap_sec": mean_non_punct_gap,
            "signal_a":               float(signal_a),
            "signal_b_f1":            float(f1),
        },
    }