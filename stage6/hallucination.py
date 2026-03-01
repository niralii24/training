"""
stage6/hallucination.py
-----------------------
Detect three classes of alignment anomaly in WhisperX output:

1. **Hallucinated segments**
   Text that Whisper generated with no strong acoustic grounding.
   Signals:
     a. Very low average word alignment confidence (below threshold).
     b. Words clustered uniformly in time (repeating-pattern hallucination).
     c. Words cluster exclusively near segment boundaries (boundary artefact).

2. **Skipped regions**
   Contiguous stretches of audio where no word was aligned.
   These represent speech that Whisper completely missed (under-transcription).

3. **Overlapping misalignments**
   Adjacent words (across any segments) whose aligned time windows overlap
   by more than a small tolerance, indicating the aligner forced words into
   the wrong frames.
"""

from __future__ import annotations
import numpy as np
from typing import Any

# ── Thresholds (tunable) ───────────────────────────────────────────────────

# Word score below which a segment is suspicious (hallucination signal)
HALLUCINATION_CONF_THRESHOLD: float = 0.35

# Coefficient of variation below which word-spacing is "too uniform"
UNIFORM_SPACING_CV_THRESHOLD: float = 0.20

# Fraction of words near a boundary for "boundary clustering" flag
BOUNDARY_CLUSTER_FRACTION: float = 0.65
BOUNDARY_PROXIMITY_SEC: float = 0.15   # within 150 ms of segment edge

# Minimum gap (seconds) to report as a skipped region
SKIP_MIN_GAP_SEC: float = 2.0

# Minimum word overlap (seconds) to report as a misalignment
OVERLAP_MIN_SEC: float = 0.02


# ---------------------------------------------------------------------------
# 1. Hallucinated segments
# ---------------------------------------------------------------------------

def detect_hallucinated_segments(
    original_segments: list[dict],
    aligned_segments: list[dict],
    audio_duration: float,
) -> list[dict[str, Any]]:
    """
    Returns a list of segments flagged as likely hallucinations.

    Each entry:
    {
        "segment_idx":    int,
        "start":          float,
        "end":            float,
        "text":           str,
        "avg_confidence": float | None,
        "flags":          list[str],   # e.g. ["low_confidence", "uniform_word_spacing"]
    }
    """
    hallucinated: list[dict] = []

    n = min(len(original_segments), len(aligned_segments))

    for idx in range(n):
        orig = original_segments[idx]
        aln  = aligned_segments[idx]

        words  = aln.get("words", [])
        scores = [w["score"] for w in words if w.get("score") is not None]
        timed  = [w for w in words
                  if w.get("start") is not None and w.get("end") is not None]

        flags: list[str] = []

        # ── Flag A: low average alignment confidence ──────────────────────
        avg_score = float(np.mean(scores)) if scores else None
        if avg_score is not None and avg_score < HALLUCINATION_CONF_THRESHOLD:
            flags.append("low_confidence")

        # ── Flag B: uniform word spacing (repeating hallucination) ────────
        if len(timed) >= 3:
            centers = np.array([
                (w["start"] + w["end"]) / 2.0 for w in timed
            ])
            centers.sort()
            gaps = np.diff(centers)
            mean_gap = float(np.mean(gaps))
            if mean_gap > 0:
                cv = float(np.std(gaps)) / mean_gap
                if cv < UNIFORM_SPACING_CV_THRESHOLD:
                    flags.append("uniform_word_spacing")

        # ── Flag C: words cluster near segment boundary ───────────────────
        if timed:
            seg_start = orig.get("start", 0.0)
            seg_end   = orig.get("end",   0.0)
            centers   = [(w["start"] + w["end"]) / 2.0 for w in timed]
            near = sum(
                1 for c in centers
                if abs(c - seg_start) < BOUNDARY_PROXIMITY_SEC
                or abs(c - seg_end)   < BOUNDARY_PROXIMITY_SEC
            )
            if len(centers) > 0 and (near / len(centers)) >= BOUNDARY_CLUSTER_FRACTION:
                flags.append("boundary_clustering")

        if flags:
            hallucinated.append({
                "segment_idx":    idx,
                "start":          orig.get("start"),
                "end":            orig.get("end"),
                "text":           orig.get("text", ""),
                "avg_confidence": avg_score,
                "flags":          flags,
            })

    return hallucinated


# ---------------------------------------------------------------------------
# 2. Skipped regions
# ---------------------------------------------------------------------------

def detect_skipped_regions(
    aligned_segments: list[dict],
    audio_duration: float,
    min_gap: float = SKIP_MIN_GAP_SEC,
) -> list[dict[str, Any]]:
    """
    Find contiguous audio regions not covered by any aligned word.

    Candidate audio spans are identified from word-level timestamps across
    all segments.  Any gap ≥ ``min_gap`` seconds is reported.

    Returns a list of:
    {
        "start":    float,
        "end":      float,
        "duration": float,
    }
    """
    # Collect all aligned word spans
    spans: list[tuple[float, float]] = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                spans.append((float(w["start"]), float(w["end"])))

    if not spans:
        # Nothing aligned at all → the entire file is "skipped"
        return [{"start": 0.0, "end": audio_duration,
                 "duration": audio_duration}]

    spans.sort(key=lambda x: x[0])

    gaps: list[dict] = []

    # Gap before the first word
    if spans[0][0] >= min_gap:
        gaps.append({
            "start":    0.0,
            "end":      spans[0][0],
            "duration": spans[0][0],
        })

    # Gaps between consecutive words
    for i in range(len(spans) - 1):
        gap_start = spans[i][1]
        gap_end   = spans[i + 1][0]
        duration  = gap_end - gap_start
        if duration >= min_gap:
            gaps.append({
                "start":    float(gap_start),
                "end":      float(gap_end),
                "duration": float(duration),
            })

    # Gap after the last word
    tail = audio_duration - spans[-1][1]
    if tail >= min_gap:
        gaps.append({
            "start":    float(spans[-1][1]),
            "end":      float(audio_duration),
            "duration": float(tail),
        })

    return gaps


# ---------------------------------------------------------------------------
# 3. Overlapping misalignments
# ---------------------------------------------------------------------------

def detect_overlapping_misalignments(
    aligned_segments: list[dict],
    min_overlap: float = OVERLAP_MIN_SEC,
) -> list[dict[str, Any]]:
    """
    Find pairs of successive words (globally, across all segments) whose
    aligned time windows overlap.

    Overlaps > ``min_overlap`` seconds are reported.

    Returns a list of:
    {
        "word_a":          str,
        "word_b":          str,
        "segment_a":       int,
        "segment_b":       int,
        "overlap_sec":     float,
        "at_time":         float,   # absolute position (start of word_a)
        "score_a":         float | None,
        "score_b":         float | None,
    }
    """
    # Flatten all timed words with their segment index
    flat: list[dict] = []
    for seg_idx, seg in enumerate(aligned_segments):
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                flat.append({
                    "seg_idx": seg_idx,
                    "word":    w.get("word", ""),
                    "start":   float(w["start"]),
                    "end":     float(w["end"]),
                    "score":   w.get("score"),
                })

    flat.sort(key=lambda x: x["start"])

    overlaps: list[dict] = []
    for i in range(len(flat) - 1):
        a = flat[i]
        b = flat[i + 1]

        # True overlap = intersection of [start, end] intervals
        overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
        if overlap > min_overlap:
            overlaps.append({
                "word_a":      a["word"],
                "word_b":      b["word"],
                "segment_a":   a["seg_idx"],
                "segment_b":   b["seg_idx"],
                "overlap_sec": float(overlap),
                "at_time":     max(a["start"], b["start"]),
                "score_a":     a["score"],
                "score_b":     b["score"],
            })

    return overlaps
