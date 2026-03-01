"""
stage7/stage7_runner.py
-----------------------
Stage 7 – Acoustic Similarity Metrics (Text-level, Multi-reference)

Motivation
----------
Stage 6 scores each transcript option *acoustically* (does WhisperX align
the words to the right audio frames?).  Stage 7 scores them *textually*
(how similar is each option to what multiple ASR models actually heard?).

Together they catch two different failure modes:
  • A word-for-word correct transcript with bad punctuation  → Stage 6 catches it
  • A paraphrase that aligns well but uses different words  → Stage 7 catches it

Inputs
------
excel_options   : {"option_1": "text", …, "option_5": "text"}
                  From Stage 6 / Excel loader.
asr_references  : [str, str, …]
                  All ASR transcripts from Stage 5 (one per model).
                  With a single model this degrades gracefully (variance=0).
language        : ISO-639-1 code for normalisation (e.g. "ar").
correct_option  : int | None  from the Excel sheet.

Per-option metrics
------------------
per_ref_wer      : WER against each individual ASR reference
mean_wer         : mean over all references
var_wer          : variance of WER across references (0 = all models agree)
per_ref_cer      : CER against each reference
mean_cer         : mean CER
var_cer          : variance of CER
fuzzy_similarity : mean difflib token-ratio across references
consistency_score: how stable the option's error rate is across models
                   = 1 − clip(var_wer × 20, 0, 1)
tss              : Transcript Similarity Score [0, 1] – see _compute_tss()
confusion        : token-level confusion matrix (top substitutions / del / ins)

Return dict top-level keys
--------------------------
options         : {option_key: metrics_dict}
ranked          : option keys sorted best→worst by TSS
best_option     : str
best_tss        : float
correct_option  : int | None
asr_reference_count : int
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .text_metrics import wer, cer, fuzzy_similarity, token_confusion_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage7(
    excel_options:  dict[str, str],
    asr_references: list[str],
    language:       str,
    correct_option: Optional[int] = None,
) -> dict:
    """
    Score all Excel transcript options against the multi-model ASR references.

    Parameters
    ----------
    excel_options   : mapping of option key → transcript text.
    asr_references  : list of ASR output strings (one per Stage 5 model).
    language        : ISO-639-1 code used for text normalisation.
    correct_option  : 1-based int if the correct option is known, else None.

    Returns
    -------
    dict – see module docstring.
    """
    if not asr_references:
        raise ValueError("Stage 7: asr_references must contain at least one string.")

    # Filter out None / blank references
    refs = [r for r in asr_references if r and r.strip()]
    if not refs:
        raise ValueError("Stage 7: all asr_references are blank.")

    logger.info(
        "Stage 7 ▶ scoring %d option(s) against %d ASR reference(s)  lang=%s",
        len(excel_options), len(refs), language,
    )

    results: dict[str, Optional[dict]] = {}

    for key in [f"option_{i}" for i in range(1, 6)]:
        text = excel_options.get(key, "")
        if not text or not text.strip():
            logger.info("Stage 7 ▶ %s is empty — skipping.", key)
            results[key] = None
            continue

        logger.info("Stage 7 ▶ computing metrics for %s …", key)

        # ── WER per reference ──────────────────────────────────────────────
        per_ref_wer_vals = [wer(ref, text, language) for ref in refs]
        mean_wer_val = float(np.mean(per_ref_wer_vals))
        var_wer_val  = float(np.var(per_ref_wer_vals)) if len(refs) > 1 else 0.0

        # ── CER per reference ──────────────────────────────────────────────
        per_ref_cer_vals = [cer(ref, text, language) for ref in refs]
        mean_cer_val = float(np.mean(per_ref_cer_vals))
        var_cer_val  = float(np.var(per_ref_cer_vals)) if len(refs) > 1 else 0.0

        # ── Fuzzy similarity (mean across references) ──────────────────────
        fuzzy_vals = [fuzzy_similarity(ref, text, language) for ref in refs]
        mean_fuzzy = float(np.mean(fuzzy_vals))

        # ── Consistency: how stable the WER is across different ASR models ─
        # High variance  → models disagree about how similar this option is
        #               → the option is probably not strongly correct
        # WER variance is typically 0–0.05; multiply by 20 to scale to 0–1
        consistency = float(np.clip(1.0 - var_wer_val * 20.0, 0.0, 1.0))

        # ── Token confusion matrix (all refs × [this option]) ─────────────
        confusion = token_confusion_matrix(refs, [text], language=language)

        # ── TSS (Transcript Similarity Score) ─────────────────────────────
        tss = _compute_tss(
            mean_wer      = mean_wer_val,
            mean_cer      = mean_cer_val,
            fuzzy         = mean_fuzzy,
            consistency   = consistency,
        )

        results[key] = {
            "transcript": text,
            # WER
            "per_ref_wer": per_ref_wer_vals,
            "mean_wer":    mean_wer_val,
            "var_wer":     var_wer_val,
            # CER
            "per_ref_cer": per_ref_cer_vals,
            "mean_cer":    mean_cer_val,
            "var_cer":     var_cer_val,
            # Token similarity
            "fuzzy_similarity": mean_fuzzy,
            "per_ref_fuzzy":    fuzzy_vals,
            # Cross-model consistency
            "consistency_score": consistency,
            # Confusion
            "confusion": confusion,
            # Final score
            "tss": tss,
        }

        logger.info(
            "Stage 7 ▶ %s │ mean_wer=%.3f  mean_cer=%.3f  "
            "fuzzy=%.3f  consistency=%.3f  TSS=%.4f",
            key, mean_wer_val, mean_cer_val, mean_fuzzy, consistency, tss,
        )

    # ── Rank by TSS ────────────────────────────────────────────────────────
    scored = {k: v["tss"] for k, v in results.items() if v is not None}
    ranked = sorted(scored, key=scored.get, reverse=True)  # type: ignore[arg-type]
    best   = ranked[0] if ranked else None

    return {
        "options":              results,
        "ranked":               ranked,
        "best_option":          best,
        "best_tss":             scored.get(best) if best else None,
        "correct_option":       correct_option,
        "asr_reference_count":  len(refs),
        "asr_references":       refs,
    }


# ---------------------------------------------------------------------------
# Combined final score (AQS from Stage 6 + TSS from Stage 7)
# ---------------------------------------------------------------------------

def compute_combined_score(aqs: float, tss: float) -> float:
    """
    Blend the acoustic alignment score (Stage 6) with the transcript
    similarity score (Stage 7) into a single final ranking metric.

    Weights
    -------
    0.55 × AQS   Acoustic forced-alignment quality (harder to fake)
    0.45 × TSS   Text similarity to ASR references (content correctness)
    """
    return float(np.clip(0.55 * aqs + 0.45 * tss, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TSS formula
# ---------------------------------------------------------------------------

def _compute_tss(
    mean_wer:    float,
    mean_cer:    float,
    fuzzy:       float,
    consistency: float,
) -> float:
    """
    Transcript Similarity Score [0, 1].

    Component weights (sum = 1.0)
    -----------------
    0.45 × (1 − mean_wer)   – primary word-level accuracy signal
    0.20 × (1 − mean_cer)   – character-level accuracy (catches partial errors)
    0.15 × fuzzy_similarity  – token-order/overlap via difflib ratio
    0.20 × consistency_score – stability of WER across ASR models
                               (high variance → ASR models disagree → lower score)
    """
    tss = (
        0.45 * (1.0 - mean_wer)
        + 0.20 * (1.0 - mean_cer)
        + 0.15 * fuzzy
        + 0.20 * consistency
    )
    return float(np.clip(tss, 0.0, 1.0))
