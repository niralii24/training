"""
stage8/stage8_runner.py
-----------------------
Stage 8 – Multilingual Linguistic Grammar Scoring

Loads XLM-RoBERTa once, scores all five Excel transcript options via
pseudo-perplexity (model-driven, 100+ languages) plus Unicode-based
structural integrity checks, and returns a Linguistic Grammar Score (LGS)
for each option.

Public functions
----------------
run_stage8(excel_options, language, device, model_name, correct_option)
    → dict with per-option LGS metrics and a final ranking.

compute_final_score(aqs, tss, lgs)
    → single combined float blending all three stage scores.
"""

from __future__ import annotations

import difflib
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from .linguistic_scorer import (
    _load_model,
    compute_linguistic_score,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tie-breaking helpers
# ---------------------------------------------------------------------------

def _find_odd_one_out(texts: dict) -> Optional[str]:
    """
    Among a group of 3+ transcript options with identical LGS, find the one
    that is most textually *different* from the rest (the "odd one out").

    Algorithm
    ---------
    Compute the average character-level SequenceMatcher ratio between each
    option and all others in the group.  The option with the **lowest**
    average similarity is the odd one out — it differs from the majority in
    annotation style, bracket choice, speaker labels, whitespace, etc.

    Returns None when all texts are essentially identical (ratio ≥ 0.999).
    """
    keys = list(texts.keys())
    if len(keys) < 3:
        return None

    avg_sim = {}
    for k in keys:
        sims = [
            difflib.SequenceMatcher(None, texts[k], texts[other], autojunk=False).ratio()
            for other in keys if other != k
        ]
        avg_sim[k] = sum(sims) / len(sims) if sims else 1.0

    odd = min(avg_sim, key=avg_sim.get)   # type: ignore[arg-type]

    # Only return if it is *genuinely* different from the majority
    other_sims = [v for k, v in avg_sim.items() if k != odd]
    majority_avg = sum(other_sims) / len(other_sims) if other_sims else 1.0
    if majority_avg - avg_sim[odd] < 0.005:   # < 0.5 % difference → too subtle
        return None

    logger.info(
        "Stage 8 ▶ tie-break odd-one-out: %s  (avg_sim=%.4f vs majority %.4f)",
        odd, avg_sim[odd], majority_avg,
    )
    return odd


def _apply_tie_boost(
    scored:   dict,
    results:  dict,
    tie_boost: float = 0.015,
) -> dict:
    """
    Detect groups of >=3 options sharing the same LGS (within 1e-4) and apply
    a tiny boost to the odd one out, making it the preferred selection when
    the acoustic and text stages cannot distinguish between them.

    Iterates until stable -- so if boosting one option out of a 4-way tie
    leaves a 3-way tie, that remainder is resolved in the next pass automatically.

    The boost is intentionally tiny (default 0.015) -- just enough to break
    the tie without overriding meaningful score differences elsewhere.

    Parameters
    ----------
    scored    : {option_key: lgs_float}  mutated in-place
    results   : {option_key: metrics_dict | None}  lgs field updated too
    tie_boost : score added to the odd-one-out LGS

    Returns
    -------
    Updated scored dict.
    """
    max_iters = 10  # safety cap -- 5 options cannot need more than a few passes
    for iteration in range(max_iters):
        changed = False

        # Rebuild buckets each iteration (scores change after each boost)
        buckets = defaultdict(list)
        for k, v in scored.items():
            buckets[round(v, 4)].append(k)

        for bucket_score, group in buckets.items():
            if len(group) < 3:
                continue

            texts = {
                k: results[k]["transcript"]
                for k in group
                if results.get(k) and results[k].get("transcript")
            }
            odd = _find_odd_one_out(texts)
            if not odd:
                continue

            new_lgs = float(np.clip(scored[odd] + tie_boost, 0.0, 1.0))
            scored[odd]               = new_lgs
            results[odd]["lgs"]       = new_lgs
            results[odd]["tiebreak"]  = True
            changed = True
            logger.info(
                "Stage 8 tie-break pass %d: %s boosted by %.4f -> LGS %.4f",
                iteration + 1, odd, tie_boost, new_lgs,
            )
            # One boost per pass -- rebuild buckets before the next boost
            break

        if not changed:
            break

    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage8(
    excel_options:  dict,
    language:       str,
    device:         str   = "cpu",
    model_name:     str   = DEFAULT_MODEL,
    correct_option: Optional[int] = None,
    max_samples:    int   = 40,
    break_ties:     bool  = True,
    tie_boost:      float = 0.015,
) -> dict:
    """
    Score all five transcript options linguistically using XLM-RoBERTa.

    Parameters
    ----------
    excel_options   : {"option_1": text, …}  (blank/None entries are skipped)
    language        : ISO-639-1 code (informational only — model is multilingual)
    device          : "cuda" | "cpu"
    model_name      : HuggingFace model id (default: xlm-roberta-base)
    correct_option  : 1-based ground-truth index if known
    max_samples     : tokens sampled per text for pseudo-perplexity (speed/quality tradeoff)
    break_ties      : when True, apply a tiny boost to the odd-one-out when
                      3+ options share the same LGS (default: True)
    tie_boost       : size of the tie-breaking boost (default: 0.015)

    Returns
    -------
    dict:
        options         : {option_key: full metrics dict or None}
        ranked          : option keys sorted best→worst by LGS
        best_option     : str
        best_lgs        : float
        correct_option  : int | None
        model_used      : str
    """
    logger.info(
        "Stage 8 ▶ loading linguistic model %s on %s …", model_name, device
    )
    model, tokenizer = _load_model(model_name, device)

    results = {}

    for key in [f"option_{i}" for i in range(1, 6)]:
        text = excel_options.get(key, "")
        if not text or not text.strip():
            logger.info("Stage 8 ▶ %s is empty — skipping.", key)
            results[key] = None
            continue

        logger.info("Stage 8 ▶ scoring %s (len=%d chars) …", key, len(text))
        try:
            metrics = compute_linguistic_score(
                text        = text,
                model       = model,
                tokenizer   = tokenizer,
                device      = device,
                max_samples = max_samples,
            )
            metrics["transcript"] = text
            results[key] = metrics
            logger.info(
                "Stage 8 ▶ %s │ PPPL=%.2f  pppl_score=%.3f  "
                "struct=%.3f  LGS=%.4f",
                key,
                metrics["pppl"] or -1,
                metrics["pppl_score"],
                metrics["structural"]["score"],
                metrics["lgs"],
            )
        except Exception as exc:
            logger.error("Stage 8 ▶ %s failed: %s", key, exc, exc_info=True)
            results[key] = None

    # ── Build initial score lookup ─────────────────────────────────────────
    scored = {k: v["lgs"] for k, v in results.items() if v is not None}

    # ── Tie-break: boost the odd-one-out when 3+ options are tied ─────────
    if break_ties:
        scored = _apply_tie_boost(scored, results, tie_boost)

    # ── Rank by LGS ────────────────────────────────────────────────────────
    ranked = sorted(scored, key=scored.get, reverse=True)   # type: ignore[arg-type]
    best   = ranked[0] if ranked else None

    return {
        "options":        results,
        "ranked":         ranked,
        "best_option":    best,
        "best_lgs":       scored.get(best) if best else None,
        "correct_option": correct_option,
        "model_used":     model_name,
    }


# ---------------------------------------------------------------------------
# Final combined score (AQS + TSS + LGS)
# ---------------------------------------------------------------------------

def compute_final_score(
    aqs: float,
    tss: float,
    lgs: float,
    weights: dict | None = None,
) -> float:
    """
    Blend all three stage scores into a single final ranking metric.

    If weights dict provided → use it (keys: "aqs", "tss", "lgs")
    else → use defaults (0.45 / 0.35 / 0.20)
    """
    if weights:
        w_aqs = float(weights.get("aqs", 0.45))
        w_tss = float(weights.get("tss", 0.35))
        w_lgs = float(weights.get("lgs", 0.20))
    else:
        w_aqs, w_tss, w_lgs = 0.45, 0.35, 0.20

    # normalize so weights don't have to sum to 1
    total = w_aqs + w_tss + w_lgs
    if total <= 0:
        w_aqs, w_tss, w_lgs = 0.45, 0.35, 0.20
        total = 1.0

    return float(np.clip(
        (w_aqs * aqs + w_tss * tss + w_lgs * lgs) / total,
        0.0, 1.0,
    ))
