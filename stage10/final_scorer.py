"""
final_scorer.py — Stage 10 core

Combines four per-candidate signals from the preceding stages into one
final weighted score and selects the Golden Transcript.

Signal sources
--------------
Stage 6  alignment_quality_score (AQS)
    A single float [0,1] reflecting how reliably Whisper's ASR reference was
    forced-aligned to the audio.  A high AQS means the acoustic reference is
    trustworthy, so we amplify the Stage 7 acoustic contribution accordingly.
    AQS is NOT a per-candidate score — it applies equally to the whole batch
    as a confidence modifier on Stage 7.

Stage 7  acoustic_score  (per candidate, key "score")
    How closely each candidate matches the ASR reference at word/character
    level (WER + CER), penalised when multiple ASR models disagree.

Stage 8  consensus_score  (per candidate)
    Levenshtein-based agreement with the dominant cluster of candidates.
    Penalises outliers.

Stage 9  grammar_score  (per candidate)
    mT5 cross-entropy fluency: how probable / natural the text sounds to
    a multilingual language model.

Combination formula
-------------------
    aqs_mod       = AQS_FLOOR + (1 - AQS_FLOOR) * aqs
        → scales from AQS_FLOOR (weakest trust) to 1.0 (full trust)

    final_score = w7 * acoustic * aqs_mod
                + w8 * consensus
                + w9 * grammar

Default weights  :  w7=0.30  w8=0.40  w9=0.30  (sum = 1.0)
AQS_FLOOR        :  0.50  (even terrible alignment still contributes 50%)

Candidates whose acoustic data is unavailable receive acoustic_score=0.5
(neutral, neither rewarded nor penalised).
"""

# AQS modifier constants
AQS_FLOOR = 0.50   # minimum multiplier on Stage 7 when AQS = 0.0

DEFAULT_WEIGHTS = {
    "acoustic":  0.30,   # Stage 7
    "consensus": 0.40,   # Stage 8
    "grammar":   0.30,   # Stage 9
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _index_stage7(stage7_out: dict) -> dict:
    """
    Return a dict keyed by 0-based index → acoustic_score.

    Stage 7 uses 1-based index in its output dicts; we normalise here.
    Falls back gracefully to an empty dict when stage7_out is absent.
    """
    if not stage7_out:
        return {}
    mapping = {}
    for item in stage7_out.get("acoustic_scores", []):
        zero_based = item["index"] - 1          # Stage 7 is 1-indexed
        mapping[zero_based] = float(item["score"])
    return mapping


def _index_stage8(stage8_out: dict) -> dict:
    """Return 0-based index → consensus_score from Stage 8 output."""
    if not stage8_out:
        return {}
    return {
        item["index"]: float(item["consensus_score"])
        for item in stage8_out.get("scored_candidates", [])
    }


def _index_stage9(stage9_out: dict) -> dict:
    """Return 0-based index → grammar_score from Stage 9 output."""
    if not stage9_out:
        return {}
    return {
        item["index"]: float(item["grammar_score"])
        for item in stage9_out.get("scored_candidates", [])
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_final_scores(
    candidates:   list,
    stage6_out:   dict,
    stage7_out:   dict,
    stage8_out:   dict,
    stage9_out:   dict,
    weights:      dict | None = None,
) -> list[dict]:
    """
    Compute a final weighted score for every candidate.

    Parameters
    ----------
    candidates  : list of raw candidate strings (same order used in stages 7-9)
    stage6_out  : output of run_stage6()   — provides AQS
    stage7_out  : output of run_stage7()   — provides per-candidate acoustic score
    stage8_out  : output of run_stage8()   — provides per-candidate consensus score
    stage9_out  : output of run_stage9()   — provides per-candidate grammar score
    weights     : optional dict with keys "acoustic", "consensus", "grammar"
                  (values must sum to 1.0; defaults applied for missing keys)

    Returns
    -------
    list[dict] sorted by final_score descending — each dict:
    {
        "index":           int,    # 0-based
        "text":            str,
        "acoustic_score":  float,
        "consensus_score": float,
        "grammar_score":   float,
        "aqs":             float,
        "aqs_modifier":    float,
        "final_score":     float,
        "rank":            int,    # 1 = best
    }
    """
    # ── Resolve weights ────────────────────────────────────────────────────
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    w_acoustic  = float(w.get("acoustic",  DEFAULT_WEIGHTS["acoustic"]))
    w_consensus = float(w.get("consensus", DEFAULT_WEIGHTS["consensus"]))
    w_grammar   = float(w.get("grammar",   DEFAULT_WEIGHTS["grammar"]))

    # ── AQS modifier (global for this batch) ───────────────────────────────
    aqs = float(stage6_out.get("alignment_quality_score", 1.0)) if stage6_out else 1.0
    aqs_modifier = AQS_FLOOR + (1.0 - AQS_FLOOR) * aqs

    # ── Per-candidate score lookups ────────────────────────────────────────
    acoustic_map  = _index_stage7(stage7_out)
    consensus_map = _index_stage8(stage8_out)
    grammar_map   = _index_stage9(stage9_out)

    # ── Score each candidate ───────────────────────────────────────────────
    results = []
    for i, text in enumerate(candidates):
        acoustic  = acoustic_map.get(i,  0.5)   # 0.5 = neutral if unavailable
        consensus = consensus_map.get(i, 0.0)
        grammar   = grammar_map.get(i,  0.0)

        final = (
            w_acoustic  * acoustic  * aqs_modifier
            + w_consensus * consensus
            + w_grammar   * grammar
        )

        results.append({
            "index":           i,
            "text":            text,
            "acoustic_score":  round(acoustic,  4),
            "consensus_score": round(consensus, 4),
            "grammar_score":   round(grammar,   4),
            "aqs":             round(aqs,        4),
            "aqs_modifier":    round(aqs_modifier, 4),
            "final_score":     round(final,      4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    return results
