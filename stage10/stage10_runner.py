"""
stage10/stage10_runner.py
-------------------------
Stage 10: Final Score Combination → Golden Transcript

Merges the four scoring signals produced by stages 6–9 into a single
ranked list.  The top-ranked candidate is the Golden Transcript.

    Stage 6  alignment_quality_score  → AQS modifier on acoustic signal
    Stage 7  acoustic_score           → how well candidate matches ASR
    Stage 8  consensus_score          → agreement with candidate cluster
    Stage 9  grammar_score            → mT5 language-model fluency

Default weights:
    acoustic   = 0.30  (Stage 7, scaled by AQS from Stage 6)
    consensus  = 0.40  (Stage 8)
    grammar    = 0.30  (Stage 9)
"""

from .final_scorer import compute_final_scores, DEFAULT_WEIGHTS


def run_stage10(
    candidates:  list,
    stage6_out:  dict,
    stage7_out:  dict,
    stage8_out:  dict,
    stage9_out:  dict,
    weights:     dict | None = None,
) -> dict:
    """
    Stage 10: Final Score Combination — selects the Golden Transcript.

    Parameters
    ----------
    candidates  : list of raw candidate strings (same list fed to stages 7-9)
    stage6_out  : output of run_stage6()
    stage7_out  : output of run_stage7()   (pass None if stage 7 was skipped)
    stage8_out  : output of run_stage8()
    stage9_out  : output of run_stage9()
    weights     : optional override dict, e.g. {"acoustic": 0.2, "consensus": 0.5,
                  "grammar": 0.3}  — values should sum to 1.0

    Returns
    -------
    {
        "golden_transcript" : str          — text of the top-ranked candidate
        "golden_index"      : int          — 0-based index of the winner
        "final_scores"      : list[dict]   — all candidates, sorted best → worst
        "weights_used"      : dict         — weights applied
        "aqs"               : float        — alignment quality score from Stage 6
    }
    """
    print("\n" + "=" * 60)
    print("STAGE 10: FINAL SCORE COMBINATION")
    print("=" * 60)
    print(f"  Candidates : {len(candidates)}")

    if not candidates:
        print("  ⚠️  No candidates to score.")
        return {
            "golden_transcript": "",
            "golden_index":      -1,
            "final_scores":      [],
            "weights_used":      weights or DEFAULT_WEIGHTS,
            "aqs":               0.0,
        }

    # ── Resolve and display weights ────────────────────────────────────────
    w_used = dict(DEFAULT_WEIGHTS)
    if weights:
        w_used.update(weights)

    aqs = float(stage6_out.get("alignment_quality_score", 1.0)) if stage6_out else 1.0
    aqs_mod = 0.50 + 0.50 * aqs

    print(f"\n  Weights:")
    print(f"    Acoustic  (Stage 7, w={w_used['acoustic']:.2f})  "
          f"× AQS modifier {aqs_mod:.4f}  (AQS={aqs:.4f})")
    print(f"    Consensus (Stage 8, w={w_used['consensus']:.2f})")
    print(f"    Grammar   (Stage 9, w={w_used['grammar']:.2f})")

    # ── Compute scores ─────────────────────────────────────────────────────
    final_scores = compute_final_scores(
        candidates  = candidates,
        stage6_out  = stage6_out,
        stage7_out  = stage7_out,
        stage8_out  = stage8_out,
        stage9_out  = stage9_out,
        weights     = weights,
    )

    # ── Print ranking table ────────────────────────────────────────────────
    print(f"\n  {'Rank':<5}  {'Idx':<4}  {'Final':>7}  "
          f"{'Acoustic':>9}  {'Consensus':>9}  {'Grammar':>8}  Text")
    print("  " + "-" * 90)

    for s in final_scores:
        crown = " 🏆" if s["rank"] == 1 else ""
        print(
            f"  #{s['rank']:<4}  [{s['index']}]   "
            f"{s['final_score']:>7.4f}  "
            f"{s['acoustic_score']:>9.4f}  "
            f"{s['consensus_score']:>9.4f}  "
            f"{s['grammar_score']:>8.4f}  "
            f"\"{s['text'][:55]}\"{crown}"
        )

    # ── Winner ─────────────────────────────────────────────────────────────
    winner = final_scores[0]
    print("\n" + "=" * 60)
    print("  🏆  GOLDEN TRANSCRIPT")
    print("=" * 60)
    print(f"  Candidate index : [{winner['index']}]")
    print(f"  Final score     : {winner['final_score']:.4f}")
    print(f"    Acoustic      : {winner['acoustic_score']:.4f}  "
          f"(× AQS mod {winner['aqs_modifier']:.4f})")
    print(f"    Consensus     : {winner['consensus_score']:.4f}")
    print(f"    Grammar       : {winner['grammar_score']:.4f}")
    print(f"\n  Text:")
    # Print full text, wrapped at 70 chars
    text = winner["text"]
    for start in range(0, len(text), 70):
        print(f"    {text[start:start + 70]}")
    print("=" * 60)

    return {
        "golden_transcript": winner["text"],
        "golden_index":      winner["index"],
        "final_scores":      final_scores,
        "weights_used":      w_used,
        "aqs":               aqs,
    }
