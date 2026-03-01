import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from grammar_scorer import score_grammar


def run_stage9(
    candidates:  list,
    language:    str  = "en",
    mt5_model:   str  = "google/mt5-small",
    device:      str  = "cpu",
) -> dict:
    """
    Stage 9: Language / Grammar Quality Check.

    Uses Google's mT5 to score how grammatically natural each candidate
    transcript sounds.  Works for any of mT5's 101 supported languages
    (Arabic, English, French, etc.) with no language-specific config needed.

    Scoring method:
        - Feed each candidate through mT5 as a seq2seq fluency task
        - Measure cross-entropy loss (how unexpected the text is to the model)
        - Convert to fluency score:  raw = exp(−loss)
        - Min-max normalise across candidates so the best scores 1.0

    Args:
        candidates:  list of raw transcript strings (from Excel / Stage 8)
        language:    ISO-639-1 code — used for logging only, mT5 auto-detects
        mt5_model:   HuggingFace model ID (default: google/mt5-small)
        device:      "cpu" or "cuda"

    Returns:
        {
            "scored_candidates":  list[dict] sorted by grammar_score desc
            "best_candidate":     dict — highest grammar score
            "worst_candidate":    dict — lowest grammar score
        }
    """
    print("\n" + "=" * 60)
    print("STAGE 9: LANGUAGE / GRAMMAR QUALITY CHECK")
    print("=" * 60)
    print(f"  Candidates : {len(candidates)}")
    print(f"  Language   : {language}")
    print(f"  Model      : {mt5_model}")
    print(f"  Device     : {device}")

    if not candidates:
        print("  ⚠️  No candidates to score.")
        return {
            "scored_candidates": [],
            "best_candidate":    None,
            "worst_candidate":   None,
        }

    # ── Score ─────────────────────────────────────────────
    print()
    scored = score_grammar(
        candidates = candidates,
        language   = language,
        mt5_model  = mt5_model,
        device     = device,
    )

    # ── Report ────────────────────────────────────────────
    print("\n  Results (sorted by grammar score):")
    for s in scored:
        bar  = "█" * int(s["grammar_score"] * 20)
        print(
            f"    [{s['index']}] grammar={s['grammar_score']:.4f}  "
            f"loss={s['loss']:.4f}  raw={s['raw_score']:.4f}  "
            f"|{bar:<20}|"
        )
        print(f"         \"{s['text'][:80]}\"")

    best  = scored[0]
    worst = scored[-1]

    print(f"\n  Best  candidate : [{best['index']}]  "
          f"grammar_score={best['grammar_score']:.4f}")
    print(f"  Text            : \"{best['text'][:80]}\"")

    if len(scored) > 1:
        print(f"\n  Worst candidate : [{worst['index']}]  "
              f"grammar_score={worst['grammar_score']:.4f}")
        print(f"  Text            : \"{worst['text'][:80]}\"")

    print("=" * 60)

    return {
        "scored_candidates": scored,
        "best_candidate":    best,
        "worst_candidate":   worst,
    }
