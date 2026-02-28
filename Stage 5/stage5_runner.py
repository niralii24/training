import sys
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from acoustic_scorer import run_acoustic_reference
from text_normalizer import normalize_all_candidates_light

def run_stage5(stage4_result):

    def run_stage5(stage4_result):
        print("\nDEBUG:")
        raw = stage4_result.get("raw_candidates", [])
        for i, c in enumerate(raw):
            print(f"  raw[{i}]: {c[:80]}")


    """
    Stage 5: Independent Acoustic Reference Scoring (Multi-Model).

    Takes Stage 4 output, scores all valid candidates against
    multi-model ASR references, returns best candidate with
    full scoring breakdown.
    """
    print("\n" + "=" * 60)
    print("STAGE 5: SCORING & SELECTION")
    print("=" * 60)

    waveform   = stage4_result["waveform"]
    sample_rate= stage4_result["sample_rate"]
    language   = stage4_result["language"]

    # Use valid candidates if any passed Stage 4,
    # otherwise fall back to all normalized candidates
    valid = stage4_result.get("valid_candidates", [])

    if valid:
        # valid_candidates is a list of (index, text) tuples
        candidates = [text for _, text in valid]
        print(f"  Using {len(candidates)} candidates that passed Stage 4")
    else:
        original_candidates = stage4_result["raw_candidates"]
        candidates_for_scoring = normalize_all_candidates_light(
            original_candidates, language
        )
        print(f"  ⚠️  No candidates passed Stage 4 — scoring all {len(candidates)} normalized")

    stage5 = run_acoustic_reference(
        waveform, sample_rate, candidates, language
    )

    # Merge into result dict
    stage4_result["asr_results"]       = stage5["asr_results"]
    stage4_result["stability_score"]   = stage5["stability_score"]
    stage4_result["stability_reason"]  = stage5["stability_reason"]
    stage4_result["reference"]         = stage5["reference"]
    stage4_result["scored_candidates"] = stage5["scored_candidates"]
    stage4_result["best_candidate"]    = stage5["best_candidate"]

    return stage4_result