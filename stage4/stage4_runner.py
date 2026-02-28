import sys
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from candidate_filter import filter_candidates


def run_stage4(stage3_result):
    """
    Stage 4: Smart Candidate Filtering

    Takes Stage 3 output, runs all filter checks on normalized candidates,
    and returns only the valid ones.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: CANDIDATE FILTERING")
    print("=" * 60)

    language         = stage3_result["metadata"]["language"]
    audio_duration   = stage3_result["metadata"]["trimmed_duration"]
    candidates       = stage3_result["normalized_candidates"]

    valid_candidates, filtered_out = filter_candidates(
        candidates, audio_duration, language
    )

    print(f"\n  Valid candidates:   {len(valid_candidates)}")
    print(f"  Filtered out:       {len(filtered_out)}")
    print("=" * 60)

    stage3_result["valid_candidates"] = valid_candidates
    stage3_result["filtered_out"]     = filtered_out

    return stage3_result