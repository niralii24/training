import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from acoustic_similarity import compute_acoustic_similarity


def run_stage7(candidates: list, stage5_out: dict, language: str = "en") -> dict:
    """
    Stage 7: Acoustic Similarity Scoring.

    Adapts to the Stage 5 output format from the other team member:
        stage5_out = {
            "reference_transcript":  consensus string,
            "reference_transcripts": list of ASR transcript strings,
            "rss":                   float,
            "agreement":             float,
            "details":               list of raw ASR output dicts,
        }

    Converts their format into the asr_results format that
    compute_acoustic_similarity expects, then scores all candidates.

    Args:
        candidates:  list of raw candidate transcript strings
        stage5_out:  output dict from run_stage5()
        language:    ISO language code

    Returns:
        dict with acoustic scores and best candidate
    """
    print("\n" + "=" * 60)
    print("STAGE 7: ACOUSTIC SIMILARITY")
    print("=" * 60)

    # ── Map their Stage 5 output to our asr_results format ─
    # Their "details" is a list of raw ASR outputs per model
    # Each has "text", "confidence", "entropy", "no_speech_prob"
    details = stage5_out.get("details", [])
    texts   = stage5_out.get("reference_transcripts", [])

    # Build asr_results list in our format
    asr_results = []
    for i, detail in enumerate(details):
        asr_results.append({
            "transcript":     detail.get("text", texts[i] if i < len(texts) else ""),
            "confidence":     detail.get("confidence", 0.5),
            "no_speech_prob": detail.get("no_speech_prob", 0.0),
            "entropy":        detail.get("entropy", 0.0),
            "model":          detail.get("model", f"model_{i+1}"),
        })

    # Fallback: if details is empty but reference_transcripts exists
    if not asr_results and texts:
        for i, text in enumerate(texts):
            asr_results.append({
                "transcript":     text,
                "confidence":     0.5,
                "no_speech_prob": 0.0,
                "entropy":        0.0,
                "model":          f"model_{i+1}",
            })

    print(f"  ASR references:  {len(asr_results)}")
    print(f"  RSS score:       {stage5_out.get('rss', 'N/A')}")
    print(f"  Agreement:       {stage5_out.get('agreement', 'N/A')}")
    print(f"  Consensus:       \"{stage5_out.get('reference_transcript', '')[:60]}\"")

    # ── Run acoustic similarity scoring ───────────────────
    scored = compute_acoustic_similarity(candidates, asr_results, language)

    return {
        "acoustic_scores": scored,
        "best_acoustic":   scored[0] if scored else None,
        "asr_results":     asr_results,
        "rss":             stage5_out.get("rss"),
        "agreement":       stage5_out.get("agreement"),
        "reference":       stage5_out.get("reference_transcript", ""),
    }