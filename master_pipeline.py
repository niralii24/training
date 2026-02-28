import os, sys, requests, tempfile

# ── Add stage folders to path ─────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
STAGE1_DIR = os.path.join(BASE_DIR, "Stage 1")
STAGE2_DIR = os.path.join(BASE_DIR, "Stage 2")
STAGE3_DIR = os.path.join(BASE_DIR, "Stage 3")
STAGE4_DIR = os.path.join(BASE_DIR, "Stage 4")
STAGE5_DIR = os.path.join(BASE_DIR, "Stage 5")

sys.path.append(STAGE1_DIR)
sys.path.append(STAGE2_DIR)
sys.path.append(STAGE3_DIR)
sys.path.append(STAGE4_DIR)
sys.path.append(STAGE5_DIR)

# ── Import stage runners ──────────────────────────────────
from stage1_runner  import run_stage1
from f_language_detector import detect_language
from text_normalizer import normalize_all_candidates
from transcript_loader import load_transcripts, get_language_code
from stage4_runner import run_stage4
from stage5_runner import run_stage5

def run_pipeline(audio_file, candidates):
    """
    Master pipeline — runs all stages in order.
    Each stage receives and builds on the previous output.
    """
    print("\n" + "=" * 60)
    print("MASTER PIPELINE STARTED")
    print(f"Audio: {os.path.basename(audio_file)}")
    print(f"Candidates: {len(candidates)}")
    print("=" * 60)

    # ── STAGE 1: Audio Loading & Analysis ─────────────────
    print("\n>>> STAGE 1: Audio Loading & Analysis")
    stage1 = run_stage1(audio_file)

    waveform    = stage1["waveform"]
    sample_rate = stage1["sample_rate"]
    metadata    = stage1["metadata"]

    # ── STAGE 2: Language Detection ────────────────────────
    print("\n>>> STAGE 2: Language Detection")
    language, confidence, probs, method = detect_language(
        waveform, sample_rate
    )
    metadata["language"]            = language
    metadata["language_confidence"] = confidence
    metadata["language_probs"]      = probs
    metadata["language_method"]     = method


    # ── STAGE 3: Text Normalization ────────────────────────
    print("\n>>> STAGE 3: Text Normalization")
    normalized = normalize_all_candidates(candidates, language)

    # ── Build stage3 result dict for Stage 4 ──────────────
    stage3_result = {
    "audio_file":            audio_file,
    "waveform":              waveform,
    "sample_rate":           sample_rate,
    "metadata":              metadata,
    "language":              language,
    "raw_candidates":        candidates,       
    "candidates":            candidates,
    "normalized_candidates": normalized,
    }

    # ── STAGE 4: Candidate Filtering ──────────────────────
    print("\n>>> STAGE 4: Candidate Filtering")
    stage4 = run_stage4(stage3_result)

    # ── STAGE 5: Coming soon ───────────────────────────────
    print("\n>>> STAGE 5: Scoring & Selection")
    stage5 = run_stage5(stage4)

    # ── Build final result dict ────────────────────────────
    return {
    "audio_file":        audio_file,
    "language":          language,        # ← make sure this is here
    "metadata":          metadata,
    "normalized_candidates": normalized,
    "valid_candidates":  stage4["valid_candidates"],
    "filtered_out":      stage4["filtered_out"],
    "best_candidate":    stage5["best_candidate"],
    "scored_candidates": stage5["scored_candidates"],
    "stability_score":   stage5["stability_score"],
}

# ── Run on all files in audio_input folder ────────────────
if __name__ == "__main__":

    EXCEL_PATH   = r"C:\Users\Admin\Desktop\golden_transcription_system\training\transcripts.xlsx"
    CACHE_FOLDER = r"C:\Users\Admin\Desktop\golden_transcription_system\training\downloaded_audio"

    os.makedirs(CACHE_FOLDER, exist_ok=True)

    records = load_transcripts(EXCEL_PATH)
    print(f"\nFound {len(records)} records in Excel")

    for record in records:
        audio_url  = record["audio_url"]
        candidates = record["candidates"]
        audio_id   = record["audio_id"]

        filename   = os.path.basename(audio_url)
        audio_file = os.path.join(CACHE_FOLDER, filename)

        # ── Download if not already cached ────────────────
        if not os.path.exists(audio_file):
            print(f"\n⬇️  Downloading {filename}...")
            try:
                response = requests.get(audio_url, timeout=30)
                response.raise_for_status()
                with open(audio_file, "wb") as f:
                    f.write(response.content)
                print(f"   ✅ Saved to cache: {filename}")
            except Exception as e:
                print(f"   ❌ Failed to download {audio_id}: {e}")
                continue
        else:
            print(f"\n📁 Using cached file: {filename}")

        result = run_pipeline(audio_file, candidates)

        print(f"\n✅ Done: Record {audio_id}")
        print(f"   Language:    {result['language']}")
        print(f"   Candidates:  {len(result['normalized_candidates'])} normalized")
        print(f"   Correct ans: Option {record['correct_option']}")