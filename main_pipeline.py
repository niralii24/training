import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

CACHE = "C:/Users/Omega/hf_cache"
os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["XDG_CACHE_HOME"] = CACHE

# ---------- IMPORT PIPELINE ----------
from stage5.stage5_runner import run_stage5
from stage2.language_detector import detect_language
from stage1.stage1_runner import run_stage1
from stage7.stage7_runner import run_stage7

# ── Add your training folder to path for transcript_loader ─
TRAINING_DIR = r"C:\Users\Admin\Desktop\golden_transcription_system\training"
sys.path.append(TRAINING_DIR)
from transcript_loader import load_transcripts

import torch
import torchaudio

# ---------- CONFIG ----------
cfg = {
    "models": [
        {"name": "whisper:medium", "device": "cuda"},
    ]
}

# ---------- LOAD CANDIDATES FROM EXCEL ----------
EXCEL_PATH = r"C:\Users\Admin\Desktop\golden_transcription_system\training\transcripts.xlsx"
records    = load_transcripts(EXCEL_PATH)
print(f"Loaded {len(records)} records from Excel")

# ---------- RUN PIPELINE PER RECORD ----------
for record in records:
    audio_url     = record["audio_url"]
    raw_candidates = record["candidates"]
    audio_id      = record["audio_id"]
    correct       = record["correct_option"]

    # Use the local cached audio file
    filename   = os.path.basename(audio_url)
    audio_path = os.path.join(
        r"C:\Users\Admin\Desktop\golden_transcription_system\training\downloaded_audio",
        filename
    )

    if not os.path.exists(audio_path):
        print(f"⚠️  Skipping {audio_id} — file not found: {filename}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing record {audio_id}: {filename}")
    print(f"{'='*60}")

    # ── STAGE 1 → PREPROCESS ─────────────────────────────
    stage1   = run_stage1(audio_path)
    waveform = stage1["waveform"]
    sr       = stage1["sample_rate"]

    clean_path = "stage1_clean.wav"
    torchaudio.save(clean_path, waveform, sr)

    # ── STAGE 2 → LANGUAGE DETECTION ─────────────────────
    lang, conf, probs, method = detect_language(waveform, sr)
    print(f"Detected language: {lang} (confidence {conf:.2%}, method={method})")

    # ── STAGE 5 → ASR ─────────────────────────────────────
    out = run_stage5(clean_path, cfg)
    print(f"Reference: {out['reference_transcript'][:80]}")
    print(f"RSS: {out['rss']:.3f} | Agreement: {out['agreement']:.3f}")

    # ── STAGE 7 → ACOUSTIC SIMILARITY ────────────────────
    stage7 = run_stage7(
        candidates = raw_candidates,
        stage5_out = out,
        language   = lang
    )

    print(f"\n✅ Record {audio_id} done")
    print(f"   Best candidate: #{stage7['best_acoustic']['index']} "
          f"(score={stage7['best_acoustic']['score']:.4f})")
    print(f"   Correct answer: Option {correct}")
    print(f"\n   Full ranking:")
    for s in stage7["acoustic_scores"]:
        print(f"     Candidate {s['index']} — score={s['score']:.4f} | "
              f"WER={s['mean_wer']:.3f} | CER={s['mean_cer']:.3f}")