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

import torch
import torchaudio


# ---------- CONFIG ----------
cfg = {
    "models": [
        {"name": "whisper:medium", "device": "cuda"},
        {"name": "wav2vec2", "device": "cpu"},
    ]
}


# ---------- INPUT ----------
audio_path = "1.mp3"


# =========================================================
# STAGE 1 → PREPROCESS
# =========================================================
stage1 = run_stage1(audio_path)

waveform = stage1["waveform"]
sr = stage1["sample_rate"]


# ---------- SAVE CLEAN AUDIO ----------
clean_path = "stage1_clean.wav"
torchaudio.save(clean_path, waveform, sr)
print(f"\nSaved cleaned audio → {clean_path}")


# =========================================================
# STAGE 2 → LANGUAGE DETECTION
# =========================================================
lang, conf, probs, method = detect_language(waveform, sr)

print(f"Detected language: {lang} (confidence {conf:.2%}, method={method})")


# =========================================================
# STAGE 5 → ASR
# =========================================================
out = run_stage5(clean_path, cfg, language=lang)


# ---------- RESULTS ----------
print(out["reference_transcript"])
print(out["reference_transcripts"])
print(out["details"])
print(out["rss"], out["agreement"])