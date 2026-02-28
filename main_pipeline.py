import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

CACHE = "C:/Users/Omega/hf_cache"
os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["XDG_CACHE_HOME"] = CACHE


# ---------- IMPORT PIPELINE ----------
from stage6.stage6_runner import run_stage6
from stage5.stage5_runner import run_stage5
from stage2.language_detector import detect_language
from stage1.stage1_runner import run_stage1

import torch
import torchaudio


# ---------- CONFIG ----------
cfg = {
    "models": [
        {"name": "whisper:medium", "device": "cuda"},
    ]
}


# ---------- INPUT ----------
audio_path = "2.mp3"


# =========================================================
# STAGE 1 → PREPROCESS
# =========================================================
stage1 = run_stage1(audio_path)

waveform = stage1["waveform"]
sr = stage1["sample_rate"]


# ---------- SAVE CLEAN AUDIO ----------
# Save as 16-bit PCM WAV (the universal format expected by librosa / Whisper).
# Saving the float32 tensor directly would produce a 32-bit float WAV that
# some decoders read back with subtle level differences and prepended silence.
clean_path = "stage1_clean.wav"
torchaudio.save(clean_path, waveform, sr, encoding="PCM_S", bits_per_sample=16)
print(f"\nSaved cleaned audio → {clean_path}")


# =========================================================
# STAGE 2 → LANGUAGE DETECTION (optional)
# =========================================================
# language detection can still be run for diagnostics if you like, but
# the transcription step no longer uses the hint.  Whisper will autodetect
# the language itself, which keeps the behavior consistent and avoids the
# "random" outputs you were seeing when a wrong hint was supplied.
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


# =========================================================
# STAGE 6 → FORCED ALIGNMENT SCORING (WhisperX)
# =========================================================
stage6_device = cfg.get("stage6", {}).get("device", "cuda")
stage6_skip   = cfg.get("stage6", {}).get("skip_gap_sec", 2.0)

align_out = run_stage6(
    audio_path    = clean_path,
    stage5_output = out,
    language      = lang,             # ISO-639-1 from stage 2
    device        = stage6_device,
    skip_gap_sec  = stage6_skip,
)

# ---------- ALIGNMENT RESULTS ----------
print("\n===== STAGE 6 – FORCED ALIGNMENT SCORING =====")
print(f"  Word alignment ratio      : {align_out['word_alignment_ratio']:.3f}")
dev = align_out['timing_deviation']
print(f"  Timing deviation          : mean={dev['mean']:.3f}s  std={dev['std']:.3f}s  "
      f"max={dev['max']:.3f}s  p90={dev['p90']:.3f}s")
print(f"  Unaligned segment ratio   : {align_out['unaligned_segment_ratio']:.3f}")
print(f"  Avg alignment confidence  : {align_out['avg_alignment_confidence']:.3f}")
if align_out['phoneme_confidence']:
    pc = align_out['phoneme_confidence']
    print(f"  Phoneme-level confidence  : mean={pc['mean']:.3f}  "
          f"std={pc['std']:.3f}  p10={pc['p10']:.3f}  p90={pc['p90']:.3f}  "
          f"n_chars={pc['n_chars']}")
else:
    print("  Phoneme-level confidence  : not available (model fallback)")

# Anomalies
print(f"\n  Hallucinated segments ({len(align_out['hallucinated_segments'])})  :")
for h in align_out['hallucinated_segments']:
    print(f"    [{h['start']:.2f}s – {h['end']:.2f}s]  flags={h['flags']}  "
          f"conf={h['avg_confidence']}  text={repr(h['text'][:60])}")

print(f"\n  Skipped regions ({len(align_out['skipped_regions'])})  :")
for r in align_out['skipped_regions']:
    print(f"    [{r['start']:.2f}s – {r['end']:.2f}s]  duration={r['duration']:.2f}s")

print(f"\n  Overlapping misalignments ({len(align_out['overlapping_misalignments'])})  :")
for o in align_out['overlapping_misalignments'][:10]:   # cap display at 10
    print(f"    '{o['word_a']}' / '{o['word_b']}'  overlap={o['overlap_sec']:.3f}s  "
          f"at {o['at_time']:.2f}s  (segs {o['segment_a']}/{o['segment_b']})")

print(f"\n  Alignment Quality Score (AQS) : {align_out['alignment_quality_score']:.4f}")