import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

CACHE = os.environ.get("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache"))
os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["XDG_CACHE_HOME"] = CACHE
os.makedirs(CACHE, exist_ok=True)


# ---------- IMPORT PIPELINE ----------
from stage1.stage1_runner   import run_stage1
from stage2.language_detector import detect_language
from stage3.stage3_runner   import run_stage3
from stage4.stage4_runner   import run_stage4
from stage5.stage5_runner   import run_stage5
from stage6.stage6_runner   import run_stage6
from stage7.stage7_runner   import run_stage7
from stage8.stage8_runner   import run_stage8
from stage9.stage9_runner   import run_stage9
from stage10.stage10_runner import run_stage10

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

# Enrich stage1 metadata with language info so it can feed into Stage 3
stage1["metadata"]["language"]            = lang
stage1["metadata"]["language_confidence"] = conf
stage1["metadata"]["language_probs"]      = probs
stage1["metadata"]["language_method"]     = method


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


# =========================================================
# STAGE 3 → LANGUAGE-AWARE TEXT NORMALIZATION
# =========================================================
# Normalizes all candidate transcripts (punctuation, diacritics, casing, etc.)
# according to the detected language.
raw_candidates = out.get("reference_transcripts") or [out["reference_transcript"]]

stage3 = run_stage3(stage1, raw_candidates)

print("\n===== STAGE 3 – TEXT NORMALIZATION =====")
for i, text in enumerate(stage3["normalized_candidates"]):
    print(f"  [{i}] {text[:100]}")


# =========================================================
# STAGE 4 → SMART CANDIDATE FILTERING
# =========================================================
# Removes candidates that are too short/long, mostly non-target-language,
# repetitive, or otherwise low-quality.
stage4 = run_stage4(stage3)

print("\n===== STAGE 4 – CANDIDATE FILTERING =====")
print(f"  Valid   : {len(stage4['valid_candidates'])}")
print(f"  Filtered: {len(stage4['filtered_out'])}")
for f in stage4["filtered_out"]:
    print(f"  [dropped] {f[:80]}")

# Use filtered+normalized candidates for all downstream stages.
# Fall back to all normalized candidates if the filter removes everything.
candidates_for_scoring = stage4["valid_candidates"] or stage3["normalized_candidates"]


# =========================================================
# STAGE 7 → ACOUSTIC SIMILARITY SCORING
# =========================================================
# Compares each candidate against the ASR reference(s) from Stage 5 using
# WER + CER, weighted by ASR model agreement variance.

stage7 = run_stage7(
    candidates = candidates_for_scoring,
    stage5_out = out,
    language   = lang,
)

print("\n===== STAGE 7 – ACOUSTIC SIMILARITY =====")
for s in stage7["acoustic_scores"]:
    print(f"  [{s['index']}] acoustic={s['score']:.4f}  "
          f"WER={s['mean_wer']:.3f}  CER={s['mean_cer']:.3f}")
best7 = stage7.get("best_acoustic")
if best7:
    print(f"\n  Best candidate : [{best7['index']}] score={best7['score']:.4f}")


# =========================================================
# STAGE 8 → CROSS-TRANSCRIPT CONSENSUS MODELING
# =========================================================
# In production: pass the raw human-annotated candidate list loaded from Excel.
# In single-file test mode: the ASR reference transcripts from Stage 5 are used
# as stand-in candidates so the consensus logic can still be exercised.
candidates_for_consensus = candidates_for_scoring

stage8 = run_stage8(
    candidates = candidates_for_consensus,
    language   = lang,
)

# ---------- CONSENSUS RESULTS ----------
print("\n===== STAGE 8 – CROSS-TRANSCRIPT CONSENSUS =====")
print(f"  Clusters found  : {stage8['cluster_count']}")
print(f"  Dominant cluster: {sorted(stage8['dominant_cluster'])}")
print("\n  Candidate scores:")
for s in stage8["scored_candidates"]:
    tag = "dominant" if s["in_dominant_cluster"] else "outlier"
    print(f"    [{s['index']}] consensus={s['consensus_score']:.4f}  "
          f"cluster_sim={s['avg_cluster_sim']:.4f}  "
          f"global_avg={s['global_avg_sim']:.4f}  ({tag})")
best = stage8["best_candidate"]
print(f"\n  Best candidate  : [{best['index']}] score={best['consensus_score']:.4f}")
print(f"  Text            : \"{best['text'][:80]}\"")


# =========================================================
# STAGE 9 → LANGUAGE / GRAMMAR QUALITY CHECK (mT5)
# =========================================================
# In production: pass the raw human-annotated candidate list loaded from Excel.
# In single-file test mode: Stage 5 ASR references are used as stand-in candidates.
candidates_for_grammar = candidates_for_consensus

stage9 = run_stage9(
    candidates = candidates_for_grammar,
    language   = lang,
    mt5_model  = "google/mt5-small",   # swap to mt5-base/large for better quality
    device     = "cpu",                # swap to "cuda" if GPU available
)

# ---------- GRAMMAR RESULTS ----------
print("\n===== STAGE 9 – GRAMMAR / FLUENCY QUALITY =====")
for s in stage9["scored_candidates"]:
    print(f"  [{s['index']}] grammar={s['grammar_score']:.4f}  "
          f"loss={s['loss']:.4f}  raw={s['raw_score']:.4f}")
best9 = stage9["best_candidate"]
print(f"\n  Best candidate  : [{best9['index']}] grammar_score={best9['grammar_score']:.4f}")
print(f"  Text            : \"{best9['text'][:80]}\"")


# =========================================================
# STAGE 10 → FINAL SCORE COMBINATION → GOLDEN TRANSCRIPT
# =========================================================
stage10 = run_stage10(
    candidates = candidates_for_consensus,
    stage6_out = align_out,
    stage7_out = stage7,
    stage8_out = stage8,
    stage9_out = stage9,
    # Optional: override default weights
    # weights = {"acoustic": 0.30, "consensus": 0.40, "grammar": 0.30},
)

print(f"\n  GOLDEN TRANSCRIPT → \"{stage10['golden_transcript'][:100]}\"")
print(f"  Final score : {stage10['final_scores'][0]['final_score']:.4f}")