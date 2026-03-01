"""
main_pipeline.py
----------------
Full pipeline: Stage 1  ->  Stage 2  ->  Stage 5  ->  Stage 6

Stage 1  - Audio loading & cleaning (16 kHz mono WAV)
Stage 2  - Language detection
Stage 5  - ASR transcription (Whisper, multi-model consensus)
Stage 6  - Forced alignment scoring (WhisperX) against EACH transcript
            option stored in transcripts.xlsx for the given audio_id.
            Each option is scored and compared to identify the best match.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

CACHE = "C:/Users/Omega/hf_cache"
os.environ["HF_HOME"]               = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["XDG_CACHE_HOME"]        = CACHE

import torchaudio

from stage1.stage1_runner     import run_stage1
from stage2.language_detector import detect_language
from stage5.stage5_runner     import run_stage5
from stage6.stage6_runner     import run_stage6_excel_options

# ================================================================
# CONFIG
# ================================================================
AUDIO_ID   =  16            # matches the audio_id column in Excel
AUDIO_FILE = f"{AUDIO_ID}.mp3"  # e.g. "1.mp3"
EXCEL_FILE = "transcripts.xlsx"
CLEAN_WAV  = "stage1_clean.wav" # intermediate cleaned audio

CFG = {
    "models": [
        {"name": "whisper:medium", "device": "cuda"},
    ],
    "stage6": {
        "device":       "cuda",  
        "skip_gap_sec": 2.0,
    },
}


# ================================================================
# STAGE 1 - AUDIO LOADING & CLEANING
# ================================================================
print("\n" + "="*60)
print("  STAGE 1 - Audio Loading & Cleaning")
print("="*60)

stage1_out = run_stage1(AUDIO_FILE)
waveform   = stage1_out["waveform"]
sr         = stage1_out["sample_rate"]
meta       = stage1_out["metadata"]

torchaudio.save(CLEAN_WAV, waveform, sr, encoding="PCM_S", bits_per_sample=16)
print(f"  Saved cleaned audio -> {CLEAN_WAV}")
print(f"  Duration : {meta['trimmed_duration']:.2f} s  |  SNR : {meta['snr_db']:.1f} dB"
      f"  |  Speech ratio : {meta['speech_ratio']:.1%}")


# ================================================================
# STAGE 2 - LANGUAGE DETECTION
# ================================================================
print("\n" + "="*60)
print("  STAGE 2 - Language Detection")
print("="*60)

lang, conf, probs, method = detect_language(waveform, sr)
print(f"  Language : {lang}  (confidence {conf:.2%}, method={method})")


# ================================================================
# STAGE 5 - ASR TRANSCRIPTION
# ================================================================
print("\n" + "="*60)
print("  STAGE 5 - ASR Transcription")
print("="*60)

stage5_out = run_stage5(CLEAN_WAV, CFG, language=lang)

print(f"\n  Reference transcript:")
print(f"    {stage5_out['reference_transcript']}")
print(f"\n  RSS={stage5_out['rss']:.4f}  |  Agreement={stage5_out['agreement']:.4f}")


# ================================================================
# STAGE 6 - FORCED ALIGNMENT SCORING (WhisperX vs Excel options)
# ================================================================
print("\n" + "="*60)
print("  STAGE 6 - Forced Alignment Scoring (WhisperX x Excel)")
print("="*60)
print(f"  Excel file : {EXCEL_FILE}")
print(f"  audio_id   : {AUDIO_ID}")

s6_device   = CFG["stage6"]["device"]
s6_skip_gap = CFG["stage6"]["skip_gap_sec"]

stage6_result = run_stage6_excel_options(
    audio_wav    = CLEAN_WAV,
    excel_path   = EXCEL_FILE,
    audio_id     = AUDIO_ID,
    language     = lang,
    device       = s6_device,
    skip_gap_sec = s6_skip_gap,
)

# -- Per-option detailed output ------------------------------------
cell_refs    = stage6_result["cell_refs"]      # {"option_1": "D2", ...}
excel_row    = stage6_result["excel_row"]
correct_opt  = stage6_result["correct_option"]
lang_tag     = stage6_result["language_tag"]

print(f"\n  Excel row found   : {excel_row}")
print(f"  Language tag      : {lang_tag}")
print(f"  Correct option    : option_{correct_opt}" if correct_opt else "  Correct option    : (not specified)")
print(f"  Cell references   : { {k: v for k, v in cell_refs.items()} }")

for opt_key in [f"option_{i}" for i in range(1, 6)]:
    result = stage6_result["options"].get(opt_key)
    cell   = cell_refs.get(opt_key, "?")
    is_correct = bool(correct_opt and opt_key == f"option_{correct_opt}")
    rank_mark  = ""
    for rank_i, rk in enumerate(stage6_result["ranked"], 1):
        if rk == opt_key:
            rank_mark = f"  #{rank_i}"
            break

    print(f"\n  {'_'*56}")
    correct_label = "  <- CORRECT ANSWER" if is_correct else ""
    print(f"  {opt_key.upper()}  [Excel cell {cell}]{correct_label}{rank_mark}")
    print(f"  {'_'*56}")

    if result is None:
        print("    (empty or failed - skipped)")
        continue

    preview = result["transcript"][:120].replace("\n", " ")
    print(f"  Transcript  : {preview}{'...' if len(result['transcript'])>120 else ''}")
    print()
    print(f"  Word alignment ratio      : {result['word_alignment_ratio']:.3f}")
    dev = result["timing_deviation"]
    print(f"  Timing deviation          : mean={dev['mean']:.3f}s  std={dev['std']:.3f}s  "
          f"max={dev['max']:.3f}s  p90={dev['p90']:.3f}s")
    print(f"  Unaligned segment ratio   : {result['unaligned_segment_ratio']:.3f}")
    print(f"  Avg alignment confidence  : {result['avg_alignment_confidence']:.3f}")

    pc = result.get("phoneme_confidence")
    if pc:
        print(f"  Phoneme-level confidence  : mean={pc['mean']:.3f}  std={pc['std']:.3f}  "
              f"p10={pc['p10']:.3f}  p90={pc['p90']:.3f}  n_chars={pc['n_chars']}")
    else:
        print("  Phoneme-level confidence  : n/a")

    pps = result.get("punctuation_pause_score", {})
    if pps:
        prec_s = f"{pps['precision']:.3f}" if pps.get("precision") is not None else "n/a"
        rec_s  = f"{pps['recall']:.3f}"    if pps.get("recall")    is not None else "n/a"
        print(f"  Punct-pause alignment     : score={pps['score']:.3f}  "
              f"gap_ratio={pps['gap_ratio']:.2f}x  "
              f"coverage={pps['punct_coverage']:.2f}  "
              f"f1={pps['f1']:.3f}  "
              f"prec={prec_s}  rec={rec_s}  "
              f"n_punct={pps['n_punct_boundaries']}  "
              f"n_pauses={pps['n_audio_pauses']}")

    # Anomalies
    halls  = result["hallucinated_segments"]
    skips  = result["skipped_regions"]
    ovlps  = result["overlapping_misalignments"]

    print(f"\n  Hallucinated segments ({len(halls)}):")
    for h in halls:
        conf_s = f"{h['avg_confidence']:.4f}" if h["avg_confidence"] is not None else "n/a"
        print(f"    [{h['start']:.2f}s-{h['end']:.2f}s]  flags={h['flags']}  "
              f"conf={conf_s}  text={repr(h['text'][:60])}")

    print(f"\n  Skipped regions ({len(skips)}):")
    for r in skips:
        print(f"    [{r['start']:.2f}s-{r['end']:.2f}s]  gap={r['duration']:.2f}s")

    print(f"\n  Overlapping misalignments ({len(ovlps)}):")
    for o in ovlps[:5]:
        print(f"    '{o['word_a']}' <-> '{o['word_b']}'  overlap={o['overlap_sec']:.3f}s"
              f"  at {o['at_time']:.2f}s")
    if len(ovlps) > 5:
        print(f"    ... and {len(ovlps)-5} more")

    print(f"\n  * AQS (Alignment Quality Score) : {result['alignment_quality_score']:.4f}")


# -- Summary table -------------------------------------------------
print("\n\n" + "="*60)
print("  STAGE 6 SUMMARY  -  Ranked by Alignment Quality Score")
print("="*60)
print(f"  {'Rank':<6}{'Option':<12}{'Cell':<8}{'AQS':>8}  {'Note'}")
print(f"  {'_'*52}")

for rank_i, opt_key in enumerate(stage6_result["ranked"], 1):
    result = stage6_result["options"][opt_key]
    aqs    = result["alignment_quality_score"]
    cell   = cell_refs.get(opt_key, "?")
    note   = "<- correct answer" if (correct_opt and opt_key == f"option_{correct_opt}") else ""
    print(f"  {rank_i:<6}{opt_key:<12}{cell:<8}{aqs:>8.4f}  {note}")

for opt_key, result in stage6_result["options"].items():
    if result is None:
        cell = cell_refs.get(opt_key, "?")
        print(f"  {'--':<6}{opt_key:<12}{cell:<8}{'SKIP/ERR':>8}")

best = stage6_result["best_option"]
if best:
    best_aqs  = stage6_result["best_aqs"]
    best_cell = cell_refs.get(best, "?")
    print(f"\n  Best  -> {best}  (cell {best_cell})  AQS = {best_aqs:.4f}")
    if correct_opt:
        expected = f"option_{correct_opt}"
        if best == expected:
            print("  [OK] Best-scoring option IS the correct transcription.")
        else:
            exp_cell = cell_refs.get(expected, "?")
            print(f"  [!!] Best-scoring option differs from correct answer "
                  f"({expected}, cell {exp_cell}).")
