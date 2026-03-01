"""
main_pipeline.py
----------------
Full pipeline: Stage 1  ->  Stage 2  ->  Stage 5  ->  Stage 6 -> Stage 7 -> Stage 8

Stage 1  - Audio loading & cleaning (16 kHz mono WAV)
Stage 2  - Language detection
Stage 5  - ASR transcription (Whisper, multi-model consensus)
Stage 6  - Forced alignment scoring (WhisperX) against EACH transcript
            option stored in transcripts.xlsx for the given audio_id.
Stage 7  - Transcript similarity metrics
Stage 8  - Linguistic grammar scoring (XLM-RoBERTa)

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import torchaudio
import yaml

from stage1.stage1_runner     import run_stage1
from stage2.language_detector import detect_language
from stage5.stage5_runner     import run_stage5
from stage6.stage6_runner     import run_stage6_excel_options
from stage7.stage7_runner     import run_stage7
from stage8.stage8_runner     import run_stage8, compute_final_score
from write_results_to_excel   import add_or_update_columns


# ================================================================
# LOAD & VALIDATE CONFIG
# ================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """Load and validate YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["global", "paths", "stage5", "stage6", "final_score"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section in config: '{section}'")
    
    return cfg


def setup_environment(cfg: dict):
    """Configure environment variables from config."""
    cache_dir = cfg["global"]["cache_dir"]
    os.environ["HF_HOME"]               = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["XDG_CACHE_HOME"]        = cache_dir


def resolve_device(device_str: str, default: str = "cuda") -> str:
    """Resolve 'auto' device specification."""
    if device_str == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ================================================================
# PIPELINE FUNCTION  (runs all stages for one audio ID)
# ================================================================
def process_one(audio_id: int, cfg: dict):
    """
    Process a single audio file through the full pipeline.
    
    Args:
        audio_id: Numeric ID of the audio file
        cfg: Configuration dictionary from config.yaml
    
    Returns:
        True if successful, False if file not found and should stop batch
    """
    audio_dir    = cfg["paths"]["audio_dir"]
    excel_file   = cfg["paths"]["excel_file"]
    clean_wav    = cfg["paths"]["clean_wav"]
    
    audio_file = os.path.join(audio_dir, f"{audio_id}.mp3")
    if not os.path.exists(audio_file):
        print(f"\n[SKIP] {audio_file} not found — stopping batch.")
        return False   # signal caller to stop

    print("\n" + "="*60)
    print(f"  PROCESSING AUDIO ID: {audio_id}   ({audio_file})")
    print("="*60)

    # ── STAGE 1 ────────────────────────────────────────────────
    print("\n[1/8] Loading and standardizing audio...")
    stage1_out = run_stage1(audio_file)
    waveform   = stage1_out["waveform"]
    sr         = stage1_out["sample_rate"]
    meta       = stage1_out["metadata"]

    torchaudio.save(clean_wav, waveform, sr, encoding="PCM_S", bits_per_sample=16)
    print(f"  Saved cleaned audio -> {clean_wav}")
    print(f"  Duration : {meta['trimmed_duration']:.2f} s  |  SNR : {meta['snr_db']:.1f} dB"
          f"  |  Speech ratio : {meta['speech_ratio']:.1%}")

    # ── STAGE 2 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 2 - Language Detection")
    print("="*60)
    lang, conf, probs, method = detect_language(waveform, sr)
    print(f"  Language : {lang}  (confidence {conf:.2%}, method={method})")

    # ── STAGE 5 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 5 - ASR Transcription")
    print("="*60)
    stage5_out = run_stage5(clean_wav, cfg["stage5"], language=lang)
    print(f"\n  Reference transcript:")
    print(f"    {stage5_out['reference_transcript']}")
    print(f"\n  RSS={stage5_out['rss']:.4f}  |  Agreement={stage5_out['agreement']:.4f}")

    # ── STAGE 6 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 6 - Forced Alignment Scoring (WhisperX x Excel)")
    print("="*60)
    s6_device   = resolve_device(cfg["stage6"]["device"])
    s6_skip_gap = cfg["stage6"]["skip_gap_sec"]

    stage6_result = run_stage6_excel_options(
        audio_wav    = clean_wav,
        excel_path   = excel_file,
        audio_id     = audio_id,
        language     = lang,
        device       = s6_device,
        skip_gap_sec = s6_skip_gap,
    )

    cell_refs   = stage6_result["cell_refs"]
    excel_row   = stage6_result["excel_row"]
    correct_opt = stage6_result["correct_option"]
    lang_tag    = stage6_result["language_tag"]

    print(f"\n  Excel row found   : {excel_row}")
    print(f"  Language tag      : {lang_tag}")
    print(f"  Correct option    : option_{correct_opt}" if correct_opt else "  Correct option    : (not specified)")

    for opt_key in [f"option_{i}" for i in range(1, 6)]:
        result = stage6_result["options"].get(opt_key)
        cell   = cell_refs.get(opt_key, "?")
        print(f"\n  {'_'*56}")
        print(f"  {opt_key.upper()}  [Excel cell {cell}]")
        print(f"  {'_'*56}")
        if result is None:
            print("    (empty or failed - skipped)")
            continue
        preview = result["transcript"][:120].replace("\n", " ")
        print(f"  Transcript  : {preview}{'...' if len(result['transcript'])>120 else ''}")
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
        pps = result.get("punctuation_pause_score", {})
        if pps:
            prec_s = f"{pps['precision']:.3f}" if pps.get("precision") is not None else "n/a"
            rec_s  = f"{pps['recall']:.3f}"    if pps.get("recall")    is not None else "n/a"
            print(f"  Punct-pause alignment     : score={pps['score']:.3f}  "
                  f"gap_ratio={pps['gap_ratio']:.2f}x  "
                  f"f1={pps['f1']:.3f}  prec={prec_s}  rec={rec_s}")
        halls = result["hallucinated_segments"]
        skips = result["skipped_regions"]
        ovlps = result["overlapping_misalignments"]
        print(f"  Hallucinated segs: {len(halls)}  Skipped regions: {len(skips)}  Overlaps: {len(ovlps)}")
        print(f"\n  * AQS : {result['alignment_quality_score']:.4f}")

    # Stage 6 summary
    print("\n\n" + "="*60)
    print("  STAGE 6 SUMMARY  -  Ranked by AQS")
    print("="*60)
    for rank_i, opt_key in enumerate(stage6_result["ranked"], 1):
        result = stage6_result["options"][opt_key]
        aqs    = result["alignment_quality_score"]
        cell   = cell_refs.get(opt_key, "?")
        print(f"  #{rank_i}  {opt_key:<12}  cell={cell:<4}  AQS={aqs:.4f}")

    # ── STAGE 7 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 7 - Transcript Similarity Metrics")
    print("="*60)
    _excel_options = {
        k: (v["transcript"] if v is not None else "")
        for k, v in stage6_result["options"].items()
    }
    _asr_refs = stage5_out["reference_transcripts"]
    stage7_result = run_stage7(
        excel_options  = _excel_options,
        asr_references = _asr_refs,
        language       = lang,
        correct_option = correct_opt,
    )
    for opt_key in [f"option_{i}" for i in range(1, 6)]:
        s7   = stage7_result["options"].get(opt_key)
        cell = cell_refs.get(opt_key, "?")
        print(f"\n  {opt_key.upper()}  [cell {cell}]")
        if s7 is None:
            print("    (empty or skipped)")
            continue
        print(f"  Mean WER={s7['mean_wer']:.4f}  Mean CER={s7['mean_cer']:.4f}  "
              f"Fuzzy={s7['fuzzy_similarity']:.4f}  TSS={s7['tss']:.4f}")

    # ── STAGE 8 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 8 - Linguistic Grammar Scoring (XLM-RoBERTa)")
    print("="*60)
    s8_device = resolve_device(cfg["stage8"]["device"])
    stage8_result = run_stage8(
        excel_options  = _excel_options,
        language       = lang,
        device         = s8_device,
        correct_option = correct_opt,
    )
    print(f"  Model used : {stage8_result['model_used']}")
    for opt_key in [f"option_{i}" for i in range(1, 6)]:
        s8   = stage8_result["options"].get(opt_key)
        cell = cell_refs.get(opt_key, "?")
        print(f"\n  {opt_key.upper()}  [cell {cell}]")
        if s8 is None:
            print("    (empty or skipped)")
            continue
        print(f"  PPPL={s8['pppl']:.2f}  pppl_score={s8['pppl_score']:.4f}  "
              f"LGS={s8['lgs']:.4f}")
    print(f"\n  Stage 8 ranked: {' > '.join(stage8_result['ranked'])}")

    # ── FINAL COMBINED RANKING ─────────────────────────────────
    print("\n\n" + "="*60)
    print("  FINAL COMBINED RANKING")
    weights = cfg["final_score"]["weights"]
    w_aqs = weights["aqs"]
    w_tss = weights["tss"]
    w_lgs = weights["lgs"]
    print(f"  AQS×{w_aqs:.2f}  +  TSS×{w_tss:.2f}  +  LGS×{w_lgs:.2f}")
    print("="*60)
    print(f"  {'Rank':<6}{'Option':<12}{'AQS':>7}  {'TSS':>7}  {'LGS':>7}  {'FINAL':>9}")
    print(f"  {'_'*60}")

    final_scores = {}
    for opt_key in [f"option_{i}" for i in range(1, 6)]:
        s6_res = stage6_result["options"].get(opt_key)
        s7_res = stage7_result["options"].get(opt_key)
        s8_res = stage8_result["options"].get(opt_key)
        if s6_res is None and s7_res is None and s8_res is None:
            continue
        aqs_val = s6_res["alignment_quality_score"] if s6_res else 0.0
        tss_val = s7_res["tss"] if s7_res else 0.0
        lgs_val = s8_res["lgs"] if s8_res else 0.0
        final_scores[opt_key] = compute_final_score(aqs_val, tss_val, lgs_val, weights)

    final_ranked = sorted(final_scores, key=final_scores.get, reverse=True)  # type: ignore[arg-type]
    for rank_i, opt_key in enumerate(final_ranked, 1):
        s6_res  = stage6_result["options"].get(opt_key)
        s7_res  = stage7_result["options"].get(opt_key)
        s8_res  = stage8_result["options"].get(opt_key)
        aqs_val = s6_res["alignment_quality_score"] if s6_res else 0.0
        tss_val = s7_res["tss"] if s7_res else 0.0
        lgs_val = s8_res["lgs"] if s8_res else 0.0
        fin     = final_scores[opt_key]
        print(f"  {rank_i:<6}{opt_key:<12}{aqs_val:>7.4f}  {tss_val:>7.4f}  {lgs_val:>7.4f}  {fin:>9.4f}")

    winner = final_ranked[0] if final_ranked else None
    if winner:
        print(f"\n  DETECTED -> {winner}  FINAL SCORE = {final_scores[winner]:.4f}")

    # ── Write results to Excel immediately ────────────────────
    row_result = {}
    for opt_key in [f"option_{i}" for i in range(1, 6)]:
        s6_res = stage6_result["options"].get(opt_key)
        s7_res = stage7_result["options"].get(opt_key)
        s8_res = stage8_result["options"].get(opt_key)
        row_result[opt_key] = {
            "aqs":         s6_res["alignment_quality_score"] if s6_res else None,
            "wer":         s7_res["mean_wer"]               if s7_res else None,
            "cer":         s7_res["mean_cer"]               if s7_res else None,
            "tss":         s7_res["tss"]                    if s7_res else None,
            "lgs":         s8_res["lgs"]                    if s8_res else None,
            "final_score": final_scores.get(opt_key),
        }
    detected_opt = int(winner.split("_")[1]) if winner else None
    row_result["detected_option"] = detected_opt
    print(f"\n  Writing results for audio_id={audio_id} to {excel_file} …")
    add_or_update_columns(excel_file, {str(audio_id): row_result})

    return True   # success


# ================================================================
# BATCH LOOP  -  process 1.mp3, 2.mp3, … until no file found
# ================================================================
if __name__ == "__main__":
    import torch as _torch
    
    # Load configuration
    cfg = load_config("config.yaml")
    setup_environment(cfg)
    
    batch_cfg = cfg.get("batch", {})
    audio_id = batch_cfg.get("start_id", 1)
    clear_cuda = batch_cfg.get("clear_cuda_cache", True)
    stop_on_missing = batch_cfg.get("stop_on_missing_file", True)
    
    start_id = audio_id
    while True:
        try:
            ok = process_one(audio_id, cfg)
        except Exception as exc:
            print(f"\n[ERROR] audio_id={audio_id} failed: {exc}")
            import traceback
            traceback.print_exc()
            ok = True  
        
        if clear_cuda and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        
        if not ok:
            if stop_on_missing:
                break
        
        audio_id += 1

    print(f"\n{'='*60}")
    print(f"  Batch complete — processed {audio_id - start_id} audio file(s).")
    print(f"{'='*60}")