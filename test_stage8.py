"""
test_stage8.py
--------------
Quick standalone test for Stage 8 (Linguistic Grammar Scoring).
Runs against five hard-coded sample Arabic transcript options so you
can verify the model loads and scores correctly without running the
full pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import yaml

# ── Load config first so we can set env vars before importing transformers ──
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cache_dir = cfg.get("global", {}).get("cache_dir", "hf_cache")
os.environ["HF_HOME"]               = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"]        = cache_dir

from stage8.stage8_runner import run_stage8

s8_cfg = cfg.get("stage8", {})
device     = s8_cfg.get("device", "cpu")
model_name = s8_cfg.get("model",  "xlm-roberta-base")
max_samples = s8_cfg.get("max_samples", 20)

# ── Sample options (Arabic) ────────────────────────────────────
# Replace these with real transcript options from your Excel file.
excel_options = {
    "option_1": "السلام عليكم ورحمة الله وبركاته، أهلاً وسهلاً بكم في هذا البرنامج.",
    "option_2": "السلام عليكم ورحمه الله وبركاته اهلاً وسهلاً بكم في هذا البرنامج",
    "option_3": "السلام عليكم ورحمة الله، أهلاً وسهلاً بكم",
    "option_4": "سلام عليكم ورحمة الله وبركاته اهلاً وسهلاً بكم في البرنامج.",
    "option_5": "السلام عليكم ورحمة الله وبركاته أهلاً وسهلاً بكم فى هذا البرنامج",
}

language = "ar"

print("=" * 60)
print("  STAGE 8 TEST — Linguistic Grammar Scoring")
print(f"  Model  : {model_name}")
print(f"  Device : {device}")
print("=" * 60)

result = run_stage8(
    excel_options  = excel_options,
    language       = language,
    device         = device,
    model_name     = model_name,
    max_samples    = max_samples,
)

print(f"\n  Model used : {result['model_used']}")
print(f"  Ranked     : {' > '.join(result['ranked'])}")
print(f"  Best       : {result['best_option']}  (LGS={result['best_lgs']:.4f})")
print()

for opt_key in [f"option_{i}" for i in range(1, 6)]:
    s8 = result["options"].get(opt_key)
    if s8 is None:
        print(f"  {opt_key.upper()} : (empty / skipped)")
        continue
    tiebreak_note = "  [tie-break boost applied]" if s8.get("tiebreak") else ""
    print(f"  {opt_key.upper()} : PPPL={s8['pppl']:.2f}  "
          f"pppl_score={s8['pppl_score']:.4f}  "
          f"LGS={s8['lgs']:.4f}{tiebreak_note}")

print("\nDone.")
