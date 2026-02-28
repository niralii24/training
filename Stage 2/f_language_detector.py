import os
import torch
import torchaudio
from faster_whisper import WhisperModel

# ── Load Whisper models once at startup ───────────────────
print("Loading Whisper models...")
whisper_primary  = WhisperModel("small", device="cpu", compute_type="int8")
whisper_fallback = WhisperModel("medium", device="cpu", compute_type="int8")
print("Whisper models loaded ✅")

CONFIDENCE_THRESHOLD = 0.70


# ── Helper: save waveform temporarily ────────────────────
def _save_temp(waveform, sample_rate, path="temp_lang.wav"):
    torchaudio.save(path, waveform, sample_rate)
    return path


# ── Primary Detection ─────────────────────────────────────
def detect_with_primary(waveform, sample_rate):
    """
    Primary detection using Whisper small model.
    Fast and accurate for high-confidence cases.

    Returns language, confidence, probability dict.
    """
    temp = _save_temp(waveform, sample_rate)
    segments, info = whisper_primary.transcribe(temp)
    os.remove(temp)

    language   = info.language
    confidence = info.language_probability
    probs      = {language: confidence, "other": round(1.0 - confidence, 4)}

    return language, confidence, probs


# ── Fallback Detection ────────────────────────────────────
def detect_with_fallback(waveform, sample_rate):
    """
    Fallback using Whisper medium model on multiple audio chunks.
    More accurate but slower — only runs when primary is unsure.

    Strategy:
    - Split audio into 3 chunks (start, middle, end)
    - Run medium model on each chunk
    - Aggregate results by weighted voting
    """
    audio    = waveform.squeeze()
    total    = len(audio)
    chunk_sz = min(total // 3, 16000 * 10)  # max 10s per chunk

    chunks = [
        audio[0          : chunk_sz],                        # start
        audio[total//2   : total//2 + chunk_sz],             # middle
        audio[total-chunk_sz : total],                       # end
    ]

    vote_probs = {}

    for i, chunk in enumerate(chunks):
        if len(chunk) < 1600:  # skip if chunk too short
            continue

        chunk_waveform = chunk.unsqueeze(0)
        temp = _save_temp(chunk_waveform, sample_rate, f"temp_chunk_{i}.wav")

        try:
            segments, info = whisper_fallback.transcribe(temp)
            lang = info.language
            conf = info.language_probability

            # Weight middle chunk higher (most representative)
            weight = 1.5 if i == 1 else 1.0
            vote_probs[lang] = vote_probs.get(lang, 0.0) + (conf * weight)

        except Exception as e:
            print(f"  ⚠️ Chunk {i+1} failed: {e}")
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    if not vote_probs:
        return None, 0.0, {}

    # Normalize votes to get probabilities
    total_votes  = sum(vote_probs.values())
    norm_probs   = {k: round(v / total_votes, 4) for k, v in vote_probs.items()}
    top_language = max(norm_probs, key=norm_probs.get)
    top_conf     = norm_probs[top_language]

    return top_language, top_conf, norm_probs


# ── Ensemble ──────────────────────────────────────────────
def ensemble(primary_lang, primary_conf, primary_probs,
             fallback_lang, fallback_conf, fallback_probs):
    """
    Combines primary and fallback Whisper results.

    Weights:
    - Primary (small):  40%
    - Fallback (medium): 60%  ← medium is more accurate
    """
    PRIMARY_W  = 0.40
    FALLBACK_W = 0.60

    all_langs    = set(list(primary_probs.keys()) + list(fallback_probs.keys()))
    merged_probs = {}

    for lang in all_langs:
        p = primary_probs.get(lang, 0.0)
        f = fallback_probs.get(lang, 0.0)
        merged_probs[lang] = round(PRIMARY_W * p + FALLBACK_W * f, 4)

    final_lang = max(merged_probs, key=merged_probs.get)
    final_conf = merged_probs[final_lang]
    agreed     = primary_lang == fallback_lang

    print(f"  Primary  (small):  {primary_lang} ({primary_conf:.2%})")
    print(f"  Fallback (medium): {fallback_lang} ({fallback_conf:.2%})")
    print(f"  Agreement: {'✅ Yes' if agreed else '⚠️ No — using weighted ensemble'}")
    print(f"  Final:     {final_lang} ({final_conf:.2%})")

    return final_lang, final_conf, merged_probs


# ── Main Function ─────────────────────────────────────────
def detect_language(waveform, sample_rate):
    """
    Dual-mode language detection using Whisper only.

    Mode 1 — High confidence:
        Whisper small → confidence ≥ 0.70 → return result

    Mode 2 — Low confidence:
        Whisper small + Whisper medium (chunk voting) → ensemble

    Args:
        waveform:    audio tensor from Stage 1
        sample_rate: 16000

    Returns:
        language:   detected language code (e.g. 'ar', 'en')
        confidence: final confidence (0.0 to 1.0)
        probs:      full probability distribution dict
        method:     'whisper_primary' or 'whisper_ensemble'
    """
    print("\n--- Language Detection ---")

    # Step 1: Primary detection
    print("Running primary detection (Whisper small)...")
    p_lang, p_conf, p_probs = detect_with_primary(waveform, sample_rate)
    print(f"Primary result: {p_lang} ({p_conf:.2%})")

    # Step 2: High confidence → done
    if p_conf >= CONFIDENCE_THRESHOLD:
        print(f"Confidence above threshold ✅ → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary"

    # Step 3: Low confidence → run fallback ensemble
    print(f"⚠️ Low confidence ({p_conf:.2%}) → running fallback (Whisper medium)...")
    f_lang, f_conf, f_probs = detect_with_fallback(waveform, sample_rate)

    if not f_lang:
        print("⚠️ Fallback also failed → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary_fallback"

    # Step 4: Ensemble both
    print("Running ensemble...")
    final_lang, final_conf, final_probs = ensemble(
        p_lang, p_conf, p_probs,
        f_lang, f_conf, f_probs
    )

    return final_lang, final_conf, final_probs, "whisper_ensemble"