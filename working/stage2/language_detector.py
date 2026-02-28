import os
import torch
import torchaudio
# WhisperModel is imported lazily once cache locations are configured

# ensure cache dirs exist before any HF download attempts
# prefer environment variables already set (main_pipeline sets them)
CACHE = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["TRANSFORMERS_CACHE"] = CACHE
os.environ.setdefault("XDG_CACHE_HOME", CACHE)

try:
    os.makedirs(CACHE, exist_ok=True)
except Exception:
    pass

# models will be loaded lazily on first use; this avoids spending time
# during import and also makes it easier to catch download errors gracefully
whisper_primary = None
whisper_fallback = None

CONFIDENCE_THRESHOLD = 0.70  # primary confidence threshold


# ── Helper: save waveform temporarily ────────────────────
def _save_temp(waveform, sample_rate, path="temp_lang.wav"):
    """Write tensor to disk; helper for faster-whisper interface.

    We don't try to avoid the filesystem because the library only accepts a
    filename.  Caller is responsible for removing the file afterwards.
    """
    torchaudio.save(path, waveform, sample_rate)
    return path


# ── Primary Detection ─────────────────────────────────────
def detect_with_primary(waveform, sample_rate):
    """
    Primary detection using Whisper small model.

    Fast and accurate for high‑confidence cases.  The model is loaded lazily
    on first call, which also gives us a chance to fall back if the download
    fails (e.g. no network).

    Returns a tuple ``(language, confidence, probs_dict, segments)`` where the
    segments are the raw Whisper output; callers may inspect them for quality
    heuristics.
    """
    global whisper_primary

    # guard against empty audio
    if waveform.numel() == 0 or waveform.shape[-1] < 1600:
        return "unknown", 0.0, {}, []

    if whisper_primary is None:
        try:
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_primary = WhisperModel("small", device=device, compute_type="int8")
        except Exception as e:
            print(f"⚠️ Failed to load primary Whisper model: {e}")
            return "unknown", 0.0, {}, []

    temp = _save_temp(waveform, sample_rate)
    segments = []
    try:
        segments, info = whisper_primary.transcribe(temp)
        language   = info.language
        confidence = info.language_probability
    except Exception as e:
        print(f"⚠️ Primary detection error: {e}")
        language, confidence = "unknown", 0.0
    finally:
        if os.path.exists(temp):
            os.remove(temp)

    probs = {language: confidence, "other": round(1.0 - confidence, 4)}
    return language, confidence, probs, segments


# ── Fallback Detection ────────────────────────────────────
def detect_with_fallback(waveform, sample_rate):
    """
    Fallback using Whisper medium model on multiple audio chunks.

    The small model may be uncertain on noisy or very short files; in that
    case we split the audio into three parts and run the larger medium model on
    each, then combine the results by a simple weighted vote (middle chunk
    carries slightly more weight).

    Returns ``(language, confidence, probs_dict, transcript_text)`` where the
    final element is the concatenated text from all chunks; this is useful for
    performing a quick sanity check on whether any speech was actually present.
    """
    global whisper_fallback

    if waveform.numel() == 0 or waveform.shape[-1] < 1600:
        return "unknown", 0.0, {}

    if whisper_fallback is None:
        try:
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_fallback = WhisperModel("medium", device=device, compute_type="int8")
        except Exception as e:
            print(f"⚠️ Failed to load fallback Whisper model: {e}")
            return "unknown", 0.0, {}

    audio = waveform.squeeze()
    total = len(audio)
    if total == 0:
        return None, 0.0, {}

    chunk_sz = min(total // 3, 16000 * 10)  # max 10s per chunk
    chunks = []
    # start
    chunks.append(audio[0:chunk_sz])
    # middle
    mid_start = max(0, total // 2 - chunk_sz // 2)
    chunks.append(audio[mid_start:mid_start + chunk_sz])
    # end
    chunks.append(audio[max(0, total - chunk_sz):total])

    vote_probs = {}
    all_text = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < 1600:
            continue

        chunk_waveform = chunk.unsqueeze(0)
        temp = _save_temp(chunk_waveform, sample_rate, f"temp_chunk_{i}.wav")

        try:
            segments, info = whisper_fallback.transcribe(temp)
            lang = info.language
            conf = info.language_probability
            weight = 1.5 if i == 1 else 1.0
            vote_probs[lang] = vote_probs.get(lang, 0.0) + (conf * weight)
            # gather text for quality check
            all_text.append("".join(s.get("text", "") for s in segments))
        except Exception as e:
            print(f"  ⚠️ Chunk {i+1} failed: {e}")
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    if not vote_probs:
        return "unknown", 0.0, {}, ""

    total_votes = sum(vote_probs.values())
    norm_probs = {k: round(v / total_votes, 4) for k, v in vote_probs.items()}
    top_language = max(norm_probs, key=norm_probs.get)
    top_conf = norm_probs[top_language]

    combined_text = " ".join(all_text)
    return top_language, top_conf, norm_probs, combined_text


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


# ── Utility helpers ────────────────────────────────────

def _is_nonsense_text(text: str) -> bool:
    """Return True if transcription looks like non‑speech (laughter, noise).

    Heuristic rules:
    * More than 70% of characters are the same symbol (common for "hahaha"
      / "هههههه" laughs).
    * The string contains very few alphanumeric characters.
    * Compression ratio reported by Whisper is extremely high (caller can
      supply this if available).
    """
    if not text:
        return True
    # strip spaces to examine raw char distribution
    chars = text.replace(" ", "")
    if not chars:
        return True
    most = max(chars.count(c) for c in set(chars))
    if most / len(chars) > 0.7:
        return True
    # if there are fewer than 10 letters/digits, treat as garbage
    alnum = sum(c.isalnum() for c in chars)
    if alnum / len(chars) < 0.1:
        return True
    return False


# ── Main Function ─────────────────────────────────────────
def detect_language(waveform, sample_rate, metadata=None):
    """
    Dual-mode language detection using Whisper only.

    ``metadata`` may be provided by stage1 and should contain
    ``speech_ratio`` (among other quality metrics).  When speech_ratio is
    low we skip detection entirely and return ``"unknown"`` so that
    downstream components can avoid spurious language hints.

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

    # if we have quality metadata from stage1, use it to bail out early
    if metadata and isinstance(metadata, dict):
        sr_val = metadata.get("speech_ratio")
        if sr_val is not None and sr_val < 0.2:
            print(f"Low speech ratio ({sr_val:.1%}) → skipping language detection")
            return "unknown", 0.0, {}, "none"

    # Step 1: Primary detection
    print("Running primary detection (Whisper small)...")
    p_lang, p_conf, p_probs, p_segs = detect_with_primary(waveform, sample_rate)
    print(f"Primary result: {p_lang} ({p_conf:.2%})")

    # look at text quality, if available
    transcript = "".join(s.get("text", "") for s in p_segs)
    if _is_nonsense_text(transcript):
        print("  ⚠️ Primary transcript appears to be non-speech or garbage")
        p_conf = 0.0


    # Step 2: High confidence → done
    if p_conf >= CONFIDENCE_THRESHOLD:
        print(f"Confidence above threshold ✅ → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary"

    # Step 3: Low confidence → run fallback ensemble
    print(f"⚠️ Low confidence ({p_conf:.2%}) → running fallback (Whisper medium)...")
    f_lang, f_conf, f_probs, f_text = detect_with_fallback(waveform, sample_rate)

    # check whether fallback text looks like garbage
    if _is_nonsense_text(f_text):
        print("  ⚠️ Fallback transcript appears to be non-speech; abandoning detection")
        return p_lang, p_conf, p_probs, "whisper_primary_fallback"

    if f_lang is None or f_lang == "unknown":
        print("⚠️ Fallback also failed or returned unknown → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary_fallback"

    # Step 4: Ensemble both
    print("Running ensemble...")
    final_lang, final_conf, final_probs = ensemble(
        p_lang, p_conf, p_probs,
        f_lang, f_conf, f_probs
    )

    return final_lang, final_conf, final_probs, "whisper_ensemble"