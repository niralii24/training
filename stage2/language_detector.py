import os
import torch
import torchaudio

CACHE = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
os.environ["TRANSFORMERS_CACHE"] = CACHE
os.environ.setdefault("XDG_CACHE_HOME", CACHE)

try:
    os.makedirs(CACHE, exist_ok=True)
except Exception:
    pass

whisper_primary = None
whisper_fallback = None

CONFIDENCE_THRESHOLD = 0.70


def _save_temp(waveform, sample_rate, path="temp_lang.wav"):
    """Write tensor to disk; helper for faster-whisper interface."""
    torchaudio.save(path, waveform, sample_rate)
    return path


def detect_with_primary(waveform, sample_rate):
    """
    Primary language detection using Whisper small model.
    Returns: (language, confidence, probs_dict, segments)
    """
    global whisper_primary

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
        seg_gen, info = whisper_primary.transcribe(temp)
        segments   = list(seg_gen)   # materialise – Segment namedtuples, not dicts
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


def detect_with_fallback(waveform, sample_rate):
    """
    Fallback detection using Whisper medium model on audio chunks.
    Returns: (language, confidence, probs_dict, transcript_text)
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

    chunk_sz = min(total // 3, 16000 * 10)
    chunks = [
        audio[0:chunk_sz],
        audio[max(0, total // 2 - chunk_sz // 2):max(0, total // 2 - chunk_sz // 2) + chunk_sz],
        audio[max(0, total - chunk_sz):total]
    ]

    vote_probs = {}
    all_text = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < 1600:
            continue

        chunk_waveform = chunk.unsqueeze(0)
        temp = _save_temp(chunk_waveform, sample_rate, f"temp_chunk_{i}.wav")

        try:
            seg_gen, info = whisper_fallback.transcribe(temp)
            lang = info.language
            conf = info.language_probability
            weight = 1.5 if i == 1 else 1.0
            vote_probs[lang] = vote_probs.get(lang, 0.0) + (conf * weight)
            all_text.append("".join(s.text for s in seg_gen))
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


def ensemble(primary_lang, primary_conf, primary_probs,
             fallback_lang, fallback_conf, fallback_probs):
    """
    Combines primary and fallback Whisper results using weighted ensemble.
    """
    PRIMARY_W = 0.40
    FALLBACK_W = 0.60

    all_langs = set(list(primary_probs.keys()) + list(fallback_probs.keys()))
    merged_probs = {}

    for lang in all_langs:
        p = primary_probs.get(lang, 0.0)
        f = fallback_probs.get(lang, 0.0)
        merged_probs[lang] = round(PRIMARY_W * p + FALLBACK_W * f, 4)

    final_lang = max(merged_probs, key=merged_probs.get)
    final_conf = merged_probs[final_lang]
    agreed = primary_lang == fallback_lang

    print(f"  Primary  (small):  {primary_lang} ({primary_conf:.2%})")
    print(f"  Fallback (medium): {fallback_lang} ({fallback_conf:.2%})")
    print(f"  Agreement: {'✅ Yes' if agreed else '⚠️ No — using weighted ensemble'}")
    print(f"  Final:     {final_lang} ({final_conf:.2%})")

    return final_lang, final_conf, merged_probs


def _is_nonsense_text(text: str) -> bool:
    """Check if transcription looks like non-speech (laughter, noise)."""
    if not text:
        return True
    chars = text.replace(" ", "")
    if not chars:
        return True
    most = max(chars.count(c) for c in set(chars))
    if most / len(chars) > 0.7:
        return True
    alnum = sum(c.isalnum() for c in chars)
    if alnum / len(chars) < 0.1:
        return True
    return False


def detect_language(waveform, sample_rate, metadata=None):
    """
    Dual-mode language detection using Whisper (small + medium if needed).
    Returns: (language, confidence, probs_dict, method)
    """
    print("\n--- Language Detection ---")

    # if we have quality metadata from stage1, use it to bail out early
    if metadata and isinstance(metadata, dict):
        sr_val = metadata.get("speech_ratio")
        if sr_val is not None and sr_val < 0.2:
            print(f"Low speech ratio ({sr_val:.1%}) → skipping language detection")
            return "unknown", 0.0, {}, "none"

    print("Running primary detection (Whisper small)...")
    p_lang, p_conf, p_probs, p_segs = detect_with_primary(waveform, sample_rate)
    print(f"Primary result: {p_lang} ({p_conf:.2%})")

    # look at text quality, if available
    transcript = "".join(s.text for s in p_segs)
    if _is_nonsense_text(transcript):
        print("  ⚠️ Primary transcript appears to be non-speech or garbage")
        p_conf = 0.0

    if p_conf >= CONFIDENCE_THRESHOLD:
        print(f"Confidence above threshold ✅ → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary"

    print(f"⚠️ Low confidence ({p_conf:.2%}) → running fallback (Whisper medium)...")
    f_lang, f_conf, f_probs, f_text = detect_with_fallback(waveform, sample_rate)

    if _is_nonsense_text(f_text):
        print("  ⚠️ Fallback transcript appears to be non-speech; abandoning detection")
        return p_lang, p_conf, p_probs, "whisper_primary_fallback"

    if f_lang is None or f_lang == "unknown":
        print("⚠️ Fallback also failed or returned unknown → using primary result")
        return p_lang, p_conf, p_probs, "whisper_primary_fallback"

    print("Running ensemble...")
    final_lang, final_conf, final_probs = ensemble(
        p_lang, p_conf, p_probs,
        f_lang, f_conf, f_probs
    )

    return final_lang, final_conf, final_probs, "whisper_ensemble"