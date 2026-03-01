"""
asr_models_conservative.py
──────────────────────────
Conservative cleaning that preserves legitimate Arabic/English words
while only removing obvious Whisper hallucination artifacts.

Key differences from original:
  1. Tatweel (kashida) removal is disabled by default
  2. Character run squashing only for EXTREME repetitions (50+ chars)
  3. Latin hyphen removal is more conservative
  4. Option to disable all cleaning if needed
  5. Separate "strict" mode for when hallucinations are detected
"""

import whisper
import numpy as np
import torch
import librosa
import os
import re
import logging
from typing import Tuple, Dict, List, Optional, Any

torch.set_grad_enabled(False)


# =========================================================
# CONSERVATIVE CLEANING (default - preserves words)
# =========================================================

def _clean_transcript_conservative(text: str, strict: bool = False) -> str:
    """
    Remove ONLY obvious Whisper hallucination artifacts.
    
    Conservative mode (strict=False) preserves:
      ✓ Natural character repetition (e.g., "ااااه" - stretched vocalizations)
      ✓ Legitimate Arabic words with tatweel
      ✓ Hyphenated English words
      ✓ Most punctuation
    
    Removes only:
      ✗ Multiple consecutive ellipses/dots
      ✗ Excessive whitespace
      ✗ Obvious repetition loops (if strict=True)
    
    Parameters
    ----------
    text : str
        Raw Whisper output
    strict : bool
        If True, also remove character runs > 20 (hallucination recovery mode)
        If False, preserve all character repetition (default)
    
    Returns
    -------
    str
        Cleaned text
    """
    
    if not text or not text.strip():
        return ""
    
    # 1. Collapse ONLY multiple ellipses (2+ consecutive dots)
    #    Preserve single dots for abbreviations, sentence endings, etc.
    text = re.sub(r'\.{2,}', '.', text)
    
    # 2. Collapse multiple spaces (always safe)
    text = re.sub(r' {2,}', ' ', text).strip()
    
    # 3. If strict mode: remove excessive repetition (> 20 chars)
    #    Otherwise: preserve natural character repetition
    if strict:
        # Only for extreme pathological cases
        text = _squash_extreme_runs(text, max_run=20)
    
    return text


def _squash_extreme_runs(text: str, max_run: int = 20) -> str:
    """
    Collapse ONLY pathologically extreme character repetitions.
    
    Preserves natural Arabic phonetic stretching like:
      ✓ "ااااه" (4 alefs) - natural elongation
      ✓ "هممممم" (5 meems) - vocal stretching
    
    Removes obvious hallucinations like:
      ✗ "هاااااااااااااااااااااااا" (50+ repetitions) - pathological
      ✗ "............................." (infinite dots) - glitch
    
    Parameters
    ----------
    text : str
        Input text
    max_run : int
        Only collapse runs longer than this (default: 20)
        Increase to 50+ for even more conservative approach
    
    Returns
    -------
    str
        Text with extreme runs collapsed
    """
    return re.sub(r"(.)\1{%d,}" % max_run, r"\1\1", text)


# =========================================================
# STRICT CLEANING (only use when hallucinations detected)
# =========================================================

def _clean_transcript_strict(text: str) -> str:
    """
    Aggressive cleaning when hallucinations are CONFIRMED.
    
    Use this ONLY when:
      • Compression ratio > 2.8 (repetition detected)
      • Mean confidence < -1.5 (low quality)
      • Manual inspection shows garbage output
    
    Removes:
      ✗ Tatweel/kashida characters
      ✗ Character runs > 10
      ✗ Latin tokens with hyphens
      ✗ Multiple ellipses
      ✗ Excessive whitespace
    """
    
    if not text or not text.strip():
        return ""
    
    # 1. Remove Arabic tatweel (kashida)
    #    Only do this when we KNOW output is corrupted
    text = re.sub(r'ـ+', '', text)
    
    # 2. Remove Latin tokens ending with hyphen (boundary artifacts)
    text = re.sub(r'\b[A-Za-z]+\-\s*', ' ', text)
    
    # 3. Collapse extreme repetitions (strict mode)
    text = _squash_extreme_runs(text, max_run=10)
    
    # 4. Remove multiple ellipses
    text = re.sub(r'\.{2,}', '.', text)
    
    # 5. Normalize whitespace
    text = re.sub(r' {2,}', ' ', text).strip()
    
    return text


def _detect_hallucination(
    confidence: float,
    entropy: float,
    no_speech_prob: float,
    text: str
) -> Tuple[bool, str]:
    """
    Detect if Whisper output is likely hallucinated garbage.
    
    Returns
    -------
    (is_hallucinated: bool, reason: str)
    """
    
    reasons = []
    
    # Check 1: Pathological repetition (compression ratio > 2.8)
    if entropy > 2.8:
        reasons.append(f"extreme_repetition(ratio={entropy:.2f})")
    
    # Check 2: Extremely low confidence (< -1.5)
    if confidence < -1.5:
        reasons.append(f"very_low_confidence(conf={confidence:.2f})")
    
    # Check 3: High no-speech probability
    if no_speech_prob > 0.8:
        reasons.append(f"no_speech(prob={no_speech_prob:.2f})")
    
    # Check 4: Suspicious text patterns
    if text:
        # Excessive single-character repetition in output
        if re.search(r'(.)\1{50,}', text):
            reasons.append("pathological_character_loop")
        
        # All punctuation (no actual words)
        if re.match(r'^[\s\.\!؟،؛\-\—]+$', text):
            reasons.append("only_punctuation")
    
    is_hallucinated = len(reasons) > 0
    reason = " + ".join(reasons) if reasons else ""
    
    return is_hallucinated, reason


# =========================================================
# Whisper Model (updated with conservative cleaning)
# =========================================================

# Language-specific prompts (keep the same)
_DIALECT_PROMPTS: dict = {
    "ar": "ويش. ثيو. اليكس. ابيك. ارسلى. بالظبط. اخس. تشوف. ليه. وين. ازا.",
}

CACHE_DIR = os.environ.get("HF_HOME", "C:/Users/Omega/hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class WhisperModel:
    def __init__(self, size, device, conservative_cleaning=True):
        """
        Initialize Whisper model.
        
        Parameters
        ----------
        size : str
            Model size (tiny, base, small, medium, large)
        device : str
            "cuda" or "cpu"
        conservative_cleaning : bool
            If True (default), use conservative cleaning that preserves words
            If False, use strict aggressive cleaning
        """
        self.device = device
        self.conservative_cleaning = conservative_cleaning
        self.model = whisper.load_model(
            size,
            device=device,
            download_root=CACHE_DIR
        )

    def transcribe(self, audio_path=None, audio=None, sr: int = 16000,
                   offset: float = 0.0, language: str = None):
        """
        Transcribe audio with conservative cleaning by default.
        """
        WHISPER_SR = 16000

        if audio is not None:
            if sr != WHISPER_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)
            audio_np = audio.astype("float32")
        else:
            audio_np, _ = librosa.load(audio_path, sr=WHISPER_SR, mono=True)

        use_fp16 = (self.device != "cpu")

        kwargs = dict(
            fp16=use_fp16,
            language=language,
            task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-2.0,
            compression_ratio_threshold=2.4,
            initial_prompt=_DIALECT_PROMPTS.get(language) if language else None,
        )

        if torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=True
            ):
                result = self.model.transcribe(audio_np, **kwargs)
        else:
            result = self.model.transcribe(audio_np, **kwargs)

        segments = result.get("segments", [])

        if not segments:
            return dict(
                text="",
                confidence=0,
                entropy=1,
                no_speech_prob=1,
                confidence_vector=[],
                entropy_vector=[],
                no_speech_vector=[],
                segments=[],
                cleaning_applied="none",
                hallucination_detected=False,
            )

        if offset and segments:
            for s in segments:
                if "start" in s:
                    s["start"] += offset
                if "end" in s:
                    s["end"] += offset

        confs = [s["avg_logprob"] for s in segments]
        ents = [s["compression_ratio"] for s in segments]
        nos = [s["no_speech_prob"] for s in segments]

        mean_conf = float(np.mean(confs))
        mean_ent = float(np.mean(ents))
        mean_no_speech = float(np.mean(nos))
        raw_text = result.get("text", "")

        # ── Detect if output is hallucinated ─────────────────────────────
        is_hallucinated, halluc_reason = _detect_hallucination(
            mean_conf, mean_ent, mean_no_speech, raw_text
        )

        # ── Choose cleaning strategy ─────────────────────────────────────
        if is_hallucinated:
            # Use STRICT cleaning for confirmed hallucinations
            cleaned_text = _clean_transcript_strict(raw_text)
            cleaning_applied = f"strict({halluc_reason})"
        elif self.conservative_cleaning:
            # Use CONSERVATIVE cleaning by default
            cleaned_text = _clean_transcript_conservative(raw_text, strict=False)
            cleaning_applied = "conservative"
        else:
            # No cleaning
            cleaned_text = raw_text
            cleaning_applied = "none"

        # Apply cleaning to segments too
        for s in segments:
            if "text" in s:
                if is_hallucinated:
                    s["text"] = _clean_transcript_strict(s["text"])
                elif self.conservative_cleaning:
                    s["text"] = _clean_transcript_conservative(s["text"], strict=False)

        # ── Final check: if still pathological after cleaning, flag as empty ──
        if is_hallucinated and (mean_ent > 2.8 or mean_conf < -1.5):
            return dict(
                text="",
                confidence=0,
                entropy=1,
                no_speech_prob=1,
                confidence_vector=[float(c) for c in confs],
                entropy_vector=[float(e) for e in ents],
                no_speech_vector=[float(n) for n in nos],
                segments=segments,
                cleaning_applied=cleaning_applied,
                hallucination_detected=True,
            )

        return dict(
            text=cleaned_text,
            confidence=mean_conf,
            entropy=mean_ent,
            no_speech_prob=mean_no_speech,
            confidence_vector=[float(c) for c in confs],
            entropy_vector=[float(e) for e in ents],
            no_speech_vector=[float(n) for n in nos],
            segments=segments,
            cleaning_applied=cleaning_applied,
            hallucination_detected=is_hallucinated,
        )


def load_asr_models(config, conservative_cleaning=True):
    """
    Create Whisper ASR model instances with optional conservative cleaning.
    
    Parameters
    ----------
    config : dict
        Configuration dict with "models" key
    conservative_cleaning : bool
        If True (default), use conservative cleaning
        If False, use strict cleaning
    """
    models = []

    for entry in config["models"]:
        name = entry["name"]
        device = entry.get("device", "cpu")

        if name.startswith("whisper"):
            size = name.split(":", 1)[1]
            models.append(
                WhisperModel(size, device, conservative_cleaning=conservative_cleaning)
            )

    return models