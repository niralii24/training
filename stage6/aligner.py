"""
aligner.py [FIXED]
------------------
Thin wrapper around WhisperX forced-alignment with text normalization.

FIX: Text is now normalized before alignment to remove punctuation/diacritics
that have no acoustic signal. This prevents WhisperX from trying to align
silent tokens, which was causing low confidence scores.

The normalization is applied ONLY to the text passed to WhisperX; the original
text is preserved in the results for display.

Public API
----------
WhisperXAligner(language, device)
    .align(audio_path, segments, preserve_original=True) -> dict
        {
            "segments": [
                {
                    "words": [{word, start, end, score}],
                    "chars": [...],
                    "original_text": "...",  # if preserve_original=True
                }
            ],
            "word_segments": [...]
        }
"""
from __future__ import annotations

import os
import logging
import inspect
import numpy as np
import librosa

logger = logging.getLogger(__name__)

# Point WhisperX's Hugging Face downloads to the same local cache
CACHE_DIR = os.environ.get("HF_HOME", "C:/Users/Omega/hf_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", CACHE_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization for alignment
# ─────────────────────────────────────────────────────────────────────────────

# Arabic diacritics (no acoustic signal)
ARABIC_DIACRITICS = {
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
    '\u0651', '\u0652', '\u0653', '\u0654', '\u0655', '\u0656',
    '\u0657', '\u0658', '\u0670',
}

# Punctuation that confuses WhisperX alignment
ALIGNMENT_PUNCT = set('؟،؛:!.\'"()-[]{}…‹›«»„"‟—–')


def _normalize_text_for_alignment(text: str) -> str:
    """
    Remove punctuation and diacritics before WhisperX alignment.

    These elements have no acoustic signal, so they confuse CTC-based alignment.
    Removing them allows WhisperX to focus on actual phonetic content.
    """
    if not text:
        return ""
    # Remove diacritics
    text = ''.join(c for c in text if c not in ARABIC_DIACRITICS)
    # Replace punctuation with spaces
    text = ''.join(c if c not in ALIGNMENT_PUNCT else ' ' for c in text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


class WhisperXAligner:
    """
    Forced-alignment engine backed by WhisperX with smart text normalization.

    Parameters
    ----------
    language : str
        ISO-639-1 language code, e.g. ``"ar"``, ``"en"``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    cache_dir : str | None
        Override the Hugging Face cache directory.
    """

    def __init__(
        self,
        language: str,
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> None:
        try:
            import whisperx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "WhisperX is required for Stage 6.\n"
                "Install with:\n"
                "  pip install whisperx\n"
                "or for the latest version:\n"
                "  pip install git+https://github.com/m-bain/whisperX.git"
            ) from exc

        self._wx      = whisperx
        self.language = language
        self.device   = device
        _cache        = cache_dir or CACHE_DIR

        logger.info(
            "Stage6 Aligner: loading WhisperX alignment model "
            "(language=%r, device=%s) …", language, device
        )

        # ── load_align_model API varies across WhisperX versions ──────────
        try:
            sig = inspect.signature(whisperx.load_align_model)
            if "model_dir" in sig.parameters:
                self.model, self.metadata = whisperx.load_align_model(
                    language_code=language,
                    device=device,
                    model_dir=_cache,
                )
            else:
                self.model, self.metadata = whisperx.load_align_model(
                    language_code=language,
                    device=device,
                )
        except TypeError:
            self.model, self.metadata = whisperx.load_align_model(
                language, device
            )

        logger.info("Stage6 Aligner: alignment model ready.")

    # ------------------------------------------------------------------
    def align(
        self,
        audio_path: str,
        segments: list[dict],
        preserve_original: bool = True,
    ) -> dict:
        """
        Run forced alignment on *audio_path* using *segments* from stage 5.

        Text normalization (removal of punctuation/diacritics) is applied
        before alignment to improve alignment confidence, since punctuation
        has no acoustic signal and confuses the CTC-based aligner.

        Parameters
        ----------
        audio_path : str
            Path to a mono WAV (any sample rate; WhisperX resamples internally).
        segments : list[dict]
            Stage-5 Whisper segments – must contain ``start``, ``end``, ``text``.
        preserve_original : bool
            If True, save the original (non-normalized) text in the output
            segments for display purposes.

        Returns
        -------
        dict with at minimum a ``"segments"`` key:
            {
                "segments": [
                    {
                        "start": float, "end": float, "text": str,
                        "original_text": str,  # if preserve_original=True
                        "words": [{"word", "start", "end", "score"}, …],
                        "chars": [{"char", "start", "end", "score"}, …],
                    },
                    …
                ],
                "word_segments": […],   # flat list; may be absent in old API
            }
        """
        if not segments:
            logger.warning("Stage6 Aligner: received empty segment list; "
                           "returning empty alignment.")
            return {"segments": [], "word_segments": []}

        # ── Load audio with librosa ──────────────────────────────────────
        WHISPERX_SR = 16000
        audio_np, _ = librosa.load(audio_path, sr=WHISPERX_SR, mono=True)
        audio_np = audio_np.astype("float32")

        # ── Normalize text for alignment ──────────────────────────────────
        # Save original text before normalizing so we can restore it later
        original_texts = [s.get("text", "") for s in segments]
        clean_segs = _normalise_segments(segments, normalize_text=True)

        # ── Run WhisperX alignment on normalized text ────────────────────
        logger.info(
            "Stage6 Aligner: aligning %d segment(s) with normalized text …",
            len(clean_segs)
        )
        result = self._wx.align(
            clean_segs,
            self.model,
            self.metadata,
            audio_np,
            self.device,
            return_char_alignments=True,
        )

        # ── Restore original text in output ──────────────────────────────
        if isinstance(result, list):
            result = {"segments": result, "word_segments": []}
        elif isinstance(result, dict):
            result = dict(result)
            result.setdefault("segments", clean_segs)
            result.setdefault("word_segments", [])

        if preserve_original and isinstance(result.get("segments"), list):
            for i, seg in enumerate(result["segments"]):
                if i < len(original_texts):
                    seg["original_text"] = original_texts[i]

        logger.info("Stage6 Aligner: alignment complete.")
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_segments(
    segments: list[dict],
    normalize_text: bool = True,
) -> list[dict]:
    """
    Ensure every segment dict has the mandatory keys WhisperX expects
    and optionally normalize text for better alignment.

    Parameters
    ----------
    segments : list[dict]
        Input segments.
    normalize_text : bool
        If True, remove punctuation/diacritics from text before alignment.
    """
    out: list[dict] = []
    for s in segments:
        start = s.get("start", 0.0)
        end   = s.get("end",   0.0)
        text  = s.get("text",  "")

        # Normalize text if requested
        if normalize_text:
            text = _normalize_text_for_alignment(text)

        clean: dict = {
            "start": float(start) if start is not None else 0.0,
            "end":   float(end)   if end   is not None else 0.0,
            "text":  str(text)    if text  is not None else "",
        }

        # Pass through any existing word-level data
        if "words" in s and isinstance(s["words"], list):
            clean["words"] = [
                {
                    "word":  str(w.get("word", "")),
                    "start": float(w["start"]) if w.get("start") is not None else None,
                    "end":   float(w["end"])   if w.get("end")   is not None else None,
                    "score": float(w["score"]) if w.get("score") is not None else None,
                }
                for w in s["words"]
            ]

        out.append(clean)
    return out