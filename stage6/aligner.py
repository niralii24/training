"""
stage6/aligner.py
-----------------
Thin wrapper around WhisperX forced-alignment.

WhisperX uses a language-specific HuBERT / wav2vec2/MMS model trained with
CTC labels to snap every word (and optionally every character) to the exact
frame in the audio waveform.

Public API
----------
WhisperXAligner(language, device)
    .align(audio_path, segments) -> dict
        {
            "segments": [ { "words": [{word, start, end, score}], "chars": [...] } ],
            "word_segments": [...]
        }
"""
from __future__ import annotations  # MUST be first line: makes all annotations
                                     # lazy strings on Python 3.8/3.9

import os
import logging
import inspect
import numpy as np
import librosa

logger = logging.getLogger(__name__)

# Point WhisperX's Hugging Face downloads to the same local cache used by the
# rest of the pipeline, so we never download to two different places.
CACHE_DIR = os.environ.get("HF_HOME", "C:/Users/Omega/hf_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", CACHE_DIR)


class WhisperXAligner:
    """
    Forced-alignment engine backed by WhisperX.

    Parameters
    ----------
    language : str
        ISO-639-1 language code, e.g. ``"ar"``, ``"en"``.
        WhisperX selects the appropriate pre-trained alignment model.
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
        # Newer builds accept model_dir; older ones only take language_code
        # and device.  Inspect the signature first, fall back gracefully.
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
            # Absolute fallback – positional-only old API
            self.model, self.metadata = whisperx.load_align_model(
                language, device
            )

        logger.info("Stage6 Aligner: alignment model ready.")

    # ------------------------------------------------------------------
    def align(self, audio_path: str, segments: list[dict]) -> dict:
        """
        Run forced alignment on *audio_path* using *segments* from stage 5.

        Parameters
        ----------
        audio_path : str
            Path to a mono WAV (any sample rate; WhisperX resamples internally).
        segments : list[dict]
            Stage-5 Whisper segments – must contain ``start``, ``end``, ``text``.

        Returns
        -------
        dict with at minimum a ``"segments"`` key:
            {
                "segments": [
                    {
                        "start": float, "end": float, "text": str,
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

        # ── Load audio with librosa (no FFmpeg required) ──────────────────
        # whisperx.load_audio() calls FFmpeg under the hood which may not be
        # on PATH.  librosa reads WAV/MP3 via soundfile/audioread without any
        # external binary.  WhisperX align() accepts a raw float32 numpy array
        # at 16 kHz, identical to what load_audio() would have returned.
        WHISPERX_SR = 16000
        audio_np, _ = librosa.load(audio_path, sr=WHISPERX_SR, mono=True)
        audio_np = audio_np.astype("float32")

        clean_segs = _normalise_segments(segments)

        # align() API is stable across recent WhisperX versions
        result = self._wx.align(
            clean_segs,
            self.model,
            self.metadata,
            audio_np,
            self.device,
            return_char_alignments=True,   # enables phoneme-proxy metrics
        )

        # ── Normalise return type ─────────────────────────────────────────
        # Some pre-release builds returned a plain list; current builds return
        # a TypedDict / plain dict with "segments" and "word_segments" keys.
        if isinstance(result, list):
            return {"segments": result, "word_segments": []}
        if isinstance(result, dict):
            result = dict(result)                     # shallow copy
            result.setdefault("segments", clean_segs)
            result.setdefault("word_segments", [])
            return result

        logger.error(
            "Stage6 Aligner: unexpected align() return type %s; "
            "returning empty alignment.", type(result)
        )
        return {"segments": [], "word_segments": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_segments(segments: list[dict]) -> list[dict]:
    """
    Ensure every segment dict has the mandatory keys WhisperX expects
    (``start``, ``end``, ``text``) and scrub any heavy numpy/tensor objects
    that would confuse WhisperX's internal JSON-like processing.
    """
    out: list[dict] = []
    for s in segments:
        start = s.get("start", 0.0)
        end   = s.get("end",   0.0)
        text  = s.get("text",  "")

        clean: dict = {
            "start": float(start) if start is not None else 0.0,
            "end":   float(end)   if end   is not None else 0.0,
            "text":  str(text)    if text  is not None else "",
        }

        # Pass through any existing word-level data (from a prior partial align)
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
