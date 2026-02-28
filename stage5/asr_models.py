import whisper
import numpy as np
import torch
import librosa
import os
import re
import logging
torch.set_grad_enabled(False)


def _clean_transcript(text: str) -> str:
    """Remove common Whisper hallucination artifacts."""
    # 1. Strip Arabic tatweel / kashida – never a real ASR output
    text = re.sub(r'ـ+', '', text)
    # 2. Remove Latin tokens ending with a hyphen (boundary hallucinations)
    text = re.sub(r'\b[A-Za-z]+\-\s*', ' ', text)
    # 3. Collapse sequences of dots/ellipses left by chunk-boundary artifacts
    text = re.sub(r'\.{2,}', '.', text)
    # 4. Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()
    return text

# Ensure cache folder exists – read from the same env-var that main_pipeline sets
CACHE_DIR = os.environ.get("HF_HOME", "C:/Users/Omega/hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# Whisper Model
# =========================================================

class WhisperModel:
    def __init__(self, size, device):
        self.device = device
        self.model = whisper.load_model(
            size,
            device=device,
            download_root=CACHE_DIR
        )

    def transcribe(self, audio_path=None, audio=None, sr: int = 16000,
                   offset: float = 0.0):
        """Transcribe audio using Whisper with stable, language-agnostic settings.

        Key flags that prevent the "random output" problem:
          - condition_on_previous_text=False  → stops the model conditioning on
            its own prior output, which is the #1 cause of hallucinations and
            repeating/random text, especially on long or chunked audio.
          - temperature=0                     → greedy (deterministic) decoding.
          - fp16 matched to device            → avoids precision-related artefacts.
        Whisper auto-detects the language; no hint is passed.

        Audio is always loaded with librosa at 16 kHz and passed as a numpy
        array.  This avoids whisper's internal load_audio() which shells out to
        FFmpeg and fails when FFmpeg is not on PATH.
        """
        WHISPER_SR = 16000  # whisper always expects 16 kHz float32

        if audio is not None:
            # already a numpy array – resample if needed
            if sr != WHISPER_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SR)
            audio_np = audio.astype("float32")
        else:
            # load from file with librosa (no FFmpeg required for WAV/MP3 via soundfile/audioread)
            audio_np, _ = librosa.load(audio_path, sr=WHISPER_SR, mono=True)

        use_fp16 = (self.device != "cpu")

        kwargs = dict(
            fp16=use_fp16,
            # Temperature schedule: start greedy, escalate on repetition/low-confidence.
            # A scalar 0 disables Whisper's internal retry logic entirely.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=False,   # prevents context-drift hallucinations
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            # Anchor to conversational speech so Whisper doesn't drift into
            # English tokens or padding hallucinations at segment boundaries.
            initial_prompt="محادثة.",
        )

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

        # clean tatweel and boundary-hallucination tokens from every text field
        for s in segments:
            if "text" in s:
                s["text"] = _clean_transcript(s["text"])

        return dict(
            text=_clean_transcript(result.get("text", "")),
            confidence=float(np.mean(confs)),
            entropy=float(np.mean(ents)),
            no_speech_prob=float(np.mean(nos)),
            confidence_vector=[float(c) for c in confs],
            entropy_vector=[float(e) for e in ents],
            no_speech_vector=[float(n) for n in nos],
            segments=segments,
        )


# =========================================================
# Generic Wav2Vec2 ASR wrapper (works for XLSR, MMS, etc.)
# =========================================================

class Wav2Vec2ASR:
    """Wrapper around any Wav2Vec2-style CTC model that has a tokenizer/vocab.

    ``model_name`` can be a Hugging Face identifier such as
    ``facebook/wav2vec2-large-xlsr-53-multilingual`` or ``facebook/mms-1b-all``.
    The underlying class uses ``AutoProcessor`` and ``AutoModelForCTC`` so the
    same code works for all fine‑tuned checkpoints. If you accidentally pass a
    pretrained *base* checkpoint (e.g. ``facebook/wav2vec2-large-xlsr-53`` or
    ``facebook/mms-300m``) the constructor will raise a helpful error telling
    you that the model needs to be fine‑tuned first.
    """

    def __init__(self, model_name: str, device="cpu"):
        from transformers import AutoProcessor, AutoModelForCTC

        self.model_name = model_name
        self.device = device

        # load processor (feature extractor + tokenizer)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"failed to load processor for {model_name}: {e}\n"
                "make sure this is a *fine-tuned* ASR checkpoint that has a tokenizer/vocab."
            )

        # load the model itself
        self.model = AutoModelForCTC.from_pretrained(model_name).to(device)
        self.model.eval()

    def transcribe(self, audio_path=None, audio=None, sr=16000,
                   offset=0.0, language: str = None):
        # same audio handling as before
        if audio is None:
            speech, sr = librosa.load(audio_path, sr=sr)
        else:
            speech = audio

        proc_kwargs = {
            "audio": speech,
            "sampling_rate": sr,
            "return_tensors": "pt",
            "padding": True,
        }
        if language is not None:
            # some multilingual models take an explicit language token/field
            proc_kwargs["language"] = language

        inputs = self.processor(**proc_kwargs).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        # use processor to decode, this takes care of blanks/padding
        text = self.processor.batch_decode(pred_ids)[0]

        confidence = float(torch.mean(torch.max(probs, dim=-1).values))
        entropy = float((-probs * torch.log(probs + 1e-10)).sum(dim=-1).mean())

        return dict(
            text=text,
            confidence=confidence,
            entropy=entropy,
            no_speech_prob=0.0,
            confidence_vector=[],
            entropy_vector=[],
            no_speech_vector=[],
        )


def load_asr_models(config):
    """Create a list of Whisper ASR model instances based on configuration.

    Only Whisper models are supported.  Language detection and language-specific
    routing have been removed; Whisper auto-detects the language reliably.
    """

    models = []

    for entry in config["models"]:
        name = entry["name"]
        device = entry.get("device", "cpu")   # default cpu

        if name.startswith("whisper"):
            size = name.split(":", 1)[1]
            models.append(WhisperModel(size, device))

        # elif name.startswith("wav2vec2"):
        #     parts = name.split(":", 1)

        #     if len(parts) > 1 and parts[1].strip():
        #         model_id = parts[1].strip()
        #     else:
        #         # no explicit model requested - try language-aware guess
        #         if language:
        #             candidate = f"jonatasgrosman/wav2vec2-large-xlsr-53-{language}"
        #             try:
        #                 models.append(Wav2Vec2ASR(candidate, device))
        #                 continue
        #             except RuntimeError:
        #                 # language-specific model not available; fall back
        #                 pass

        #         # fall back to a generic multilingual ASR checkpoint that is
        #         # known to include a tokenizer.  ``mms-1b-all`` covers 1000+
        #         # languages and is public.
        #         model_id = "facebook/mms-1b-all"

        #     models.append(Wav2Vec2ASR(model_id, device))

    return models