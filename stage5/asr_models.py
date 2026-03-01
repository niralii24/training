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


def _squash_long_runs(text: str, max_run: int = 10) -> str:
    """Collapse pathologically repeated character runs (e.g. هاااااااااا → هاا).

    max_run is set to 10 so that natural Arabic phonetic stretching like
    'ааааه' (4 alefs) is preserved while genuine Whisper repetition loops
    (100+ identical tokens) are still caught and collapsed.
    """
    return re.sub(r"(.)\1{%d,}" % max_run, r"\1\1", text)


# Language-specific initial prompts to anchor Whisper's decoder vocabulary
# toward the relevant dialect/register.  These are passed as `initial_prompt`
# only when the stage-2 detected language matches the key.
#
# The prompt should contain words that are characteristic of the dialect so
# that Whisper's language model does not normalise them toward MSA forms.
# The prompt must be SHORT (≤ 224 tokens) and must NOT include full sentences
# that would bleed into the transcript.
_DIALECT_PROMPTS: dict = {
    # Gulf / Saudi Arabic – keep dialectal forms like ويش، ثيو، بالظبط، اليكس،
    # ارسل‌لي, etc.  Without this, Whisper normalises them to MSA equivalents.
    "ar": "ويش. ثيو. اليكس. ابيك. ارسلى. بالظبط. اخس. تشوف. ليه. وين. ازا.",
}

# Ensure cache folder exists – read from the same env-var that main_pipeline sets
CACHE_DIR = os.environ.get("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf_cache"))
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
                   offset: float = 0.0, language: str = None):
        """Transcribe audio using Whisper with stable, language-pinned settings.

        Key flags that prevent the "random output" / wrong-translation problem:
          - language                           → pins the source language so
            Whisper does not mis-detect Arabic dialects as Farsi/Urdu.
          - task="transcribe"                  → always transcribe in the
            source language; never translate to English.
          - condition_on_previous_text=False   → stops the model conditioning on
            its own prior output, which is the #1 cause of hallucinations and
            repeating/random text, especially on long or chunked audio.
          - temperature=0                      → greedy (deterministic) decoding.
          - fp16 matched to device             → avoids precision-related artefacts.
          - initial_prompt removed             → the Arabic-biased prompt was
            corrupting transcriptions for non-Arabic and mixed-language audio.

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
            # Pin source language – prevents mis-detection of Arabic dialects
            # as Farsi / Urdu / other close languages.  Passing None means
            # Whisper will still auto-detect when no language was provided.
            language=language,
            # Always transcribe in the source language; never translate to English.
            task="transcribe",
            # Temperature fallback sequence: Whisper starts at 0.0 (greedy) and
            # automatically retries at higher temperatures when it detects
            # compression_ratio > threshold (repetition loop) or logprob < threshold.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # beam_size=5 at temp 0 gives better quality than greedy;
            # best_of=5 selects the best among sampled candidates at temp > 0.
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            # Loosened from -1.0 → -2.0 so that valid low-confidence segments
            # at the end of the audio are NOT silently dropped, which was causing
            # transcripts to be truncated mid-sentence.
            logprob_threshold=-2.0,
            compression_ratio_threshold=2.4,
            # Dialect-anchoring prompt: if we have a language-specific prompt it
            # steers the decoder vocabulary away from MSA normalisation (e.g.
            # "ويش" being rewritten as "وش", "بالظبط" → "بالضبط", etc.).
            initial_prompt=_DIALECT_PROMPTS.get(language) if language else None,
        )

        # Disable flash SDP kernels on Windows to avoid unstable attention paths
        # seen in PyTorch's scaled_dot_product_attention on some driver stacks.
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
                s["text"] = _squash_long_runs(_clean_transcript(s["text"]))

        # If Whisper reports pathological repetition or extremely low confidence,
        # treat it as unusable speech to avoid propagating garbage downstream.
        if np.mean(ents) > 2.8 or np.mean(confs) < -1.5:
            return dict(
                text="",
                confidence=0,
                entropy=1,
                no_speech_prob=1,
                confidence_vector=[float(c) for c in confs],
                entropy_vector=[float(e) for e in ents],
                no_speech_vector=[float(n) for n in nos],
                segments=segments,
            )

        return dict(
            text=_squash_long_runs(_clean_transcript(result.get("text", ""))),
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