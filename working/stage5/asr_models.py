import whisper
import numpy as np
import torch
import librosa
import os
import logger
torch.set_grad_enabled(False)

# Ensure cache folder exists
CACHE_DIR = "D:/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# Whisper Model
# =========================================================

class WhisperModel:
    def __init__(self, size, device):
        self.model = whisper.load_model(
            size,
            device=device,
            download_root=CACHE_DIR
        )

    def transcribe(self, audio_path=None, audio=None, sr: int = 16000,
                   offset: float = 0.0, language: str = None):
        """Transcribe audio with optional language hint.

        ``language`` should be a short code such as ``'ar'`` or ``'en'``.  Whisper
        will use it to force decoding in that language instead of detecting it.
        """

        kwargs = dict(
            temperature=0.2,      # Allow slight randomness for better decoding stability
            beam_size=10,         # Larger search space (default in many APIs is 5-10)
            best_of=5,            # Sample 5 hypotheses and pick best — dramatically improves quality
            condition_on_previous_text=False,
        )
        if language:
            kwargs["language"] = language
            # logger.info(f"Transcribing with language hint: {language}")
        # else:
            # logger.warning("No language specified. Whisper will auto-detect (may be unreliable).")

        if audio is not None:
            result = self.model.transcribe(audio, **kwargs)
        else:
            result = self.model.transcribe(audio_path, **kwargs)

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

        return dict(
            text=result.get("text", ""),
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

# def load_asr_models(config):
## This loader only uses one type : either cpu or gpu 
#     models = []

#     for name in config["models"]:

#         if name.startswith("whisper"):
#             size = name.split(":")[1]
#             models.append(WhisperModel(size, config["device"]))

#         # elif name.startswith("wav2vec2"):
#         #     # allow specifying the checkpoint e.g. "wav2vec2:facebook/mms-1b-all"
#         #     parts = name.split(":", 1)
#         #     if len(parts) > 1 and parts[1].strip():
#         #         model_id = parts[1].strip()
#         #     else:
#         #         # sensible multilingual default if none provided
#         #         model_id = "facebook/wav2vec2-large-xlsr-53-multilingual"

#         #     models.append(Wav2Vec2ASR(model_id, config["device"]))

#         # other ASR backends could be added here

#     return models

def load_asr_models(config, language: str = None):
    """Create a list of ASR model instances based on configuration.

    ``language`` is an optional two-letter code.  When an entry specifies
    ``"wav2vec2"`` without an explicit checkpoint ID we attempt to pick a
    language-specific version first (following the common naming pattern
    ``jonatasgrosman/wav2vec2-large-xlsr-53-<lang>``).  If that fails or no
    language is provided we fall back to a generic multilingual model.
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