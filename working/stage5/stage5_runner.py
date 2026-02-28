from .asr_models import load_asr_models
from .utils import normalize_text, word_agreement
from .consensus import build_consensus
from .scoring import compute_rss, compute_agreement_score
import numpy as np


def run_stage5(audio_path: str, config: dict, language: str = None) -> dict:
    """
    Main entry point for Stage 5.  Loads the file once, splits it into chunks
    if it is long, and runs each configured ASR on every chunk.

    ``language`` is an optional two‑letter code (e.g. 'ar', 'en') obtained from
    Stage 2 language detection.  If provided, models that accept a language
    hint (Whisper, MMS, etc.) will receive it for improved accuracy.

    The results from each model are concatenated back together; timestamps are
    preserved for Whisper segments via an ``offset`` argument so that later
    alignment stages (stage 6) can operate on the full signal.
    """

    models = load_asr_models(config, language)

    # load and chop long audio
    import librosa

    sr = config.get("sample_rate", 16000)
    signal, _ = librosa.load(audio_path, sr=sr)

    chunk_sec = config.get("chunk_size", 30.0)
    overlap = config.get("chunk_overlap", 1.0)
    from .utils import chunk_audio

    chunks = list(chunk_audio(signal, sr, chunk_sec, overlap))

    # prepare per-model buckets
    model_outputs = [[] for _ in models]

    for chunk, offset in chunks:
        for i, model in enumerate(models):
            # pass language hint to transcription if we have one
            result = model.transcribe(audio=chunk, sr=sr, offset=offset, language=language)
            model_outputs[i].append(result)

    # flatten outputs for summary metrics
    outputs = [o for outs in model_outputs for o in outs]

    # combine the text from each model across chunks; normalize before scoring
    # so punctuation and case differences don't artificially reduce agreement
    raw_texts = [" ".join(o["text"] for o in outs) for outs in model_outputs]
    texts = [normalize_text(t) for t in raw_texts]

    agreement_score = compute_agreement_score(texts)

    confidences = [o["confidence"] for o in outputs]
    entropies = [o["entropy"] for o in outputs]
    no_speech = [o["no_speech_prob"] for o in outputs]

    rss = compute_rss(agreement_score, confidences, entropies, no_speech)

    consensus = build_consensus(texts)

    return {
        "reference_transcript": consensus,
        "reference_transcripts": raw_texts,  # return originals for readability
        "rss": float(rss),
        "agreement": float(agreement_score),
        "details": outputs,
    }