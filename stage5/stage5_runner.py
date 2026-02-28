from .asr_models import load_asr_models
from .utils import normalize_text, word_agreement
from .consensus import build_consensus
from .scoring import compute_rss, compute_agreement_score
import numpy as np


def run_stage5(audio_path: str, config: dict, language: str = None) -> dict:
    """
    Main entry point for Stage 5 - runs Whisper ASR on the audio file.

    The audio path is passed directly to Whisper so it can use its own
    built-in long-form transcription logic (30-second sliding window with
    proper context management).  Manual chunking was removed because it
    bypassed Whisper's internal context handling and was the root cause of
    the 'random output' problem.

    The ``language`` argument is intentionally ignored - Whisper auto-detects
    the language reliably without any hint.
    """

    models = load_asr_models(config)

    outputs = []
    for model in models:
        result = model.transcribe(audio_path=audio_path)
        outputs.append(result)

    texts = [o["text"] for o in outputs]

    agreement_score = compute_agreement_score(texts)

    confidences = [o["confidence"]    for o in outputs]
    entropies   = [o["entropy"]       for o in outputs]
    no_speech   = [o["no_speech_prob"] for o in outputs]

    rss = compute_rss(agreement_score, confidences, entropies, no_speech)

    consensus = build_consensus(texts)

    return {
        "reference_transcript":  consensus,
        "reference_transcripts": texts,
        "rss":       float(rss),
        "agreement": float(agreement_score),
        "details":   outputs,
    }
