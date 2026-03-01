import numpy as np


def normalize_confidence(confidences):
    """
    Normalize different confidence scales so that every model contributes
    roughly in the [0,1] range.

    * Whisper avg_logprob comes out negative, so we exponentiate it.
    * wav2vec2 already produces probabilities.
    """

    normalized = []

    for c in confidences:
        if c < 0:  # Whisper logprob
            # avoid overflow on very small numbers
            normalized.append(np.exp(c))
        else:
            normalized.append(c)

    return np.mean(normalized)


def compute_rss(agreement, confidences, entropies, no_speech):
    """
    Reference Stability Score (RSS) attempts to quantify how reliable the
    acoustic reference is. Higher values indicate all models are in agreement
    and produced confident, low‑entropy, speech‑like output.  This score is
    intentionally conservative so the downstream pipeline can down‑weight the
    acoustic channel when the score is low.
    """

    avg_conf = normalize_confidence(confidences)

    avg_entropy = np.mean(entropies)
    entropy_factor = 1 / (1 + avg_entropy)

    speech_factor = np.mean([1 - p for p in no_speech])

    return float(agreement * avg_conf * entropy_factor * speech_factor)


def compute_agreement_score(texts):
    """
    Token‑level agreement across multiple transcripts.  A value of 1.0 means
    every model produced the exact same sequence after normalization; lower
    values correspond to disagreement.
    """

    if not texts:
        return 1.0

    from collections import Counter
    token_lists = [t.split() for t in texts]
    max_len = max(len(t) for t in token_lists)
    scores = []

    for i in range(max_len):
        tokens = [tokens[i] for tokens in token_lists if i < len(tokens)]
        if not tokens:
            continue
        top = Counter(tokens).most_common(1)[0][1]
        scores.append(top / len(tokens))

    return float(np.mean(scores))