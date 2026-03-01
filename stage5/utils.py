import re
from difflib import SequenceMatcher


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\|", " ", text)  # remove CTC separator if any
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_agreement(t1, t2):
    w1 = t1.split()
    w2 = t2.split()

    matcher = SequenceMatcher(None, w1, w2)
    matches = sum(m.size for m in matcher.get_matching_blocks())

    total = max(len(w1), len(w2))
    return matches / total if total else 0


def chunk_audio(signal, sr, chunk_size=30.0, overlap=1.0):
    """Yield (segment, offset) pairs for a long waveform.

    ``chunk_size`` and ``overlap`` are in seconds.
    """
    if chunk_size <= 0 or len(signal) == 0:
        yield signal, 0.0
        return

    step = int((chunk_size - overlap) * sr)
    chunk_len = int(chunk_size * sr)
    total_len = len(signal)
    start = 0
    while start < total_len:
        end = min(start + chunk_len, total_len)
        yield signal[start:end], start / sr
        start += step