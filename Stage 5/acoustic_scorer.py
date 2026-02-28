import torch
import numpy as np
from collections import Counter

# ── Optional imports (graceful fallback if not installed) ─
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  whisper not installed — acoustic scoring unavailable")

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False


# ── Model Registry ────────────────────────────────────────
# Each entry: (model_type, model_name, weight)
WHISPER_MODELS = [
    ("whisper", "base",  0.4),
    ("whisper", "small", 0.6),
    # ("whisper", "large", 1.0),  # uncomment if compute allows
]

WAV2VEC_MODELS = [
    # ("wav2vec2", "facebook/wav2vec2-large-960h", 0.5),
    # ("wav2vec2", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", 0.5),
]


# ── Whisper Runner ────────────────────────────────────────
# ── Model cache — loaded once, reused forever ─────────────
_WHISPER_CACHE = {}

def get_whisper_model(model_name):
    if model_name not in _WHISPER_CACHE:
        print(f"  Loading Whisper {model_name} (first time)...")
        _WHISPER_CACHE[model_name] = whisper.load_model(model_name)
        print(f"  ✅ Whisper {model_name} cached")
    return _WHISPER_CACHE[model_name]

def run_whisper(waveform, sample_rate, model_name, language=None):
    """
    Runs a single Whisper model and returns transcript + confidence metrics.
    """
    print(f"    Running Whisper {model_name}...")
    model = get_whisper_model(model_name)
    
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.squeeze().numpy()
    else:
        audio_np = np.array(waveform).squeeze()

    if sample_rate != 16000:
        import torchaudio
        waveform_tensor = torch.tensor(audio_np).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_np  = resampler(waveform_tensor).squeeze().numpy()

    result = model.transcribe(
        audio_np,
        language=language,
        verbose=False,
        fp16=False,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    transcript = result["text"].strip()
    segments   = result.get("segments", [])

    if segments:
        avg_logprob    = np.mean([s["avg_logprob"]    for s in segments])
        no_speech_prob = np.mean([s["no_speech_prob"] for s in segments])
        logprobs       = [s["avg_logprob"] for s in segments]
        entropy        = float(np.var(logprobs)) if len(logprobs) > 1 else 0.0
    else:
        avg_logprob    = -1.0
        no_speech_prob = 0.5
        entropy        = 1.0

    confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))

    return {
        "model":          f"whisper-{model_name}",
        "transcript":     transcript,
        "confidence":     confidence,
        "avg_logprob":    float(avg_logprob),
        "no_speech_prob": float(no_speech_prob),
        "entropy":        entropy,
        "language":       result.get("language", "unknown"),
    }


# ── Wav2Vec2 Runner ───────────────────────────────────────

def run_wav2vec2(waveform, sample_rate, model_name):
    """
    Runs a wav2vec2 model and returns transcript + confidence.
    """
    print(f"    Running wav2vec2 {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name)

    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.squeeze().numpy()
    else:
        audio_np = np.array(waveform).squeeze()

    inputs = processor(
        audio_np, sampling_rate=sample_rate,
        return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcript    = processor.decode(predicted_ids[0])

    probs      = torch.softmax(logits, dim=-1)
    max_probs  = probs.max(dim=-1).values
    confidence = float(max_probs.mean().item())
    log_probs  = torch.log(probs + 1e-9)
    entropy    = float(-(probs * log_probs).sum(dim=-1).mean().item())

    return {
        "model":          f"wav2vec2-{model_name.split('/')[-1]}",
        "transcript":     transcript.strip(),
        "confidence":     confidence,
        "avg_logprob":    float(torch.log(max_probs.mean()).item()),
        "no_speech_prob": 0.0,
        "entropy":        entropy,
        "language":       "unknown",
    }


# ── Reference Stability Score ─────────────────────────────

def compute_stability_score(asr_results):
    """
    Measures agreement between all ASR models using pairwise
    character-level Jaccard similarity.

    High agreement → high stability → trust acoustic reference more.
    Low agreement  → low stability  → compress scores toward center.
    """
    transcripts = [r["transcript"] for r in asr_results if r["transcript"]]

    if len(transcripts) < 2:
        return 1.0, "Only one model ran — stability assumed"

    def char_jaccard(a, b):
        set_a = set(a.replace(" ", ""))
        set_b = set(b.replace(" ", ""))
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union        = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    similarities = []
    for i in range(len(transcripts)):
        for j in range(i + 1, len(transcripts)):
            similarities.append(char_jaccard(transcripts[i], transcripts[j]))

    stability = float(np.mean(similarities))
    level     = "high" if stability > 0.7 else "medium" if stability > 0.4 else "low"
    reason    = (
        f"Pairwise similarity across {len(transcripts)} models: "
        f"{stability:.2%} ({level} stability)"
    )
    return stability, reason


def compute_weighted_reference(asr_results, model_weights):
    """
    Picks the transcript from the highest weighted+confident model
    as the consensus reference.
    """
    if not asr_results:
        return "", 0.0

    best_score  = -1
    best_result = asr_results[0]

    for result in asr_results:
        weight = model_weights.get(result["model"], 0.5)
        score  = weight * result["confidence"] * (1.0 - result["no_speech_prob"])
        if score > best_score:
            best_score  = score
            best_result = result

    return best_result["transcript"], best_score


# ── Similarity Metrics ────────────────────────────────────

def levenshtein_similarity(a, b):
    """
    Normalized Levenshtein (edit distance) similarity.
    Returns 1.0 for identical strings, 0.0 for completely different.

    Order-aware — the key advantage over Jaccard/ngrams for
    differentiating near-identical candidates.
    """
    a = a.replace(" ", "")
    b = b.replace(" ", "")

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    len_a, len_b = len(a), len(b)
    dp = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len_b + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])

    return 1.0 - (dp[len_b] / max(len_a, len_b))


def score_candidate_against_reference(candidate, reference, stability):
    """
    Scores a candidate against the acoustic reference using 3 metrics:

    1. Levenshtein similarity (0.50 weight)
       Order-aware edit distance — most discriminative for near-identical
       candidates that only differ in a few words or characters.

    2. Character bigram F1 (0.30 weight)
       Partial match — handles Arabic morphology where words share
       character patterns even when not identical.

    3. Word Jaccard (0.20 weight)
       Content overlap — catches gross content differences.

    Final score is scaled by stability so an unreliable reference
    doesn't confidently pick the wrong candidate.
    """
    if not reference or not candidate:
        return 0.0, "Empty candidate or reference"

    # ── Metric 1: Levenshtein ─────────────────────────────
    lev_sim = levenshtein_similarity(candidate, reference)

    # ── Metric 2: Character bigram F1 ────────────────────
    def get_bigrams(text):
        t = text.replace(" ", "")
        return [t[i:i+2] for i in range(len(t) - 1)]

    ref_bigrams = get_bigrams(reference)
    can_bigrams = get_bigrams(candidate)

    if ref_bigrams and can_bigrams:
        ref_counts = Counter(ref_bigrams)
        can_counts = Counter(can_bigrams)
        overlap    = sum((ref_counts & can_counts).values())
        precision  = overlap / len(can_bigrams)
        recall     = overlap / len(ref_bigrams)
        bigram_f1  = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
    else:
        bigram_f1 = 0.0

    # ── Metric 3: Word Jaccard ────────────────────────────
    ref_words = set(reference.split())
    can_words = set(candidate.split())
    if ref_words or can_words:
        word_jaccard = len(ref_words & can_words) / len(ref_words | can_words)
    else:
        word_jaccard = 0.0

    # ── Weighted combination ──────────────────────────────
    raw_score = (
        0.50 * lev_sim +
        0.30 * bigram_f1 +
        0.20 * word_jaccard
    )

    # Scale by stability — compress toward 0.5 if reference is unreliable
    scaled_score = 0.5 + (raw_score - 0.5) * stability

    reason = (
        f"lev={lev_sim:.3f}, bigram_f1={bigram_f1:.3f}, "
        f"word_jac={word_jaccard:.3f} → raw={raw_score:.3f}, "
        f"stability={stability:.2f}, final={scaled_score:.3f}"
    )
    return float(scaled_score), reason


# ── Main Stage 5 Function ─────────────────────────────────

def run_acoustic_reference(waveform, sample_rate, candidates, language=None):
    """
    Stage 5: Independent Acoustic Reference (Multi-Model).

    1. Runs all configured ASR models on the raw audio
    2. Computes Reference Stability Score across models
    3. Builds weighted consensus reference transcript
    4. Scores each candidate against the reference
    5. Returns ranked candidates with full scoring breakdown
    """
    print("\n" + "=" * 60)
    print("STAGE 5: ACOUSTIC REFERENCE SCORING")
    print("=" * 60)

    asr_results   = []
    model_weights = {}

    # ── Run Whisper models ────────────────────────────────
    if WHISPER_AVAILABLE:
        for _, model_name, weight in WHISPER_MODELS:
            try:
                result = run_whisper(waveform, sample_rate, model_name, language)
                asr_results.append(result)
                model_weights[result["model"]] = weight
                print(f"    ✅ {result['model']}: \"{result['transcript'][:60]}\"")
                print(f"       confidence={result['confidence']:.3f}, "
                      f"no_speech={result['no_speech_prob']:.3f}, "
                      f"entropy={result['entropy']:.3f}")
            except Exception as e:
                print(f"    ❌ whisper-{model_name} failed: {e}")
    else:
        print("  ⚠️  Whisper unavailable — skipping")

    # ── Run Wav2Vec2 models ───────────────────────────────
    if WAV2VEC_AVAILABLE and WAV2VEC_MODELS:
        for _, model_name, weight in WAV2VEC_MODELS:
            try:
                result = run_wav2vec2(waveform, sample_rate, model_name)
                asr_results.append(result)
                model_weights[result["model"]] = weight
                print(f"    ✅ {result['model']}: \"{result['transcript'][:60]}\"")
            except Exception as e:
                print(f"    ❌ wav2vec2-{model_name} failed: {e}")

    # ── Fallback: no models ran ───────────────────────────
    if not asr_results:
        print("\n  ❌ No ASR models available — cannot score candidates")
        return {
            "asr_results":       [],
            "stability_score":   0.0,
            "stability_reason":  "No models ran",
            "reference":         "",
            "scored_candidates": [(i+1, c, 0.0) for i, c in enumerate(candidates)],
            "best_candidate":    (1, candidates[0], 0.0) if candidates else None,
        }

    # ── Stability Score ───────────────────────────────────
    print(f"\n  Computing Reference Stability Score...")
    stability, stability_reason = compute_stability_score(asr_results)
    print(f"  Stability: {stability:.2%} — {stability_reason}")

    # ── Weighted Reference ────────────────────────────────
    reference, ref_score = compute_weighted_reference(asr_results, model_weights)
    print(f"\n  Reference transcript (weighted best):")
    print(f"  \"{reference[:100]}{'...' if len(reference) > 100 else ''}\"")

    # ── Score Each Candidate ──────────────────────────────
    print(f"\n  Scoring {len(candidates)} candidates against reference...")
    scored_candidates = []

    for i, candidate in enumerate(candidates):
        score, reason = score_candidate_against_reference(
            candidate, reference, stability
        )
        scored_candidates.append((i + 1, candidate, score))
        print(f"\n  -- Candidate {i+1} --")
        print(f"     Score:  {score:.4f}")
        print(f"     Detail: {reason}")
        print(f"     Text:   \"{candidate[:70]}{'...' if len(candidate) > 70 else ''}\"")

    # ── Rank & Select Best ────────────────────────────────
    scored_candidates.sort(key=lambda x: x[2], reverse=True)
    best_idx, best_text, best_score = scored_candidates[0]

    print(f"\n{'=' * 60}")
    print(f"STAGE 5 RESULTS")
    print(f"{'=' * 60}")
    print(f"  Stability:      {stability:.2%}")
    print(f"  Reference:      \"{reference[:60]}\"")
    print(f"  Best candidate: #{best_idx} (score={best_score:.4f})")
    print(f"  Text:           \"{best_text[:60]}\"")
    print(f"\n  Full ranking:")
    for rank, (idx, text, score) in enumerate(scored_candidates, 1):
        print(f"    {rank}. Candidate {idx} — score={score:.4f}")
    print(f"{'=' * 60}")

    return {
        "asr_results":       asr_results,
        "stability_score":   stability,
        "stability_reason":  stability_reason,
        "reference":         reference,
        "scored_candidates": scored_candidates,
        "best_candidate":    (best_idx, best_text, best_score),
    }