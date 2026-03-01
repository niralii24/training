import re
import unicodedata
from collections import Counter


# ── Text Cleaning ─────────────────────────────────────────

def clean_for_comparison(text):
    """
    Minimal cleaning before comparison — removes punctuation
    and normalizes whitespace but preserves letter differences.
    Works for any script (Arabic, Latin, CJK etc.)
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── WER (Word Error Rate) ─────────────────────────────────

def compute_wer(reference, hypothesis):
    """
    Word Error Rate — measures word-level differences.

    WER = (Substitutions + Deletions + Insertions) / len(reference)

    Lower WER = better match.
    Returns WER value and breakdown of error types.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0, {"substitutions": 0, "deletions": 0, "insertions": len(hyp_words)}

    # Dynamic programming edit distance at word level
    r, h = len(ref_words), len(hyp_words)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    # Traceback to count error types
    i, j = r, h
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1

    wer = (substitutions + deletions + insertions) / r
    return wer, {
        "substitutions": substitutions,
        "deletions":     deletions,
        "insertions":    insertions,
        "ref_words":     r,
        "hyp_words":     h
    }


# ── CER (Character Error Rate) ────────────────────────────

def compute_cer(reference, hypothesis):
    """
    Character Error Rate — measures character-level differences.

    More granular than WER — catches small spelling/diacritic
    differences that WER would count as full word errors.

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    Lower CER = better match.
    """
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if not ref_chars:
        return 1.0

    r, h = len(ref_chars), len(hyp_chars)
    dp = list(range(h + 1))

    for i in range(1, r + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, h + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])

    return dp[h] / r


# ── WER Variance Across Multiple References ───────────────

def compute_mean_wer(candidate, asr_references):
    """
    Computes mean WER across ALL ASR reference transcripts.

    Instead of comparing to just one reference, we compare to
    all ASR outputs from Stage 5 and average the WER scores.

    High variance in WER = ASR models disagree = less reliable signal.
    Low variance = ASR models agree = more reliable signal.
    """
    if not asr_references:
        return 1.0, 0.0, []

    wers = []
    for ref in asr_references:
        ref_clean = clean_for_comparison(ref)
        can_clean = clean_for_comparison(candidate)
        wer, _    = compute_wer(ref_clean, can_clean)
        wers.append(wer)

    mean_wer = sum(wers) / len(wers)
    variance = sum((w - mean_wer) ** 2 for w in wers) / len(wers)

    return mean_wer, variance, wers


# ── Token Confusion Matrix ────────────────────────────────

def compute_token_confusion(candidate, reference):
    """
    Builds a simple token-level confusion summary.

    Shows which words in the reference were:
    - matched correctly
    - substituted with something else
    - deleted entirely

    Useful for understanding WHERE the candidate diverges.
    """
    ref_words = clean_for_comparison(reference).split()
    can_words = clean_for_comparison(candidate).split()

    ref_counts = Counter(ref_words)
    can_counts = Counter(can_words)

    matched     = {}
    substituted = {}
    deleted     = {}

    for word, count in ref_counts.items():
        if word in can_counts:
            matched[word] = min(count, can_counts[word])
        else:
            deleted[word] = count

    # Words in candidate not in reference = insertions/substitutions
    inserted = {
        word: count for word, count in can_counts.items()
        if word not in ref_counts
    }

    return {
        "matched":     matched,
        "deleted":     deleted,
        "inserted":    inserted,
        "match_rate":  sum(matched.values()) / max(len(ref_words), 1)
    }


# ── Acoustic Score ────────────────────────────────────────

def compute_acoustic_score(mean_wer, mean_cer, wer_variance,
                           wer_weight=0.5, cer_weight=0.5):
    """
    Acoustic Score = 1 - (weighted WER + weighted CER)

    Scaled by WER variance — high variance means ASR references
    disagree, so we compress the score toward 0.5 (less confident).

    Returns score in [0.0, 1.0] where 1.0 = perfect match.
    """
    raw_score = 1.0 - (wer_weight * mean_wer + cer_weight * mean_cer)
    raw_score = max(0.0, min(1.0, raw_score))

    # Variance penalty: compress toward 0.5 when ASRs disagree
    # variance of 0 = no compression, variance of 1 = full compression
    variance_penalty = min(wer_variance * 2, 0.5)
    score = 0.5 + (raw_score - 0.5) * (1.0 - variance_penalty)

    return float(score)


# ── Main Stage 7 Function ─────────────────────────────────

def compute_acoustic_similarity(candidates, asr_results, language="en"):
    """
    Stage 7: Acoustic Similarity Scoring.

    For each candidate:
    1. Computes WER against each ASR reference from Stage 5
    2. Computes CER against the best ASR reference
    3. Computes mean WER + variance across all references
    4. Builds token confusion matrix
    5. Combines into final Acoustic Score

    Args:
        candidates:  list of raw/lightly normalized candidate strings
        asr_results: list of ASR result dicts from Stage 5
                     each has "transcript" and "confidence" keys
        language:    ISO language code

    Returns:
        list of dicts, one per candidate, sorted best → worst
    """
    print("\n" + "=" * 60)
    print("STAGE 7: ACOUSTIC SIMILARITY SCORING")
    print("=" * 60)

    # Extract reference transcripts from Stage 5 ASR results
    asr_references = [
        r["transcript"] for r in asr_results
        if r.get("transcript", "").strip()
    ]

    # Use highest-confidence ASR as primary reference for CER
    primary_reference = max(
        asr_results,
        key=lambda r: r.get("confidence", 0)
    )["transcript"] if asr_results else ""

    print(f"  ASR references available: {len(asr_references)}")
    print(f"  Primary reference: \"{primary_reference[:60]}\"")
    print(f"  Candidates to score: {len(candidates)}")

    scored = []

    for i, candidate in enumerate(candidates):
        print(f"\n  -- Candidate {i+1} --")

        can_clean  = clean_for_comparison(candidate)
        ref_clean  = clean_for_comparison(primary_reference)

        # ── WER across all references ─────────────────────
        mean_wer, wer_variance, all_wers = compute_mean_wer(
            can_clean, asr_references
        )

        # ── CER against primary reference ─────────────────
        mean_cer = compute_cer(ref_clean, can_clean)

        # ── Token confusion ───────────────────────────────
        confusion = compute_token_confusion(can_clean, ref_clean)

        # ── Final acoustic score ──────────────────────────
        score = compute_acoustic_score(mean_wer, mean_cer, wer_variance)

        result = {
            "index":        i + 1,
            "candidate":    candidate,
            "mean_wer":     mean_wer,
            "mean_cer":     mean_cer,
            "wer_variance": wer_variance,
            "all_wers":     all_wers,
            "confusion":    confusion,
            "score":        score,
        }
        scored.append(result)

        print(f"     Mean WER:     {mean_wer:.3f}")
        print(f"     WER variance: {wer_variance:.4f}")
        print(f"     Mean CER:     {mean_cer:.3f}")
        print(f"     Match rate:   {confusion['match_rate']:.2%}")
        print(f"     Score:        {score:.4f}")
        print(f"     Text:         \"{candidate[:60]}\"")

    # ── Rank ──────────────────────────────────────────────
    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"STAGE 7 SUMMARY")
    print(f"{'=' * 60}")
    for rank, s in enumerate(scored, 1):
        print(f"  {rank}. Candidate {s['index']} — "
              f"score={s['score']:.4f} | "
              f"WER={s['mean_wer']:.3f} | "
              f"CER={s['mean_cer']:.3f}")
    print(f"{'=' * 60}")

    return scored