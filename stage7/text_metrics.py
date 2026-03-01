"""
stage7/text_metrics.py
-----------------------
Low-level text comparison metrics for Stage 7.

All functions are self-contained (no external dependencies beyond the
standard library and numpy).  Arabic diacritics are stripped automatically
before any comparison so that harakat differences do not inflate error rates.

Functions
---------
normalize_for_comparison(text, language)
    Strip punctuation, diacritics, case-fold.

wer(reference, hypothesis, language) -> float
    Word Error Rate via Levenshtein edit distance.

cer(reference, hypothesis, language) -> float
    Character Error Rate (same algorithm, char-level).

fuzzy_similarity(a, b) -> float
    difflib SequenceMatcher ratio in [0, 1].

edit_alignment(ref_tokens, hyp_tokens) -> list[tuple]
    Return the full token-level alignment as (op, ref_tok, hyp_tok) triples.
    op ∈ {"equal", "substitute", "delete", "insert"}.

token_confusion_matrix(references, hypotheses, language) -> dict
    Aggregate substitution/deletion/insertion counts across all ref↔hyp pairs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Arabic diacritic / harakat strip (Unicode combining-mark removal)
# ---------------------------------------------------------------------------

# Arabic harakat (short vowels, tanwin, shadda, sukun, etc.) Unicode range
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]"
)

# Arabic/Latin punctuation to remove before comparison
_PUNCT_RE = re.compile(r"[؟،؛\!\?\.\,\:\;\'\"\(\)\[\]\{\}\-\_\+\=\/\\\|]")


def normalize_for_comparison(text: str, language: str = "") -> str:
    """
    Normalise text for error-rate comparison.

    Steps applied:
    1. Strip Arabic diacritics (harakat) — they are often inconsistently
       present between ASR output and human transcripts.
    2. Unicode NFKD normalisation to decompose composed characters.
    3. Remove punctuation (comparison is word/content focused).
    4. Lowercase (relevant for Latin/mixed-script content).
    5. Collapse whitespace.

    Parameters
    ----------
    text     : Raw transcript string.
    language : ISO-639-1 hint (e.g. ``"ar"``).  Currently used to decide
               whether to apply Arabic-specific normalisation.
    """
    if not text:
        return ""

    # Strip Arabic diacritics
    text = _ARABIC_DIACRITICS_RE.sub("", text)

    # Optional: strip tatweel (U+0640 — Arabic letter elongation)
    text = text.replace("\u0640", "")

    # Unicode decompose then drop combining marks (general normalisation)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    # Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # Lowercase, collapse whitespace
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein) — generic token sequence
# ---------------------------------------------------------------------------

def _levenshtein_counts(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """
    Compute (hits, substitutions, deletions, insertions) between two
    token sequences using the standard Levenshtein DP table.

    Returns
    -------
    (H, S, D, I) counts  (H = exact matches)
    """
    n = len(ref)
    m = len(hyp)

    # DP table of costs  dp[i][j] = min edits for ref[:i], hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace to count operation types
    S = D = I = H = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            H += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1

    return H, S, D, I


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def wer(reference: str, hypothesis: str, language: str = "") -> float:
    """
    Word Error Rate = (S + D + I) / len(reference_words).

    Both strings are normalised before comparison.

    Returns 0.0 if reference is empty, 1.0 if hypothesis is empty but
    reference is not (treat as all deletions).
    """
    ref_norm = normalize_for_comparison(reference, language)
    hyp_norm = normalize_for_comparison(hypothesis, language)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return 0.0
    if not hyp_words:
        return 1.0

    _, S, D, I = _levenshtein_counts(ref_words, hyp_words)
    error_rate = (S + D + I) / len(ref_words)
    return float(min(error_rate, 1.0))   # cap at 1.0


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def cer(reference: str, hypothesis: str, language: str = "") -> float:
    """
    Character Error Rate — same as WER but at character level.
    Spaces are preserved as characters after normalisation.
    """
    ref_norm = normalize_for_comparison(reference, language)
    hyp_norm = normalize_for_comparison(hypothesis, language)

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if not ref_chars:
        return 0.0
    if not hyp_chars:
        return 1.0

    _, S, D, I = _levenshtein_counts(ref_chars, hyp_chars)
    error_rate = (S + D + I) / len(ref_chars)
    return float(min(error_rate, 1.0))


# ---------------------------------------------------------------------------
# Fuzzy token similarity
# ---------------------------------------------------------------------------

def fuzzy_similarity(a: str, b: str, language: str = "") -> float:
    """
    difflib SequenceMatcher ratio on normalised token sequences.

    Returns float in [0, 1] where 1.0 = identical.
    Uses token (word) level rather than character level so short word
    differences are more visible.
    """
    a_norm = normalize_for_comparison(a, language)
    b_norm = normalize_for_comparison(b, language)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    a_toks = a_norm.split()
    b_toks = b_norm.split()

    return SequenceMatcher(None, a_toks, b_toks).ratio()


# ---------------------------------------------------------------------------
# Full edit alignment (for confusion matrix)
# ---------------------------------------------------------------------------

def edit_alignment(
    ref_tokens: list[str],
    hyp_tokens: list[str],
) -> list[tuple[str, Optional[str], Optional[str]]]:
    """
    Produce the token-level alignment as a list of operation triples.

    Each triple: (op, ref_token, hyp_token)
      "equal"      : (op, word, word)
      "substitute" : (op, ref_word, hyp_word)
      "delete"     : (op, ref_word, None)
      "insert"     : (op, None,     hyp_word)
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    # Build DP cost + backtrace tables
    dp  = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0]  = i
        ops[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j]  = j
        ops[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j]  = dp[i - 1][j - 1]
                ops[i][j] = "M"  # match
            else:
                d_cost = dp[i - 1][j]     + 1   # delete
                i_cost = dp[i][j - 1]     + 1   # insert
                s_cost = dp[i - 1][j - 1] + 1   # substitute
                best = min(d_cost, i_cost, s_cost)
                dp[i][j] = best
                if s_cost == best:
                    ops[i][j] = "S"
                elif d_cost == best:
                    ops[i][j] = "D"
                else:
                    ops[i][j] = "I"

    # Backtrace
    alignment: list[tuple] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] == "M":
            alignment.append(("equal", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and ops[i][j] == "S":
            alignment.append(("substitute", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ops[i][j] == "D"):
            alignment.append(("delete", ref_tokens[i - 1], None))
            i -= 1
        else:
            alignment.append(("insert", None, hyp_tokens[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Token confusion matrix
# ---------------------------------------------------------------------------

def token_confusion_matrix(
    references: list[str],
    hypotheses: list[str],
    language:   str = "",
    top_n:      int = 10,
) -> dict:
    """
    Build a token-level confusion analysis by aligning *each* reference to
    *each* hypothesis via the edit-alignment algorithm.

    Parameters
    ----------
    references : list of reference transcripts (e.g. all ASR model outputs).
    hypotheses : list of hypothesis strings to evaluate (e.g. one Excel option).
                 Typically you call this once per option.
    language   : ISO-639-1 code for normalisation.
    top_n      : Number of top substitution pairs to include in output.

    Returns
    -------
    dict with:
        substitutions      : Counter of (ref_word, hyp_word) pairs
        deletions          : Counter of deleted ref words
        insertions         : Counter of inserted hyp words
        total_substitutions: int
        total_deletions    : int
        total_insertions   : int
        top_substitutions  : list of {"ref": str, "hyp": str, "count": int}
        top_deletions      : list of {"token": str, "count": int}
        top_insertions     : list of {"token": str, "count": int}
        total_pairs        : int  total (ref, hyp) pairs evaluated
    """
    sub_counter  : Counter = Counter()
    del_counter  : Counter = Counter()
    ins_counter  : Counter = Counter()
    total_pairs  = 0

    for ref in references:
        for hyp in hypotheses:
            ref_norm = normalize_for_comparison(ref, language)
            hyp_norm = normalize_for_comparison(hyp, language)
            ref_toks = ref_norm.split()
            hyp_toks = hyp_norm.split()

            if not ref_toks:
                continue

            alignment = edit_alignment(ref_toks, hyp_toks)
            for op, r, h in alignment:
                if op == "substitute":
                    sub_counter[(r, h)] += 1
                elif op == "delete":
                    del_counter[r] += 1
                elif op == "insert":
                    ins_counter[h] += 1
            total_pairs += 1

    top_subs = [
        {"ref": r, "hyp": h, "count": c}
        for (r, h), c in sub_counter.most_common(top_n)
    ]
    top_dels = [
        {"token": tok, "count": c}
        for tok, c in del_counter.most_common(top_n)
    ]
    top_ins = [
        {"token": tok, "count": c}
        for tok, c in ins_counter.most_common(top_n)
    ]

    return {
        "substitutions":       dict(sub_counter),
        "deletions":           dict(del_counter),
        "insertions":          dict(ins_counter),
        "total_substitutions": sum(sub_counter.values()),
        "total_deletions":     sum(del_counter.values()),
        "total_insertions":    sum(ins_counter.values()),
        "top_substitutions":   top_subs,
        "top_deletions":       top_dels,
        "top_insertions":      top_ins,
        "total_pairs":         total_pairs,
    }
