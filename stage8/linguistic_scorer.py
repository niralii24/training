"""
stage8/linguistic_scorer.py
----------------------------
Multilingual linguistic quality scoring using XLM-RoBERTa pseudo-perplexity.

Why pseudo-perplexity?
----------------------
A masked language model (XLM-RoBERTa) is trained to predict masked tokens from
context.  A grammatically natural sentence lets the model predict each masked
token with high confidence → low pseudo-perplexity (PPPL).  A garbled,
ungrammatical, or structurally broken transcript confuses the model → high PPPL.

This requires ZERO language-specific rules or hardcoded vocabulary — the model
itself encodes grammar, punctuation conventions, and word-order expectations for
100+ languages including Arabic (MSA + dialects), English, French, Spanish, etc.

Algorithm (PLL-masked, sampled for speed)
------------------------------------------
For a text of N tokens, we sample up to ``max_samples`` positions.  For each:
  1. Replace that token with <mask>.
  2. Run one forward pass.
  3. Record log P(original_token | context).
Pseudo-PPL = exp(  -mean(log_probs)  ).

Structural integrity checks (language-agnostic, Unicode-based)
--------------------------------------------------------------
Completely separate from the model — purely rule-based but genuinely
universal because they work on Unicode codepoint properties:
  • Bracket balance  : () [] {} «» ‹›
  • Quotation balance: " " ' '  (handles Arabic and Latin)
  • Repeated punct   : more than 3 of the same punct in a row is suspicious
  • Script consistency: flags mixing of unrelated scripts (e.g. Arabic + Cyrillic)
    Uses Unicode script blocks via unicodedata.name() — no hardcoding needed.
"""

from __future__ import annotations

import logging
import math
import random
import re
import unicodedata
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name — XLM-RoBERTa base (100 languages, 125 M params, fast enough on CPU)
# Switch to "xlm-roberta-large" for higher quality at the cost of speed.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "xlm-roberta-base"


# ---------------------------------------------------------------------------
# Model loader (singleton pattern — load once per process)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}   # model_name -> (model, tokenizer)


def _load_model(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load and cache XLM-RoBERTa tokenizer + model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    import os
    cache_dir = os.environ.get("HF_HOME", None)

    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        logger.info("Stage 8 ▶ loading %s on %s …", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
        model.eval()
        model.to(device)
        logger.info("Stage 8 ▶ %s ready.", model_name)
        _MODEL_CACHE[model_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as exc:
        logger.error("Stage 8 ▶ could not load %s: %s", model_name, exc)
        raise


# ---------------------------------------------------------------------------
# Pseudo-perplexity via masked language modelling
# ---------------------------------------------------------------------------

def compute_pseudo_perplexity(
    text:        str,
    model,
    tokenizer,
    device:      str  = "cpu",
    max_samples: int  = 40,
    seed:        int  = 42,
) -> Optional[float]:
    """
    Compute pseudo-perplexity (PPPL) for *text* using a masked LM.

    Returns
    -------
    float ≥ 1.0, or None if the text is too short to score.
    Lower = more linguistically natural.
    """
    if not text or not text.strip():
        return None

    encoding = tokenizer(
        text,
        return_tensors       = "pt",
        truncation           = True,
        max_length           = 512,
        add_special_tokens   = True,
        return_attention_mask= True,
    )
    input_ids       = encoding["input_ids"].to(device)          # (1, L)
    attention_mask  = encoding["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    # Special tokens (CLS, SEP, PAD) should not be masked
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    } - {None}

    maskable = [
        i for i in range(seq_len)
        if input_ids[0, i].item() not in special_ids
    ]

    if len(maskable) < 2:
        return None   # too short

    # Sample positions to avoid O(N²) cost on long texts
    rng = random.Random(seed)
    positions = rng.sample(maskable, min(max_samples, len(maskable)))

    mask_id   = tokenizer.mask_token_id
    log_probs: list[float] = []

    with torch.no_grad():
        for pos in positions:
            masked_ids               = input_ids.clone()
            original_token_id        = masked_ids[0, pos].item()
            masked_ids[0, pos]       = mask_id

            outputs = model(
                input_ids      = masked_ids,
                attention_mask = attention_mask,
            )
            # logits: (1, L, vocab)
            logits_at_pos = outputs.logits[0, pos]           # (vocab,)
            log_prob = torch.log_softmax(logits_at_pos, dim=-1)
            lp = log_prob[original_token_id].item()
            log_probs.append(lp)

    if not log_probs:
        return None

    pppl = math.exp(-sum(log_probs) / len(log_probs))
    return pppl


def pppl_to_score(pppl: Optional[float], max_pppl: float = 200.0) -> float:
    """
    Map pseudo-perplexity → quality score in [0, 1].

    PPPL = 1   → score = 1.0  (perfect prediction, completely natural)
    PPPL = max_pppl → score ≈ 0.0

    Uses log-scale so differences at low PPPL (good text) matter more than
    differences at high PPPL (already clearly bad text).

    max_pppl is calibrated on XLM-RoBERTa-base:
      • Grammatically natural Arabic prose:   PPPL ≈ 5–30
      • Random/garbled Arabic:                PPPL ≈ 100–400
    """
    if pppl is None:
        return 0.5   # neutral fallback when scoring impossible

    pppl  = max(1.0, pppl)
    score = 1.0 - math.log(pppl) / math.log(max(max_pppl, 2.0))
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Structural integrity checks (pure Unicode, no language rules)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Annotation stripping — remove inline markers before PPPL
# ---------------------------------------------------------------------------
# Matches bracketed annotations up to 60 chars:  {صمت}  [صمت]  [ثيو:]  (SFX) …
# Longer spans are left alone so we don't accidentally eat real sentence content.
_ANNOTATION_RE = re.compile(r'[\[{(][^\]})]{0,60}[\]})]')


def _strip_annotations(text: str) -> str:
    """
    Remove inline annotation markers from a transcript before PPPL scoring.

    Examples removed
    ----------------
    {صمت}   [صمت]   [ثيو:]   [مايا:]   (laughs)   (SFX: noise)

    Rationale
    ---------
    XLM-RoBERTa finds *structured* repetitive patterns like
    ``[speaker:] utterance  [speaker:] utterance`` easy to predict, which
    gives transcripts with speaker labels an artificially low PPPL and
    therefore an inflated LGS — even though the *spoken* words are identical
    to options without labels.  Stripping annotations ensures PPPL reflects
    only the linguistic quality of the spoken content.
    """
    cleaned = _ANNOTATION_RE.sub(' ', text)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned).strip()
    return cleaned


# Matching bracket pairs
_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "«": "»",
    "‹": "›",
    "\u201c": "\u201d",   # " "
    "\u2018": "\u2019",   # ' '
}
_CLOSERS = set(_BRACKET_PAIRS.values())
_OPENERS = set(_BRACKET_PAIRS.keys())

# Punctuation characters whose repetition is suspect
_PUNCT_CHARS = set('؟،؛.!?,;:"\'-–—')


def check_structural_integrity(text: str) -> dict:
    """
    Language-agnostic structural checks on Unicode text.

    Returns
    -------
    dict:
        unmatched_brackets  : int   count of unmatched open/close brackets
        repeated_punct      : int   count of runs of 4+ identical punct chars
        score               : float [0,1] — 1.0 = fully clean structure
        issues              : list[str]   human-readable issue descriptions
    """
    issues: list[str] = []

    # ── Bracket balance ────────────────────────────────────────────────────
    stack: list[str] = []
    unmatched = 0
    for ch in text:
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSERS:
            # Find the expected opener for this closer
            expected_opener = next(
                (o for o, c in _BRACKET_PAIRS.items() if c == ch), None
            )
            if stack and expected_opener and stack[-1] == expected_opener:
                stack.pop()
            else:
                unmatched += 1
    unmatched += len(stack)  # unclosed openers
    if unmatched:
        issues.append(f"{unmatched} unmatched bracket(s)/quote(s)")

    # ── Repeated punctuation runs ──────────────────────────────────────────
    repeated_runs = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _PUNCT_CHARS:
            run_len = 1
            while i + run_len < len(text) and text[i + run_len] == ch:
                run_len += 1
            if run_len >= 4:
                repeated_runs += 1
                issues.append(f"repeated '{ch}' × {run_len}")
            i += run_len
        else:
            i += 1

    # ── Script consistency (Unicode block detection) ───────────────────────
    # Determine which Unicode scripts are present (ignoring punctuation/space)
    scripts: set[str] = set()
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch, "")
                # Extract script from Unicode name: first word(s) before the
                # specific character name (e.g. "ARABIC LETTER BA" → "ARABIC")
                script = name.split()[0] if name else "UNKNOWN"
                scripts.add(script)
            except Exception:
                pass

    # Flag if incompatible scripts are mixed (allow ARABIC+LATIN for loanwords,
    # but flag truly alien combos like ARABIC+CYRILLIC or ARABIC+HEBREW)
    suspicious_mixes = {
        frozenset({"ARABIC", "CYRILLIC"}),
        frozenset({"ARABIC", "HEBREW"}),
        frozenset({"LATIN", "CYRILLIC"}),
        frozenset({"LATIN", "HEBREW"}),
    }
    script_issue = False
    for pair in suspicious_mixes:
        if pair.issubset(scripts):
            issues.append(f"suspicious script mix: {sorted(pair)}")
            script_issue = True
            break

    # ── Composite structural score ─────────────────────────────────────────
    # Each unmatched bracket: −0.10  (capped at −0.40)
    # Each repeated-punct run: −0.05 (capped at −0.20)
    # Script mix: −0.15
    penalty = (
        min(0.40, unmatched * 0.10)
        + min(0.20, repeated_runs * 0.05)
        + (0.15 if script_issue else 0.0)
    )
    struct_score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    return {
        "unmatched_brackets": unmatched,
        "repeated_punct":     repeated_runs,
        "scripts_found":      sorted(scripts),
        "score":              struct_score,
        "issues":             issues,
    }


# ---------------------------------------------------------------------------
# Combined linguistic quality score
# ---------------------------------------------------------------------------

def compute_linguistic_score(
    text:      str,
    model,
    tokenizer,
    device:    str   = "cpu",
    max_pppl:  float = 200.0,
    max_samples: int = 40,
) -> dict:
    """
    Compute the full linguistic quality assessment for one transcript option.

    Returns
    -------
    dict:
        pppl               : float | None   raw pseudo-perplexity
        pppl_score         : float [0,1]    normalised naturalness score
        structural         : dict           from check_structural_integrity()
        lgs                : float [0,1]    Linguistic Grammar Score (final)
                             = 0.75 × pppl_score  +  0.25 × structural_score
    """
    # Strip inline annotations (speaker labels, silence markers) before PPPL
    # so the model scores the *spoken words* rather than annotation style.
    pppl_text  = _strip_annotations(text)
    pppl       = compute_pseudo_perplexity(pppl_text, model, tokenizer, device, max_samples)
    pppl_score = pppl_to_score(pppl, max_pppl)
    # Structural check runs on the ORIGINAL text (bracket balance still matters)
    structural = check_structural_integrity(text)

    # Blend: model-based naturalness carries most weight;
    # structural is a hard safety check for obviously broken text.
    lgs = 0.75 * pppl_score + 0.25 * structural["score"]

    return {
        "pppl":         pppl,
        "pppl_score":   pppl_score,
        "structural":   structural,
        "lgs":          float(np.clip(lgs, 0.0, 1.0)),
    }
