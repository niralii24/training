"""
grammar_scorer.py — Stage 9 core

Uses Google's mT5 (Multilingual T5) as a fluency / grammar judge.

How it works
------------
mT5 is a seq2seq language model trained on 101 languages.
We exploit its learned probability distribution:

  Input  →  "grammar: <candidate text>"
  Target →  "<candidate text>"

The model's cross-entropy loss over the target tokens measures
how "expected" (natural) the sequence is.  Grammatically fluent,
human-sounding text scores low loss.  Garbled or unnatural text
scores high loss.

We convert loss → fluency score:
    raw_score  = exp(−loss)        (perplexity-based, in (0, 1])
    final_score = min-max normalize across all candidates

Model size options (passed as mt5_model):
    "google/mt5-small"   (~300 MB, fastest, good for CPU)
    "google/mt5-base"    (~580 MB, better quality)
    "google/mt5-large"   (~1.2 GB, best quality, GPU recommended)
"""

import math
import torch

# ── Lazy model cache — load once per process ─────────────
_model     = None
_tokenizer = None
_loaded_model_name = None


def _load_model(mt5_model: str, device: str):
    """
    Loads mT5 tokenizer and model into module-level cache.
    Subsequent calls with the same model name are instant.
    """
    global _model, _tokenizer, _loaded_model_name

    if _model is not None and _loaded_model_name == mt5_model:
        return _tokenizer, _model

    print(f"  Loading mT5 model: {mt5_model} …")

    # Import here so the rest of the pipeline can still import
    # grammar_scorer even if transformers isn't installed yet.
    try:
        from transformers import T5Tokenizer, MT5ForConditionalGeneration
        import transformers
        transformers.logging.set_verbosity_error()   # suppress tie_weights warnings
    except ImportError as e:
        raise ImportError(
            "transformers is required for Stage 9.\n"
            "Install it with:  pip install transformers sentencepiece"
        ) from e

    _tokenizer = T5Tokenizer.from_pretrained(mt5_model)
    _model     = MT5ForConditionalGeneration.from_pretrained(mt5_model)
    _model.to(device)
    _model.eval()
    _loaded_model_name = mt5_model

    print(f"  ✅ mT5 loaded on {device}")
    return _tokenizer, _model


# ── Single-candidate scorer ───────────────────────────────

def _cross_entropy_loss(text: str, tokenizer, model, device: str,
                        prompt_prefix: str = "grammar: ",
                        max_length: int = 256) -> float:
    """
    Computes the seq2seq cross-entropy loss for a single text.

    A lower loss means the model found the text more probable /
    natural, i.e. better grammar and fluency.

    Returns: float (loss value, lower is better)
    """
    input_text = prompt_prefix + text

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    tgt = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    # Replace padding token id in labels with -100 so it's ignored in loss
    tgt[tgt == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            labels         = tgt,
        )
    return outputs.loss.item()


def compute_raw_score(text: str, tokenizer, model, device: str) -> float:
    """
    Converts cross-entropy loss → raw fluency score in (0, 1].
    score = exp(−loss)

    A perfectly predicted sequence has loss→0, score→1.
    A completely unexpected sequence has high loss, score→0.
    """
    loss = _cross_entropy_loss(text, tokenizer, model, device)
    return math.exp(-loss)


# ── Multi-candidate scorer ────────────────────────────────

def score_grammar(
    candidates: list,
    language:   str  = "en",
    mt5_model:  str  = "google/mt5-small",
    device:     str  = "cpu",
) -> list:
    """
    Grammar / fluency scores for all candidates using mT5.

    Args:
        candidates:  list of raw candidate strings
        language:    ISO-639-1 code (used for logging; mT5 auto-detects)
        mt5_model:   HuggingFace model name
        device:      "cpu" or "cuda"

    Returns:
        list[dict] sorted by grammar_score descending — each dict:
        {
            "index":         int,
            "text":          str,
            "loss":          float,   # raw cross-entropy (lower = better)
            "raw_score":     float,   # exp(−loss), in (0,1]
            "grammar_score": float,   # min-max normalised, in [0,1]
        }
    """
    if not candidates:
        return []

    tokenizer, model = _load_model(mt5_model, device)

    raw_scores = []
    for i, text in enumerate(candidates):
        print(f"  Scoring candidate {i + 1}/{len(candidates)} …", end=" ")
        loss  = _cross_entropy_loss(text, tokenizer, model, device)
        score = math.exp(-loss)
        raw_scores.append((i, text, loss, score))
        print(f"loss={loss:.4f}  raw={score:.4f}")

    # ── Softmax over negated losses (numerically stable) ─
    # Works correctly even when all exp(-loss) values round to 0.
    # Step 1: shift losses by subtracting the minimum (log-sum-exp trick)
    losses     = [loss for _, _, loss, _ in raw_scores]
    min_loss   = min(losses)
    exp_vals   = [math.exp(-(l - min_loss)) for l in losses]  # shifted
    exp_sum    = sum(exp_vals)
    softmax    = [e / exp_sum for e in exp_vals]              # sums to 1.0

    results = []
    for idx, (i, text, loss, raw) in enumerate(raw_scores):
        results.append({
            "index":         i,
            "text":          text,
            "loss":          round(loss,           4),
            "raw_score":     round(raw,            4),
            "grammar_score": round(softmax[idx],   4),
        })

    results.sort(key=lambda x: x["grammar_score"], reverse=True)
    return results
