"""
Stage 7 – Acoustic Similarity Metrics (Text-level, Multi-reference)
====================================================================
Scores each transcript option textually against the Stage 5 ASR outputs,
computing mean/variance WER, CER, fuzzy token similarity, cross-model
consistency, and a token-level confusion matrix.  Combined with Stage 6's
acoustic alignment score (AQS) into a single final ranking.

Exports
-------
run_stage7(excel_options, asr_references, language, correct_option)
    Score all 5 Excel options against the ASR reference pool.

compute_combined_score(aqs, tss)
    Blend AQS (Stage 6) and TSS (Stage 7) into a final score.
"""

from .stage7_runner import run_stage7, compute_combined_score

__all__ = ["run_stage7", "compute_combined_score"]
