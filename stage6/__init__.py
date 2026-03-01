"""
Stage 6 – Forced Alignment Scoring (WhisperX)
=============================================
Consumes the cleaned audio (Stage 1) and the ASR transcript / segments
(Stage 5) to produce fine-grained word-level and phoneme-level alignment
quality metrics, anomaly detection, and a composite Alignment Quality Score.

Exports
-------
run_stage6(audio_path, stage5_output, language, ...)
    Score a single transcript against an audio file.

run_stage6_excel_options(audio_wav, excel_path, audio_id, language, ...)
    Score option_1 … option_5 from a transcripts Excel sheet and rank them.
"""

from .stage6_runner import run_stage6, run_stage6_excel_options

__all__ = ["run_stage6", "run_stage6_excel_options"]
