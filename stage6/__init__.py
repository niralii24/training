"""
Stage 6 – Forced Alignment Scoring (WhisperX)
=============================================
Consumes the cleaned audio (stage 1) and the ASR transcript/segments
(stage 5) to produce fine-grained alignment quality metrics.
"""

from .stage6_runner import run_stage6

__all__ = ["run_stage6"]
