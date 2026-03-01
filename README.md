# Golden Transcription System: Stage-by-Stage Overview

This project is a modular pipeline for high-quality audio transcription, language detection, and text normalization. Each stage is designed to process audio and text data with precision, leveraging state-of-the-art models and custom logic. Below is a detailed breakdown of each stage:

---

## Stage 1: Audio Loading & Analysis
- **Purpose:** Load audio files, standardize format, and analyze quality.
- **Process:**
  - Converts input audio to 16kHz mono float32 using FFmpeg.
  - Analyzes audio for signal-to-noise ratio (SNR), background noise, voice activity, and average energy.
  - Trims silence using neural Voice Activity Detection (VAD) and energy-based methods, preserving natural pauses.
  - Outputs: Cleaned waveform and detailed acoustic metadata for downstream stages.

## Stage 2: Language Detection
- **Purpose:** Identify the spoken language in the audio.
- **Process:**
  - Uses Whisper (small and medium models) for robust language detection.
  - Primary detection with Whisper small; if confidence is low, runs fallback ensemble with Whisper medium on audio chunks.
  - Combines results using weighted ensemble for best accuracy.
  - Skips detection if speech content is too low.
  - Outputs: Language code, confidence score, probability distribution, and detection method.

## Stage 3: Transcript Loading & Text Normalization
- **Purpose:** Prepare and clean transcript text for further processing.
- **Process:**
  - Loads raw transcript data.
  - Applies normalization routines to standardize punctuation, remove artifacts, and ensure text consistency.
  - Outputs: Cleaned transcript ready for candidate filtering and scoring.

## Stage 4: Candidate Filtering
- **Purpose:** Filter and select the best transcript candidates.
- **Process:**
  - Applies custom filtering logic to remove low-quality or irrelevant transcript candidates.
  - May use heuristics, confidence scores, or metadata from previous stages.
  - Outputs: Filtered set of candidate transcripts for scoring.

## Stage 5: ASR Models & Consensus Scoring
- **Purpose:** Score transcript candidates and build consensus.
- **Process:**
  - Runs multiple Automatic Speech Recognition (ASR) models for diverse predictions.
  - Scores candidates based on model outputs, confidence, and agreement.
  - Uses consensus logic to select the most reliable transcript.
  - Outputs: Final transcript and scoring details.

## Stage 6: Alignment & Metrics
- **Purpose:** Align transcript with audio and compute quality metrics.
- **Process:**
  - Aligns transcript text to audio waveform using pause alignment and acoustic features.
  - Detects hallucinations and computes metrics such as accuracy, timing, and pause correspondence.
  - Outputs: Aligned transcript, hallucination flags, and detailed metrics.

## Stage 7: Acoustic Similarity & Finalization
- **Purpose:** Assess how closely each candidate transcript matches what multiple ASR models actually heard, using text-level metrics.
- **Process:**
  - Compares each candidate transcript to all ASR outputs from Stage 5 (multi-model references).
  - Computes per-option metrics:
    - Word Error Rate (WER) and Character Error Rate (CER) against each ASR reference.
    - Mean and variance of WER/CER across references (variance shows model agreement).
    - Fuzzy similarity (difflib token-ratio) to capture paraphrase similarity.
    - Consistency score: how stable the error rate is across models.
    - Transcript Similarity Score (TSS): a composite score for ranking.
    - Token-level confusion matrix: highlights common substitutions, deletions, insertions.
  - Ranks all candidates by TSS and selects the best match.
- **Outputs:**
  - Per-candidate metrics and scores
  - Ranked list of candidates
  - Best option and its similarity score
## Stage 8: Multilingual Linguistic Grammar Scoring
- **Purpose:** Score transcript candidates for linguistic quality and structural integrity across 100+ languages.
- **Process:**
  - Loads XLM-RoBERTa (multilingual masked language model) for grammar scoring.
  - Computes pseudo-perplexity (PPPL) for each candidate transcript, measuring how grammatically natural each option is.
  - Applies Unicode-based structural integrity checks: bracket/quotation balance, repeated punctuation, script consistency.
  - Uses tie-breaking logic to resolve cases where multiple candidates have identical scores, favoring the most textually distinct option.
  - Outputs: Linguistic Grammar Score (LGS) for each candidate, with a final ranking and combined score for selection.

---

## Working Directory Structure
- **stage1/**: Audio analysis modules
- **stage2/**: Language detection logic
- **stage3/**: Transcript normalization and loading
- **stage4/**: Candidate filtering
- **stage5/**: ASR models, consensus, and scoring
- **stage6/**: Alignment, hallucination detection, metrics
- **stage7/**: Acoustic similarity and finalization
- **stage8/**: Multilingual linguistic grammar scoring
- **working/**: Intermediate files, cache, and results

---

## Usage
1. Place your audio files in the designated input folder.
2. Run the pipeline starting from Stage 1; each stage processes and passes data to the next.
3. Inspect outputs and logs for detailed analysis and results.

---

## Dependencies
- Python 3.8+
- PyTorch, torchaudio
- FFmpeg
- faster-whisper
- Other dependencies as specified in each stage's module

---

