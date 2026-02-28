---

# Golden Transcription Selection System

## Training & System Design Document

---

## 1. Problem Statement

We are given:

* **1 audio file**
* **5 candidate transcriptions** (possibly multilingual)

Our task:

1. Transcribe the audio using our internal ASR system.
2. Compare our transcription with the 5 provided candidates.
3. Select the candidate that most closely matches our transcription.
4. Achieve maximum accuracy across noisy, multilingual, and varying speaker conditions.

This is not a generic ASR task.
This is a **transcription matching and selection problem**.

---

## 2. Core Strategy

We reduce the problem to:

> Generate a high-quality internal transcription, then select the closest candidate using robust multilingual similarity scoring.

The system has 3 main components:

1. Internal Transcription Engine
2. Text Normalization Pipeline
3. Multilingual Similarity Scoring Engine

No heavy training. No large-scale fine-tuning. Fully laptop-compatible.

---

## 3. System Architecture

```
Audio
  ↓
Internal ASR (Whisper-small / faster-whisper)
  ↓
Normalized Internal Transcript
  ↓
Compare with each of 5 candidates
  ↓
Similarity Scoring
  ↓
Select Best Match
```

---

## 4. Component Design

---

## 4.1 Internal Transcription Engine

### Model Choice

* `faster-whisper`
* Model size: `small` or `medium`
* CPU mode

Why:

* Strong multilingual capability
* Robust to noise
* Gives token-level log probabilities
* Runs on laptop

We are NOT training this model.

---

## 4.2 Text Normalization Pipeline

All comparisons must happen on normalized text.

For both:

* Internal transcript
* All 5 candidates

Apply:

* Lowercasing
* Unicode normalization (NFKC)
* Remove punctuation (optional toggle)
* Normalize whitespace
* Remove filler words (language-aware optional)
* Strip repeated tokens
* Normalize numbers (e.g., "five" → "5")

This reduces mismatch noise.

---

## 4.3 Multilingual Handling

Because candidates may be multilingual:

1. Detect language of internal transcript.
2. Detect language of each candidate.
3. If language mismatch:

   * Translate both into a pivot language (English) using a lightweight translation model.
   * OR compare using multilingual embeddings (preferred).

We avoid hard reliance on translation by using multilingual embeddings.

---

## 4.4 Similarity Scoring Engine

We combine multiple similarity signals.

### 1. Semantic Similarity (Primary Signal)

Use sentence-transformers:

* `paraphrase-multilingual-MiniLM-L12-v2`

Process:

* Generate embedding for internal transcript
* Generate embedding for each candidate
* Compute cosine similarity

Higher similarity = better match

---

### 2. Token-Level Overlap

Compute:

* Word-level Levenshtein similarity
* Character-level similarity
* Jaccard similarity
* N-gram overlap (1-gram, 2-gram)

This helps when wording is similar but not identical.

---

### 3. Optional: Whisper Log-Probability Check

Advanced (optional but powerful):

Force-align each candidate transcript with the audio using Whisper and compute:

* Average token log probability

If a candidate has very low acoustic likelihood, penalize it.

---

## 5. Final Scoring Formula

For each candidate:

```
FinalScore =
0.50 * SemanticSimilarity +
0.25 * TokenSimilarity +
0.15 * NgramOverlap +
0.10 * AcousticScore (optional)
```

Weights can be tuned on validation data.

Select candidate with highest FinalScore.

---

## 6. Training Plan

We are NOT training Whisper.

We only train:

* Weight tuning
* Optional meta-classifier

---

## 6.1 Dataset Preparation

Use open datasets:

* Common Voice
* FLEURS
* LibriSpeech (English baseline)

For each audio sample:

1. Generate transcript using Whisper.
2. Create 4 synthetic variations:

   * Slight paraphrase
   * Minor noise injection
   * Partial truncation
   * Different ASR output (wav2vec2)

Now we simulate:

1 audio + 5 candidate transcripts
Label the correct one.

---

## 6.2 Train Meta-Selector (Optional Upgrade)

Feature vector per candidate:

```
semantic_similarity
levenshtein_ratio
jaccard_score
ngram_overlap
length_ratio
acoustic_logprob
```

Train:

* XGBoost (CPU)
* LightGBM (CPU)

Target:

* 1 if correct transcript
* 0 otherwise

This allows learning optimal weighting.

---

## 7. Evaluation Strategy

Metrics:

* Selection Accuracy (primary metric)
* Top-1 accuracy
* Robustness across:

  * Noise
  * Language variation
  * Accent variation

Test across multiple languages separately.

---

## 8. Edge Case Handling

### Case 1: All candidates very different

Fallback:

* Select highest semantic similarity.

### Case 2: Multilingual mismatch

* Rely more heavily on embeddings than token overlap.

### Case 3: Noisy audio

* Increase weight of acoustic confidence.

---

## 9. Computational Requirements

Runs fully on laptop:

* 8–16 GB RAM
* CPU inference
* No GPU required
* No paid APIs

---

## 10. Final Deliverable Behavior

Given:

* Audio
* 5 transcripts

System will:

1. Transcribe audio.
2. Normalize all texts.
3. Compute similarity signals.
4. Score each candidate.
5. Output:

   * Selected transcription
   * Scores for all candidates
   * Confidence score

---

## 11. Why This Will Work

* Whisper provides strong multilingual base transcript.
* Multilingual embeddings allow cross-language comparison.
* Token metrics capture structural similarity.
* Acoustic scoring filters semantically similar but incorrect transcripts.
* Ensemble scoring avoids reliance on a single metric.

This design maximizes robustness while remaining lightweight.

---

## 12. Final Summary

We are not building a better ASR model.

We are building:

> A robust multilingual transcription arbitration system powered by semantic embeddings, structural similarity, and optional acoustic validation.

Fully offline.
Fully laptop-compatible.
Scalable.
Research-grade architecture.

---

