import sys
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.join(BASE_DIR, "..", "Stage 1")
STAGE2_DIR = os.path.join(BASE_DIR, "..", "Stage 2")

sys.path.append(STAGE1_DIR)
sys.path.append(STAGE2_DIR)

from stage1_runner     import run_stage1
from language_detector import detect_language
from text_normalizer   import normalize_all_candidates


def run_stage3(stage2_result, candidates):
    """
    Stage 3: Language-Aware Text Normalization

    Takes Stage 2 output + raw candidates,
    normalizes all candidates using detected language.

    Returns stage2_result + normalized candidates.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: TEXT NORMALIZATION")
    print("=" * 60)

    language = stage2_result["metadata"]["language"]

    normalized = normalize_all_candidates(candidates, language)

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print(f"  Language:   {language}")
    print(f"  Candidates: {len(normalized)} normalized")
    print("=" * 60)

    stage2_result["candidates"]            = candidates
    stage2_result["normalized_candidates"] = normalized

    return stage2_result


# # ── Test ──────────────────────────────────────────────────
# if __name__ == "__main__":

#     AUDIO_FOLDER      = r"C:\Users\Admin\Desktop\golden_transcription_system\audio_input"
#     SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

#     candidates = [
#         "ااااه ثيو يا الله لازم تشوف ذا الحين لا اخس شوف ليه اليكس ارسلى ابيك بكلمة",
#         "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف ، ليه اليكس ارسلى ابيك بكلمة؟",
#         "aaaaaah theo ya allah",  # bad candidate
#     ]

#     audio_files = [
#         os.path.join(AUDIO_FOLDER, f)
#         for f in os.listdir(AUDIO_FOLDER)
#         if f.lower().endswith(SUPPORTED_FORMATS)
#     ]

#     if not audio_files:
#         print("No audio files found!")
#     else:
#         for audio_file in audio_files:
#             # Stage 1
#             s1 = run_stage1(audio_file)

#             # Stage 2
#             language, confidence, probs, method = detect_language(
#                 s1["waveform"], s1["sample_rate"]
#             )
#             s1["metadata"]["language"]            = language
#             s1["metadata"]["language_confidence"] = confidence
#             s1["metadata"]["language_probs"]      = probs
#             s1["metadata"]["language_method"]     = method

#             # Stage 3
#             result = run_stage3(s1, candidates)

#             print(f"\n✅ {os.path.basename(audio_file)}")
#             print(f"   Language: {language}")
#             print("\n   Normalized candidates:")
#             for i, text in enumerate(result["normalized_candidates"]):
#                 print(f"   [{i+1}] {text[:80]}")