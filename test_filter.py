from audio_loader import load_and_standardize_audio
from language_detector import detect_language
from candidate_filter import filter_candidates

# Load audio to get duration
waveform, sample_rate, duration = load_and_standardize_audio("sample.mp3")

# Detect language
language, confidence = detect_language(waveform, sample_rate)

# Your real candidates + 1 fake bad one to test filtering
candidates = [
    "ااااه ثيو يا الله لازم تشوف ذا الحين لا اخس شوف ليه اليكس ارسلى ابيك بكلمة بالظبط صح ويش يقصد بها ابيك بكلمة عن ويش وليه متى وين وعن ويش",
    
    "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف ، ليه اليكس ارسلى ابيك بكلمة؟ بالظبط . صح؟ ويش يقصد بها؟ ابيك بكلمة. عن ويش؟ وليه؟ متى ؟وين؟ وعن ويش",
    
    "aaaaaah theo",  # ❌ bad candidate - wrong script, too short
    
    "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف ، ليه اليكس ارسلى ابيك بكلمة؟ بالظبط . صح؟ ويش يقصد بها؟ ابيك بكلمة. عن ويش؟ وليه؟ متى ؟وين؟ وعن ويش",
]

# Run filtering
valid, removed = filter_candidates(candidates, duration, language)

print("\n--- Valid Candidates ---")
for idx, text in valid:
    print(f"Candidate {idx}: {text[:60]}...")

print("\n--- Removed Candidates ---")
for idx, text, reasons in removed:
    print(f"Candidate {idx} removed because: {reasons}")