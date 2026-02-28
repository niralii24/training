from audio_loader import load_and_standardize_audio
from language_detector import detect_language

# Step 1: Load audio
waveform, sample_rate, duration = load_and_standardize_audio("sample.mp3")

# Step 2: Detect language
language, confidence = detect_language(waveform, sample_rate)

print(f"\n--- Language Detection Result ---")
print(f"Language: {language}")
print(f"Confidence: {confidence:.2%}")