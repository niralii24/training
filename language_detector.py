from faster_whisper import WhisperModel
import torchaudio
import os

# Load faster-whisper model
print("Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model loaded ✅")

def detect_language(waveform, sample_rate):
    """
    Detects language from audio using faster-whisper.
    Returns language code and confidence score.
    """
    print("\nDetecting language...")

    # Save waveform temporarily for faster-whisper to read
    torchaudio.save("temp_for_detection.wav", waveform, sample_rate)

    # Transcribe - faster-whisper automatically detects language
    segments, info = model.transcribe("temp_for_detection.wav")

    language = info.language
    confidence = info.language_probability

    print(f"Detected language: {language}")
    print(f"Confidence: {confidence:.2%}")

    if confidence < 0.70:
        print("⚠️ Low confidence in language detection")
    else:
        print("Language detection confident ✅")

    # Clean up temp file
    os.remove("temp_for_detection.wav")

    return language, confidence