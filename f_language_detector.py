from faster_whisper import WhisperModel
import os

# ================= CONFIG =================
os.environ["HF_HOME"] = "D:/Hackathon/hf_cache"   # model cache location
audio_folder = r"D:\Hackathon\2\training\downloaded_audio"  # folder path

supported_formats = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


# ================= LOAD MODEL =================
print("Loading Whisper model...")
model = WhisperModel("medium", device="cuda", compute_type="float16")
print("Model loaded successfully")


# ================= LANGUAGE DETECTOR =================
def detect_language(file_path):
    filename = os.path.basename(file_path)
    print(f"\nProcessing: {filename}")

    segments, info = model.transcribe(file_path)

    language = info.language
    confidence = info.language_probability

    print(f"Detected language: {language}")
    print(f"Confidence: {confidence:.2%}")

    if confidence < 0.70:
        print("Low confidence")
    else:
        print("Language detection confident")

    return language, confidence


# ================= MAIN LOOP =================
print("\nScanning folder...")

if not os.path.exists(audio_folder):
    print("ERROR: Folder not found:", audio_folder)
    exit()

files = os.listdir(audio_folder)

if not files:
    print("Folder is empty")
    exit()

print(f"Found {len(files)} files")

processed = 0

for file in files:
    if file.lower().endswith(supported_formats):
        try:
            path = os.path.join(audio_folder, file)
            detect_language(path)
            processed += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")

print(f"\nDone. Processed {processed} audio files.")