from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe("Segment_001.wav")

print("Detected language:", info.language)

for segment in segments:
    print(segment.text)