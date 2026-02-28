from faster_whisper import WhisperModel
from selector import score_candidates

# Load Whisper
model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path)
    return " ".join([seg.text for seg in segments])


audio = "audio.wav"

candidates = [
    "candidate transcript 1",
    "candidate transcript 2",
    "candidate transcript 3",
    "candidate transcript 4",
    "candidate transcript 5"
]

internal_transcript = transcribe(audio)

best_index, scores = score_candidates(internal_transcript, candidates)

print("Internal Transcript:\n", internal_transcript)
print("\nScores:", scores)
print("\nSelected Candidate:", best_index + 1)