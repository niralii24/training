import torchaudio
import torchaudio.transforms as T
import torch
import subprocess
import os

def load_and_standardize_audio(file_path):
    """
    Takes any audio file and converts it to:
    - 16kHz sample rate
    - Mono (single channel)
    - float32 format
    """
    print(f"Loading audio file: {file_path}")

    # Convert mp3 to wav first using ffmpeg (most reliable on Windows)
    converted_path = "converted_audio.wav"
    subprocess.run([
        "ffmpeg", "-y",        # -y means overwrite if exists
        "-i", file_path,       # input file
        "-ar", "16000",        # set sample rate to 16kHz
        "-ac", "1",            # set to mono (1 channel)
        "-f", "wav",           # output format
        converted_path
    ], check=True)
    print("Converted to WAV using FFmpeg ✅")

    # Now load the clean wav file
    waveform, sample_rate = torchaudio.load(converted_path)

    # Convert to float32
    waveform = waveform.to(torch.float32)
    print("Converted to float32 ✅")

    # Calculate duration
    duration = waveform.shape[1] / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")

    # Clean up temp file
    os.remove(converted_path)

    print("Audio standardization complete! ✅")
    return waveform, sample_rate, duration