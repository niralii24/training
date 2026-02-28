import torchaudio
import torch
import subprocess
import os


def load_and_standardize_audio(file_path):
    """
    Loads any audio file and converts it to:
    - 16kHz sample rate
    - Mono channel
    - float32 format

    Returns: waveform, sample_rate, duration
    """
    print(f"\nLoading audio file: {file_path}")

    converted_path = "converted_audio.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", file_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        converted_path
    ], check=True, capture_output=True)
    print("Converted to WAV using FFmpeg ✅")

    waveform, sample_rate = torchaudio.load(converted_path)
    waveform = waveform.to(torch.float32)
    duration = waveform.shape[1] / sample_rate
    os.remove(converted_path)

    print(f"Duration: {duration:.2f}s")
    print("Audio standardization complete ✅")
    return waveform, sample_rate, duration