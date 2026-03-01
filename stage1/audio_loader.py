import torchaudio
import torch
import subprocess
import os


def load_and_standardize_audio(file_path):
    """
    Loads audio and standardizes to 16kHz mono float32.
    Returns: waveform, sample_rate, duration
    """
    print(f"Loading audio file: {file_path}")
    converted_path = "converted_audio.wav"
    subprocess.run([
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "-y",
        "-i", file_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        converted_path
    ], check=True, capture_output=True)
    waveform, sample_rate = torchaudio.load(converted_path)
    waveform = waveform.to(torch.float32)
    duration = waveform.shape[1] / sample_rate
    os.remove(converted_path)
    return waveform, sample_rate, duration