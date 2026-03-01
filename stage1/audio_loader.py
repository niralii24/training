import torch
import librosa
import numpy as np


def load_and_standardize_audio(file_path):
    """
    Loads any audio file and converts it to:
    - 16kHz sample rate
    - Mono channel
    - float32 format

    Returns: waveform (1, N) torch tensor, sample_rate (int), duration (float)
    """
    print(f"\nLoading audio file: {file_path}")

    TARGET_SR = 16000
    audio_np, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio_np = audio_np.astype(np.float32)

    # Shape to (1, N) to match torchaudio convention expected by downstream stages
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    duration = audio_np.shape[0] / TARGET_SR

    print(f"Duration: {duration:.2f}s")
    print("Audio standardization complete ✅")
    return waveform, TARGET_SR, duration