import torch
import numpy as np


TARGET_SR = 16000


def _load_with_av(file_path, target_sr=TARGET_SR):
    """Decode audio using PyAV (bundled with faster-whisper) — handles MP3, AAC,
    M4A, OGG, FLAC, WAV and virtually every other container/codec.

    Uses an AudioResampler to normalise to float32 mono at target_sr so we
    don't need to handle every possible source format manually.

    Returns a float32 torch tensor of shape (1, samples) and target_sr.
    """
    import av

    resampler = av.AudioResampler(
        format="fltp",       # float32 planar
        layout="mono",
        rate=target_sr,
    )

    frames = []
    with av.open(file_path) as container:
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"), None
        )
        if audio_stream is None:
            raise ValueError(f"No audio stream found in '{file_path}'")

        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()   # (1, samples) float32
                frames.append(arr)

        # flush the resampler
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            frames.append(arr)

    if not frames:
        raise ValueError(f"No audio decoded from '{file_path}'")

    audio = np.concatenate(frames, axis=1)          # (1, total_samples)
    waveform = torch.from_numpy(audio.copy())        # float32 tensor
    return waveform, target_sr


def load_and_standardize_audio(file_path):
    """
    Loads any audio file and converts it to:
    - 16 kHz sample rate
    - Mono channel
    - float32 format

    Uses PyAV (already installed via faster-whisper) so no external
    ffmpeg binary is required.

    Returns: waveform (Tensor), sample_rate (int), duration (float)
    """
    print(f"\nLoading audio file: {file_path}")

    waveform, sr = _load_with_av(file_path, target_sr=TARGET_SR)

    duration = waveform.shape[1] / sr

    print(f"Duration: {duration:.2f}s")
    print("Audio standardization complete ✅")
    return waveform, sr, duration