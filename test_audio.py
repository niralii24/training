from audio_loader import load_and_standardize_audio

waveform, sample_rate, duration = load_and_standardize_audio("sample.mp3")

print(f"\n--- Final Output ---")
print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}")
print(f"Duration: {duration:.2f} seconds")