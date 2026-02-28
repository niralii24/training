from .audio_loader import load_and_standardize_audio
from .audio_analyzer import analyze_audio


def run_stage1(audio_file):
    """
    Stage 1: Audio Loading & Analysis

    1. Load & standardize audio (16kHz, mono, float32)
    2. Analyze audio (trim silence, SNR, VAD, noise, metadata)

    Returns waveform + full acoustic metadata for downstream stages.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: AUDIO LOADING & ANALYSIS")
    print("=" * 60)

    # Step 1: Load and standardize
    print("\n[1/2] Loading and standardizing audio...")
    waveform, sample_rate, raw_duration = load_and_standardize_audio(audio_file)

    # Step 2: Analyze
    print("\n[2/2] Analyzing audio quality...")
    waveform, metadata = analyze_audio(waveform, sample_rate, raw_duration)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"  Audio quality: {metadata['audio_quality']}")
    print(f"  SNR:           {metadata['snr_db']:.1f} dB")
    print(f"  Speech ratio:  {metadata['speech_ratio']:.1%}")
    print(f"  Duration (raw): {metadata['raw_duration']:.2f}s")
    print(f"  Duration (trimmed): {metadata['trimmed_duration']:.2f}s")
    print("=" * 60)

    return {
        "waveform":    waveform,
        "sample_rate": sample_rate,
        "metadata":    metadata
    }


# ── Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # ── Set your audio folder path here ──────────────────────
    AUDIO_FOLDER = r"C:\Users\Admin\Desktop\golden_transcription_system\training\test_audio"

    # Supported audio formats
    SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4")

    # Get all audio files from the folder
    audio_files = [
        f for f in os.listdir(AUDIO_FOLDER)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if not audio_files:
        print(f"No audio files found in: {AUDIO_FOLDER}")
    else:
        print(f"Found {len(audio_files)} audio file(s) to process\n")
        all_results = {}

        for i, filename in enumerate(audio_files, 1):
            full_path = os.path.join(AUDIO_FOLDER, filename)
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(audio_files)}: {filename}")
            print(f"{'='*60}")

            result = run_stage1(full_path)
            all_results[filename] = result

        print(f"\n{'='*60}")
        print(f"ALL FILES PROCESSED ✅")
        print(f"Total files: {len(all_results)}")
        for filename, result in all_results.items():
            print(f"  {filename} → {result['metadata']['audio_quality']} | "
                  f"SNR: {result['metadata']['snr_db']:.1f}dB | "
                  f"Duration: {result['metadata']['trimmed_duration']:.2f}s")
        print(f"{'='*60}")