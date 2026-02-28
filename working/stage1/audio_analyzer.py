import torch
def trim_silence_vad(waveform, sample_rate):
    """
    Perform a two‑stage trimming pass:

    1.  Quick energy‑based removal of leading/trailing silence using
        :func:`trim_silence` so that the VAD doesn't waste time on long
        empty regions.
    2.  Run the Silero neural VAD to chop out *non‑speech* segments within
        the remaining audio and concatenate the speech chunks.

    The return value is always a waveform that is no longer than the input
    (empty audio is possible when nothing is detected) and the duration of
    the trimmed signal in seconds.

    If the Silero library cannot be imported or the model call fails the
    function will gracefully fall back to the simple energy‑based
    ``trim_silence`` implementation.
    """
    # stage‑0 energy trimming to help the VAD
    waveform, _ = trim_silence(waveform, sample_rate)

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        audio = waveform.squeeze()  # shape -> (N,) for the VAD helper

        speech_timestamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=sample_rate,
            threshold=0.3,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )

        # if the model returned no segments we simply keep whatever the
        # energy trimming produced earlier (which may be the entire file)
        if not speech_timestamps:
            print("⚠️ No speech detected by VAD – returning energy‑trimmed audio")
            duration = waveform.shape[1] / sample_rate
            return waveform, duration

        speech_chunks = []
        padding_samples = int(sample_rate * 0.2)  # 200ms padding

        for ts in speech_timestamps:
            start = ts["start"]
            end = ts["end"]
            # sanity clamp just in case
            padding_samples = int(sample_rate * 0.1)  # 100ms
            start = max(0, start - padding_samples)
            end = min(waveform.shape[1], end + padding_samples)
            speech_chunks.append(waveform[:, start:end])

        if speech_chunks:
            trimmed = torch.cat(speech_chunks, dim=1)
        else:
            trimmed = waveform

        trimmed_duration = trimmed.shape[1] / sample_rate
        print(f"VAD trimming complete → {trimmed_duration:.2f}s remaining ✅")
        return trimmed, trimmed_duration

    except Exception as e:
        print(f"⚠️ VAD trimming failed ({e}), using basic trimming...")
        return trim_silence(waveform, sample_rate)
def trim_silence(waveform, sample_rate, threshold_db=-40.0):
    """
    Removes silence from the beginning and end of audio using an energy
    threshold expressed in decibels below the *peak* level of the file.

    The original implementation used a fixed linear threshold which meant
    quiet recordings were often treated as entirely silent.  By scaling the
    threshold to the maximum amplitude we get consistent behaviour regardless
    of gain.
    """
    # convert threshold from dB to linear scale relative to the peak
    peak = waveform.abs().max().item()
    threshold = peak * (10 ** (threshold_db / 20))
    amplitude = waveform.abs().squeeze()
    above_threshold = amplitude > threshold
    indices = above_threshold.nonzero(as_tuple=True)[0]

    if len(indices) == 0:
        print("⚠️ Audio appears to be entirely silent!")
        duration = waveform.shape[1] / sample_rate
        return waveform, duration

    start = indices[0].item()
    end = indices[-1].item() + 1
    trimmed = waveform[:, start:end]
    trimmed_duration = trimmed.shape[1] / sample_rate

    print(f"Silence trimmed → {trimmed_duration:.2f}s remaining ✅")
    return trimmed, trimmed_duration


def estimate_snr(waveform):
    """
    Estimates Signal-to-Noise Ratio (SNR) in dB.
    - Above 20dB = clean
    - 10-20dB = moderate noise
    - Below 10dB = noisy
    """
    amplitude = waveform.abs().squeeze()
    signal_level = torch.quantile(amplitude, 0.90).item()
    noise_level = torch.quantile(amplitude, 0.10).item()

    if noise_level < 1e-10:
        snr_db = 999.0
    else:
        snr_db = 20 * torch.log10(
            torch.tensor(signal_level / noise_level)
        ).item()

    print(f"SNR: {snr_db:.1f} dB → ", end="")
    if snr_db > 20:
        print("Clean audio ✅")
    elif snr_db > 10:
        print("Moderate noise ⚠️")
    else:
        print("Noisy audio ❌")

    return snr_db


def estimate_noise_level(waveform):
    """
    Estimates background noise floor as RMS value.
    """
    noise_floor = torch.quantile(waveform.abs().squeeze(), 0.20).item()
    print(f"Noise floor: {noise_floor:.6f} RMS")
    return noise_floor


def detect_voice_activity(waveform, sample_rate):
    """
    Voice Activity Detection using Silero VAD.
    A neural network trained specifically to detect speech.
    Works accurately for any language including Arabic.
    
    Returns:
    - speech_ratio: 0.0 to 1.0
    - speech_frames: number of frames with speech
    - total_frames: total frames checked
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()

        # Silero needs mono float32 at 16kHz — we already have that ✅
        audio = waveform.squeeze()

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=sample_rate,
            threshold=0.5,          # confidence threshold (0-1)
            min_speech_duration_ms=250,   # ignore very short sounds
            min_silence_duration_ms=100,  # ignore very short silences
        )

        # Calculate how much of audio is speech
        total_samples = len(audio)
        speech_samples = sum(
            ts["end"] - ts["start"] for ts in speech_timestamps
        )
        speech_ratio = speech_samples / total_samples if total_samples > 0 else 0.0
        speech_frames = len(speech_timestamps)
        total_frames = total_samples

        print(f"Voice activity: {speech_ratio:.1%} → ", end="")
        if speech_ratio > 0.6:
            print("Good speech content ✅")
        elif speech_ratio > 0.3:
            print("Moderate speech content ⚠️")
        else:
            print("Low speech content ❌")

        return speech_ratio, speech_frames, total_frames

    except Exception as e:
        # Fallback to energy-based if Silero fails
        print(f"⚠️ Silero VAD failed ({e}), falling back to energy-based...")
        threshold = 10 ** (-45.0 / 20)
        frame_size = int(sample_rate * 30 / 1000)
        audio = waveform.squeeze()

        total_frames = 0
        speech_frames = 0

        for start in range(0, len(audio) - frame_size, frame_size):
            frame = audio[start:start + frame_size]
            if frame.abs().mean().item() > threshold:
                speech_frames += 1
            total_frames += 1

        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0.0
        print(f"Voice activity (fallback): {speech_ratio:.1%}")
        return speech_ratio, speech_frames, total_frames


def compute_average_energy(waveform):
    """Computes average energy (loudness) of the audio."""
    energy = (waveform ** 2).mean().item()
    print(f"Average energy: {energy:.8f}")
    return energy


def analyze_audio(waveform, sample_rate, raw_duration):
    """
    Runs all analysis on a standardized waveform.

    Quality metrics such as SNR/voice‑activity are calculated on the *original*
    recording; trimming is performed afterwards so that the returned waveform
    represents the version that downstream stages should consume.

    Returns waveform (trimmed) + metadata dict.
    """
    print("\n--- Running Audio Analysis ---")

    # --- metrics on raw audio -------------------------------------------------
    snr_db = estimate_snr(waveform)
    noise_level = estimate_noise_level(waveform)
    speech_ratio, speech_frames, total_frames = detect_voice_activity(
        waveform, sample_rate
    )
    avg_energy = compute_average_energy(waveform)

    # --- trim the waveform for downstream use --------------------------------
    waveform, trimmed_duration = trim_silence_vad(waveform, sample_rate)

    metadata = {
        "raw_duration":     raw_duration,
        "trimmed_duration": trimmed_duration,
        "snr_db":           snr_db,
        "noise_level":      noise_level,
        "speech_ratio":     speech_ratio,
        "speech_frames":    speech_frames,
        "total_frames":     total_frames,
        "avg_energy":       avg_energy,
        "audio_quality":    "clean"    if snr_db > 20 else
                            "moderate" if snr_db > 10 else "noisy"
    }

    return waveform, metadata