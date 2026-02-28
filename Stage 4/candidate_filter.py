import re
import unicodedata
from collections import Counter

# Speaking speeds by language (words per minute)
SPEAKING_SPEEDS = {
    "ar": 130, "fa": 130, "ur": 130,
    "en": 150, "fr": 150, "it": 150, "pt": 150,
    "de": 130, "ru": 140, "tr": 140, "hi": 130,
    "es": 160, "zh": 200, "ja": 200, "ko": 140,
    "default": 140
}

# Speaking rate tolerance: valid range as multiplier of expected WPM
# e.g. 0.4x to 2.0x of expected rate is acceptable
SPEAKING_RATE_MIN_RATIO = 0.4
SPEAKING_RATE_MAX_RATIO = 2.0


# ── Script Detection ──────────────────────────────────────

def get_dominant_script(text):
    """
    Detects the dominant Unicode script in a text.
    Works for ANY language — no hardcoding needed.
    """
    script_counts = {}

    for char in text:
        if char.isspace() or not char.strip():
            continue
        try:
            char_name = unicodedata.name(char, "")
            script = char_name.split()[0] if char_name else "UNKNOWN"
        except:
            script = "UNKNOWN"

        script_counts[script] = script_counts.get(script, 0) + 1

    if not script_counts:
        return "UNKNOWN", 0.0

    dominant_script = max(script_counts, key=script_counts.get)
    total_chars = sum(script_counts.values())
    dominant_ratio = script_counts[dominant_script] / total_chars

    return dominant_script, dominant_ratio


def check_script(text, language):
    """
    Checks if the transcript script is internally consistent.
    Passes if dominant script covers ≥50% of characters.
    """
    if not text.strip():
        return False, "Empty transcript"

    dominant_script, ratio = get_dominant_script(text)
    passed = ratio >= 0.5
    reason = f"Dominant script: {dominant_script} ({ratio:.2%} of characters)"
    return passed, reason


def check_script_match(text1, text2):
    """
    Checks if two texts use the same dominant script.
    """
    script1, _ = get_dominant_script(text1)
    script2, _ = get_dominant_script(text2)
    passed = script1 == script2
    reason = f"Candidate script: {script2} | Reference script: {script1}"
    return passed, reason


# ── Duration & Speaking Rate ──────────────────────────────

def estimate_speaking_duration(text, language="en"):
    """
    Estimates how long it would take to speak this text
    using language-specific speaking speed.
    """
    words = len(text.split())
    wpm = SPEAKING_SPEEDS.get(language, SPEAKING_SPEEDS["default"])
    estimated_seconds = (words / wpm) * 60
    return estimated_seconds, words


def check_duration(text, audio_duration, language="en", tolerance=3.0):
    """
    Checks if transcript length is realistic for the audio duration.
    """
    estimated_seconds, word_count = estimate_speaking_duration(text, language)
    ratio = estimated_seconds / audio_duration if audio_duration > 0 else 999
    passed = (1 / tolerance) <= ratio <= tolerance
    reason = (
        f"words={word_count}, "
        f"estimated={estimated_seconds:.1f}s, "
        f"audio={audio_duration:.1f}s, "
        f"ratio={ratio:.2f}"
    )
    return passed, reason


def check_speaking_rate(text, audio_duration, language="en"):
    """
    ✅ NEW: Speaking Rate Consistency Check.

    Instead of estimating duration from word count, this goes the other
    direction — computes actual words-per-minute given the real audio
    duration, then checks if that WPM falls within a plausible range
    for the detected language.

    This catches transcripts that are technically within a loose duration
    tolerance but are still implausibly fast or slow for a native speaker.

    Example: A 5-second clip with 40 Arabic words = 480 WPM — impossible.
    """
    words = text.split()
    word_count = len(words)

    if audio_duration <= 0:
        return False, "Invalid audio duration"

    actual_wpm = (word_count / audio_duration) * 60
    expected_wpm = SPEAKING_SPEEDS.get(language, SPEAKING_SPEEDS["default"])

    ratio = actual_wpm / expected_wpm
    passed = SPEAKING_RATE_MIN_RATIO <= ratio <= SPEAKING_RATE_MAX_RATIO

    reason = (
        f"actual={actual_wpm:.0f} WPM, "
        f"expected~{expected_wpm} WPM, "
        f"ratio={ratio:.2f} "
        f"(valid: {SPEAKING_RATE_MIN_RATIO}x–{SPEAKING_RATE_MAX_RATIO}x)"
    )
    return passed, reason


# ── Outlier Detection ─────────────────────────────────────

def check_outlier(candidates, index):
    """
    Checks if this transcript is dramatically different in length
    from the other candidates.

    Uses MEDIAN instead of mean — resistant to junk candidates that
    are extremely long/short and would poison a mean-based average.

    Example: [30, 30, 30, 30, 5802] → median=30, not mean=1184
    So the 4 correct candidates get ratio=1.0 ✅
    And the junk candidate gets ratio=193 ❌
    """
    lengths = sorted([len(t.split()) for t in candidates])
    this_length = len(candidates[index].split())

    if len(lengths) < 2:
        return True, "Not enough candidates to compare"

    mid = len(lengths) // 2
    median_length = (
        lengths[mid] if len(lengths) % 2 != 0
        else (lengths[mid - 1] + lengths[mid]) / 2
    )

    ratio = this_length / median_length if median_length > 0 else 0
    passed = 0.33 <= ratio <= 3.0
    reason = f"words={this_length}, median={median_length:.1f}, ratio={ratio:.2f}"
    return passed, reason


# ── NEW: Text Quality Checks ──────────────────────────────

def check_lexical_diversity(text, min_ttr=0.3):
    """
    ✅ NEW: Minimum Lexical Diversity Check (Type-Token Ratio).

    TTR = unique words / total words.
    A very low TTR means the transcript is highly repetitive or
    filled with filler tokens — likely a bad transcription.

    Example: "yes yes yes yes yes yes" → TTR = 0.17 → FAIL
    Example: "لازم تشوف ذا الحين لا اخس" → TTR = 1.0 → PASS

    Threshold of 0.3 is conservative — even naturally repetitive
    speech rarely drops below this unless something is very wrong.
    """
    words = text.lower().split()
    if len(words) == 0:
        return False, "Empty transcript"
    if len(words) <= 3:
        # Too short to judge diversity — give benefit of the doubt
        return True, f"Too short to assess (words={len(words)})"

    ttr = len(set(words)) / len(words)
    passed = ttr >= min_ttr
    reason = (
        f"unique={len(set(words))}, total={len(words)}, "
        f"TTR={ttr:.2f} (min={min_ttr})"
    )
    return passed, reason


def check_empty_token_ratio(text, max_ratio=0.3):
    """
    ✅ NEW: Empty Token Ratio Check.

    Counts tokens that are effectively empty or meaningless:
    pure punctuation, single characters (outside CJK), lone digits,
    or strings of repeated characters like '....' or '-----'.

    A high ratio of such tokens signals a poorly formed transcript.

    Example: ". . . . . . text" → 6/7 tokens are junk → FAIL
    """
    tokens = text.split()
    if not tokens:
        return False, "Empty transcript"

    def is_empty_token(tok):
        # Pure punctuation
        if re.fullmatch(r'[\W_]+', tok):
            return True
        # Repeated single character (e.g. 'aaaaa', '.....')
        if len(set(tok)) == 1 and len(tok) > 2:
            return True
        # Single non-CJK character (lone letters are often noise)
        if len(tok) == 1 and not ('\u4e00' <= tok <= '\u9fff'):
            return True
        return False

    empty_count = sum(1 for t in tokens if is_empty_token(t))
    ratio = empty_count / len(tokens)
    passed = ratio <= max_ratio
    reason = (
        f"empty_tokens={empty_count}/{len(tokens)}, "
        f"ratio={ratio:.2f} (max={max_ratio})"
    )
    return passed, reason


def check_repetition(text, max_repeat_ratio=0.4, window=5):
    """
    ✅ NEW: Repetition Detection.

    Detects two types of repetition that signal transcription errors:

    1. Word-level repetition: any single word appearing more than
       40% of the time in the transcript (e.g. hallucinated loops).

    2. N-gram repetition: checks consecutive windows of `window` words
       — if the same window appears 3+ times, it's likely a looping
       hallucination (common in Whisper on noisy audio).

    Example: "الله الله الله الله الله" → top word ratio = 1.0 → FAIL
    """
    words = text.lower().split()
    if len(words) <= 3:
        return True, "Too short to assess repetition"

    # Check 1: Single word dominance
    counts = Counter(words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    word_ratio = most_common_count / len(words)

    if word_ratio > max_repeat_ratio:
        return False, (
            f"Word '{most_common_word}' appears {most_common_count}/"
            f"{len(words)} times (ratio={word_ratio:.2f}, max={max_repeat_ratio})"
        )

    # Check 2: Repeated n-gram windows
    if len(words) >= window * 2:
        ngrams = [
            tuple(words[i:i + window])
            for i in range(len(words) - window + 1)
        ]
        ngram_counts = Counter(ngrams)
        most_common_ngram, ngram_count = ngram_counts.most_common(1)[0]
        if ngram_count >= 3:
            return False, (
                f"Repeated {window}-gram detected "
                f"({ngram_count}x): '{' '.join(most_common_ngram)}'"
            )

    return True, (
        f"No excessive repetition "
        f"(top word ratio={word_ratio:.2f}, ngrams OK)"
    )


# ── Main Filter ───────────────────────────────────────────

def filter_candidates(candidates, audio_duration, language="en"):
    """
    Main filtering function — works for ANY language.
    Runs all checks on each candidate and returns valid ones.

    Checks (in order):
      1. Duration check         — is transcript length plausible?
      2. Speaking rate check    — is WPM realistic for this language? [NEW]
      3. Script check           — is the script internally consistent?
      4. Outlier check          — is length wildly different from peers?
      5. Lexical diversity      — is TTR above minimum? [NEW]
      6. Empty token ratio      — too many junk tokens? [NEW]
      7. Repetition detection   — looping words or n-grams? [NEW]
    """
    print(f"\nFiltering {len(candidates)} candidates...")
    print(f"Audio duration: {audio_duration:.1f}s | Language: {language}")

    valid_candidates = []
    filtered_out    = []

    for i, transcript in enumerate(candidates):
        print(f"\n-- Candidate {i+1} --")
        failed_reasons = []

        # ── Check 1: Duration ─────────────────────────────
        dur_passed, dur_reason = check_duration(
            transcript, audio_duration, language
        )
        print(f"  Duration check:       {'✅' if dur_passed else '❌'} ({dur_reason})")
        if not dur_passed:
            failed_reasons.append(f"Duration: {dur_reason}")

        # ── Check 2: Speaking Rate (NEW) ──────────────────
        rate_passed, rate_reason = check_speaking_rate(
            transcript, audio_duration, language
        )
        print(f"  Speaking rate check:  {'✅' if rate_passed else '❌'} ({rate_reason})")
        if not rate_passed:
            failed_reasons.append(f"SpeakingRate: {rate_reason}")

        # ── Check 3: Script ───────────────────────────────
        script_passed, script_reason = check_script(transcript, language)
        print(f"  Script check:         {'✅' if script_passed else '❌'} ({script_reason})")
        if not script_passed:
            failed_reasons.append(f"Script: {script_reason}")

        # ── Check 4: Outlier ──────────────────────────────
        outlier_passed, outlier_reason = check_outlier(candidates, i)
        print(f"  Outlier check:        {'✅' if outlier_passed else '❌'} ({outlier_reason})")
        if not outlier_passed:
            failed_reasons.append(f"Outlier: {outlier_reason}")

        # ── Check 5: Lexical Diversity (NEW) ──────────────
        lex_passed, lex_reason = check_lexical_diversity(transcript)
        print(f"  Lexical diversity:    {'✅' if lex_passed else '❌'} ({lex_reason})")
        if not lex_passed:
            failed_reasons.append(f"LexicalDiversity: {lex_reason}")

        # ── Check 6: Empty Token Ratio (NEW) ──────────────
        token_passed, token_reason = check_empty_token_ratio(transcript)
        print(f"  Empty token ratio:    {'✅' if token_passed else '❌'} ({token_reason})")
        if not token_passed:
            failed_reasons.append(f"EmptyTokens: {token_reason}")

        # ── Check 7: Repetition Detection (NEW) ───────────
        rep_passed, rep_reason = check_repetition(transcript)
        print(f"  Repetition check:     {'✅' if rep_passed else '❌'} ({rep_reason})")
        if not rep_passed:
            failed_reasons.append(f"Repetition: {rep_reason}")

        # ── Final Decision ────────────────────────────────
        if failed_reasons:
            print(f"  Result: FILTERED OUT ❌")
            filtered_out.append((i + 1, transcript, failed_reasons))
        else:
            print(f"  Result: PASSED ✅")
            valid_candidates.append((i + 1, transcript))

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"FILTER SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Passed:  {len(valid_candidates)}/{len(candidates)}")
    print(f"  Removed: {len(filtered_out)}/{len(candidates)}")
    if filtered_out:
        print(f"\n  Filtered candidates:")
        for idx, text, reasons in filtered_out:
            print(f"    Candidate {idx}: {reasons}")
    print(f"{'=' * 60}")

    return valid_candidates, filtered_out