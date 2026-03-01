import os
import pandas as pd


def load_transcripts(excel_path):
    """
    Loads candidate transcripts from Excel file.

    Expected format:
    | audio_id | language | audio (URL) | option_1 | option_2 | option_3 | option_4 | option_5 | correct_option |

    Returns:
        list of dicts, one per row:
        [
            {
                "audio_id":      "1",
                "language":      "Arabic_SA",
                "audio_url":     "https://...",
                "candidates":    ["option1 text", "option2 text", ...],
                "correct_option": "5"
            },
            ...
        ]
    """
    print(f"\nLoading transcripts from: {os.path.basename(excel_path)}")

    df = pd.read_excel(excel_path)

    print(f"Found {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Identify option columns automatically
    option_cols = sorted([
        col for col in df.columns
        if str(col).lower().startswith("option")
    ])
    print(f"Option columns found: {option_cols}")

    records = []

    for _, row in df.iterrows():
        # Collect all non-empty options
        candidates = []
        for col in option_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                candidates.append(str(value).strip())

        if not candidates:
            continue

        record = {
            "audio_id":      str(row.get("audio_id", "")).strip(),
            "language":      str(row.get("language", "")).strip(),
            "audio_url":     str(row.get("audio", "")).strip(),
            "candidates":    candidates,
            "correct_option": str(row.get("correct_option", "")).strip()
        }

        records.append(record)
        print(f"  Row {record['audio_id']} → {len(candidates)} candidates | "
              f"lang: {record['language']} | "
              f"correct: {record['correct_option']}")

    print(f"\n✅ Loaded {len(records)} records from Excel")
    return records


def get_language_code(language_str):
    """
    Converts Excel language values to ISO language codes
    that Whisper and our normalizer understand.

    Examples:
        'Arabic_SA' → 'ar'
        'English_US' → 'en'
        'French' → 'fr'
    """
    language_map = {
        "arabic_sa": "ar",
        "arabic_eg": "ar",
        "arabic":    "ar",
        "english_us": "en",
        "english_uk": "en",
        "english":   "en",
        "french":    "fr",
        "german":    "de",
        "spanish":   "es",
        "italian":   "it",
        "portuguese": "pt",
        "russian":   "ru",
        "chinese":   "zh",
        "japanese":  "ja",
        "korean":    "ko",
        "hindi":     "hi",
        "turkish":   "tr",
    }

    key = language_str.lower().strip()
    code = language_map.get(key, None)

    if not code:
        # Try matching by first part before underscore
        # e.g. 'Arabic_SA' → 'arabic' → 'ar'
        base = key.split("_")[0]
        code = language_map.get(base, "unknown")

    return code


# ── Test ──────────────────────────────────────────────────
if __name__ == "__main__":

    EXCEL_PATH = r"C:\Users\Admin\Desktop\golden_transcription_system\transcripts.xlsx"

    records = load_transcripts(EXCEL_PATH)

    print("\n--- Sample Output (first 2 rows) ---")
    for record in records[:2]:
        print(f"\nAudio ID:  {record['audio_id']}")
        print(f"Language:  {record['language']} → {get_language_code(record['language'])}")
        print(f"Audio URL: {record['audio_url'][:60]}...")
        print(f"Correct:   Option {record['correct_option']}")
        print(f"Candidates:")
        for i, t in enumerate(record["candidates"], 1):
            print(f"  [{i}] {t[:80]}")