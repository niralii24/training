import re
import unicodedata
import pyarabic.araby as araby

def normalize_text(text, language="en"):
    """
    Cleans and standardizes a transcript based on its language.
    
    Steps:
    - Lowercase everything
    - Remove punctuation
    - Standardize numbers
    - Remove extra whitespace
    - Apply language-specific rules
    
    Returns: cleaned text string
    """
    print(f"\nNormalizing text (language: {language})...")
    print(f"Original: {text[:80]}...")  # show first 80 chars

    # Step 1: Lowercase (only for non-Arabic scripts)
    if language != "ar":
        text = text.lower()

    # Step 2: Remove punctuation
    # This removes all punctuation marks like . , ! ? " ' etc.
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Standardize numbers
    # Converts written numbers to digits for consistency
    number_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10"
    }
    if language == "en":
        for word, digit in number_map.items():
            text = re.sub(rf'\b{word}\b', digit, text)

    # Step 4: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 5: Arabic-specific normalization
    if language == "ar":
        # Remove Arabic diacritics (tashkeel) - small marks above/below letters
        text = araby.strip_tashkeel(text)
        # Remove Arabic tatweel (elongation dashes)
        text = araby.strip_tatweel(text)
        # Normalize Arabic letters (e.g. different forms of Alef → one standard form)
        text = araby.normalize_alef(text)
        text = araby.normalize_hamza(text)
        print("Applied Arabic normalization ✅")

    # Step 6: Unicode normalization (handles special characters across all languages)
    text = unicodedata.normalize('NFKC', text)

    print(f"Normalized: {text[:80]}...")
    print("Text normalization complete ✅")

    return text


def normalize_all_candidates(candidates, language="en"):
    """
    Takes a list of candidate transcripts and normalizes all of them.
    
    candidates: list of strings (the transcripts)
    language: detected language code
    
    Returns: list of normalized strings
    """
    print(f"\nNormalizing {len(candidates)} candidate transcripts...")
    normalized = []

    for i, transcript in enumerate(candidates):
        print(f"\n-- Candidate {i+1} --")
        cleaned = normalize_text(transcript, language)
        normalized.append(cleaned)

    print("\nAll candidates normalized ✅")
    return normalized