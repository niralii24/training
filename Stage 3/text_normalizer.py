import re
import unicodedata
import langcodes

# ── Optional imports (graceful fallback if not installed) ─
try:
    import pyarabic.araby as araby
    HAS_PYARABIC = True
except ImportError:
    HAS_PYARABIC = False

try:
    from sacremoses import MosesPunctNormalizer
    HAS_MOSES = True
except ImportError:
    HAS_MOSES = False


# ── Script detection ──────────────────────────────────────
def get_script(language):
    """
    Returns the writing script for a given language code.
    Uses langcodes library — works for any language.
    """
    try:
        lang = langcodes.get(language)
        script = lang.script
        if not script:
            # Guess script from language
            script = langcodes.tag_is_rtl(language)
        return script or "Unknown"
    except Exception:
        return "Unknown"


def is_rtl(language):
    """
    Returns True if language is right-to-left.
    Works for Arabic, Hebrew, Persian, Urdu, and all RTL languages.
    """
    try:
        return langcodes.tag_is_rtl(language)
    except Exception:
        return False


# ── Number maps ───────────────────────────────────────────
NUMBER_MAPS = {
    "en": {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20"
    },
    "fr": {
        "zéro": "0", "un": "1", "deux": "2", "trois": "3",
        "quatre": "4", "cinq": "5", "six": "6", "sept": "7",
        "huit": "8", "neuf": "9", "dix": "10"
    },
    "de": {
        "null": "0", "ein": "1", "zwei": "2", "drei": "3",
        "vier": "4", "fünf": "5", "sechs": "6", "sieben": "7",
        "acht": "8", "neun": "9", "zehn": "10"
    },
    "es": {
        "cero": "0", "uno": "1", "dos": "2", "tres": "3",
        "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7",
        "ocho": "8", "nueve": "9", "diez": "10"
    },
    "ar": {
        "صفر": "0", "واحد": "1", "اثنان": "2", "ثلاثة": "3",
        "أربعة": "4", "خمسة": "5", "ستة": "6", "سبعة": "7",
        "ثمانية": "8", "تسعة": "9", "عشرة": "10"
    }
}

# ── Filler words per language ─────────────────────────────
FILLER_WORDS = {
    "en": ["uh", "um", "erm", "hmm", "ah", "like", "you know",
           "i mean", "basically", "literally", "actually", "so"],
    "ar": ["اه", "ااه", "ايه", "يعني", "اممم", "هممم", "طب"],
    "fr": ["euh", "ben", "bah", "hein", "quoi"],
    "de": ["äh", "ähm", "halt", "also", "ne"],
    "es": ["eh", "este", "o sea", "bueno", "pues"],
}


# ── Core normalization steps ──────────────────────────────

def step_unicode(text):
    """Unicode NFKC normalization — handles accents, special chars for ALL languages."""
    return unicodedata.normalize("NFKC", text)


def step_lowercase(text, language):
    """Lowercase — skips RTL scripts where case doesn't apply."""
    if is_rtl(language):
        return text
    return text.lower()


def step_punctuation(text, language):
    """
    Removes punctuation universally.
    Uses Moses punct normalizer if available for better handling,
    falls back to regex for all languages.
    """
    if HAS_MOSES:
        try:
            mpn = MosesPunctNormalizer(lang=language)
            text = mpn.normalize(text)
        except Exception:
            pass

    # Remove all punctuation characters universally
    # \w matches word chars in any Unicode script
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return text


def step_numbers(text, language):
    """
    Standardizes written numbers to digits.
    Uses language-specific number map if available,
    skips gracefully for unknown languages.
    """
    number_map = NUMBER_MAPS.get(language, {})
    for word, digit in number_map.items():
        text = re.sub(
            rf'(?<!\w){re.escape(word)}(?!\w)',
            digit,
            text,
            flags=re.IGNORECASE | re.UNICODE
        )
    return text


def step_fillers(text, language):
    """
    Removes filler words consistently.
    Uses language-specific filler list if available,
    skips gracefully for unknown languages.
    """
    fillers = FILLER_WORDS.get(language, [])
    for filler in fillers:
        text = re.sub(
            rf'(?<!\w){re.escape(filler)}(?!\w)',
            '',
            text,
            flags=re.IGNORECASE | re.UNICODE
        )
    return text


def step_arabic(text):
    """
    Arabic-specific normalization.
    Only runs if pyarabic is installed AND language is Arabic.
    Handles diacritics, tatweel, and letter normalization.
    """
    if not HAS_PYARABIC:
        return text

    text = araby.strip_tashkeel(text)   # remove diacritics
    text = araby.strip_tatweel(text)    # remove elongation
    text = araby.normalize_alef(text)   # unify Alef forms
    text = araby.normalize_hamza(text)  # unify Hamza forms
    return text


def step_whitespace(text):
    """Collapses multiple spaces and strips edges."""
    return re.sub(r'\s+', ' ', text, flags=re.UNICODE).strip()


def step_tokenize(text, language):
    """
    Basic tokenization — splits text into tokens and rejoins.
    Ensures consistent spacing around all word boundaries.
    Works for any language using Unicode word boundaries.
    
    Note: For CJK languages (Chinese, Japanese, Korean),
    characters are naturally space-separated after this step.
    """
    # CJK character block — insert space between each character
    cjk_pattern = re.compile(
        r'([\u4E00-\u9FFF'   # Chinese
        r'\u3040-\u30FF'     # Japanese hiragana/katakana
        r'\uAC00-\uD7AF])'   # Korean
    )
    text = cjk_pattern.sub(r' \1 ', text)
    return step_whitespace(text)


# ── Main normalization function ───────────────────────────

def normalize_text(text, language="en"):
    """
    Normalizes a single transcript for any language.

    Steps applied in order:
    1. Unicode normalization (universal)
    2. Lowercase (skipped for RTL scripts)
    3. Language-specific rules (Arabic, CJK etc.)
    4. Filler word removal
    5. Number standardization
    6. Punctuation removal
    7. Tokenization
    8. Whitespace cleanup

    Args:
        text:     raw transcript string
        language: ISO language code (e.g. 'en', 'ar', 'fr', 'zh')

    Returns:
        normalized text string
    """
    if not text or not text.strip():
        return ""

    print(f"  Original:   {text[:80]}")

    # 1. Unicode
    text = step_unicode(text)

    # 2. Lowercase
    text = step_lowercase(text, language)

    # 3. Language-specific rules
    if language in ("ar", "fa", "ur"):
        text = step_arabic(text)

    # 4. Filler words
    text = step_fillers(text, language)

    # 5. Numbers
    text = step_numbers(text, language)

    # 6. Punctuation
    text = step_punctuation(text, language)

    # 7. Tokenization
    text = step_tokenize(text, language)

    # 8. Whitespace
    text = step_whitespace(text)

    print(f"  Normalized: {text[:80]}")
    return text


def normalize_all_candidates(candidates, language="en"):
    """
    Normalizes all candidate transcripts.

    Args:
        candidates: list of raw transcript strings
        language:   detected language code from Stage 2

    Returns:
        list of normalized strings (same order as input)
    """
    print(f"\nNormalizing {len(candidates)} candidates (language: {language})...")
    normalized = []

    for i, transcript in enumerate(candidates):
        print(f"\n  -- Candidate {i+1} --")
        cleaned = normalize_text(transcript, language)
        normalized.append(cleaned)

    print(f"\nAll {len(candidates)} candidates normalized ✅")
    return normalized