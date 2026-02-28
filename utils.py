import re
import unicodedata

def normalize_text(text):
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text