import re
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()

def tokenize(text):
    """Lowercase, tokenize, remove common stopwords and lemmatize."""
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = ENGLISH_STOP_WORDS
    except ImportError:
        stop_words = set()

    cleaned = []
    for word in words:
        if word not in stop_words:
            try:
                word = _lemmatizer.lemmatize(word)
            except LookupError:
                # If WordNet data is not installed, continue without lemmatization.
                pass
            cleaned.append(word)
    return cleaned
