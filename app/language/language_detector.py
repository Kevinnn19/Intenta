from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

SUPPORTED_LANGUAGES = {
    "en", "hi", "gu"
}

def detect_language(text):
    try:
        predictions = detect_langs(text)
        language = predictions[0].lang
        if language not in SUPPORTED_LANGUAGES:
            return "en"
        return language
    except LangDetectException:
        return "en"
