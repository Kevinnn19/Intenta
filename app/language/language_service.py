from app.language.language_detector import detect_language
from app.language.translator import translator

class LanguageService:
    def process_query(self, text):
        language = detect_language(text)

        translated_text = text
        if language != "en":
            translated_text = translator.translate_to_english(text)

        return {
            "original_text": text,
            "processed_text": translated_text,
            "language": language
        }

language_service = LanguageService()