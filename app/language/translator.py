from deep_translator import GoogleTranslator

class Translator:
    def __init__(self):
        self.translator = GoogleTranslator(
            source='auto',
            target='en'
        )

    def translate_to_english(self, text):
        return self.translator.translate(text)

translator = Translator()