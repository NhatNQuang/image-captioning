# model/translator.py
from deep_translator import GoogleTranslator

class Translator:
    """
    Class handles text translation from English to Vietnamese.
    """
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='vi')

    def translate(self, text: str) -> str:
        """
        Translates a given English text to Vietnamese.
        """
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Error during translation: {e}")
            return f"Translation error: {text}"