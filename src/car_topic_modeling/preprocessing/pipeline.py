from .cleaner import TextCleaner


class PreprocessingPipeline:
    def __init__(self):
        self.cleaner = TextCleaner()

    def preprocess(self, text: str):
        cleaned_text = self.cleaner.clean(text)
        return cleaned_text
