from ..utils.constants import CleanType
from .cleaner import TextCleaner


class PreprocessingPipeline:
    def __init__(self, company: str, mode: CleanType):
        self.company = company
        self.mode = mode
        self.cleaner = TextCleaner(company, mode)

    def preprocess(self, text: str, lang: str, author: str):
        cleaned_text = self.cleaner.clean(text, lang, author)
        return cleaned_text
