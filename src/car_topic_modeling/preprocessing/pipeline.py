from ..utils.constants import CleanType
from .cleaner import TextCleaner


class PreprocessingPipeline:
    def __init__(self, company: str):
        self.company = company
        self.cleaner = TextCleaner(company)

    def preprocess(self, text: str, lang: str, author: str, mode: CleanType):
        cleaned_text = self.cleaner.clean(text, lang, author, mode)
        return cleaned_text
