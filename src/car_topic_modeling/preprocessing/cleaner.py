from emoji import demojize
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class TextCleaner:
    def __init__(self):
        self._mention_pattern = re.compile(r"@\w+")
        self._hashtag_pattern = re.compile(r"#\w+")
        self._url_pattern = re.compile(r"http\S+")
        self._connector_pattern = re.compile(r"\b(RT|via)\b", flags=re.IGNORECASE)
        self._punctuation = string.punctuation
        self._nlp = spacy.load("en_core_web_sm")

    def clean(self, text: str) -> str:
        text = self.remove_emojis(text)
        text = self.remove_mentions(text)
        text = self.remove_urls(text)
        text = self.remove_connectors(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.remove_short_queries(text)
        text = self.remove_numbers(text)
        text = text.lower()
        return text.strip()

    def filter_by_lang(self, text: str, lang: str = "en") -> str:
        if lang == "en":
            return text
        return ""

    def remove_links(self, text: str) -> str:
        return self._url_pattern.sub(r"", text)

    def remove_emojis(self, text: str) -> str:
        return demojize(text, delimiters=(" ", " "))

    def remove_mentions(self, text: str) -> str:
        return self._mention_pattern.sub(r"", text)

    def remove_urls(self, text: str) -> str:
        return self._url_pattern.sub(r"", text)

    def remove_connectors(self, text: str) -> str:
        return self._connector_pattern.sub(r"", text)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", self._punctuation))

    def remove_stopwords(self, text: str) -> str:
        doc = self._nlp(text)
        return " ".join(
            [token.text for token in doc if token.text.lower() not in STOP_WORDS]
        )

    def remove_short_queries(self, text: str) -> str:
        """Removes short queries (less than 3 words)."""
        if len(text.split()) < 3:
            return ""
        return text

    def remove_numbers(self, text: str) -> str:
        """Removes numbers from the text."""
        return re.sub(r"\d+", "", text)
