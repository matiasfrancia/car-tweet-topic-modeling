from ..utils.constants import CleanType
from typing import Set
from emoji import demojize
import re
import string

from ..config.settings import get_settings
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


settings = get_settings()


class TextCleaner:
    def __init__(self, company: str, mode: CleanType):
        self.company = company
        self.mode = mode
        self._mention_pattern = re.compile(r"@\w+")
        self._hashtag_pattern = re.compile(r"#\w+")
        self._url_pattern = re.compile(r"http\S+")
        self._connector_pattern = re.compile(r"\b(RT|via)\b", flags=re.IGNORECASE)
        self._control_chars_pattern = re.compile(
            r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]"
        )
        self._punctuation = string.punctuation
        self._nlp = spacy.load(settings.spacy_model)
        self._seen_tweets: Set[str] = set()
        self._punctuation_collapse = re.compile(r"([!?.,;:])\1{1,}")

    def clean(self, text: str, lang: str, author: str) -> str:
        if not text:
            print("An empty text was given for cleaning.")
            return ""
        text = self.remove_company_tweets(text, author)
        text = self.filter_by_lang(text, lang)
        text = self.remove_control_chars(text)
        text = self.transform_emojis(text)
        text = self.remove_mentions(text)
        text = self.remove_urls(text)
        text = self.remove_short_queries(text)
        text = self.remove_numbers(text)
        text = self.remove_consecutive_duplicated_chars(text)
        if self.mode == CleanType.AGGRESSIVE:
            text = self.remove_connectors(text)
            text = self.remove_punctuation(text)
            text = self.remove_stopwords(text)
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = self.remove_duplicates(text)
        return text.strip()

    def remove_consecutive_duplicated_chars(self, text: str) -> str:
        """
        Collapses duplicated characters or words into just one of them.
        """
        text = self._punctuation_collapse.sub("\1", text)
        tokens = text.split()
        deduped = [tokens[0]] if tokens else []
        deduped.extend(
            tok for i, tok in enumerate(tokens[1:], 1) if tok != tokens[i - 1]
        )
        return " ".join(deduped)

    def remove_duplicates(self, clean_text: str) -> str:
        """
        Removes tweets that have duplicated text.
        This function has to be called after cleaning the text.
        """
        if clean_text in self._seen_tweets:
            return ""
        self._seen_tweets.add(clean_text)
        return clean_text

    def remove_company_tweets(self, text: str, author: str) -> str:
        """
        Removes tweets of the company itself.
        """
        if isinstance(author, str) and author.lower() == self.company.lower():
            return ""
        return text

    def filter_by_lang(self, text: str, lang: str) -> str:
        if lang == "en":
            return text
        return ""

    def remove_control_chars(self, text: str) -> str:
        """
        Removes control characters given by twitter from the text.
        """
        return self._control_chars_pattern.sub(r"", text)

    def transform_emojis(self, text: str) -> str:
        """
        Replaces the emojis in the text with a string describing it.
        """
        # return emoji.replace_emoji(text) # removes the emoji
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
        """
        Removes short queries (less than 3 words).
        """
        if len(text.split()) < 3:
            return ""
        return text

    def remove_numbers(self, text: str) -> str:
        """
        Removes numbers from the text.
        """
        return re.sub(r"\d+", "", text)
