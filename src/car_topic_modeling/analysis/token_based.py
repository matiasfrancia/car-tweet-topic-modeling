"""
Takes clean data and extracts intents from it by using a token-based approach.
"""

import logging
from pathlib import Path
from typing import Dict, List

from .config.types import TokenPaths

from ..utils.io import read_csv_in_chunks
from wordcloud import WordCloud
from ..config.settings import get_settings
from collections import Counter
import spacy
import spacy_ngram
import pandas as pd
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)
_settings = get_settings()


class TokenExtractor:
    def __init__(self, paths: TokenPaths) -> None:
        self.paths = paths
        self._n_gram = _settings.n_gram

        self._nlp = spacy.load(_settings.spacy_model)
        self._nlp.add_pipe(
            "spacy-ngram",
            config={"ngrams": tuple(range(1, self._n_gram + 1))},
            last=True,
        )

        # counters[k] counts k-grams (k = 1 ... n)
        self.counters: list[Counter[str]] = [Counter() for _ in range(self._n_gram)]

    def extract_ngrams(self) -> None:
        """
        Process the whole dataset and extract n-grams
        """
        for chunk in read_csv_in_chunks(self.paths.dataset, "tweet_clean_text"):
            for doc in self._nlp.pipe(chunk, batch_size=512, n_process=1):
                for k in range(1, self._n_gram + 1):
                    self.counters[k - 1].update(getattr(doc._, f"ngram_{k}"))

        self.paths.ngrams.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving ngram counters to {self.paths.ngrams.parent}")
        for i, counter in enumerate(self.counters):
            filename: str = (
                f"{self.paths.ngrams.stem}_{i + 1}{self.paths.ngrams.suffix}"
            )
            filepath: Path = self.paths.ngrams.parent / (filename)
            log.info(
                f"\nSaving most common 20 of the {len(counter)} {i + 1}-grams to {filename}."
            )
            with open(filepath, "w") as f:
                for ngram, count in counter.most_common(20):
                    f.write(f"{ngram}: {count},\n")

    def assign_ngram_intents(
        self,
        intent_mapping: Dict[
            str, str
        ],  # set made with the counter keys and the intent's name
    ) -> None:
        """
        Adds the column for the intents to the dataset and saves it to a new file
        """
        intents: List[str] = []  # given the same csv the order will be the same

        for chunk in read_csv_in_chunks(self.paths.dataset, "tweet_clean_text"):
            for doc in self._nlp.pipe(chunk, batch_size=512, n_process=1):
                intent: str = ""
                for k in range(1, self._n_gram + 1):
                    k_grams: List[str] = getattr(doc._, f"ngram_{k}")
                    if not set(intent_mapping.keys()).isdisjoint(k_grams):
                        intersection: set[str] = set(
                            intent_mapping.keys()
                        ).intersection(k_grams)
                        intent_list = list(
                            {intent_mapping[k_gram] for k_gram in list(intersection)}
                        )
                        intent = ",".join(intent_list)
                        if len(intent_list) > 1:
                            log.debug(
                                f"Found {intent} ({' '.join(intent_list)}) in {doc}"
                            )
                        break
                intents.append(intent)

        tweets_df = pd.read_csv(self.paths.dataset)
        tweets_df["intent"] = intents
        self.paths.labelled.parent.mkdir(parents=True, exist_ok=True)
        tweets_df.to_csv(self.paths.labelled, index=False)

    def generate_word_cloud(
        self,
        max_words: int = 200,
        background: str = "white",
        width: int = 800,
        height: int = 400,
    ) -> None:
        """
        Generate a word cloud from the extracted n-grams
        """
        if not self.counters[0]:
            raise ValueError("No n-grams found. Please run extract_ngrams() first.")
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background,
            max_words=max_words,
            collocations=False,  # keep “new car” instead of “new car car”
        ).generate_from_frequencies(self.counters[0])

        self.paths.wordcloud.parent.mkdir(parents=True, exist_ok=True)
        wc.to_file(self.paths.wordcloud)
        log.info(f"Word cloud saved to {self.paths.wordcloud}")

        plt.figure(figsize=(width / 100, height / 100))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
