"""
Takes clean data and extracts intents from it by using a token-based approach.
"""

from pathlib import Path
from typing import Dict, List

from ..utils.io import read_csv_in_chunks
from wordcloud import WordCloud
from ..config.settings import get_settings
from collections import Counter
import spacy
import pandas as pd
import matplotlib.pyplot as plt


settings = get_settings()


class TokenExtractor:
    def __init__(
        self,
        dataset_path: Path,
        labelled_path: Path,
        ngrams_path: Path,
        wordcloud_path: Path,
    ):
        self._n_gram = settings.n_gram
        self._nlp = spacy.load(settings.spacy_model)
        self._nlp.add_pipe(
            "spacy-ngram",
            config={
                "ngrams": tuple(range(1, self._n_gram + 1)),
            },
            last=True,
        )
        self._dataset_path = dataset_path
        self._labelled_path = labelled_path
        self._ngrams_path = ngrams_path
        self._wordcloud_path = wordcloud_path
        self.counters: List[Counter[str]] = [Counter() for _ in range(self._n_gram)]

    def extract_n_grams(self) -> None:
        """
        Process the whole dataset and extract n-grams
        """
        for chunk in read_csv_in_chunks(self._dataset_path, "tweet_clean_text"):
            for doc in self._nlp.pipe(chunk, batch_size=512, n_process=1):
                for k in range(1, self._n_gram + 1):
                    self.counters[k - 1].update(getattr(doc._, f"ngram_{k}"))

        self._ngrams_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving ngram counters to {self._ngrams_path.parent}")
        for i, counter in enumerate(self.counters):
            filename: str = (
                f"{self._ngrams_path.stem}_{i + 1}{self._ngrams_path.suffix}"
            )
            filepath: Path = self._ngrams_path.parent / (filename)
            print(
                f"\nSaving most common 20 of the {len(counter)} {i + 1}-grams to {filename}."
            )
            with open(filepath, "w") as f:
                for ngram, count in counter.most_common(20):
                    f.write(f"{ngram}: {count},\n")

    def assign_ngram_based_intents(
        self,
        intent_mapping: Dict[
            str, str
        ],  # set made with the counter keys and the intent name
    ) -> None:
        """
        Adds the column for the intents to the dataset and saves it to a new file
        """
        intents: List[str] = []  # given the same csv the order will be the same

        for chunk in read_csv_in_chunks(self._dataset_path, "tweet_clean_text"):
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
                            print(f"Found {intent} ({' '.join(intent_list)}) in {doc}")
                        break
                intents.append(intent)

        tweets_df = pd.read_csv(self._dataset_path)
        tweets_df["intent"] = intents
        self._labelled_path.parent.mkdir(parents=True, exist_ok=True)
        tweets_df.to_csv(self._labelled_path, index=False)

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
            raise ValueError("No n-grams found. Please run extract_n_grams() first.")
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background,
            max_words=max_words,
            collocations=False,  # keep “new car” instead of “new car car”
        ).generate_from_frequencies(self.counters[0])

        self._wordcloud_path.parent.mkdir(parents=True, exist_ok=True)
        wc.to_file(self._wordcloud_path)
        print(f"Word cloud saved to {self._wordcloud_path}")

        plt.figure(figsize=(width / 100, height / 100))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
