from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set
from datasketch import MinHash, MinHashLSH
from difflib import SequenceMatcher
import networkx as nx
import pandas as pd
from ..config.settings import get_settings
from ..utils.io import read_csv_in_chunks


settings = get_settings()


class SemanticExtractor:
    def __init__(
        self,
        dataset_path: Path,
        labelled_path: Path,
    ):
        self._dataset_path = dataset_path
        self._labelled_path = labelled_path

        self._num_perm = settings.num_perm
        self._jaccard_th = settings.jaccard_threshold
        self._ro_th = settings.ro_threshold
        self._n_shingles = settings.n_gram_shingles
        self._misc_count: int = 0

        self.raw_strings: Dict[str, str] = {}
        self.token_sets: Dict[str, Set[str]] = {}
        self.signatures: Dict[str, MinHash] = {}
        self.lsh = MinHashLSH(threshold=self._jaccard_th, num_perm=self._num_perm)
        self.G_clusters = nx.Graph()

    def _shingle(self, tokens: List[str]) -> Set[str]:
        return {
            " ".join(tokens[i : i + self._n_shingles])
            for i in range(len(tokens) - self._n_shingles + 1)
        }

    def _read_chunk(
        self, reassign_intent: bool, chunk_size: int = 1_000
    ) -> Iterable[tuple[str, List[str]]]:
        for rows in read_csv_in_chunks(
            self._dataset_path,
            text_col=["tweet_id", "tweet_clean_text", "intent"],
            chunk_size=chunk_size,
        ):
            for tid, text, intent in rows:
                if not reassign_intent and intent != "nan" and intent.strip():
                    continue
                tokens: List[str] = text.split()
                if len(tokens) >= 3:
                    yield tid, tokens

    def _representative_label(self, cluster_ids: set[str]) -> str:
        """
        Uses the top-2 most common words concatenated to
        make the cluster label.
        """
        freq = Counter()
        for tid in cluster_ids:
            freq.update(self.raw_strings[tid].split())
        parts = [s for s, _ in freq.most_common(2)]
        return " | ".join(parts)

    def _write_labelled_dataset(
        self, intent_map: Dict[str, str], reassign: bool
    ) -> None:
        """
        Rewrite original dataset file with updated `intent` column
        and by dropping tweets that have repeated texts.
        """
        first = True
        self._labelled_path.parent.mkdir(parents=True, exist_ok=True)

        for chunk in pd.read_csv(self._dataset_path, chunksize=10_000, dtype=str):
            if not reassign:
                mask = chunk["intent"].str.strip().eq("") | chunk[
                    "intent"
                ].str.lower().eq("nan")
            else:
                mask = slice(None)  # overwrite all rows

            chunk.loc[mask, "intent"] = (
                chunk.loc[mask, "tweet_id"]
                .map(intent_map)
                .fillna(chunk.loc[mask, "intent"])
            )

            mode = "w" if first else "a"
            chunk.to_csv(self._labelled_path, mode=mode, index=False, header=first)
            first = False

        print(f"Saved labelled dataset to {self._labelled_path}")

    def cluster(self, reassign_intent: bool = False) -> List[Set[str]]:
        # create shingles for each document/tweet
        for tid, tokens in self._read_chunk(reassign_intent):
            shingles = self._shingle(tokens)
            self.token_sets[tid] = shingles
            self.raw_strings[tid] = " ".join(tokens)

            # use minhash to find the best hashing for all the shingles
            mh = MinHash(self._num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))
            self.signatures[tid] = mh

            # insert the tweet with the hash to the minlsh object
            self.lsh.insert(tid, mh)

        # get the tweets that have a jaccard_th >= 0.8
        for tid, mh in self.signatures.items():
            for nid in self.lsh.query(mh):
                if tid >= nid:
                    continue

                # check jaccard score
                A, B = self.token_sets[tid], self.token_sets[nid]
                jacc = len(A & B) / len(A | B)
                if jacc < self._jaccard_th:
                    continue

                # check ro score
                A_string, B_string = self.raw_strings[tid], self.raw_strings[nid]
                seq_matcher = SequenceMatcher(None, A_string, B_string)
                ro = seq_matcher.ratio()

                if ro >= self._ro_th:
                    self.G_clusters.add_edge(tid, nid)

        clusters: List[Set[str]] = list(nx.connected_components(self.G_clusters))

        intent_map: Dict[str, str] = {}  # maps tid to intent
        for cluster in clusters:
            intent: str = (
                self._representative_label(cluster)
                if len(cluster) >= settings.min_cluster_size
                else ""
            )
            for tid in cluster:
                if intent == "":
                    self._misc_count += 1
                intent_map[tid] = intent

        print(f"""Found {len(clusters)} clusters. Each with the 
              following number of elements:""")
        for i, cluster in enumerate(clusters):
            print(f"============== Cluster {i}: {len(cluster)} ==============")
            for tid in cluster:
                print(f"{self.raw_strings[tid]}")

        # filter the df based on the tweet ids gotten in token_sets
        self._write_labelled_dataset(intent_map, reassign=reassign_intent)
        print(f"Number of miscelaneous labels: {self._misc_count}")

        return clusters
