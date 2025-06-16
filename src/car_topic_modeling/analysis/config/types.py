from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic.backend import OpenAIBackend


# ============== Paths ==============
@dataclass(frozen=True)
class TokenPaths:
    dataset: Path
    labelled: Path
    wordcloud: Path
    ngrams: Path


@dataclass(frozen=True)
class SemanticPaths:
    dataset: Path
    labelled: Path


@dataclass(frozen=True)
class TopicModelingPaths:
    dataset: Path
    models_dir: Path
    labelled: Path
    report: Path
    datamap: Path


@dataclass(frozen=True)
class CompanyPaths:
    token: TokenPaths
    semantic: SemanticPaths
    topic: TopicModelingPaths


@dataclass(frozen=True)
class StagePaths:
    """
    Paths for ONE hash belonging to ONE stage.
    """

    file: Path  # .npy or .bin
    meta: Path  # .json


@dataclass(frozen=True)
class BERTopicPaths:
    """
    All stages for a single run (may mix hashes).
    """

    embeddings: StagePaths
    pca: StagePaths | None  # None if PCA disabled
    umap: StagePaths
    bertopic: StagePaths


# ============== Config ==============
@dataclass(frozen=True, slots=True)
class EmbedCfg:
    provider: Literal["sentence_transformer", "openai"] = "sentence_transformer"

    model_name: Literal["all-mpnet-base-v2", "text-embedding-3-small"] = (
        "all-mpnet-base-v2"
    )


@dataclass(frozen=True, slots=True)
class PCACfg:
    active: bool = True
    dimensions: int = 50


@dataclass(frozen=True, slots=True)
class UMAPCfg:
    init: Literal["spectral", "random"] = "spectral"
    n_neighbors: int = 30
    n_components: int = 5
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class HDBSCANCfg:
    min_cluster_size: int = 10
    min_samples: int = 5
    metric: str = "euclidean"
    cluster_selection_method: Literal["eom", "leaf"] = "eom"


@dataclass(frozen=True, slots=True)
class BERTopicCfg:
    top_n_words: int = 15
    language: str = "english"
    calculate_probabilities: bool = True
    nr_topics: int = 20


@dataclass(frozen=True, slots=True)
class BERTopicConfig:
    """
    Configuration class for BERTopic model, considering the configuration
    of all the different models it has to generate during the process.
    """

    embeddings: EmbedCfg
    pca: Optional[PCACfg]
    umap: UMAPCfg
    hdbscan: HDBSCANCfg
    bertopic: BERTopicCfg


# ============== Interfaces ==============
@runtime_checkable
class Embedder(Protocol):
    """
    Interface that any embedder passed to embed the documents
    in BERTopic must satisfy.
    """

    def encode(
        self,
        docs: Iterable[str],
    ) -> Sequence[np.ndarray]: ...


DocEmbedder = Union[SentenceTransformer, OpenAIBackend]


class CacheMode(Enum):
    AUTO = auto()
    FORCE_LOAD = auto()
    FORCE_COMPUTE = auto()


# ============== Results ==============
@dataclass
class BERTopicSearchResult:
    score: float
    cfg: Dict[str, Any]
    paths: BERTopicPaths
    n_topics: int
