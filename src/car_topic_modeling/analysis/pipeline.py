from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from car_topic_modeling.utils.constants import CleanType

from .config.factories import build_company_paths
from .config.types import (
    BERTopicCfg,
    BERTopicConfig,
    CompanyPaths,
    EmbedCfg,
    HDBSCANCfg,
    PCACfg,
    UMAPCfg,
)
from .topic_modeling.modeler import TopicModeler
from .semantic_based import SemanticExtractor
from .token_based import TokenExtractor

log = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    High-level orchestrator for token-based, semantic-based and topic-model
    intent extraction on a *single* company corpus.

    Parameters
    ----------
    company : str
        Company identifier used to build path layout.
    paths : CompanyPaths, optional
        Inject pre-built paths (useful for tests).
    token_extractor : TokenExtractor, optional
        Inject a custom extractor (for mocks / experiments).
    semantic_extractor : SemanticExtractor, optional
        Same for semantic step.
    """

    def __init__(
        self,
        company: str,
        *,
        paths: Optional[CompanyPaths] = None,
        token_extractor: Optional[TokenExtractor] = None,
        semantic_extractor: Optional[SemanticExtractor] = None,
    ) -> None:
        self.company = company
        self.paths = paths or build_company_paths(company)

        self.token_extractor = token_extractor or TokenExtractor(self.paths.token)

        self.semantic_extractor = semantic_extractor or SemanticExtractor(
            self.paths.semantic
        )

    def extract_token_intents(self, *, max_words: int = 200) -> Path:
        """
        Extract n-grams, build word-cloud and return the path of the token-labelled CSV.
        """
        log.info("Extracting n-grams for %s...", self.company)
        self.token_extractor.extract_ngrams()

        log.info("Generating word-cloud...")
        self.token_extractor.generate_word_cloud(max_words=max_words)

    def assign_token_intents(self, intent_mapping: Dict[str, str]) -> Path:
        """
        Apply a user-defined mapping (ngram -> intent) to the tweets.

        Returns
        -------
        Path
            CSV of tweets now including the ``intent`` column.
        """
        self.token_extractor.assign_ngram_intents(intent_mapping)
        return self.token_extractor.paths.labelled

    def extract_semantic_intents(self, *, reassign_intent: bool = False) -> Path:
        """
        Cluster tweets semantically and adds value to / overwrite the *intent*
        column.

        Parameters
        ----------
        reassign_intent : bool
            If True, previously assigned token-based intents can be overwritten.
        """
        self.semantic_extractor.cluster(reassign_intent=reassign_intent)
        return self.semantic_extractor.paths.labelled

    def search_topic_modeler(
        self,
        *,
        param_list: Optional[BERTopicConfig] = None,
        clean_type: CleanType = CleanType.SOFT,
    ) -> None:
        """
        Run a grid-search over a list of parameters to test and save every
        model under a hash-identified folder.

        Uses whichever scoring each strategy provides; BERTopic runs will be
        cached for manual inspection.
        """

        if param_list is None:
            base_cfg = BERTopicConfig(
                embeddings=EmbedCfg(
                    # provider="sentence_transformer",
                    # model_name="all-mpnet-base-v2",
                    provider="openai",
                    model_name="text-embedding-3-small",
                ),
                pca=PCACfg(
                    active=False,  # disable PCA
                ),
                umap=UMAPCfg(n_neighbors=15, n_components=10, min_dist=0.1),
                hdbscan=HDBSCANCfg(min_cluster_size=30, min_samples=1),
                bertopic=BERTopicCfg(top_n_words=10, nr_topics=30),
            )
            param_list: List[BERTopicConfig] = [base_cfg]

        modeler = TopicModeler(self.paths.topic, clean_type=clean_type)

        log.info("Grid-searching the params in %s", param_list)
        modeler.search(param_list=param_list)

    def visualize_2d_embeddings(
        self,
        *,
        n_neighbors: Optional[int] = None,
        n_components: Optional[int] = None,
        min_dist: Optional[float] = None,
        metric: Optional[str] = None,
        sentence_model: Optional[str] = None,
    ) -> None:
        """
        Produce an interactive 2-D UMAP scatter of document embeddings.
        All parameters map 1-to-1 to BERTopic's `visualize_document_datamap`.
        """
        TopicModeler(self.paths.topic).visualize_2d_embedding(
            n_neighbors, n_components, min_dist, metric, sentence_model
        )

    def visualize_topic_modeler_results(
        self,
    ) -> None:
        """
        Open topic bars, interactive topics plot and document datamap for a
        *previously saved* run.

        Parameters
        ----------
        model_name : str, optional
            Traditional backend name ('lda', 'nmf', ...).  Ignored by BERTopic.
        n_topics : int, optional
            Number of topics of the saved model.
        """
        TopicModeler(
            self.paths.topic,
        ).visualize_model_results()
