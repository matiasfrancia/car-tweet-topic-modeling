import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from bertopic import BERTopic
from car_topic_modeling.analysis.config.factories import build_bertopic_pipeline_paths
from car_topic_modeling.analysis.topic_modeling.artifact_store import (
    ArtifactStore,
    run_or_load,
)
from car_topic_modeling.utils.constants import CleanType
from ..config.types import (
    BERTopicConfig,
    BERTopicPaths,
    BERTopicSearchResult,
    CacheMode,
    TopicModelingPaths,
)
from .visualization import visualize_UMAP_2d_embeddings
from car_topic_modeling.utils.io import read_csv_in_chunks
import numpy as np
import webbrowser

from .grid_search import grid_search_topic_modeling_embedding_models
from plotly.io import renderers

renderers.default = "browser"

log = logging.getLogger(__name__)


class TopicModeler:
    """
    Train or load a BERTopic-based clustering model and open basic visualisations.

    Parameters
    ----------
    paths : TopicModelingPaths
        Folder layout for dataset, models and reports.
    model_name, n_topics : str | int, optional
        If provided, the constructor tries to *load* a previously saved
        traditional (textacy) model instead of training BERTopic.
    """

    def __init__(
        self,
        paths: TopicModelingPaths,
        *,
        clean_type: Optional[CleanType] = CleanType.SOFT,
    ):
        self.paths = paths
        self.clean_type = clean_type

        self.docs, self.raw_text = self._load_docs(self.paths.dataset)

    def _load_docs(self, csv_path: Path) -> Tuple[List[str], List[str]]:
        """
        Read tweets in chunks; return (cleaned_texts, raw_texts).
        Only rows whose `intent` column is empty are used.
        """
        # TODO: implement the actual iterable logic by adding the chunk of tweets
        # to the clustering model
        raw_texts, cleaned = [], []
        clean_type: str = (
            "aggressive" if self.clean_type == CleanType.AGGRESSIVE else "soft"
        )
        for rows in read_csv_in_chunks(
            csv_path, ["tweet_text", f"tweet_{clean_type}_clean_text", "intent"]
        ):
            for raw, clean, intent in rows:
                if intent and str(intent).lower() != "nan":
                    continue
                raw_texts.append(raw)
                cleaned.append(clean)
        return cleaned, raw_texts

    def search(
        self,
        param_list: List[BERTopicConfig],
    ) -> None:
        """
        Train several configs for the BERTopic model.
        Saves the results to a folder hashed with the parameters passed.

        For instance, it saves the embeddings generated through UMAP with
        a name hashed using md5 based on the UMAP hyperparameters. The same
        goes for HDBSCAN and BERTopic models.

        Parameters:
        ------
        param_list : List[BERTopicConfig]
            List of dictionaries with different parameter configurations.
        """

        result: BERTopicSearchResult = grid_search_topic_modeling_embedding_models(
            self.docs, self.paths.models_dir, param_list
        )

        log.info("Saved best model (coherence = %.3f)", result.score or float("nan"))
        self.visualize_model_results(result.cfg)

    def visualize_model_results(
        self,
        cfg: BERTopicConfig,
    ) -> None:
        """
        Show topic bar-chart, interactive topic map, and document datamap.
        """
        if cfg is None:
            raise ValueError("The hyper-parameter configuration or the must be passed.")

        log.info("Starting visualization of the model results.")

        artifact_store = ArtifactStore()

        bertopic_paths: BERTopicPaths = build_bertopic_pipeline_paths(
            self.paths.models_dir, cfg
        )

        topic_model: BERTopic
        topic_model, _ = run_or_load(
            artifact_store,
            bertopic_paths.bertopic,
            cfg.bertopic,
            mode=CacheMode.FORCE_LOAD,
        )

        log.info("Loaded the topic model.")

        umap_embeddings: np.ndarray
        umap_embeddings, _ = run_or_load(
            artifact_store, bertopic_paths.umap, cfg.umap, mode=CacheMode.FORCE_LOAD
        )
        log.info("Loaded the UMAP reduced embeddings.")

        topic_model.visualize_barchart(
            top_n_topics=cfg.bertopic.nr_topics, n_words=10
        ).show()
        log.info("Visualizing barchart.")

        topic_model.visualize_topics().show()
        log.info("Visualizing topics.")

        datamap = topic_model.visualize_document_datamap(
            self.docs,
            embeddings=umap_embeddings,
            interactive=True,
        )
        log.info("Visualizing document datamap.")

        self.paths.datamap.parent.mkdir(parents=True, exist_ok=True)
        datamap.save(self.paths.datamap)

        log.info("Opening datamap in the browser.")

        webbrowser.open_new_tab(self.paths.datamap.as_uri())
        log.info("Document datamap written to %s", self.paths.datamap)

    def visualize_2d_embedding(
        self,
        n_neighbors: int | None,
        n_components: int | None,
        min_dist: float | None,
        metric: str | None,
        sentence_model: str | None,
    ) -> None:
        """
        Reduce document embeddings to 2-D with UMAP and show an interactive scatter.
        """
        cfg: Dict[str, Any] = {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "metric": metric,
            "sentence_model": sentence_model,
        }
        visualize_UMAP_2d_embeddings(self.docs, cfg)
