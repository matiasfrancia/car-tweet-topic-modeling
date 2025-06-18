from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from car_topic_modeling.analysis.topic_modeling.adapters import OpenAIBackendAdapter
from car_topic_modeling.analysis.topic_modeling.artifact_store import (
    ArtifactStore,
    run_or_load,
)

from ..config.factories import build_bertopic_pipeline_paths

from ..config.types import (
    BERTopicCfg,
    BERTopicConfig,
    BERTopicPaths,
    BERTopicSearchResult,
    DocEmbedder,
    EmbedCfg,
)
from ...config.settings import get_settings

import numpy as np
from sklearn.decomposition import PCA

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
import logging


log = logging.getLogger(__name__)

# def coherence_score(
#     model: BERTopic,
#     docs: List[str],
#     *,
#     coherence: str = "c_v", # coherence metric type
#     top_n_words: int = 10, # n words per topic to fit into the metric
# ) -> float:
#     """
#     Compute the coherence of a fitted BERTopic model.

#     Returns
#     -------
#     float : coherence
#         Higher is better. Returns -1.0 if the model has fewer than 2 topics.
#     """
#     topics = model.get_topics()
#     # drop outliers (-1) and empty topics
#     topic_words = [
#         [word for word, _ in words[:top_n_words]]
#         for tid, words in topics.items()
#         if tid >= 0 and words
#     ]

#     if len(topic_words) < 2:
#         log.warning("Less than two topics => coherence undefined; returning -1.0")
#         return -1.0

#     # gensim needs tokenised docs
#     tokenised_docs = [doc.split() for doc in docs]
#     dictionary = Dictionary(tokenised_docs)
#     corpus = [dictionary.doc2bow(text) for text in tokenised_docs]

#     cm = CoherenceModel(
#         topics=topic_words,
#         texts=tokenised_docs,
#         corpus=corpus,
#         dictionary=dictionary,
#         coherence=coherence,
#     )
#     return cm.get_coherence()


def _rescale(x, inplace=False):
    """
    Rescale an embedding so optimization will not have convergence
    issues.
    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000
    return x


def _make_embedder(cfg: EmbedCfg) -> DocEmbedder:
    """
    Returns a DocEmbedder object, i.e., with an .encode(docs) method
    """
    settings = get_settings()
    provider: str = cfg.provider
    model: str = cfg.model_name

    if provider == "openai":
        log.info(f"Generating embeddings for the docs with OpenAI's {model}")
        return OpenAIBackendAdapter(model, api_key=settings.openai_api_key)

    log.info(f"Generating embeddings for the docs with SentenceTransformer's {model}")
    return SentenceTransformer(model)


def _train_bertopic(
    cfg: BERTopicCfg,
    umap_model: UMAP,
    hdbscan_model: hdbscan.HDBSCAN,
    emb_lowdim: np.ndarray,
    docs: list[str],
) -> BERTopic:
    """
    Returns a fully-fitted BERTopic model for run_or_load function.
    """
    bertopic_kwargs: Dict[str, Any] = asdict(cfg)
    tm = BERTopic(
        embedding_model=None,  # we pass embeddings directly in the fit method
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        **bertopic_kwargs,
    )
    tm.fit(docs, embeddings=emb_lowdim)
    return tm


def grid_search_topic_modeling_embedding_models(
    docs: Iterable[str],
    models_dir_path: Path,
    param_grid: List[BERTopicConfig],  # list of bertopic hyper-param configs
) -> BERTopicSearchResult:
    """
    Train several BERTopic configurations, saves the UMAP embeddings
    and shows the best configurations

    Returns
    -------
    search_result: BERTopicSearchResult
    """
    docs = list(docs)
    search_result: BERTopicSearchResult = None
    best_score: float = -1.0

    artifact_store = ArtifactStore()
    log.info("Embeddings Grid-Search called")

    for cfg in param_grid:
        # make paths for the bertopic model with all its steps
        paths: BERTopicPaths = build_bertopic_pipeline_paths(models_dir_path, cfg)

        embeddings: np.ndarray
        embeddings, _ = run_or_load(
            artifact_store,
            paths.embeddings,
            cfg.embeddings,
            compute_fn=lambda: _make_embedder(cfg.embeddings).encode(docs),
        )

        pca_embeddings: np.ndarray | None = None
        if cfg.pca and cfg.pca.active:
            pca_dims = cfg.pca.dimensions
            pca_embeddings, _ = run_or_load(
                artifact_store,
                paths.pca,
                cfg.pca,
                compute_fn=lambda: _rescale(
                    PCA(n_components=pca_dims).fit_transform(embeddings)
                ),
            )
            log.info(
                f"Generated PCA embeddings, calculated with {pca_dims} dimensions."
            )

        umap_kwargs: Dict[str, Any] = asdict(cfg.umap).copy()
        umap_kwargs["init"] = (
            pca_embeddings if pca_embeddings is not None else cfg.umap.init
        )
        umap_model = UMAP(**umap_kwargs)

        log.info("Created UMAP model")
        umap_embeddings: np.ndarray
        umap_embeddings, _ = run_or_load(
            artifact_store,
            paths.umap,
            cfg.umap,
            compute_fn=lambda: umap_model.fit_transform(embeddings),
        )
        log.info("Generated/loaded UMAP embeddings")

        hdbscan_model = hdbscan.HDBSCAN(**asdict(cfg.hdbscan), prediction_data=True)
        log.info("Created HDBSCAN Model")

        topic_model: BERTopic
        topic_model, _ = run_or_load(
            artifact_store,
            paths.bertopic,
            cfg.bertopic,
            compute_fn=lambda: _train_bertopic(
                cfg.bertopic, umap_model, hdbscan_model, umap_embeddings, docs
            ),
        )

        topics: np.ndarray = topic_model.hdbscan_model.labels_
        log.info("BERTopic Model fitted with current configuration and docs")

        # TODO: change silhouette by coherence
        # score = coherence_score(topic_model, docs, top_n_words=cfg.bertopic.top_n_words)
        score = 0.5
        n_topics = int(np.unique(topics[topics >= 0]).size)

        log.info(f"cfg={cfg} -> coherence = {score:.3f}, (n_topics = {n_topics})")

        if score > best_score:
            best_score = score
            search_result = BERTopicSearchResult(
                score=score, cfg=cfg, paths=paths, n_topics=n_topics
            )

    if search_result is None:
        raise RuntimeError("The topic-modeling search did not produced no valid model.")

    return search_result
