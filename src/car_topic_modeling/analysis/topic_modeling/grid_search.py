from itertools import product
from typing import Any, Dict, Iterable, List, Tuple
from scipy.sparse import csr_matrix

import numpy as np
from sklearn.metrics import silhouette_score
from .vectorizer import build_document_term_matrix
from textacy.tm import TopicModel

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan


def grid_search_topic_modeling_traditional_ML(
    docs: Iterable[str],
    k_list: Tuple[int],
    models: List[str],
) -> Tuple[
    float,  # score
    TopicModel,  # best_model
    Dict[str, Any],  # best_config
    csr_matrix,  # best_dtm
    List[str],  # best_terms
]:
    """
    Does grid-search to find the best configuration for the topic
    modeling task given the docs passed as input, using nmf, lsa or
    lda. It returns the data of the best configuration based on the
    silhouette score.

    Returns
    -------
    best_score : float
        Best silhouette score obtained in the grid search.
    best_model : textacy.tm.TopicModel
        Best model found based on the silhouette score.
    best_config : Dict[str, Any]
        Best configuration of the search.
    best_dtm : csr_matrix
        Best configuration of the document-term matrix
    best_terms : List[str]
        List of terms used for creating the best dtm
    """
    best_score, best_model, best_config = -1.0, None, None
    best_dtm: csr_matrix = None
    best_terms: List[str] = []
    docs: List[str] = [doc.split() for doc in docs]

    for model_name, k in product(models, k_list):
        # create a weighted document-term matrix
        dtm, terms = build_document_term_matrix(docs)
        print(f"dtm shape = {dtm.shape}, terms shape = {len(terms)}")

        tm = TopicModel(model=model_name, n_topics=k, alpha_W=0.0, alpha_H=0.0)
        tm.fit(dtm)
        theta = tm.transform(dtm)
        labels = theta.argmax(axis=1)
        score: float = silhouette_score(theta, labels, metric="cosine")
        print(f"Silhouette score obtained with model={model_name}, k={k} was {score}.")

        if score > best_score:
            best_score = score
            best_model = tm
            best_config = dict(model=model_name, n_topics=k)
            best_dtm = dtm
            best_terms = terms

    # TODO: make grid-search also with the dtm generation's configuration
    return best_score, best_model, best_config, best_dtm, best_terms


def _umap_silhouette(emb: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0  # drop outliers (-1) -> check if this is correct
    if len(set(labels[mask])) < 2:  # silhouette needs ≥ 2 clusters
        return -1.0
    return silhouette_score(emb[mask], labels[mask], metric="euclidean")


def grid_search_topic_modeling_embedding_models(
    docs: Iterable[str],
    param_grid: List[Dict],  # list of hyper-param dicts
) -> Tuple[
    float,  # best_score
    BERTopic,  # best_model
    Dict[str, Any],  # best_config
    np.ndarray,  # umap_embeddings
    Dict[int, int],  # topics_mapping
]:
    """
    Train several BERTopic configurations and keep the one with the highest
    silhouette score (computed on UMAP embeddings, outliers dropped).

    Returns
    -------
    best_score      : float
    best_model      : BERTopic
    best_config     : hyper-parameters that produced `best_model`
    umap_embeddings : ndarray (n_docs x n_components)
    topics_mapping  : {doc_id: topic_id}
    """
    docs = list(docs)
    best_score, best_model, best_config = -1.0, None, None
    best_emb, best_map = None, None

    embedder = SentenceTransformer("all-mpnet-base-v2")

    print("Embeddings Grid-Search called")

    for cfg in param_grid:
        umap_model = UMAP(
            init="random",
            n_neighbors=cfg.get("n_neighbors", 30),
            n_components=cfg.get("n_components", 5),
            min_dist=cfg.get("min_dist", 0.0),
            metric="cosine",
            random_state=42,
        )

        print("Created UMAP Model")

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=cfg.get("min_cluster_size", 10),
            min_samples=cfg.get("min_samples", 5),
            metric="euclidean",
            prediction_data=True,
            cluster_selection_method="eom",
        )

        print("Created HDBSCAN Model")

        tm = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=15,
            language="english",
            calculate_probabilities=True,
        )

        print("Created BERTopic Model")

        topics, _ = tm.fit_transform(docs)

        print("Fitted the BERTopic Model with the data")
        emb = tm.umap_model.embedding_  # numpy array (n_docs × d)
        score = _umap_silhouette(emb, np.array(topics))
        n_topics = int(np.unique(np.array(topics)).max())
        cfg["n_topics"] = n_topics

        print(f"cfg={cfg} -> silhouette = {score:.3f}")

        if score > best_score:
            best_score, best_model, best_config = score, tm, cfg
            best_emb = emb.copy()
            best_map = {i: int(t) for i, t in enumerate(topics)}

        print("Obtained the UMAP Embeddings")

    return best_score, best_model, best_config, best_emb, best_map
