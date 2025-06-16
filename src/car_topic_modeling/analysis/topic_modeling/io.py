import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from bertopic import BERTopic
import numpy as np
from textacy.tm import TopicModel


def _paths_embeddings_models(out_dir: Path, cfg: dict) -> dict[str, Path]:
    stem = f"{cfg['model']}_{cfg['n_topics']}"
    return {
        "model_dir": out_dir / stem,
        "umap_emb": out_dir / f"{stem}_umap_emb.npz",
        "topics": out_dir / f"{stem}_topics.csv",
    }


def save_topic_model(
    topic_model: BERTopic,
    out_dir: Path,
    cfg: Dict[str, Any],
    umap_embeddings: np.ndarray,
    topics_mapping: Dict[Any, Any],
) -> None:
    """
    Persist a topic-model and its companion artefacts.

    Parameters:
    -------
    topic_model: BERTopic
    out_dir: Path
    cfg: Dict[str, Any]
    umap_embeddings: np.ndarray
    topics_mapping: {doc_id: topic_id}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _paths_embeddings_models(out_dir, cfg)
    topic_model.save(str(p["model_dir"]))  # BERTopic saves a folder

    if umap_embeddings is not None:
        np.savez_compressed(p["umap_emb"], emb=umap_embeddings)
        print("Saved UMAP embeddings ->", p["umap_emb"].relative_to(out_dir))

    if topics_mapping is not None:
        p["topics"].write_text(json.dumps(topics_mapping))
        print("Saved topics mapping ->", p["topics"].relative_to(out_dir))

    print("Saved BERTopic model ->", p["model_dir"].relative_to(out_dir))


def load_topic_model(
    model_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[
    Union[TopicModel, BERTopic],
    np.ndarray,
    Dict[Any, Any],
]:
    """
    Reload the artefacts saved with `save_topic_model()`.

    Returns
    -------
    topic_model     : BERTopic
    umap_embeddings : ndarray
    topics_json     : dict
    """
    p = _paths_embeddings_models(model_dir, cfg)
    tm = BERTopic.load(str(p["model_dir"]))

    umap_emb = None
    if p["umap_emb"].exists():
        umap_emb = np.load(p["umap_emb"])["emb"]
        print("Loaded UMAP embeddings <-", p["umap_emb"].relative_to(model_dir))

    topics_json = None
    if p["topics"].exists():
        topics_json = json.loads(p["topics"].read_text())
        print("Loaded topics mapping <-", p["topics"].relative_to(model_dir))

    print("Loaded BERTopic model <-", p["model_dir"].relative_to(model_dir))
    return tm, umap_emb, topics_json
