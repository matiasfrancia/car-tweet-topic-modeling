import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from bertopic import BERTopic
import numpy as np
from textacy.tm import TopicModel
from scipy import sparse as sp


def _paths_traditional_ml(out_dir: Path, cfg: dict) -> dict[str, Path]:
    stem = f"{cfg['model']}_k{cfg['n_topics']}"
    return {
        "model": out_dir / f"{stem}.tmx",
        "dtm": out_dir / f"{stem}_dtm.npz",
        "terms": out_dir / f"{stem}_terms.json",
    }


def _paths_embeddings_models(out_dir: Path, cfg: dict) -> dict[str, Path]:
    stem = f"{cfg['model']}_{cfg['n_topics']}"
    return {
        "model_dir": out_dir / stem,
        "umap_emb": out_dir / f"{stem}_umap_emb.npz",
        "topics": out_dir / f"{stem}_topics.csv",
    }


def save_topic_model(
    tm: Union[TopicModel, BERTopic],
    out_dir: Path,
    cfg: Dict[str, Any],
    *,
    model_type: str = "embeddings",  # "embeddings" | "traditional" # TODO: make it with constants
    dtm: sp.csr_matrix | None = None,
    terms: List[str] | None = None,
    umap_embeddings: np.ndarray | None = None,
    topics_mapping: Dict[Any, Any] | None = None,
) -> None:
    """
    Persist a topic-model and its companion artefacts.

    For `model_type="traditional"` expect:
        tm  : textacy.tm.TopicModel
        dtm : csr_matrix
        terms : list[str]

    For `model_type="embeddings"` expect:
        tm : BERTopic
        umap_embeddings : np.ndarray   (optional)
        topics_mapping  : {doc_id: topic_id} (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "traditional":
        if dtm is None or terms is None:
            raise ValueError("dtm and terms must be provided for traditional models")

        p = _paths_traditional_ml(out_dir, cfg)
        tm.save(str(p["model"]))
        sp.save_npz(p["dtm"], dtm)
        p["terms"].write_text(json.dumps(terms))

        print("Saved model ->", p["model"].relative_to(out_dir))
        print("Saved DTM ->", p["dtm"].relative_to(out_dir))
        print("Saved terms ->", p["terms"].relative_to(out_dir))

    elif model_type == "embeddings":
        p = _paths_embeddings_models(out_dir, cfg)
        tm.save(str(p["model_dir"]))  # BERTopic saves a folder

        if umap_embeddings is not None:
            np.savez_compressed(p["umap_emb"], emb=umap_embeddings)
            print("Saved UMAP embeddings ->", p["umap_emb"].relative_to(out_dir))

        if topics_mapping is not None:
            p["topics"].write_text(json.dumps(topics_mapping))
            print("Saved topics mapping ->", p["topics"].relative_to(out_dir))

        print("Saved BERTopic model ->", p["model_dir"].relative_to(out_dir))

    else:
        raise ValueError("model_type must be 'traditional' or 'embeddings'")


def load_topic_model(
    model_dir: Path,
    cfg: Dict[str, Any],
    *,
    model_type: str = "embeddings",
) -> Tuple[
    Union[TopicModel, BERTopic],
    sp.csr_matrix | np.ndarray | None,
    List[str] | Dict[Any, Any] | None,
]:
    """
    Reload the artefacts saved with `save_topic_model()`.

    Returns
    -------
    tm          : TopicModel | BERTopic
    second_obj  : csr_matrix (traditional) | ndarray (embeddings) | None
    third_obj   : list[str]  (traditional) | dict (embeddings)   | None
    """
    if model_type == "traditional":
        p = _paths_traditional_ml(model_dir, cfg)

        tm = TopicModel.load(str(p["model"]))
        dtm = sp.load_npz(p["dtm"])
        terms = json.loads(p["terms"].read_text())

        print("Loaded model <-", p["model"].relative_to(model_dir))
        print("Loaded DTM <-", p["dtm"].relative_to(model_dir))
        print("Loaded terms <-", p["terms"].relative_to(model_dir))
        return tm, dtm, terms

    elif model_type == "embeddings":
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

    else:
        raise ValueError("model_type must be 'traditional' or 'embeddings'")
