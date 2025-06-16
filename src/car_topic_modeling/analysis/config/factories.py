from dataclasses import asdict
from pathlib import Path
from typing import Optional

from car_topic_modeling.utils.hash import build_hash_with_params
from .types import (
    BERTopicConfig,
    CompanyPaths,
    SemanticPaths,
    StagePaths,
    BERTopicPaths,
    TokenPaths,
    TopicModelingPaths,
)
from ...config.settings import get_settings

_s = get_settings()

# the following code makes the paths do not rely on where is
# the factory called, instead it converts every path to its
# canonycal form
PROCESSED_ROOT = Path(_s.processed_dir).resolve()
LABELLED_ROOT = Path(_s.labelled_dir).resolve()
FIGURES_ROOT = Path(_s.figures_dir).resolve()
NGRAMS_ROOT = Path(_s.ngrams_dir).resolve()
TOPIC_MODELS_ROOT = Path(_s.topic_models_dir).resolve()


def build_company_paths(company: str) -> CompanyPaths:
    return CompanyPaths(
        token=build_token_paths(company),
        semantic=build_semantic_paths(company),
        topic=build_topic_paths(company),
    )


def build_token_paths(company: str) -> TokenPaths:
    return TokenPaths(
        dataset=PROCESSED_ROOT / company / _s.clean_tweets_file,
        labelled=LABELLED_ROOT / "token_based" / company / _s.labelled_file,
        ngrams=NGRAMS_ROOT / company / _s.ngrams_file,
        wordcloud=FIGURES_ROOT / company / _s.wordcloud_file,
    )


def build_semantic_paths(company: str) -> SemanticPaths:
    return SemanticPaths(
        dataset=LABELLED_ROOT / "token_based" / company / _s.labelled_file,
        labelled=LABELLED_ROOT / "semantic_based" / company / _s.labelled_file,
    )


def build_topic_paths(company: str) -> TopicModelingPaths:
    return TopicModelingPaths(
        dataset=PROCESSED_ROOT
        / company
        / _s.clean_tweets_file,  # execute the full intent extraction pipeline
        # dataset   = LABELLED_ROOT / "semantic_based" / company / _s.labelled_file,
        models_dir=TOPIC_MODELS_ROOT / company,
        labelled=LABELLED_ROOT / "topic_modeler" / company / _s.labelled_file,
        report=FIGURES_ROOT / company / _s.cluster_report_file,
        datamap=FIGURES_ROOT / company / _s.datamap_file,
    )


def build_bertopic_stage_paths(
    root: Path, folder: str, hash: str, suffix: str
) -> StagePaths:
    f = root / folder / f"{hash}{suffix}"
    return StagePaths(file=f, meta=f.with_suffix(".json"))


def build_bertopic_pipeline_paths(
    root: Path,
    cfg: BERTopicConfig,
) -> BERTopicPaths:
    embed_hash = build_hash_with_params(cfg.embeddings)

    if cfg.pca and cfg.pca.active:
        pca_hash: Optional[str] = build_hash_with_params(
            {"embed": embed_hash, **asdict(cfg.pca)}
        )
    else:
        pca_hash = None

    umap_hash = build_hash_with_params(
        {
            "embed": embed_hash,  # we consider it here too because pca_hash may be None
            "pca": pca_hash or "",
            **asdict(cfg.umap),
        }
    )

    bertopic_hash = build_hash_with_params(
        {"umap": umap_hash, **asdict(cfg.hdbscan), **asdict(cfg.bertopic)}
    )

    return BERTopicPaths(
        embeddings=build_bertopic_stage_paths(root, "embeddings", embed_hash, ".npy"),
        pca=build_bertopic_stage_paths(root, "pca", pca_hash, ".npy")
        if pca_hash
        else None,
        umap=build_bertopic_stage_paths(root, "umap", umap_hash, ".npy"),
        bertopic=build_bertopic_stage_paths(root, "bertopic", bertopic_hash, ".bin"),
    )
