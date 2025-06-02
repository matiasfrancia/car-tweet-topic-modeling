from typing import Any, Dict, List
from matplotlib import pyplot as plt
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from umap import UMAP


def visualize_UMAP_2d_embeddings(docs: List[str], cfg: Dict[str, Any]) -> None:
    """
    Function to calculate and visualize the embeddings reduced to 2D by UMAP.
    The default number of components is 2, more dimensions can be passed,
    but only the first two will be displayed.
    """
    embedder = SentenceTransformer(cfg.get("sentence_model", "all-mpnet-base-v2"))

    umap_model = UMAP(
        n_neighbors=cfg.get("n_neighbors", 30),
        n_components=cfg.get("n_components", 2),
        min_dist=cfg.get("min_dist", 0.0),
        metric=cfg.get("metric", "cosine"),
        random_state=42,
    )

    embeds: ndarray = embedder.encode(docs)
    reduced_embeds = umap_model.fit_transform(embeds)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        reduced_embeds[:, 0], reduced_embeds[:, 1], s=12, alpha=0.7, label="Data points"
    )
    ax.legend(frameon=False, loc="best")
    ax.set_title("UMAP projection of document embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("equal")  # keeps aspect ratio square
    ax.grid(True, linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    plt.show()
