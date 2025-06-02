from pathlib import Path
from typing import Any, Dict, List, Tuple
from bertopic import BERTopic
from .visualization import visualize_UMAP_2d_embeddings
from car_topic_modeling.utils.io import read_csv_in_chunks
import numpy as np
import scipy.sparse as sp
from textacy.tm import TopicModel
import webbrowser

from .grid_search import (
    grid_search_topic_modeling_embedding_models,
    grid_search_topic_modeling_traditional_ML,
)
from .io import load_topic_model, save_topic_model


class TopicModeler:
    """ """

    def __init__(
        self,
        docs_path: Path,
        output_model_dir: Path | None = None,
        report_path: Path | None = None,
        paradigm: str | None = None,  # "embeddings" | "traditional"
        model_name: str | None = None,
        n_topics: int | None = None,
    ):
        self.paradigm = paradigm
        self.output_model_dir = output_model_dir
        self.report_path = report_path

        # runtime attributes
        self.score: float | None = None
        self.model: BERTopic | TopicModel | None = None
        self.model_cfg: Dict[str, Any] | None = None
        self.aux_1: sp.csr_matrix | np.ndarray | None = None
        self.aux_2: List[str] | Dict[int, int] | None = None

        # preload model with its corresponding data
        if model_name and n_topics:
            cfg = {"model": model_name, "n_topics": n_topics}
            self.model_cfg = cfg
            self.model, self.aux_1, self.aux_2 = load_topic_model(
                output_model_dir,
                cfg,
                model_type="embeddings" if paradigm == "embeddings" else "traditional",
            )

            print(self.model, self.aux_1.shape)

        self.docs, self.raw_text = self._load_docs(docs_path)

    def _load_docs(self, csv_path: Path) -> Tuple[List[str], List[str]]:
        """
        Read tweets in chunks; return (cleaned_texts, raw_texts).
        Only rows whose `intent` column is empty are used.
        """
        # TODO: implement the actual iterable logic by adding the chunk of tweets
        # to the clustering model
        raw_texts, cleaned = [], []
        for rows in read_csv_in_chunks(
            csv_path, ["tweet_text", "tweet_clean_text", "intent"]
        ):
            for raw, clean, intent in rows:
                if intent != "" and intent != "nan":
                    continue
                raw_texts.append(raw)
                cleaned.append(clean)
        return cleaned, raw_texts

    def grid_search(
        self,
        *,
        # traditional:
        k_list: List[int] | None = None,
        trad_models: tuple[str, ...] = ("lsa", "lda", "nmf"),
        # embeddings:
        emb_param_grid: List[Dict[str, Any]] | None = None,
    ) -> None:
        """
        Train several configs, keep the best by Silhouette Score, then save it.

        Parameters (depend on the paradigm):
        ------
        traditional : supply `k_list` + optional `trad_models`
        embeddings  : optionally supply `emb_param_grid`
        """
        if self.paradigm == "traditional":
            if k_list is None:
                raise ValueError("k_list must be provided for traditional search")

            (self.score, self.model, self.model_cfg, dtm, terms) = (
                grid_search_topic_modeling_traditional_ML(
                    self.docs, k_list, trad_models
                )
            )

            save_topic_model(
                self.model,
                self.output_model_dir,
                self.model_cfg,
                model_type=self.paradigm,
                dtm=dtm,
                terms=terms,
            )
            self.aux_1, self.aux_2 = dtm, terms

        elif self.paradigm == "embeddings":
            if emb_param_grid is None:
                emb_param_grid = [
                    {
                        # UMAP
                        "n_neighbors": 15,
                        "n_components": 20,
                        "min_dist": 0.1,
                        # HDBSCAN
                        "min_cluster_size": 30,
                        "min_samples": 1,
                        # BERTopic
                        "model": "BERTopic",
                        "n_topics": None,  # to be filled when the model's created
                    }
                ]

            (self.score, self.model, self.model_cfg, umap_emb, topics_map) = (
                grid_search_topic_modeling_embedding_models(self.docs, emb_param_grid)
            )

            save_topic_model(
                self.model,
                self.output_model_dir,
                self.model_cfg,
                model_type=self.paradigm,
                umap_embeddings=umap_emb,
                topics_mapping=topics_map,
            )
            self.aux_1, self.aux_2 = umap_emb, topics_map

        else:
            raise ValueError("paradigm must be 'traditional' or 'embeddings'")

        print(f"\nSaved best {self.paradigm} model (silhouette = {self.score:.3f})")

        self.visualize_model_results()

    def visualize_2d_embedding(
        self,
        n_neighbors: int | None,
        n_components: int | None,
        min_dist: float | None,
        metric: str | None,
        sentence_model: str | None,
    ) -> None:
        """
        Generates an embedding reduction of 2 dimensions to have an insight
        about the clustering of the data. It also prints a plot in screen.

        As it uses UMAP and SentenceTransformer, the following arguments are required:

        Args:
        n_neighbors: int
        n_components: int
        min_dist: float
        metric: str
        sentence_model: str
        """
        cfg: Dict[str, Any] = {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "metric": metric,
            "sentence_model": sentence_model,
        }
        visualize_UMAP_2d_embeddings(self.docs, cfg)

    def visualize_model_results(self) -> None:
        """
        Visualizes the topic modeler results, depending on the type of model.
        If it is:
            - A "traditional" model: displays the documents per topic.
            - An "embedding" model: displays different plots to show the clustering.
        """
        if self.paradigm == "traditional":
            if not isinstance(self.model, TopicModel):
                raise RuntimeError(
                    "Current model is not a TopicModel, even though "
                    "traditional ML visualization was called"
                )

            top_topic_docs = self.model.top_topic_docs(
                self.model.get_doc_topic_matrix(self.aux_1)
            )

            for i, topic_docs in enumerate(top_topic_docs):
                print(f"Topic {i}:")
                for j, topic_doc in enumerate(topic_docs[1]):
                    print(f"Doc {j}: {self.raw_text[topic_doc]}")

        elif self.paradigm == "embeddings":
            if not isinstance(self.model, BERTopic):
                raise RuntimeError(
                    "Current model is not a BERTopic model, even though "
                    "embeddings visualization was called"
                )

            self.model.visualize_barchart(top_n_topics=15).show()
            self.model.visualize_topics().show(open=True)
            datamap = self.model.visualize_document_datamap(
                self.docs,
                embeddings=self.aux_1,
                interactive=True,
            )

            datamap_path: Path = self.report_path.parent / "documents_datamap.html"
            datamap.save(datamap_path)
            webbrowser.open_new_tab(datamap_path.resolve().as_uri())
