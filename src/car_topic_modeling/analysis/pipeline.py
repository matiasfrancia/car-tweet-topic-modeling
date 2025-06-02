from pathlib import Path
from typing import Dict

from .topic_modeling.modeler import TopicModeler

from .semantic_based import SemanticExtractor
from .token_based import TokenExtractor
from ..config.settings import get_settings


settings = get_settings()


class AnalysisPipeline:
    def __init__(
        self,
        company: str,
    ):
        self.company = company

        self.dataset_path = (
            Path(settings.processed_dir) / (company) / (settings.clean_tweets_file)
        )
        self.token_based_labelled_path = (
            Path(settings.labelled_dir)
            / "token_based"
            / (company)
            / (settings.labelled_file)
        )
        self.semantic_based_labelled_path = (
            Path(settings.labelled_dir)
            / "semantic_based"
            / (company)
            / (settings.labelled_file)
        )
        self.tm_clustering_labelled_path = (
            Path(settings.labelled_dir)
            / "topic_modeler"
            / (company)
            / (settings.labelled_file)
        )
        self.tm_model_dir = Path(settings.topic_models_dir) / (company)
        self.wordcloud_path = (
            Path(settings.figures_dir) / (self.company) / (settings.wordcloud_file)
        )
        self.cluster_report_path = (
            Path(settings.figures_dir) / (self.company) / (settings.cluster_report_file)
        )
        self.ngrams_path = (
            Path(settings.ngrams_dir) / (company) / (settings.ngrams_file)
        )

        self.token_extractor = TokenExtractor(
            dataset_path=self.dataset_path,
            labelled_path=self.token_based_labelled_path,
            wordcloud_path=self.wordcloud_path,
            ngrams_path=self.ngrams_path,
        )

        self.semantic_extractor = SemanticExtractor(
            dataset_path=self.token_based_labelled_path,
            labelled_path=self.semantic_based_labelled_path,
        )

    def extract_token_based_intents(self):
        self.token_extractor.extract_n_grams()
        self.token_extractor.generate_word_cloud(
            max_words=200,
            background="white",
            width=800,
            height=400,
        )

    def assign_token_based_intents(self, intent_mapping: Dict[str, str]):
        self.token_extractor.assign_ngram_based_intents(intent_mapping=intent_mapping)

    def extract_semantic_based_intents(self, reassign_tb_intents: bool):
        self.semantic_extractor.cluster(reassign_intent=reassign_tb_intents)

    def execute_grid_search_for_topic_modeler(
        self,
        paradigm: str,
        min_k: int,
        max_k: int,
        step: int,
    ):
        """
        Reads the intermediate labelled dataset given by the semantic based
        extraction and returns the best model found, using grid-search and
        the Silhouette Score metric as a reference.

        The grid-search optimizes using a list of intents' number and model.
        """
        topic_modeler = TopicModeler(
            # docs_path=self.semantic_based_labelled_path,
            docs_path=self.dataset_path,
            output_model_dir=self.tm_model_dir,
            report_path=self.cluster_report_path,
            paradigm=paradigm,
        )

        topic_modeler.grid_search(k_list=list(range(min_k, max_k, step)))

    def visualize_2d_embeddings(
        self,
        n_neighbors: int | None,
        n_components: int | None,
        min_dist: float | None,
        metric: str | None,
        sentence_model: str | None,
    ) -> None:
        topic_modeler = TopicModeler(
            docs_path=self.semantic_based_labelled_path,
        )
        topic_modeler.visualize_2d_embedding(
            n_neighbors, n_components, min_dist, metric, sentence_model
        )

    def visualize_topic_modeler_results(
        self,
        paradigm: str,
        *,
        # traditional/embeddings
        model_name: str | None,
        n_topics: int | None,
    ):
        """
        Method to visualize the results and topics given by the models
        """
        topic_modeler = TopicModeler(
            docs_path=self.semantic_based_labelled_path,
            # docs_path=self.dataset_path,
            output_model_dir=self.tm_model_dir,
            report_path=self.cluster_report_path,
            model_name=model_name,
            n_topics=n_topics,
            paradigm=paradigm,
        )
        topic_modeler.visualize_model_results()
