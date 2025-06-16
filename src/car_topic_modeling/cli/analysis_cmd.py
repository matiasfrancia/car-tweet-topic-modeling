from io import StringIO
import json
from pathlib import Path
from typing import Dict
import typer
from ..analysis.pipeline import AnalysisPipeline
import logging


logging.basicConfig(level=logging.INFO)
app = typer.Typer(help="Analysis commands")


@app.command("extract-token-intents")
def extract_token_intents(
    company: str,
):
    """
    Extract intents from the tweets using a token-based approach.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.extract_token_intents()


@app.command("assign-token-intents")
def assign_token_intents(
    company: str,
    intent_mapping_path: Path = typer.Option(
        None,
        help="Where the mapping of intents extracted by the ngram analysis are. "
        "Must be a json file containing a dict",
    ),
):
    """
    Assign intents to the tweets based on the token-based approach.
    """
    if not intent_mapping_path:
        raise ValueError("Didn't get any intent mapping filepath")

    analysis_pipeline = AnalysisPipeline(company)
    intent_mapping_file: StringIO = open(intent_mapping_path)
    intent_mapping: Dict[str, str] = json.load(intent_mapping_file)
    analysis_pipeline.assign_token_intents(intent_mapping)


@app.command("extract-semantic-intents")
def extract_semantic_intents(
    company: str,
    reassign_tb_intents: bool,  # whether to re-assign an intent already assigned by tb
):
    """
    Extract intents from the tweets using a semantic-based approach.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.extract_semantic_intents(reassign_tb_intents)


@app.command("grid-search-topic-modeler")
def grid_search_topic_modeler(company: str):
    """
    Cluster the remaining intents based on the topic modeler approach.
    It generates different BERTopic models with the configurations passed
    as parameters. It then saves them and generate plots to evaluate the
    clustering result.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.search_topic_modeler()


@app.command("visualize-docs-embedding-space")
def visualize_docs_embedding_space(
    company: str,
    n_neighbors: int = typer.Option(
        None, "--n-neighbors", "-n", help="Number of neighbors parameter of UMAP"
    ),
    n_components: int = typer.Option(
        None,
        "--n-components",
        "-c",
        help="Number of components to which the embeddings will be reduced by UMAP",
    ),
    min_dist: float = typer.Option(
        None, "--min-dist", "-d", help="Minimum distance parameter of UMAP"
    ),
    metric: str = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metric parameter of UMA, it can be 'cosine', 'euclidean', etc.",
    ),
    sentence_model: str = typer.Option(
        None,
        "--sentence-model",
        "-s",
        help="Model to use for the embedding generation, before reducing it with UMAP",
    ),
):
    """
    Generates a plot with the embeddings of the documents (tweets)
    by reducing their dimensions to 2D.

    Args:
    sentence_model: str
        The model to use to generate the first embeddings,
        before reducing them with UMAP.
    n_neighbors: int
        UMAP's 'n_neighbors' parameter
    n_components: int
        UMAP's 'n_components' parameter
    min_dist: float
        UMAP's 'min_dist' parameter
    metric: str
        UMAP's 'metric' parameter
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.visualize_2d_embeddings(
        n_neighbors, n_components, min_dist, metric, sentence_model
    )


@app.command("test-best-topic-modeler")
def test_best_topic_modeler(
    company: str,
):
    """
    Makes tests to detect the intents recognized by the topic modeler model.
    There should be a folder with all the data of the best topic modeler found
    in the same folder in which the search endpoint generates it.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.visualize_topic_modeler_results(
        # TODO: see what should be passed here to load a model's results
    )
