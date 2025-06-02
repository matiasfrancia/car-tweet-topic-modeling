from io import StringIO
import json
from pathlib import Path
from typing import Dict
import typer
from ..analysis.pipeline import AnalysisPipeline

app = typer.Typer(help="Analysis commands")


@app.command("extract-tb-intents")
def extract_token_based_intents(
    company: str,
):
    """
    Extract intents from the tweets using a token-based approach.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.extract_token_based_intents()


@app.command("assign-tb-intents")
def assign_token_based_intents(
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
    analysis_pipeline.assign_token_based_intents(intent_mapping)


@app.command("extract-sb-intents")
def extract_semantic_based_intents(
    company: str,
    reassign_tb_intents: bool,  # whether to re-assign an intent already assigned by tb
):
    """
    Extract intents from the tweets using a semantic-based approach.
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.extract_semantic_based_intents(reassign_tb_intents)


@app.command("grid-search-topic-modeler")
def get_best_topic_modeler_grid_search(company: str, paradigm: str):
    """
    Cluster the remaining intents based on the topic modeler approach.
    It finds and saves the best model based on the silhouette score by
    using grid-search.

    The paradigm must be "embeddings" or "traditional" depending on the
    type of topic modeler we want (BERTopic or LDA, LSA, NMF).
    """
    analysis_pipeline = AnalysisPipeline(company)
    analysis_pipeline.execute_grid_search_for_topic_modeler(
        paradigm, min_k=4, max_k=25, step=1
    )


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
    by reducing its dimension to 2D, rather than having it with the
    'full' dimensions.

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
    paradigm: str,
    *,
    # traditional
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Name of the saved model"
    ),
    n_topics: int | None = typer.Option(
        None, "--k", help="Number of topics of the saved model"
    ),
):
    """
    Makes tests to detect the intents recognized by the topic modeler model.
    There should be a folder with all the data of the best topic modeler found
    in the same folder that grid-search endpoint generates it.

    Args (depending on the topic modeler paradigm):
        "traditional": model_name and n_topics of the model we want to test
        "embeddings": umap embeddings (optional), topic mappings (optional)
    """
    analysis_pipeline = AnalysisPipeline(company)
    print(f"Paradigm: {paradigm}, Model name: {model_name}, N-topics: {n_topics}")
    analysis_pipeline.visualize_topic_modeler_results(
        paradigm, model_name=model_name, n_topics=n_topics
    )
