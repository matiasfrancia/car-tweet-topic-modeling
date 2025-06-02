from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------- General ---------- #
    package_name: str = "car_topic_modeling"

    # ----------- Data ---------- #
    raw_dir: str = "data/raw"
    processed_dir: str = "src/" + package_name + "/data/processed"
    figures_dir: str = "src/" + package_name + "/data/figures"
    ngrams_dir: str = "src/" + package_name + "/data/ngram"
    labelled_dir: str = "src/" + package_name + "/data/labelled"
    topic_models_dir: str = "src/" + package_name + "/data/topic_models"
    clean_tweets_file: str = "original_tweets_clean.csv"
    labelled_file: str = "original_tweets_labelled.csv"
    ngrams_file: str = "clean_tweets_ngrams.csv"
    wordcloud_file: str = "wordcloud.png"
    cluster_report_file: str = "topic_report.png"

    # ---------- Preprocessing ---------- #
    n_gram: int = 4
    spacy_model: str = "en_core_web_sm"

    # ---------- Intent Extaction ---------- #
    n_topics: int = 20
    n_gram_shingles: int = 3
    jaccard_threshold: float = 0.00
    ro_threshold: float = 0.50
    num_perm: int = 128
    min_cluster_size: int = 20

    # ----------- API ---------- #
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------- Model ---------- #
    model_dir: str = "models"
    embedder_name: str = "intfloat/e5-base-v2"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
