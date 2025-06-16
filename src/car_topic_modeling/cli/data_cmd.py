from ..utils.constants import CleanType
import typer
import pandas as pd
from pathlib import Path
from ..preprocessing.pipeline import PreprocessingPipeline
from ..config.settings import get_settings

app = typer.Typer(help="Data-related commands")
settings = get_settings()


@app.command("clean")
def clean_tweets(
    input_csv: Path,
    out: Path = typer.Option(
        None,
        help="Where to save cleaned text. Defaults to "
        "data/processed/<company_name>/<settings.clean_tweets_file>.csv",
    ),
):
    """
    Clean tweets in INPUT_CSV, drop empty rows, save csv.
    """
    if not out:
        out = (
            Path(settings.processed_dir)
            / input_csv.parent.name
            / (settings.clean_tweets_file)
        )
        print(f"The output path is {out}.")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    typer.echo(f"Loaded {len(df)} rows from {input_csv}")
    company_name = input_csv.parent.name
    pipe = PreprocessingPipeline(company_name)

    # remove rows that don't have information
    df = df[df["tweet_text"].notna()]
    df["tweet_soft_clean_text"] = df.apply(
        lambda row: pipe.preprocess(
            row["tweet_text"], row["lang"], row["user_name"], CleanType.SOFT
        ),
        axis=1,
    )
    df["tweet_aggressive_clean_text"] = df.apply(
        lambda row: pipe.preprocess(
            row["tweet_text"], row["lang"], row["user_name"], CleanType.AGGRESSIVE
        ),
        axis=1,
    )
    df = df[df["tweet_aggressive_clean_text"].str.len() > 0]
    df["intent"] = "nan"
    df.to_csv(out, index=False)

    typer.echo(f"Saved {len(df)} cleaned rows -> {out}")
