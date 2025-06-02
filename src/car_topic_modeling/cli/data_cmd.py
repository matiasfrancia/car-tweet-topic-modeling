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
    clean_mode: CleanType = typer.Option(
        CleanType.AGGRESSIVE,
        help="Whether to make an 'aggressive' or 'soft' cleaning of the text. "
        "'aggressive' should be used for traditional Topic Modeling "
        "techniques (e.g. LDA), while 'soft' cleaning for embedding models.",
    ),
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
    pipe = PreprocessingPipeline(company_name, clean_mode)

    # remove rows that don't have information
    df = df[df["tweet_text"].notna()]
    df["tweet_clean_text"] = df.apply(
        lambda row: pipe.preprocess(row["tweet_text"], row["lang"], row["user_name"]),
        axis=1,
    )
    df = df[df["tweet_clean_text"].str.len() > 0]
    df["intent"] = "nan"
    df.to_csv(out, index=False)

    typer.echo(f"Saved {len(df)} cleaned rows -> {out}")
