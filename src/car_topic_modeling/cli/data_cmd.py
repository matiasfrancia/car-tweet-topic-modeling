import typer
import pandas as pd
from pathlib import Path
from ..preprocessing.pipeline import PreprocessingPipeline

app = typer.Typer(help="Data-related commands")


@app.command("clean")
def clean_tweets(
    input_csv: Path,
    out: Path = typer.Option(
        None,
        help="Where to save cleaned text. Defaults to data/processed/<file>_clean.csv",
    ),
):
    """
    Clean tweets in INPUT_CSV, drop empty rows, save csv.
    """
    if out is None:
        out = Path("data/processed") / (input_csv.stem + "_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    typer.echo(f"Loaded {len(df)} rows from {input_csv}")
    pipe = PreprocessingPipeline()
    df["clean_text"] = df["tweet_text"].astype(str).apply(pipe.cleaner.clean)
    df = df[df["clean_text"].str.len() > 0]
    df.to_csv(out, index=False)
    typer.echo(f"Saved {len(df)} cleaned rows -> {out}")
