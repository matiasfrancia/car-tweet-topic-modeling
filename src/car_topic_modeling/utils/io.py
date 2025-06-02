from pathlib import Path
from typing import Iterable, List, Optional, Union
import pandas as pd


def read_csv_in_chunks(
    dataset_path: Path | str,
    text_col: Optional[Union[str, List[str]]] = None,
    chunk_size: int = 1_000,
) -> Iterable[list]:
    """
    Stream a large CSV in memory-friendly chunks.

    Parameters
    ----------
    dataset_path : Path | str
        File to read.
    text_col : str | list[str] | None
        • str         → yield List[str] (single column)\
        • list[str]   → yield List[tuple] (row-wise tuples of those columns)\
        • None        → yield List[tuple] of **all** columns in the file
    chunk_size : int
        Number of rows per chunk passed to pandas.

    Yields
    ------
    list
        See `text_col` rules above.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"dataset file {dataset_path!s} does not exist")

    if text_col is None:  # return all the columns
        use_cols = None
    elif isinstance(text_col, str):
        use_cols = [text_col]
    else:
        use_cols = text_col

    for chunk in pd.read_csv(dataset_path, usecols=use_cols, chunksize=chunk_size):
        chunk = chunk.astype(str)

        if text_col is None:
            yield list(map(tuple, chunk.itertuples(index=False, name=None)))

        elif isinstance(text_col, str):
            yield chunk[text_col].to_list()

        else:
            yield list(map(tuple, chunk.itertuples(index=False, name=None)))
