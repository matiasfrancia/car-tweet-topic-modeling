from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class Paradigm(Enum):
    TRADITIONAL = auto()
    EMBEDDINGS = auto()


@dataclass(frozen=True)
class DataPaths:
    raw: Path
    cleaned: Path
    labelled: Path
    figures: Path
    models: Path
