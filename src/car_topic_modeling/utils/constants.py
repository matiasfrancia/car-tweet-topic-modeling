from enum import Enum


class CleanType(str, Enum):
    """
    Cleaning regimes for the data.
    """

    SOFT = "soft"  # keep most tokens, transformer-friendly
    AGGRESSIVE = "aggressive"  # heavy normalisation for LDA/NMF/LSA/BoW
