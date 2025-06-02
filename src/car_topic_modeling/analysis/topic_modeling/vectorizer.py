# Embeds the doc to a vector in the embedding using the DTM
from typing import Iterable, List, Tuple
from textacy.representations.vectorizers import Vectorizer
from scipy.sparse import csr_matrix
from .types import TfType, IdfType, DlType, NormStr


def build_document_term_matrix(
    docs: Iterable[List[str]],
    tf_type: TfType = "sqrt",
    idf_type: IdfType = "standard",
    dl_type: DlType = "linear",
    norm: NormStr = "l1",
    min_df: int = 0.0001,
    max_df: float = 0.9,
) -> Tuple[csr_matrix, List[str]]:
    """
    Build a document-term matrix to convert the documennts
    to vectors. For default it uses the paper's configuration.

    Returns
    -------
    dtm : scipy.sparse.csr_matrix
        Row = doc, col = term, value = weighted frequency.
    terms : list[str]
        Vocabulary terms in column order of `dtm`.
    """
    vectorizer = Vectorizer(
        tf_type=tf_type,
        idf_type=idf_type,
        dl_type=dl_type,
        norm=norm,
        min_df=min_df,
        max_df=max_df,
    )

    dtm: csr_matrix = vectorizer.fit_transform(docs)
    return dtm, vectorizer.terms_list
