import logging
from typing import Sequence

import numpy as np
import openai
from bertopic.backend import BaseEmbedder

log = logging.getLogger(__name__)


class OpenAIBackendAdapter(BaseEmbedder):
    """
    Minimal embedder compatible with OpenAI Python SDK >= 1.0.

    Usage
    -----
    embedder = OpenAIv1Backend(
        model="text-embedding-3-small",
        api_key="sk-...",
        batch_size=1000,          # tune for cost / latency
    )
    """

    def __init__(self, model: str, api_key: str, *, batch_size: int = 2048):
        super().__init__()
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._bs = batch_size

    def embed(self, documents: Sequence[str], verbose: bool = False) -> np.ndarray:
        """
        Returns a 2-D np.ndarray (n_docs × dim) as required by BERTopic.
        """
        out: list[np.ndarray] = []

        for i in range(0, len(documents), self._bs):
            batch = documents[i : i + self._bs]
            resp = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            out.extend(np.asarray(e.embedding, dtype="float32") for e in resp.data)

        return np.vstack(out)

    def encode(self, documents: Sequence[str], verbose=False) -> np.ndarray:
        return self.embed(documents, verbose)
