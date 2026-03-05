"""
embedder.py — Local embeddings using BAAI/bge-large-en-v1.5.

1024-dimensional vectors, no API key, runs entirely locally.
Model is lazy-loaded on first use and cached in process memory.
~1.3 GB download on first run, then cached by sentence-transformers.

BGE models require a query instruction prefix for queries only.
Documents (chunks) are embedded without any prefix.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embedder_instance = None


class Embedder:
    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info("[EMBED] Loading %s (first use — may take ~30s)…", _MODEL_NAME)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME)
            logger.info("[EMBED] Model loaded.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks. No prefix needed."""
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with BGE instruction prefix."""
        model = self._load()
        vec = model.encode(
            _QUERY_INSTRUCTION + query.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
