"""Wrapper around the sentence-transformers embedding model.

Kept in its own module so the indexer, the retriever, and the eval harness all
load the model the same way, and so the heavy import (torch) happens only when
something actually needs to embed text.

Default model is BAAI/bge-m3: multilingual, 1024-dimensional, and better on
German compound nouns than the e5 family in the cases this project cares about.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"


def pick_device() -> str:
    """MPS on Apple Silicon, else CUDA, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized embeddings, so a dot product is cosine similarity."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
