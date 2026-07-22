"""
openai_embeddings.py  (src/utils/openai_embeddings.py)
------------------------------------------------------
Thin, shared OpenAI-embedding helper so any folder can embed text without cross-importing
data_peers. Batches inputs, truncates over-long texts, preserves order. Reads the API key from
OPEN_AI_API_KEY (the .env spelling) or OPENAI_API_KEY.
"""
from __future__ import annotations

import os

import numpy as np


def openai_api_key() -> str | None:
    return os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")


def embed_texts(texts: list[str], model: str = "text-embedding-3-small", batch_size: int = 128,
                max_chars: int = 8000, client=None) -> np.ndarray:
    """Embed `texts` -> (n, dim) float64 array, order preserved. `client` lets tests inject a stub
    (any object with `.embeddings.create(model=, input=)` returning `.data[i].embedding`)."""
    if not texts:
        return np.zeros((0, 0), dtype="float64")
    if client is None:
        from openai import OpenAI                                  # lazy: no import cost if stubbed
        client = OpenAI(api_key=openai_api_key())
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = [((t or " ").strip() or " ")[:max_chars] for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend(d.embedding for d in resp.data)                # resp.data preserves input order
    return np.asarray(vecs, dtype="float64")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (0.0 if either is degenerate)."""
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
