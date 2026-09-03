"""
embeddings.py  (src/gpt_extract/utils/embeddings.py)
----------------------------------------------------
Embedding is the second mode of the same client factory: batch inputs, truncate over-long
texts, preserve order.

The cap is measured, not chosen. `text-embedding-3-small` accepts 8,191 TOKENS and English
prose runs ~3.6 chars/token, so 28,000 chars is the model's real limit. The 2026-07 audit
measured what a stricter cap costs: 22,730 of 101,373 prepared-remarks turns (22.4%) and
1,411 Q&A turns are longer than 8,000 chars, the longest 74,550 -- an 8,000-char cap was
comparing only each turn's opening fragment.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

#: The OpenAI model limit this module enforces. It lives with the client that enforces it
#: rather than in `constants.py`, because it is a property of the model, not of a pipeline.
EMBEDDING_MAX_CHARS = 28_000
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 128


def openai_api_key() -> str | None:
    """`OPEN_AI_API_KEY` (the .env spelling) or `OPENAI_API_KEY`.

    Returns None rather than raising: a missing key is a caller's skip condition -- the
    earnings-call pass logs a warning and returns what it has.
    """
    return os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")


def embed_texts(
    texts: Sequence[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_chars: int = EMBEDDING_MAX_CHARS,
    client: Any | None = None,
) -> np.ndarray:
    """Embed `texts` -> `(n, dim)` float64, order preserved.

    `client` is the test seam: any object with `.embeddings.create(model=, input=)`
    returning `.data[i].embedding`. It is the only way to exercise this without spending
    money, so it is part of the contract rather than an implementation detail.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float64")
    if client is None:
        from openai import OpenAI                      # lazy: no import cost if stubbed
        client = OpenAI(api_key=openai_api_key())
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = [((t or " ").strip() or " ")[:max_chars] for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend(d.embedding for d in resp.data)    # resp.data preserves input order
    return np.asarray(vecs, dtype="float64")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (0.0 if either is degenerate)."""
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
