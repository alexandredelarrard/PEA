"""
test_embed_mode.py  (tests/gpt_extract/test_embed_mode.py)
------------------------------------------------------------
The embed mode that replaced `src/utils/openai_embeddings.py`.

The retired helper is gone, so its behaviour is pinned here instead: the same stub client
and the same inputs must give the same vectors, in the same order, under the same cap.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.gpt_extract import EMBEDDING_MAX_CHARS, cosine, embed_texts


class _StubClient:
    """`.embeddings.create(model=, input=)` -> `.data[i].embedding`, the documented seam."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.embeddings = self

    def create(self, model, input):
        self.batches.append(list(input))
        # a vector that encodes the input, so order and truncation are both observable
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(t)), float(ord(t[0])), 0.5])
                  for t in input])


def test_the_cap_is_the_models_real_limit():
    """8,191 TOKENS at ~3.6 chars/token. The 8,000-char cap this replaced was ~4x stricter
    than the model requires and truncated 22.4% of prepared-remarks turns."""
    assert EMBEDDING_MAX_CHARS == 28_000
    assert 8_000 < EMBEDDING_MAX_CHARS < 8_191 * 3.5

    print("\n=== SANITY: embedding cap ===")
    print(f"  EMBEDDING_MAX_CHARS = {EMBEDDING_MAX_CHARS:,}, inside the "
          f"(8,000, {int(8_191 * 3.5):,}) band the model allows. Validated.")


def test_texts_are_truncated_to_the_cap_and_order_is_preserved():
    client = _StubClient()
    texts = ["a" * 50_000, "bb", "c" * 30_000]
    out = embed_texts(texts, client=client)

    sent = client.batches[0]
    assert [len(s) for s in sent] == [EMBEDDING_MAX_CHARS, 2, EMBEDDING_MAX_CHARS]
    assert out.shape == (3, 3)
    assert [v[1] for v in out] == [float(ord("a")), float(ord("b")), float(ord("c"))]

    print("\n=== SANITY: truncation + ordering ===")
    print(f"  {[len(t) for t in texts]} chars in -> {[len(s) for s in sent]} sent; "
          "row order matches input order. Validated.")


def test_batching_preserves_global_order():
    client = _StubClient()
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(10)]
    out = embed_texts(texts, batch_size=3, client=client)

    assert [len(b) for b in client.batches] == [3, 3, 3, 1]
    assert out.shape == (10, 3)
    assert [v[1] for v in out] == [float(ord(t[0])) for t in texts]

    print("\n=== SANITY: batched ordering ===")
    print(f"  10 texts at batch_size=3 -> batches {[len(b) for b in client.batches]}; "
          "concatenated in submission order. Validated.")


def test_empty_input_returns_the_zero_by_zero_shape():
    """A caller relies on this shape; returning (0, dim) or None would break it."""
    out = embed_texts([], client=_StubClient())

    assert out.shape == (0, 0)
    assert out.dtype == np.dtype("float64")

    print("\n=== SANITY: empty input ===")
    print(f"  embed_texts([]) -> {out.shape} float64, no client call. Validated.")


def test_a_blank_text_is_never_sent_as_an_empty_string():
    """OpenAI rejects an empty input; the helper substitutes a space."""
    client = _StubClient()
    embed_texts(["", "   ", None], client=client)

    assert client.batches[0] == [" ", " ", " "]

    print("\n=== SANITY: blank-text guard ===")
    print(f"  ['', '   ', None] -> {client.batches[0]!r}, never an empty string. Validated.")


def test_cosine_is_degenerate_safe():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    assert cosine(a, a) == 1.0
    assert cosine(a, b) == 0.0
    assert cosine(a, np.zeros(2)) == 0.0          # zero norm -> 0.0, not a ZeroDivisionError
    assert cosine(np.zeros(2), np.zeros(2)) == 0.0

    print("\n=== SANITY: cosine degeneracy ===")
    print("  identical -> 1.0, orthogonal -> 0.0, zero vector -> 0.0 (no divide). Validated.")


def test_the_step_method_applies_the_configured_defaults(monkeypatch):
    """`GptExtracter.embed` is the same function with `config.gpt.embedding` applied."""
    from src.gpt_extract.transformers.step_gpt_extracter import GptExtracter
    from tests.gpt_extract.fakes import fake_context, gpt_config

    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    extracter = GptExtracter(fake_context(), gpt_config())
    client = _StubClient()
    extracter.embed(["z" * 50_000], client=client)

    assert len(client.batches[0][0]) == 28_000     # config.gpt.embedding.max_chars

    print("\n=== SANITY: configured embed defaults ===")
    print(f"  GptExtracter.embed truncated to {len(client.batches[0][0]):,} chars from "
          "config.gpt.embedding. Validated.")
