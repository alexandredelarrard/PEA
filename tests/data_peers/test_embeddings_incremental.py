"""
Incremental OpenAI embeddings (src/data_peers/utils/embeddings.py).

`ticker_embeddings` is the single "done" gate: a ticker already in that table must
NOT hit Yahoo (description) or OpenAI (embedding) again — only missing tickers are
processed, while the full universe matrix (cached + new) is still returned for the
similarity computation. OpenAI is stubbed so nothing touches the network.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from conftest import FakeStore     # the ONE shared store double
from src.data_peers.utils import embeddings as emb

_OAI_CALLS: list[list[str]] = []            # captures each embeddings.create(input=...)


class _FakeOpenAI:
    """Stub OpenAI client: records the inputs it was asked to embed."""
    def __init__(self, api_key=None):
        self.embeddings = self

    def create(self, model, input):
        _OAI_CALLS.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4])
                                     for _ in input])


def _FakeStore(embedded: dict[str, list[float]] | None = None):
    """A `ticker_embeddings` cache on the shared double. NOT `sqlite_store`: `embedding` is a
    vector column and SQLite's driver cannot bind a Python list."""
    if not embedded:
        return FakeStore()
    return FakeStore({"ticker_embeddings": pd.DataFrame(
        {"ticker": list(embedded), "embedding": [list(v) for v in embedded.values()]})})


def test_load_embedded_tickers_is_the_done_set():
    store = _FakeStore({"AAA": [1, 0, 0, 0], "BBB": [0, 1, 0, 0]})
    assert emb.load_embedded_tickers(store) == {"AAA", "BBB"}
    assert emb.load_embedded_tickers(_FakeStore()) == set()        # empty table -> no one done
    assert emb.load_embedded_tickers(None) == set()
    print("\n=== SANITY CHECK: ticker_embeddings 'done' gate ===")
    print("  load_embedded_tickers -> {AAA,BBB}; empty table / no store -> set(). Validated.")


def test_only_missing_tickers_are_embedded(monkeypatch):
    _OAI_CALLS.clear()
    monkeypatch.setattr(emb, "_api_key", lambda: "test-key")
    monkeypatch.setattr(emb, "OpenAI", _FakeOpenAI)

    store = _FakeStore({"AAA": [1, 0, 0, 0], "BBB": [0, 1, 0, 0]})   # already embedded
    universe = ["AAA", "BBB", "CCC", "DDD"]

    # the step's gate: only tickers absent from ticker_embeddings are to-do
    done = emb.load_embedded_tickers(store)
    todo = [t for t in universe if t not in done]
    assert todo == ["CCC", "DDD"]

    # feed descriptions for the to-do only; universe returns everyone's vector
    out = emb.get_openai_embeddings({"CCC": "chip maker", "DDD": "retail bank"},
                                    store=store, universe=universe)

    # OpenAI called ONCE, ONLY for the two missing tickers (AAA/BBB never re-embedded)
    assert _OAI_CALLS == [["chip maker", "retail bank"]]
    # full universe matrix returned (cached + new)
    assert set(out.index) == {"AAA", "BBB", "CCC", "DDD"} and out.shape[1] == 4
    # only the NEW vectors persisted back
    saved = store.saved_frames("ticker_embeddings")
    assert len(saved) == 1 and set(saved[0]["ticker"]) == {"CCC", "DDD"}
    print("\n=== SANITY CHECK: incremental embedding (only missing tickers) ===")
    print(f"  done={sorted(done)}, todo={todo}; OpenAI called once for {_OAI_CALLS[0]} "
          f"(cached AAA/BBB NOT re-embedded); full 4-ticker matrix returned; only CCC/DDD saved. Validated.")


def test_all_cached_makes_zero_openai_calls(monkeypatch):
    _OAI_CALLS.clear()

    def _boom(*a, **k):
        raise AssertionError("OpenAI must not be constructed when all tickers are cached")

    monkeypatch.setattr(emb, "OpenAI", _boom)
    store = _FakeStore({"AAA": [1, 0, 0, 0], "BBB": [0, 1, 0, 0]})

    # every universe ticker already done -> no descriptions needed, no OpenAI
    out = emb.get_openai_embeddings({}, store=store, universe=["AAA", "BBB"])
    assert _OAI_CALLS == []
    assert set(out.index) == {"AAA", "BBB"}
    assert not store.saved_frames("ticker_embeddings")          # nothing new saved
    print("\n=== SANITY CHECK: fully-cached universe = zero API calls ===")
    print("  all tickers in ticker_embeddings -> OpenAI never constructed, matrix served "
          "from cache, nothing re-saved. Validated.")


if __name__ == "__main__":
    test_load_embedded_tickers_is_the_done_set()
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
