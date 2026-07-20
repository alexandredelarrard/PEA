"""
embeddings.py  (src/data_aggregate/utils/embeddings.py)
-------------------------------------------------------
Fetch each ticker's business description (Yahoo) and embed it (OpenAI), with
ONCE-ONLY on-disk caching keyed by ticker:

  * fetch_business_descriptions -> Yahoo is called ONLY for tickers missing from
    the description cache; cached tickers are never re-fetched.
  * get_openai_embeddings       -> OpenAI is called ONLY for tickers missing from
    the embedding cache; cached tickers are never re-embedded.

So once all 500 tickers are cached, subsequent runs make ZERO API calls. Use
`force=True` on either to rebuild from scratch (e.g. after a universe change).

The peers step (StepDeducePeers) uses `load_embedded_tickers` as the single "done"
gate: tickers already in `ticker_embeddings` skip BOTH the Yahoo description and the
OpenAI embedding — only tickers missing from that table are (re)processed.

Env: reads OPEN_AI_API_KEY (your .env spelling) or OPENAI_API_KEY.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
import yfinance as yf
from tqdm import tqdm


def _api_key() -> str:
    key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPEN_AI_API_KEY not set in environment / .env")
    return key


# --------------------------------------------------------------------------- #
# Business descriptions (Yahoo) - cached once per ticker                       #
# --------------------------------------------------------------------------- #
def fetch_business_descriptions(
    tickers: list[str],
    store=None,
    pause: float = 0.2,
    force: bool = False,
) -> dict[str, str]:
    """
    Return {ticker: description}. Yahoo is queried ONLY for tickers not already
    in the DB cache (`ticker_descriptions`) unless force=True; new descriptions
    are merged back into the table.
    """
    cached: dict[str, str] = {}
    if store is not None:
        df = store.load("ticker_descriptions")
        if not df.empty:
            cached = dict(zip(df["ticker"], df["description"]))

    missing = [t for t in tickers if force or t not in cached]
    if missing:
        new: dict[str, str] = {}
        for t in tqdm(missing, desc=f"Fetching {len(missing)} descriptions (Yahoo)"):
            try:
                info = yf.Ticker(t).info
                text = info.get("longBusinessSummary")
                if text and isinstance(text, str) and len(text) > 40:
                    prefix = " ".join(str(info.get(k, "")) for k in ("sector", "industry"))
                    new[t] = (prefix + ". " + text).strip()
            except Exception as e:
                print(f"{t}: description fetch failed ({e})")
            time.sleep(pause)

        cached = {**cached, **new}
        if store is not None and new:
            store.save("ticker_descriptions",
                       pd.DataFrame({"ticker": list(new), "description": list(new.values())}))
    else:
        print(f"All {len(tickers)} descriptions already cached - no Yahoo calls.")

    return {t: cached[t] for t in tickers if t in cached}


# --------------------------------------------------------------------------- #
# Embeddings (OpenAI) - cached once per ticker                                 #
# --------------------------------------------------------------------------- #
def load_embedded_tickers(store) -> set[str]:
    """Tickers already present in the `ticker_embeddings` cache — the 'done' set.
    A ticker here needs NEITHER a (Yahoo) description NOR an (OpenAI) embedding, so
    the caller can skip it end-to-end and only process the rest."""
    if store is None:
        return set()
    df = store.load("ticker_embeddings", columns=["ticker"])
    return set(df["ticker"].dropna()) if not df.empty else set()


def get_openai_embeddings(
    descriptions: dict[str, str],
    model: str = "text-embedding-3-small",
    store=None,
    batch_size: int = 100,
    max_chars: int = 8000,
    force: bool = False,
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return DataFrame index=ticker, columns=embedding dims. OpenAI is called ONLY
    for tickers not already in the DB embedding cache (`ticker_embeddings`, one
    float8[] array per ticker) unless force=True. New vectors are merged back.

    `descriptions` need only cover the tickers that still need embedding; pass
    `universe` (the full ticker list) to get every ticker's vector back — cached
    ones included — so the caller can feed only the to-do descriptions here.
    """
    cached: dict[str, np.ndarray] = {}
    if store is not None:
        cache = store.load("ticker_embeddings")
        if not cache.empty:
            for _, r in cache.iterrows():
                cached[r["ticker"]] = np.asarray(r["embedding"], dtype="float64")

    todo = [t for t in descriptions if force or t not in cached]

    new: dict[str, np.ndarray] = {}
    if todo:
        client = OpenAI(api_key=_api_key())
        print(f"Embedding {len(todo)} new tickers (OpenAI); {len(cached)} already cached.")
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            inputs = [descriptions[t][:max_chars] for t in chunk]
            resp = client.embeddings.create(model=model, input=inputs)
            for t, item in zip(chunk, resp.data):      # resp.data preserves order
                new[t] = np.asarray(item.embedding, dtype="float64")
    else:
        print(f"All {len(descriptions)} embeddings already cached - no OpenAI calls.")

    all_vecs = {**cached, **new}
    if not all_vecs:
        return pd.DataFrame()

    emb = pd.DataFrame.from_dict(all_vecs, orient="index")
    emb.columns = [f"e{i}" for i in range(emb.shape[1])]

    if store is not None and new:                       # persist only newly added
        rows = pd.DataFrame({"ticker": list(new),
                             "embedding": [v.tolist() for v in new.values()]})
        store.save("ticker_embeddings", rows)

    selection = universe if universe is not None else list(descriptions)
    req = [t for t in selection if t in emb.index]
    return emb.loc[req]