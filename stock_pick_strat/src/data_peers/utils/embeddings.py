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
    cache_path: Path | None = None,
    pause: float = 0.2,
    force: bool = False,
) -> dict[str, str]:
    """
    Return {ticker: description}. Yahoo is queried ONLY for tickers not already
    in the cache (unless force=True). New descriptions are merged into the cache.
    """
    cached: dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        df = pd.read_parquet(cache_path)
        cached = df["description"].to_dict()

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
        if cache_path and new:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"description": cached}).to_parquet(cache_path)  # index = ticker
    else:
        print(f"All {len(tickers)} descriptions already cached - no Yahoo calls.")

    return {t: cached[t] for t in tickers if t in cached}


# --------------------------------------------------------------------------- #
# Embeddings (OpenAI) - cached once per ticker                                 #
# --------------------------------------------------------------------------- #
def get_openai_embeddings(
    descriptions: dict[str, str],
    model: str = "text-embedding-3-small",
    cache_path: Path | None = None,
    batch_size: int = 100,
    max_chars: int = 8000,
    force: bool = False,
) -> pd.DataFrame:
    """
    Return DataFrame index=ticker, columns=embedding dims. OpenAI is called ONLY
    for tickers not already in the embedding cache (unless force=True). New
    vectors are merged into the cache.
    """
    cached: dict[str, np.ndarray] = {}
    if cache_path and Path(cache_path).exists():
        cache = pd.read_parquet(cache_path)
        for t in cache.index:
            cached[t] = cache.loc[t].to_numpy(dtype="float64")

    todo = [t for t in descriptions if force or t not in cached]

    new: dict[str, np.ndarray] = {}
    if todo:
        client = OpenAI(api_key=_api_key())
        print(f"Embedding {len(todo)} new tickers (OpenAI); "
              f"{len(descriptions) - len(todo)} already cached.")
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

    if cache_path and new:                              # persist only if we added new
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        emb.to_parquet(cache_path)

    req = [t for t in descriptions if t in emb.index]
    return emb.loc[req]