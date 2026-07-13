"""
embeddings.py  (src/data_aggregate/utils/embeddings.py)
-------------------------------------------------------
Fetch each ticker's business description and embed it with the OpenAI
embeddings API, with on-disk caching so unchanged descriptions are never
re-embedded (embeddings are deterministic and cost money).

Env: reads OPEN_AI_API_KEY (your .env spelling) or OPENAI_API_KEY.
"""
from __future__ import annotations

import hashlib
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


def fetch_business_descriptions(tickers: list[str], pause: float = 0.2) -> dict[str, str]:
    """
    Business summary per ticker from yfinance (`longBusinessSummary`). This is a
    clean one-paragraph description of what the company does -- ideal for
    embedding. Swap in SEC 10-K Item 1 text here for a richer source if desired.
    """

    out = {}
    for t in tqdm(tickers, desc="Fetching business descriptions"):
        try:
            info = yf.Ticker(t).info
            text = info.get("longBusinessSummary")
            if text and isinstance(text, str) and len(text) > 40:
                # prepend sector/industry so the embedding anchors on taxonomy too
                prefix = " ".join(str(info.get(k, "")) for k in ("sector", "industry"))
                out[t] = (prefix + ". " + text).strip()
        except Exception as e:
            print(f"{t}: description fetch failed ({e})")
        time.sleep(pause)
    return out


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_openai_embeddings(
    descriptions: dict[str, str],
    model: str = "text-embedding-3-small",
    cache_path: Path | None = None,
    batch_size: int = 100,
    max_chars: int = 8000,
) -> pd.DataFrame:
    """
    Embed each ticker's description. Returns DataFrame index=ticker,
    columns=embedding dims. Cached by (ticker, description-hash): only new or
    changed descriptions are sent to the API.
    """
    hashes = {t: _hash(txt) for t, txt in descriptions.items()}

    cached_vecs: dict[str, np.ndarray] = {}
    cached_hashes: dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        cache = pd.read_parquet(cache_path)
        for t in cache.index:
            cached_hashes[t] = cache.loc[t, "desc_hash"]
            cached_vecs[t] = cache.drop(columns="desc_hash").loc[t].to_numpy(dtype="float64")

    reuse = {t: cached_vecs[t] for t in descriptions
             if t in cached_vecs and cached_hashes.get(t) == hashes[t]}
    todo = [t for t in descriptions if t not in reuse]

    new_vecs: dict[str, np.ndarray] = {}
    if todo:
        
        client = OpenAI(api_key=_api_key())
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            inputs = [descriptions[t][:max_chars] for t in chunk]
            resp = client.embeddings.create(model=model, input=inputs)
            for t, item in zip(chunk, resp.data):   # resp.data preserves order
                new_vecs[t] = np.asarray(item.embedding, dtype="float64")

    all_vecs = {**reuse, **new_vecs}
    if not all_vecs:
        return pd.DataFrame()
    emb = pd.DataFrame.from_dict(all_vecs, orient="index")
    emb.columns = [f"e{i}" for i in range(emb.shape[1])]

    if cache_path:
        to_save = emb.copy()
        to_save["desc_hash"] = pd.Series({t: hashes[t] for t in emb.index})
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        to_save.to_parquet(cache_path)

    return emb
