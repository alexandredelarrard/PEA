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

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# `gpt_extract` is a shared service, like `src/utils/` -- the one sanctioned cross-import
# between src/ subfolders. It owns the single OpenAI client factory in the repo, so this
# module no longer builds its own (and no longer carries its own stale character cap).
from src.gpt_extract import EMBEDDING_MAX_CHARS, embed_texts

logger = logging.getLogger(__name__)

#: Tickers whose DESCRIPTION must be fetched under a different symbol, because Yahoo's
#: `assetProfile` module is missing on the one the universe uses. Same pattern as
#: `sector_peers.DUAL_CLASS_SECONDARY_TO_PRIMARY`, and the same reason: the universe's symbol
#: is authoritative, the vendor's is a lookup key.
#:
#: ⚠ THE DEFECT IS A MISSING MODULE, NOT A RENAME -- measured, because the obvious diagnosis is
#: wrong and costs a build to discover. `yf.Ticker("FISV").info` still answers with 103 keys and
#: `shortName` "Fiserv, Inc.", so the symbol is live; it just has no `sector`, `industry` or
#: `longBusinessSummary` (a control, `MA`, returns 176 keys and an 1,831-char summary in the
#: same session). Fiserv did rebrand FISV -> FI in 2023, but `FI` is NOT a Yahoo symbol: it
#: 404s with "Quote not found", and Yahoo's own search still returns FISV as the primary NASDAQ
#: listing. So an alias to `FI` fixes nothing. The CROSS-LISTINGS do carry the module --
#: `FIV.DE` (XETRA) returns sector "Technology", industry "Information Technology Services" and
#: the same 1,690-char English description of Fiserv, Inc.
#:
#: Only the TEXT is taken, never a price, so the listing's EUR currency is irrelevant. The row
#: is STORED under the universe's symbol, so nothing downstream knows about the alias.
#: Without it the embedding is never written and at `w_corr: 0` the peer basket comes back
#: EMPTY -- 0 non-null `sector_ret` and `peer_mom_63` across FISV's whole 26-year history.
DESCRIPTION_TICKER_ALIAS: dict[str, str] = {"FISV": "FIV.DE"}   # Fiserv: US profile is empty

#: Shortest `longBusinessSummary` worth embedding. Below this Yahoo has returned a stub rather
#: than a business description, which embeds to noise.
MIN_DESCRIPTION_CHARS = 40


# --------------------------------------------------------------------------- #
# Business descriptions (Yahoo) - cached once per ticker                       #
# --------------------------------------------------------------------------- #
def fetch_business_descriptions(
    tickers: list[str],
    store=None,
    pause: float = 0.2,
    force: bool = False,
    alias: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Return {ticker: description}. Yahoo is queried ONLY for tickers not already
    in the DB cache (`ticker_descriptions`) unless force=True; new descriptions
    are merged back into the table.

    `alias` (default `DESCRIPTION_TICKER_ALIAS`) redirects the QUERY for a ticker the vendor
    has renamed; the result is still stored under the universe's own symbol.

    ⚠ A ticker that yields no usable description is logged at WARNING. It used to be silent
    unless Yahoo raised -- an absent or stub `longBusinessSummary` just fell through the `if`
    -- which is how FISV went the entire life of the feature with an empty peer basket and
    all-NaN peer features while the build reported success.
    """
    alias = DESCRIPTION_TICKER_ALIAS if alias is None else alias
    cached: dict[str, str] = {}
    if store is not None:
        df = store.load("ticker_descriptions", optional=True)
        if df is not None:
            cached = dict(zip(df["ticker"], df["description"]))

    missing = [t for t in tickers if force or t not in cached]
    if missing:
        new: dict[str, str] = {}
        for t in tqdm(missing, desc=f"Fetching {len(missing)} descriptions (Yahoo)"):
            queried = alias.get(t, t)
            try:
                info = yf.Ticker(queried).info
                text = info.get("longBusinessSummary")
                if text and isinstance(text, str) and len(text) > MIN_DESCRIPTION_CHARS:
                    prefix = " ".join(str(info.get(k, "")) for k in ("sector", "industry"))
                    new[t] = (prefix + ". " + text).strip()
                else:
                    logger.warning(
                        "%s: no usable business description from Yahoo (queried as '%s', "
                        "longBusinessSummary is %s) -> NO embedding, and at w_corr: 0 an EMPTY "
                        "peer basket. Add a DESCRIPTION_TICKER_ALIAS entry if it has been "
                        "renamed.", t, queried,
                        "absent" if not text else f"{len(str(text))} chars")
            except Exception as e:                     # noqa: BLE001 -- one ticker, not the run
                logger.warning("%s: description fetch failed (%s)", t, e)
            time.sleep(pause)

        cached = {**cached, **new}
        if store is not None and new:
            store.save("ticker_descriptions",
                       pd.DataFrame({"ticker": list(new), "description": list(new.values())}))
    else:
        logger.info("All %d descriptions already cached - no Yahoo calls.", len(tickers))

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
    df = store.load("ticker_embeddings", columns=["ticker"], optional=True)
    return set(df["ticker"].dropna()) if df is not None else set()


def get_openai_embeddings(
    descriptions: dict[str, str],
    model: str = "text-embedding-3-small",
    store=None,
    batch_size: int = 100,
    max_chars: int = EMBEDDING_MAX_CHARS,
    force: bool = False,
    universe: list[str] | None = None,
    client=None,
) -> pd.DataFrame:
    """
    Return DataFrame index=ticker, columns=embedding dims. OpenAI is called ONLY
    for tickers not already in the DB embedding cache (`ticker_embeddings`, one
    float8[] array per ticker) unless force=True. New vectors are merged back.

    `descriptions` need only cover the tickers that still need embedding; pass
    `universe` (the full ticker list) to get every ticker's vector back — cached
    ones included — so the caller can feed only the to-do descriptions here.

    `max_chars` is the model's real limit (28,000), not the 8,000 this module used to
    apply. Existing cached vectors are NOT invalidated, so no peer set moves until
    someone rebuilds with `force=True`.

    `client` injects a stub embedder for tests (see `gpt_extract.embed_texts`).
    """
    cached: dict[str, np.ndarray] = {}
    if store is not None:
        cache = store.load("ticker_embeddings", optional=True)
        if cache is not None:
            for _, r in cache.iterrows():
                cached[r["ticker"]] = np.asarray(r["embedding"], dtype="float64")

    todo = [t for t in descriptions if force or t not in cached]

    new: dict[str, np.ndarray] = {}
    if todo:
        logger.info("Embedding %d new tickers (OpenAI); %d already cached.",
                    len(todo), len(cached))
        vectors = embed_texts([descriptions[t] for t in todo], model=model,
                              batch_size=batch_size, max_chars=max_chars, client=client)
        for t, vec in zip(todo, vectors):              # embed_texts preserves order
            new[t] = np.asarray(vec, dtype="float64")
    else:
        logger.info("All %d embeddings already cached - no OpenAI calls.", len(descriptions))

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