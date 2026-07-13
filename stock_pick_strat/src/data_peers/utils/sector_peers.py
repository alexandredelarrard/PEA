"""
Build each stock's peer basket from BOTH return correlation and business-text
embedding similarity, and persist/load the resulting peer dictionary.

Why hybrid: return correlation alone rewards shared factor exposure (two large
healthcare names co-move because of the sector/market factor, not because their
businesses are alike). Embedding the business description and taking cosine
similarity captures actual business similarity (Zoetis <-> Elanco/IDEXX), which
correlation misses. We combine the two so peers must be BOTH statistically and
economically similar.

LOOK-AHEAD NOTE (unchanged): `build_peer_dict*` on full history uses the whole
return sample to define peers -- fine as a static prototype, but for a rigorous
backtest recompute on trailing windows. The embeddings use the CURRENT business
description (slowly changing, like a GICS label), a mild and acceptable static.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _weights_from_similarity(sim_row: pd.Series, top_k: int, weighting: str) -> dict:
    """Top-k peers from one row of a similarity matrix (self excluded)."""
    peers = sim_row.drop(labels=[sim_row.name], errors="ignore").dropna()
    if peers.empty:
        return {}
    top = peers.sort_values(ascending=False).head(top_k)
    top = top[top > 0]
    if top.empty:
        return {}
    if weighting == "equal":
        w = pd.Series(1.0, index=top.index)
    elif weighting == "corr":                 # weight by similarity strength
        w = top.clip(lower=0.0)
    else:
        raise ValueError("weighting must be 'equal' or 'corr'")
    w = w / w.sum()
    return {peer: float(round(wt, 6)) for peer, wt in w.items()}


# keep the old name as an alias so nothing else breaks
_weights_from_corr = _weights_from_similarity


def build_peer_dict(stock_returns, top_k=20, weighting="corr", min_obs=120) -> dict:
    """Correlation-only peer dict (unchanged; kept for fallback/comparison)."""
    corr = stock_returns.corr(min_periods=min_obs)
    return {t: _weights_from_similarity(corr[t], top_k, weighting) for t in corr.columns}


# --------------------------------------------------------------------------- #
# Embedding similarity                                                        #
# --------------------------------------------------------------------------- #
def cosine_similarity_matrix(embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity between every pair of tickers.
    embeddings: index = ticker, columns = embedding dimensions.
    Returns a symmetric DataFrame (ticker x ticker) in [-1, 1].
    """
    X = embeddings.to_numpy(dtype="float64")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    Xn = X / norms
    sim = Xn @ Xn.T
    return pd.DataFrame(sim, index=embeddings.index, columns=embeddings.index)


def combine_similarity(
    corr: pd.DataFrame,
    embed_sim: pd.DataFrame,
    w_corr: float = 0.5,
    w_embed: float = 0.5,
) -> pd.DataFrame:
    """
    Weighted blend of two similarity matrices, each mapped from [-1,1] to [0,1].
    Aligned on the union of tickers. Where a cell is missing in one matrix (e.g.
    a ticker with no embedding, or a pair with too few overlapping returns), the
    weights renormalize over whichever source IS present -- so a missing
    embedding gracefully falls back to correlation-only for that pair.
    """
    tickers = corr.index.union(embed_sim.index)
    c = ((corr.reindex(index=tickers, columns=tickers) + 1.0) / 2.0)
    e = ((embed_sim.reindex(index=tickers, columns=tickers) + 1.0) / 2.0)

    wc = c.notna().astype(float) * w_corr
    we = e.notna().astype(float) * w_embed
    denom = (wc + we).replace(0.0, np.nan)
    combined = (c.fillna(0.0) * w_corr + e.fillna(0.0) * w_embed) / denom
    return combined


def build_peer_dict_hybrid(
    stock_returns: pd.DataFrame,
    embed_sim: pd.DataFrame,
    top_k: int = 20,
    weighting: str = "corr",
    min_obs: int = 120,
    w_corr: float = 0.5,
    w_embed: float = 0.5,
) -> dict:
    """
    Peer dict from the combined (correlation + embedding) similarity.
    Falls back to correlation-only where embeddings are unavailable.
    """
    corr = stock_returns.corr(min_periods=min_obs)
    combined = combine_similarity(corr, embed_sim, w_corr, w_embed)
    return {t: _weights_from_similarity(combined[t], top_k, weighting)
            for t in combined.columns}


# --------------------------------------------------------------------------- #
# Sector returns from peers (unchanged)                                       #
# --------------------------------------------------------------------------- #
def compute_sector_returns(stock_returns: pd.DataFrame, peer_dict: dict) -> pd.DataFrame:
    sector = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns,
                          dtype="float64")
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in stock_returns.columns:
            continue
        cols = [p for p in peers if p in stock_returns.columns]
        if not cols:
            continue
        w = pd.Series({p: float(peers[p]) for p in cols}, dtype="float64")
        w = w / w.sum()
        mat = stock_returns[cols]
        weighted = mat.mul(w, axis=1).sum(axis=1, min_count=1)
        denom = mat.notna().mul(w, axis=1).sum(axis=1)
        sector[ticker] = weighted.div(denom.where(denom > 0))
    return sector


def load_peer_dict(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_peer_dict(peer_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(peer_dict, f, ensure_ascii=False, indent=2)
