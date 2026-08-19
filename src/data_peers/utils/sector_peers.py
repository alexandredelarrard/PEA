"""
Build each stock's peer basket from BOTH return correlation and business-text
embedding similarity, and persist/load the resulting peer dictionary.

Why hybrid: return correlation alone rewards shared factor exposure (two large
healthcare names co-move because of the sector/market factor, not because their
businesses are alike). Embedding the business description and taking cosine
similarity captures actual business similarity (Zoetis <-> Elanco/IDEXX), which
correlation misses. We combine the two so peers must be BOTH statistically and
economically similar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Dual-class share redundancy                                                  #
# --------------------------------------------------------------------------- #
# Some companies trade under TWO tickers for the SAME business (e.g. Alphabet
# GOOGL/GOOG, Fox FOXA/FOX, News Corp NWSA/NWS). Their returns correlate ~1.0, so
# in the peer calc the twin would be a stock's own #1 "peer" and would double-count
# that company in everyone else's basket -> flawed peers. We keep the PRIMARY
# (class A / more liquid) and map each redundant SECONDARY (class B/C) to it: the
# secondary is dropped as a peer CANDIDATE and instead inherits its primary's
# basket. Extend as the universe adds dual-class names (e.g. "UA": "UAA").
DUAL_CLASS_SECONDARY_TO_PRIMARY: dict[str, str] = {
    "GOOG": "GOOGL",   # Alphabet   class C -> class A
    "FOX": "FOXA",     # Fox        class B -> class A
    "NWS": "NWSA",     # News Corp  class B -> class A
}

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


# --------------------------------------------------------------------------- #
# Dual-class share dedup (GOOG/GOOGL, FOX/FOXA, NWS/NWSA)                       #
# --------------------------------------------------------------------------- #
def _peers_from_similarity_matrix(
    sim: pd.DataFrame,
    top_k: int,
    weighting: str,
    redundant_map: dict[str, str] | None = None,
) -> dict:
    """Top-k peers per ticker from a similarity matrix, with DUAL-CLASS dedup.

    A secondary share class (class B/C, e.g. GOOG) correlates ~1.0 with its primary
    (GOOGL), so it would otherwise be a stock's own #1 peer and double-count the
    company in every basket. It is therefore never a peer CANDIDATE; instead each
    secondary INHERITS its primary's basket, so it still has valid, non-self peers.
    """
    redundant_map = (DUAL_CLASS_SECONDARY_TO_PRIMARY if redundant_map is None
                     else redundant_map)
    secondaries = set(redundant_map)
    cand = [c for c in sim.columns if c not in secondaries]         # peer candidates
    sim_c = sim[cand]
    peer_dict = {t: _weights_from_similarity(sim_c.loc[t], top_k, weighting)
                 for t in sim.index if t not in secondaries}
    for sec, prim in redundant_map.items():                         # twin -> primary's basket
        if sec in sim.index and prim in peer_dict:
            peer_dict[sec] = dict(peer_dict[prim])
    return peer_dict


def dedupe_share_classes(peer_dict: dict,
                         redundant_map: dict[str, str] | None = None) -> dict:
    """Post-hoc dual-class dedup for an already-built / CACHED peer dict: strip every
    secondary share class out of all baskets (renormalizing the remaining weights)
    and give each secondary its primary's basket. Idempotent -- applied on load so a
    dict built before this fix is corrected without recomputing embeddings."""
    redundant_map = (DUAL_CLASS_SECONDARY_TO_PRIMARY if redundant_map is None
                     else redundant_map)
    if not redundant_map:
        return peer_dict
    secondaries = set(redundant_map)
    out: dict = {}
    for t, peers in peer_dict.items():
        kept = {p: w for p, w in (peers or {}).items() if p not in secondaries}
        s = sum(kept.values())
        out[t] = {p: float(round(w / s, 6)) for p, w in kept.items()} if s > 0 else {}
    for sec, prim in redundant_map.items():
        if prim in out:
            out[sec] = dict(out[prim])
    return out


def build_peer_dict(stock_returns, top_k=20, weighting="corr", min_obs=120,
                    redundant_map=None) -> dict:
    """Correlation-only peer dict (dual-class-deduped; kept for fallback/comparison)."""
    corr = stock_returns.corr(min_periods=min_obs)
    return _peers_from_similarity_matrix(corr, top_k, weighting, redundant_map)


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
    redundant_map: dict[str, str] | None = None,
) -> dict:
    """
    Peer dict from the combined (correlation + embedding) similarity (dual-class-
    deduped). Falls back to correlation-only where embeddings are unavailable.
    """
    corr = stock_returns.corr(min_periods=min_obs)
    combined = combine_similarity(corr, embed_sim, w_corr, w_embed)
    return _peers_from_similarity_matrix(combined, top_k, weighting, redundant_map)


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


def load_peer_dict(path: Path, redundant_map: dict[str, str] | None = None) -> dict:
    """Load a cached peer dict, applying the dual-class dedup on the way in so a dict
    built before this fix is corrected without recomputing embeddings."""
    with open(path, encoding="utf-8") as f:
        return dedupe_share_classes(json.load(f), redundant_map)


def save_peer_dict(peer_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(peer_dict, f, ensure_ascii=False, indent=2)
