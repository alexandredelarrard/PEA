"""
Build each stock's peer basket from return correlation and persist/load the
resulting peer dictionary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _weights_from_corr(corr_row: pd.Series, top_k: int, weighting: str) -> dict:
    peers = corr_row.drop(labels=[corr_row.name], errors="ignore").dropna()
    if peers.empty:
        return {}
    top = peers.sort_values(ascending=False).head(top_k)
    top = top[top > 0]
    if top.empty:
        return {}

    if weighting == "equal":
        w = pd.Series(1.0, index=top.index)
    elif weighting == "corr":
        w = top.clip(lower=0.0)
    else:
        raise ValueError("weighting must be 'equal' or 'corr'")

    w = w / w.sum()
    return {peer: float(round(wt, 6)) for peer, wt in w.items()}


def build_peer_dict(
    stock_returns: pd.DataFrame,
    top_k: int = 20,
    weighting: str = "corr",
    min_obs: int = 120,
) -> dict:
    """Static peer dict from full return history (prototype — look-ahead risk)."""
    corr = stock_returns.corr(min_periods=min_obs)
    return {
        ticker: _weights_from_corr(corr[ticker], top_k, weighting)
        for ticker in corr.columns
    }


def build_peer_dict_rolling(
    stock_returns: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 252,
    top_k: int = 10,
    weighting: str = "corr",
    min_obs: int = 120,
) -> dict:
    """Point-in-time peer dict using only data up to `as_of`."""
    window = stock_returns.loc[:as_of].tail(lookback)
    return build_peer_dict(window, top_k=top_k, weighting=weighting, min_obs=min_obs)


def compute_sector_returns(
    stock_returns: pd.DataFrame,
    peer_dict: dict,
) -> pd.DataFrame:
    """Per-stock sector daily-return series from peer baskets."""
    sector = pd.DataFrame(
        index=stock_returns.index,
        columns=stock_returns.columns,
        dtype="float64",
    )
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in stock_returns.columns:
            continue
        cols = [p for p in peers if p in stock_returns.columns]
        if not cols:
            continue
        # NaN-tolerant weighted mean: renormalize weights over the peers that
        # actually have data on each date. A plain matrix product (`@`) would
        # propagate a single missing peer into a NaN for the WHOLE date, which
        # (via the dropna in beta estimation) silently truncates a stock's whole
        # history to the short window where every peer happens to be listed.
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
