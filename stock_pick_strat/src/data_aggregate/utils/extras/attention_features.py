"""
attention_features.py  (src/data_aggregate/utils/attention_features.py)
-----------------------------------------------------------------------
Retail-ATTENTION features. Wikipedia pageviews and Google Trends search interest
are two noisy proxies of the SAME latent — how much public attention a name is
getting — so rather than ship them as two correlated features they are BLENDED
into ONE robust indicator (`build_combined_attention_panel`).

Per source we first build point-in-time trailing fields (`_attention_fields`):

    <prefix>_attn_spike   log recent (5d) avg / trailing (63d) avg attention
    <prefix>_attn_level   log trailing (21d) average attention (attention level)

Signal (Da-Engelberg-Gao "In Search of Attention"): a SPIKE in attention vs the
name's own baseline flags retail-driven attention (often short-term overpricing
-> subsequent reversal). Everything is point-in-time (trailing windows only).

The blend is a cross-sectional RANK-average: each source's field is turned into a
per-date rank in [0, 1] (scale/outlier invariant, so pageviews and search interest
become commensurable), then nan-averaged across sources (a name covered by only one
source still gets a value -> better coverage than either alone). The blended field
then flows through the shared peer-relative builder -> f_attn_{spike,level}_{vs_peers,xs}.

Weekly series (Google Trends) are forward-filled onto the daily trading calendar
(bounded ffill), so the value on date t is the last observation on/before t.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel


def _attention_fields(attn_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                      prefix: str, value_col: str, ffill_limit: int = 7) -> dict:
    piv = attn_hist.pivot_table(index="date", columns="ticker",
                                values=value_col, aggfunc="sum")
    piv.index = pd.to_datetime(piv.index).normalize()
    # onto the trading calendar; bounded ffill bridges weekly->daily and small gaps
    piv = piv.reindex(piv.index.union(idx)).ffill(limit=ffill_limit).reindex(idx)
    A = piv.clip(lower=0.0)

    recent = A.rolling(5, min_periods=3).mean()
    base = A.rolling(63, min_periods=20).mean()
    spike = (np.log1p(recent) - np.log1p(base)).replace([np.inf, -np.inf], np.nan)
    level = np.log1p(A.rolling(21, min_periods=10).mean()).replace([np.inf, -np.inf], np.nan)
    return {f"{prefix}_attn_spike": spike, f"{prefix}_attn_level": level}


def _rank_blend(frames: list[pd.DataFrame], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Robustly blend same-latent frames: per-date cross-sectional rank in [0, 1]
    (scale/outlier invariant) of each frame, then nan-mean across frames. A ticker
    covered by only one frame keeps that frame's rank; a date/ticker in none stays
    NaN. Frames are aligned to `idx` x the union of their tickers."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(index=idx)
    cols = sorted(set().union(*(f.columns for f in frames)))
    ranked = [f.rank(axis=1, pct=True).reindex(index=idx, columns=cols) for f in frames]
    with warnings.catch_warnings():                       # all-NaN cell -> NaN (not a warning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        blended = np.nanmean(np.stack([r.to_numpy(dtype=float) for r in ranked]), axis=0)
    return pd.DataFrame(blended, index=idx, columns=cols)


def build_combined_attention_panel(
    wiki_history: pd.DataFrame | None,
    trends_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    wiki_value_col: str = "pageviews",
    trends_value_col: str = "search_interest",
    prefix: str = "attn",
) -> pd.DataFrame:
    """Robust COMBINED retail-attention panel blending Wikipedia pageviews and
    Google-Trends search interest — two noisy proxies of the SAME latent (public
    attention), so a cross-sectional rank-blend denoises and improves coverage vs
    either source alone. Emits `f_<prefix>_spike_{vs_peers,xs}` and
    `f_<prefix>_level_{vs_peers,xs}`. Empty only if BOTH sources are unavailable."""
    per_source: list[tuple[str, dict]] = []
    if (wiki_history is not None and not wiki_history.empty
            and wiki_value_col in wiki_history.columns):
        per_source.append(("wiki", _attention_fields(
            wiki_history, trading_index, "wiki", wiki_value_col)))
    if (trends_history is not None and not trends_history.empty
            and trends_value_col in trends_history.columns):
        per_source.append(("gt", _attention_fields(
            trends_history, trading_index, "gt", trends_value_col)))
    if not per_source:
        return pd.DataFrame(columns=["date", "ticker"])

    fields: dict[str, pd.DataFrame] = {}
    for kind in ("spike", "level"):                       # blend spike and level separately
        blended = _rank_blend([f[f"{p}_attn_{kind}"] for p, f in per_source], trading_index)
        if not blended.empty and blended.notna().any().any():
            fields[f"{prefix}_{kind}"] = blended
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    return build_peer_relative_panel(fields, peer_dict)
