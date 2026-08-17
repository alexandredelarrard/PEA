"""
panel.py  (src/data_aggregate/utils/common/panel.py)
---------------------------------------------------
The PEER-RELATIVE panel: turn a {name: daily wide frame} dict into the long feature
panel every cube part is made of, each characteristic expressed as

    f_<name>_vs_peers    z-score against the firm's direct competitors
    f_<name>_xs          percentile against the whole universe

Shared by all thirteen panel builders -- the single hottest symbol in the package.

The generic pieces this module used to also own now live beside it in `common/`: safe
division and inf-sanitizing in `frames.py`, and every per-day cross-sectional transform in
`xs.py` (which is where the five duplicate standardizers were merged). What is left here is
the one thing that is genuinely about PEERS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.xs import XS_CLIP_PEER, winsorize_xs, xs_rank_pct


def peer_relative(
    field_df: pd.DataFrame,
    peer_dict: dict,
    min_peers: int = 3,
    clip: float = XS_CLIP_PEER,
) -> pd.DataFrame:
    """
    (stock - peer_weighted_mean) / peer_weighted_std, per date, per stock.
    Self excluded by construction of the peer dict. NaN-tolerant: peer stats use
    whichever peers have data on the date (weights renormalized).

    Robustness (critical): when only a couple of peers report or their values
    nearly coincide, the peer std collapses toward zero and the raw z-score
    explodes to ~1e13. We therefore (a) require at least `min_peers` peers with
    data on the date, (b) drop dates where the peer std is not strictly positive,
    and (c) winsorize the result to +-`clip` so a near-degenerate peer group can
    never dominate the model.
    """
    rel = pd.DataFrame(index=field_df.index, columns=field_df.columns, dtype="float64")
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in field_df.columns:
            continue
        cols = [p for p in peers if p in field_df.columns]
        if len(cols) < min_peers:
            continue
        w = pd.Series({p: float(peers[p]) for p in cols}, dtype="float64")
        w = w / w.sum()
        peer_vals = field_df[cols]
        present = peer_vals.notna()
        n_present = present.sum(axis=1)
        wsum = present.mul(w, axis=1).sum(axis=1)
        valid = (n_present >= min_peers) & (wsum > 0)

        pmean = peer_vals.mul(w, axis=1).sum(axis=1, min_count=1).div(wsum.where(valid))
        var = (peer_vals.sub(pmean, axis=0) ** 2).mul(w, axis=1).sum(axis=1, min_count=1)
        pstd = np.sqrt(var.div(wsum.where(valid)))

        z = (field_df[ticker] - pmean) / pstd.where(pstd > 0)
        z = z.where(valid)
        rel[ticker] = z.clip(-clip, clip)
    return rel.replace([np.inf, -np.inf], np.nan)


def build_peer_relative_panel(fields: dict, peer_dict: dict) -> pd.DataFrame:
    """Turn a {name: daily wide frame} dict into the long feature panel, each
    characteristic expressed as `f_<name>_vs_peers` (peer-standardized) and
    `f_<name>_xs` (universe percentile). Shared by every panel builder."""
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    long_frames = []
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        # Guarantee a numeric frame: a stray Python `None` / object cell — a KPI genuinely
        # absent for a name (e.g. sparse earnings-call coverage, "no value to compare with") —
        # must be coerced to NaN. Otherwise the NaN-tolerant peer-z math (`peer_relative`)
        # AND the `_xs` rank below both raise "unsupported operand type(s): NoneType and float"
        # the moment a single None reaches them. Coercion is the correct semantics here
        # (absent = NaN), not a workaround, and a no-op on already-float frames.
        fdf = fdf.apply(pd.to_numeric, errors="coerce")
        if fdf.empty or not fdf.notna().any().any():
            continue
        # peer z-score, then trim per-day cross-sectional 1%/99% outliers (the
        # percentile-rank `_xs` below is already outlier-proof, so it uses raw fdf).
        # The stacked long columns are cast to float32: these are z-scores / percentile ranks
        # bounded to O(1), so float64 storage is wasted — halving them (and the concat +
        # defrag copy below) is what keeps the many-feature panels off the OOM killer.
        rel = winsorize_xs(peer_relative(fdf, peer_dict))
        s = rel.stack().astype("float32")
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_peers"))

        xs = xs_rank_pct(fdf)
        s2 = xs.stack().astype("float32")
        s2.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s2.rename(f"f_{name}_xs"))
        del fdf, rel, xs, s, s2                       # free per-field intermediates promptly

    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) leaves
    # behind, so the reset_index() column insert doesn't trip the "highly fragmented
    # DataFrame" PerformanceWarning once the panel has 100+ feature columns.
    return pd.concat(long_frames, axis=1).copy().reset_index()
