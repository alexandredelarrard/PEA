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

from src.data_aggregate.utils.common.xs import (
    PEER_DISPERSION_FLOOR, XS_CLIP_PEER, winsorize_xs, xs_rank_pct,
)


def peer_relative(
    field_df: pd.DataFrame,
    peer_dict: dict,
    min_peers: int = 3,
    clip: float = XS_CLIP_PEER,
    dispersion_floor: float = PEER_DISPERSION_FLOOR,
    winsorize_inputs: bool = True,
) -> pd.DataFrame:
    """
    (stock - peer_weighted_mean) / peer_weighted_std, per date, per stock.
    Self excluded by construction of the peer dict. NaN-tolerant: peer stats use
    whichever peers have data on the date (weights renormalized).

    Robustness (critical): when only a couple of peers report or their values
    nearly coincide, the peer std collapses toward zero and the raw z-score
    explodes to ~1e13. We therefore (a) require at least `min_peers` peers with
    data on the date, (b) FLOOR the peer std at `dispersion_floor` x the day's
    cross-sectional std, (c) winsorize the INPUTS 1%/99% per day before the moments
    are taken, and (d) clip the result to +-`clip`.

    ⚠ WHY THE FLOOR IS THE LOAD-BEARING GUARD, and the input trim only the junior
    partner. Decomposing the cells that landed at the +-8 clip on five real fields over the
    full 1995-2026 history (1.8-2.5 M non-null cells each) showed those cells have FULL
    baskets -- 6 to 7 peers present, not thin -- and a peer dispersion of 0.5% to 8% of that
    day's UNIVERSE dispersion. The z explodes because the denominator collapsed, not because
    the numerator is wild: the embedding basket does its job too well, so 7 business-text-
    similar companies have nearly identical fundamentals and a subject that differs even
    modestly is divided by a near-zero std.

    Measured share of cells at the clip, on the full 1995-2026 universe. The first three
    columns are the single-treatment A/B arms, which can only be produced by re-running with
    each treatment disabled; the last is what this function does and is re-measurable from
    `cube_part_fundamentals` at any time -- the figures below are that re-measurement:

        field               base   winsor only   floor only   BOTH (shipped)   cells
        returnOnEquity      3.24%        2.70%        1.85%            1.60%   3,057,212
        debtToEquity        3.49%        2.97%        1.86%            2.11%   3,179,024
        interest_coverage   5.79%        5.63%        1.85%            2.80%   2,584,176
        cash_to_debt        5.64%        5.54%        1.80%            2.60%   2,980,320
        profitMargins       1.06%        0.76%        0.61%            0.12%   3,056,867
        mean                3.84%                                      1.85%

    (`|z| > 4` moves the same way: `interest_coverage` 9.22% -> 4.42%.)

    ⚠ CLIP RATE IS DRIVEN BY COVERAGE, NOT BY EXTREMENESS. `f_pbo_to_mcap_vs_peers` sits at
    8.59% -- 3.1x the next-worst field -- on **70,921** cells, against 2.58-3.18 M for every
    field above. It is not a wilder quantity; it is a thinner one. A sparse field's peer basket
    is drawn from the few names that also report it, those names resemble each other, basket
    dispersion collapses, and the floor binds. That is the reason a sparse field must never
    carry a peer leg, and it is also why the fix for a high clip rate is a coverage or
    definition change -- raising `XS_CLIP_PEER` would only hide the diagnostic.

    ⚠ THE TWO TREATMENTS PARTLY WORK AGAINST EACH OTHER, which is why BOTH lands above
    floor-only on four of the five fields. Trimming the inputs shrinks the day's
    cross-sectional std, the floor is a FRACTION of that std, so the floor itself drops and
    binds less often. Keeping the reference std un-trimmed would bind harder and score closer
    to the floor-only column -- but it would do so because a single mis-parsed extreme
    inflated the reference, which is the opposite of the robustness being bought. The trimmed
    reference is the honest one and the interaction is the price.

    Input winsorization alone moves `interest_coverage` 5.79% -> 5.63%, which is nothing; the
    floor is what fixes it. The trim is kept because it still removes the one genuine tail
    case the floor cannot -- a mis-parsed extreme in the SUBJECT's own value.

    The floor GENERALISES the `pstd > 0` degenerate-case guard from *zero* to *negligible*,
    and reads as: this basket is too homogeneous to resolve a difference this small.
    Residual saturation is expected and correct: a genuinely extreme name should reach the
    clip, and `f_pbo_to_mcap_vs_peers` (the worst field, 8.59%) still does.
    """
    # Trim the day's cross-section BEFORE the peer moments are taken, so a single mis-parsed
    # extreme cannot move its basket's mean and std. Done once here rather than per ticker:
    # `winsorize_xs` is per-ROW (per date) over the whole universe, so it is the same bound
    # for every basket on that day.
    #
    # The two treatments are separately switchable (`winsorize_inputs=False`,
    # `dispersion_floor=0.0`) so each can be reverted or A/B-ed on its own -- they were
    # measured as separate arms and they interact (see the table above).
    if winsorize_inputs:
        field_df = winsorize_xs(field_df)
    # The day's universe dispersion, which the floor is expressed as a fraction of. Sample
    # std (ddof=1) to match `xs.xs_z`, the repo's other standardizer.
    universe_sd = field_df.std(axis=1)
    floor = universe_sd * float(dispersion_floor)

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
        # ⚠ THE ZERO-DISPERSION GUARD IS EVALUATED ON THE RAW STD, BEFORE THE FLOOR, and the
        # order is the whole point. Flooring first would turn a basket whose peers all report
        # the SAME value into a finite z divided by the floor -- silently overriding the
        # zero-dispersion policy `xs.py` documents as a declared decision rather than an
        # accident. The floor is for NEGLIGIBLE dispersion; exactly-zero dispersion still
        # means "these peers cannot rank anything" and still yields NaN.
        degenerate = ~(pstd > 0)
        pstd = pstd.clip(lower=floor).where(~degenerate)

        z = (field_df[ticker] - pmean) / pstd
        z = z.where(valid)
        rel[ticker] = z.clip(-clip, clip)
    return rel.replace([np.inf, -np.inf], np.nan)


#: The emission modes a field may declare. The DEFAULT -- a field absent from the emission map
#: -- stays `_vs_peers` + `_xs` with no raw leg, which is what all thirteen existing builders
#: rely on. The three explicit modes exist for SPARSE / DECAYED signals, where a rank alone is
#: not a usable feature:
#:   "raw"        f_<name>                          the value itself, nothing else
#:   "raw+xs"     f_<name>, f_<name>_xs             + its universe percentile
#:   "raw+peers"  f_<name>, f_<name>_vs_peers       + its peer z-score
#: Every explicit mode emits the raw leg: nothing ships as a rank alone (D27). A decayed
#: conviction weight in [0, ~n_managers] is a quantity LightGBM can split on directly, and
#: ranking it away throws that scale out.
_EMISSION_MODES = ("raw", "raw+xs", "raw+peers")


def build_peer_relative_panel(fields: dict, peer_dict: dict,
                              emission: dict | None = None) -> pd.DataFrame:
    """Turn a {name: daily wide frame} dict into the long feature panel, each
    characteristic expressed as `f_<name>_vs_peers` (peer-standardized) and
    `f_<name>_xs` (universe percentile). Shared by every panel builder.

    `emission` overrides that per field with one of `_EMISSION_MODES`. Fields left out of the
    map -- and every caller that passes no map at all -- keep the exact two-leg behaviour
    above, unchanged.

    ⚠ The RAW leg is deliberately NOT winsorized and NOT clipped. Those treatments belong to
    the z-score, whose scale is only meaningful after outlier control; the raw value is a
    quantity in its own units and trimming it would destroy the thing it was emitted for. It is
    still cast to float32 like the other legs, to keep a 100-feature panel off the OOM killer.
    """
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = emission or {}
    unknown = {m for m in emission.values() if m not in _EMISSION_MODES}
    if unknown:
        raise ValueError(f"unknown emission mode(s) {sorted(unknown)}; "
                         f"expected one of {list(_EMISSION_MODES)}")
    stray = set(emission) - set(fields)
    if stray:
        # A name in the map that no field produces is a silent no-op -- exactly the drift the
        # parts registry was built to end. Fail loudly rather than emit a panel missing the
        # column the caller thinks it declared.
        raise KeyError(f"emission declares field(s) not present in `fields`: {sorted(stray)}")

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
        mode = emission.get(name)
        if mode is not None:
            raw = fdf.stack().astype("float32")
            raw.index.set_names(["date", "ticker"], inplace=True)
            long_frames.append(raw.rename(f"f_{name}"))
            del raw
        if mode is None or mode == "raw+peers":
            rel = winsorize_xs(peer_relative(fdf, peer_dict))
            s = rel.stack().astype("float32")
            s.index.set_names(["date", "ticker"], inplace=True)
            long_frames.append(s.rename(f"f_{name}_vs_peers"))
            del rel, s
        if mode is None or mode == "raw+xs":
            xs = xs_rank_pct(fdf)
            s2 = xs.stack().astype("float32")
            s2.index.set_names(["date", "ticker"], inplace=True)
            long_frames.append(s2.rename(f"f_{name}_xs"))
            del xs, s2
        del fdf                                       # free per-field intermediates promptly

    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) leaves
    # behind, so the reset_index() column insert doesn't trip the "highly fragmented
    # DataFrame" PerformanceWarning once the panel has 100+ feature columns.
    return pd.concat(long_frames, axis=1).copy().reset_index()
