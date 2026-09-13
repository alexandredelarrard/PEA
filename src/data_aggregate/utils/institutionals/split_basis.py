"""
split_basis.py  (src/data_aggregate/utils/institutionals/split_basis.py)
------------------------------------------------------------------------
ONE primitive: the cumulative split product as of a date, per ticker -- and the
`future_split_factor` built on it that moves an AS-FILED quantity onto today's basis.

WHY EVERY DISCLOSURE FAMILY NEEDS IT. A filing reports what was true when it was made, in the
share count and the price that existed THEN: a 13F reports the shares a manager held at the
quarter end, a Form 4 the price an insider actually paid. The price frames are the opposite --
`close_split` is restated to TODAY's basis. So any comparison that crosses a split and does not
restate one side reads the split as a trade:

  * `Sigma shares(q) / Sigma shares(q-1)` on GOOGL's 2022-07-15 20-for-1 split is **+1,900%**
    of "accumulation" with no manager having bought a share;
  * worse, the per-manager `shares(q) > shares(q-1)` test behind `ic_inst_cluster_buying` makes
    EVERY holder an increaser, so a split prints the maximum possible unanimity signal
    (`(inc - dec) / holders = +1.0`) on the one quarter nobody decided anything;
  * an insider's filed `price_per_share` of $2,800 against a post-split `close_split` of $140
    reads as a 95% loss since the purchase.

    factor(d) = PROD of ratios for splits EFFECTIVE AFTER d = cum_total / cum_at(d)

so `shares_as_filed x factor(d)` and `price_as_filed / factor(d)` are both on today's basis,
and `factor(d1) / factor(d2)` is the product of the splits between the two dates -- which is
what restates a prior quarter's count onto a later quarter's basis.

⚠ THE FACTOR IS 1.0 FOR A TICKER WITH NO SPLIT AND FOR EVERY DATE AFTER THE LAST ONE, so a
caller can apply it unconditionally. A ticker absent from `prices_splits` gets 1.0 rather than
NaN: "no split recorded" is the overwhelmingly common case, not missing data.

⚠ INCLUSIVE AT THE SPLIT DATE. `cum_at(d)` counts splits with `date <= d`, matching
`prices_splits.date` being the EFFECTIVE date (the first session that trades on the new basis).
A quantity dated on the split date is therefore already on the new basis and is not restated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_cum_product(splits: pd.DataFrame | None, tickers: pd.Series,
                      dates: pd.Series) -> np.ndarray:
    """`PROD(ratio)` over the splits of each ticker with `date <= dates`, aligned to the inputs.

    1.0 where the ticker has no split at or before the date (which includes every ticker with
    no split at all). A NaT date also resolves to 1.0: the callers pair it with a NaN quantity
    (a manager's first quarter has no previous period), so the factor it receives can never
    reach a feature.
    """
    n = len(tickers)
    out = np.ones(n, dtype="float64")
    if splits is None or splits.empty or n == 0:
        return out
    if not {"date", "ticker", "ratio"}.issubset(splits.columns):
        return out

    s = splits.dropna(subset=["date", "ticker", "ratio"]).copy()
    s["ratio"] = pd.to_numeric(s["ratio"], errors="coerce")
    # A ratio of 0 or NaN is not a split; 1.0 is a no-op that would still cost a lookup.
    s = s[s["ratio"] > 0]
    if s.empty:
        return out
    # ⚠ BOTH SIDES FORCED TO ns. `prices_splits.date` is a Postgres TIMESTAMP arriving as
    # `datetime64[us]`; periods built from a quarter-end are `datetime64[s]`. numpy compares
    # them fine but `searchsorted` on mixed resolutions silently mis-places dates.
    s["date"] = pd.to_datetime(s["date"]).dt.normalize().astype("datetime64[ns]")
    s = s.sort_values(["ticker", "date"])
    s["cum"] = s.groupby("ticker")["ratio"].cumprod()

    # ⚠ ONE GROUPBY, NOT A SCAN PER TICKER. `np.flatnonzero(want == ticker)` inside the loop is
    # O(tickers x rows) and the insider caller passes ~700k rows against ~500 split tickers --
    # 350M comparisons for a lookup table. `groupby(...).indices` builds the same row lists in
    # a single pass.
    want = pd.Series(np.asarray(tickers, dtype=object)).astype(str).reset_index(drop=True)
    when = pd.to_datetime(pd.Series(np.asarray(dates)), errors="coerce").astype("datetime64[ns]")
    when = when.fillna(pd.Timestamp("1900-01-01")).to_numpy()
    positions = want.groupby(want).indices

    for ticker, grp in s.groupby("ticker", sort=False):
        rows = positions.get(str(ticker))
        if rows is None or not len(rows):
            continue
        # `side="right"` == "splits with date <= d": the effective date is already on the new
        # basis, so a quantity dated on it needs no restatement.
        pos = grp["date"].to_numpy().searchsorted(when[rows], side="right")
        cum = np.concatenate(([1.0], grp["cum"].to_numpy()))
        out[rows] = cum[pos]
    return out


def future_split_factor(splits: pd.DataFrame | None, tickers: pd.Series,
                        dates: pd.Series) -> np.ndarray:
    """`PROD(ratio)` over the splits EFFECTIVE AFTER each date -- the factor that restates an
    as-filed quantity onto today's basis: `shares x factor`, `price / factor`.

    1.0 for a ticker with no split after the date, so it applies unconditionally.
    """
    n = len(tickers)
    if n == 0:
        return np.ones(0, dtype="float64")
    at = split_cum_product(splits, tickers, dates)
    # The all-time product per ticker, looked up rather than re-derived through a second
    # searchsorted pass over a column of sentinel dates.
    if splits is None or splits.empty or not {"date", "ticker", "ratio"}.issubset(splits.columns):
        return np.ones(n, dtype="float64")
    s = splits.dropna(subset=["date", "ticker", "ratio"])
    ratio = pd.to_numeric(s["ratio"], errors="coerce")
    s = s[ratio > 0]
    if s.empty:
        return np.ones(n, dtype="float64")
    total_by_ticker = s.groupby(s["ticker"].astype(str))["ratio"].prod()
    total = pd.Series(np.asarray(tickers, dtype=object)).astype(str).map(
        total_by_ticker).fillna(1.0).to_numpy()
    return total / at


def split_adjust_frame(splits: pd.DataFrame | None, frame: pd.DataFrame) -> pd.DataFrame:
    """`future_split_factor` as a (date x ticker) frame, aligned to `frame`.

    The wide-frame form of the same primitive, for AS-TRADED quantities that have to be divided
    by a SPLIT-ADJUSTED denominator. Multiply the as-traded frame by this and both sides are in
    today's share basis.

    ⚠ THIS IS NOT AN OPTIONAL REFINEMENT, and the case that proved it is `ic_shortvol_market_
    coverage`. FINRA's RegSHO volume is as-traded and never restated, while yfinance's `Volume`
    IS retroactively scaled by the split ratio -- measured over 746 splits, median post/pre
    volume ratio 0.869 against median R = 2.0, i.e. price / R and volume x R cancel exactly. So
    a plain `regsho_total / yf_volume` carries a factor of `1 / F(d)`, and on a REVERSE split
    (R < 1) it inflates: on the 2026-09-12 build, GE's 1-for-8 gave a "coverage" of 2.26 and
    WTW 3.71, against a universe median of 0.28. All six breaching names were corporate-action
    names, and GE's breach window ends exactly at its split date.

    Returns a frame of 1.0 when there are no splits, so the caller applies it unconditionally.
    """
    if frame is None or frame.empty:
        return frame
    if splits is None or splits.empty:
        return pd.DataFrame(1.0, index=frame.index, columns=frame.columns)
    dates = np.repeat(np.asarray(frame.index), frame.shape[1])
    tickers = np.tile(np.asarray(frame.columns, dtype=object), len(frame.index))
    factor = future_split_factor(splits, pd.Series(tickers), pd.Series(dates))
    return pd.DataFrame(factor.reshape(len(frame.index), frame.shape[1]),
                        index=frame.index, columns=frame.columns)
