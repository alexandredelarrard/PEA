"""
decay.py  (src/data_aggregate/utils/institutionals/decay.py)
-----------------------------------------------------------
Turn a sparse EVENT LOG into a dense daily state variable by exponential decay:

    value(t) = sum over events e <= t of   magnitude_e * 0.5 ** (days_since(e, t) / halflife)

WHY THIS EXISTS. Most informed-capital signals are events, not levels: a 13D is filed, an
insider buys, a manager opens a position. Fed to the panel builder raw, an event becomes a flag
that is 0 on ~99.9% of ticker-days. `peer_relative` returns NaN for any basket with zero
dispersion -- a deliberate policy, see `common/panel.py` -- and a basket of all-zeros has
exactly that. So `f_<flag>_vs_peers` would be ~100% NaN: an ABSENT feature, not a weak one.
Decay converts the event into a quantity that still differs across a peer basket weeks later,
which is the only form these signals survive peer-relativization in.

THREE PROPERTIES THAT ARE DECISIONS, NOT DETAILS

  * TRADING DAYS, not calendar days. The panel grid is trading days, so a half-life of 63 means
    "one quarter of trading", and a decay step is one row of the grid. Using calendar days would
    make the same half-life mean different things across a holiday-heavy stretch.

  * SUMMED over events. Two 13D amendments a month apart STACK -- the second adds to what is
    left of the first rather than overwriting it. Escalating campaigns are the signal; a
    last-event-wins rule would read a doubling-down as no news.

  * NaN, NEVER 0, before a ticker's first-ever event. "No 13D has ever been filed on this name"
    and "a 13D was filed in 2015 and has fully decayed" are different facts, and only the second
    is evidence about this company's shareholder base. Zero-filling the first would let the
    model read a never-targeted name as a fully-recovered one. After the first event the series
    is dense and finite -- it asymptotes toward 0 but never re-enters the NaN state.

IMPLEMENTATION. The accumulation runs as a first-order recursion down the date axis,
`v[t] = v[t-1] * 0.5**(1/halflife) + new[t]`, one vectorised numpy row-op per trading day over
all tickers at once. Two alternatives were rejected:
  * a per-event outer product (events x days x tickers) does not fit in memory at 500 tickers
    x 15 years;
  * the closed form `d**t * cumsum(new * d**-t)` is one vectorised pass, but `d**-t` reaches
    2**60 over a 15-year grid at halflife=63, and summing terms across 18 orders of magnitude
    loses the early history to floating-point cancellation. The recursion is exact and, at
    ~3,800 iterations over a 500-wide float64 row, still runs in well under a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decay_events(
    events: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    halflife: float,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    magnitude_col: str | None = None,
) -> pd.DataFrame:
    """Exponentially-decayed running total of `events`, on the `trading_index` grid.

    Returns a wide (date x ticker) float frame ready to hand to `build_peer_relative_panel`.

    `magnitude_col=None` means every event counts 1.0 -- the right default for a count-style
    signal ("a 13D was filed"). Pass a column to weight by size ($ bought, % of shares).

    An event dated on a non-trading day (a weekend filing, a holiday) lands on the NEXT trading
    day: that is the first day the information could be acted on, so rounding it backwards would
    be a look-ahead. Events after the end of the grid are dropped, not clamped onto the last
    day, for the same reason -- the grid has not reached them yet.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife!r}")
    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if events is None or events.empty or idx.empty:
        return pd.DataFrame(index=idx, dtype="float64")

    ev = events.loc[:, [c for c in (date_col, ticker_col, magnitude_col) if c]].copy()
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce").dt.normalize()
    ev = ev.dropna(subset=[date_col, ticker_col])
    if magnitude_col:
        ev["_m"] = pd.to_numeric(ev[magnitude_col], errors="coerce")
        # A NaN magnitude is an event whose SIZE is unknown, not an event that did not happen.
        # Dropping the row would erase the occurrence; counting it as 1.0 keeps the event in the
        # series on the same footing as an unweighted one.
        ev["_m"] = ev["_m"].fillna(1.0)
    else:
        ev["_m"] = 1.0

    # Snap each event onto the first trading day >= its date. `searchsorted` gives that index
    # directly; anything landing past the end of the grid is a future event and is dropped.
    pos = idx.searchsorted(ev[date_col].to_numpy(), side="left")
    keep = pos < len(idx)
    ev, pos = ev.loc[keep], pos[keep]
    if ev.empty:
        return pd.DataFrame(index=idx, dtype="float64")

    tickers = pd.Index(sorted(ev[ticker_col].astype(str).unique()), name=ticker_col)
    col = tickers.get_indexer(ev[ticker_col].astype(str))
    new = np.zeros((len(idx), len(tickers)), dtype="float64")
    # `np.add.at` (unbuffered) is what makes two events on the SAME (day, ticker) sum instead of
    # the second overwriting the first -- plain fancy-index assignment would keep only the last.
    np.add.at(new, (pos, col), ev["_m"].to_numpy(dtype="float64"))

    step = 0.5 ** (1.0 / halflife)
    out = np.empty_like(new)
    running = np.zeros(len(tickers), dtype="float64")
    for i in range(len(idx)):
        running = running * step + new[i]
        out[i] = running

    frame = pd.DataFrame(out, index=idx, columns=tickers)
    # Mask everything strictly BEFORE each ticker's first event. `cumsum(new) > 0` is true from
    # the first event onward, so its negation is exactly the never-yet-happened region. Done on
    # the event matrix rather than on `out > 0` because a decayed value can underflow to 0.0
    # while the event history is real -- that day must stay a number, not become NaN again.
    frame = frame.where(np.cumsum(new, axis=0) > 0)
    frame.index.name = None
    return frame
