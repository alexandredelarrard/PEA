"""
pit.py  (src/data_aggregate/utils/common/pit.py)
-----------------------------------------------
The POINT-IN-TIME layer: turn a `(ticker, as_of, <fields>)` filing history into daily
wide frames, forward-filled so the value on date d is the most recent filing already
public on d. Every fundamentals-derived feature in the cube goes through here.

Why these live in `common/` rather than next to the fundamentals builders:

  * `fundamentals_to_daily` / `daily_market_cap` were in `utils/target/factors.py`, which
    made the FACTOR-RETURN module a dependency of nine unrelated feature builders that
    only wanted a point-in-time pivot.
  * `infer_yoy_periods` / `fiscal_change_to_daily` / `fiscal_apply_to_daily` were private
    helpers of the 1497-line `fundamental_features.py`, and `sector_features` and
    `governance_features` reached UPWARD into it to borrow them -- importing the whole
    fundamentals monolith for two pivots. They are not fundamentals-specific: they are
    already applied to three different histories (`fundamentals_history`, `def14a_llm`,
    and the sector KPI frame), which is the proof.

Both imports are now gone and this module is a leaf (numpy/pandas only).

`PitFrames` is the memoizing accessor. `fundamentals_to_daily` is a pure function of
(frame, field, index), and in one cube sub-step every builder is handed the SAME
(history, trading_index, close) triple -- so `sharesOutstanding` was re-pivoted ~7 times
and the daily market cap recomputed 6 times per run, identically. `PitFrames` computes
each once. Sharing it changes no number; see `tests/data_aggregate/test_pit_cache.py`.
"""
from __future__ import annotations

from typing import Callable, Literal, Protocol

import numpy as np
import pandas as pd


class FieldGetter(Protocol):
    """field name -> the field's values.

    THE accessor protocol `capital.py` is written against: it composes debt / net-debt /
    invested-capital out of whatever tags are present, without caring whether it is being
    handed filing-row Series (`sector_features._col`) or daily date x ticker frames
    (`fundamental_features`'s memoized `daily`, and `PitFrames.__call__`). Was the string
    literal `Getter = "callable"` in capital.py, which documented the idea without
    expressing it."""

    def __call__(self, field: str) -> pd.Series | pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# pure functions                                                               #
# --------------------------------------------------------------------------- #
def fundamentals_to_daily(
    fundamentals_history: pd.DataFrame,
    field: str,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Turn a (ticker, as_of, <fields>) history into a daily wide frame for one
    field, forward-filled point-in-time: value on date d is the most recent
    as_of <= d. No look-ahead.
    """
    if field not in fundamentals_history.columns:
        return pd.DataFrame(index=trading_index)
    df = fundamentals_history[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    wide = df.pivot_table(index="as_of", columns="ticker", values=field, aggfunc="last")
    wide = wide.sort_index().reindex(
        wide.index.union(trading_index)
    ).ffill().reindex(trading_index)
    return wide


def daily_market_cap(fundamentals_history: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Historical daily market cap = point-in-time shares outstanding (from SEC,
    forward-filled) * daily close. This is the correct historical mcap (moves
    with price every day), replacing the old current-mcap*price-ratio proxy.
    Requires a 'sharesOutstanding' column in the fundamentals history.
    """

    shares = fundamentals_to_daily(fundamentals_history, "sharesOutstanding", close.index)
    if shares.empty:
        return pd.DataFrame(index=close.index)

    cols = [c for c in shares.columns if c in close.columns]
    if not cols:
        return pd.DataFrame(index=close.index)

    mcap = close[cols].mul(shares[cols])
    return mcap.where(mcap > 0)


def infer_yoy_periods(fund_hist: pd.DataFrame) -> int:
    """Number of filing periods that make up one year, from the median gap
    between consecutive `as_of` dates. Quarterly history -> 4, annual -> 1.
    Used so growth is always a true year-over-year comparison (no seasonality)
    regardless of the reporting cadence."""
    if "as_of" not in fund_hist.columns or fund_hist.empty:
        return 1
    d = fund_hist[["ticker", "as_of"]].copy()
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    gaps = d.sort_values(["ticker", "as_of"]).groupby("ticker")["as_of"].diff().dt.days
    med = gaps.median()
    if not np.isfinite(med) or med <= 0:
        return 1
    return int(min(4, max(1, round(365.0 / med))))


def fiscal_change_to_daily(
    fund_hist: pd.DataFrame,
    field: str,
    idx: pd.DatetimeIndex,
    kind: str = "pct",
    periods: int = 1,
) -> pd.DataFrame:
    """Change of a fiscal field over `periods` filings, forward-filled onto
    trading days. With `periods` = one year of filings this is a seasonality-free
    year-over-year change.

    Computed per ticker on ITS OWN fiscal series (ordered by filing date), then
    ffilled point-in-time so the change lands on the day the new filing is
    public. `kind='pct'` -> relative growth; `kind='diff'` -> absolute change
    (use for ratios like margins).
    """

    if field not in fund_hist.columns:
        return pd.DataFrame(index=idx)
    df = fund_hist[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    if df.empty:
        return pd.DataFrame(index=idx)

    grp = df.groupby("ticker")[field]
    if kind == "pct":
        df["chg"] = grp.pct_change(periods=periods)
    elif kind == "diff":
        df["chg"] = grp.diff(periods=periods)
    else:
        raise ValueError("kind must be 'pct' or 'diff'")

    wide = df.pivot_table(index="as_of", columns="ticker", values="chg", aggfunc="last")
    wide = wide.replace([np.inf, -np.inf], np.nan).sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


def fiscal_apply_to_daily(fund_hist, field, idx, func) -> pd.DataFrame:
    """Apply a per-ticker series transform (e.g. YoY growth, or acceleration =
    change in YoY) to a fiscal field, forward-filled point-in-time onto trading
    days. `func` receives one ticker's chronological series and returns a series
    of the same length."""
    if field not in fund_hist.columns:
        return pd.DataFrame(index=idx)
    df = fund_hist[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    if df.empty:
        return pd.DataFrame(index=idx)
    df["v"] = df.groupby("ticker")[field].transform(func)
    wide = df.pivot_table(index="as_of", columns="ticker", values="v", aggfunc="last")
    wide = wide.replace([np.inf, -np.inf], np.nan).sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


# --------------------------------------------------------------------------- #
# the memoizing accessor                                                       #
# --------------------------------------------------------------------------- #
class PitFrames:
    """ONE point-in-time view of a filing history, memoized per field.

    Built once per cube sub-step and passed to every builder that reads the SAME
    (history, trading_index, close) triple, so `sharesOutstanding` is pivoted once
    instead of ~7 times and the daily market cap built once instead of 6 times.

    Satisfies `FieldGetter` via `__call__`, so it drops straight into `capital.py`'s
    accessor protocol and into `fundamental_features`'s `daily(...)` call sites with no
    other change.

    A None / empty history is allowed and yields empty frames, exactly as
    `fundamentals_to_daily` does for an absent field -- so a caller's own
    `if history is None` guard keeps behaving as before.
    """

    def __init__(self, history: pd.DataFrame | None, trading_index: pd.DatetimeIndex,
                 close: pd.DataFrame | None = None) -> None:
        self._history = history
        self._index = trading_index
        self._close = close
        self._daily: dict[str, pd.DataFrame] = {}
        self._changes: dict[tuple[str, str, int], pd.DataFrame] = {}
        self._applied: dict[tuple[str, str], pd.DataFrame] = {}
        self._market_cap: pd.DataFrame | None = None
        self._yoy: int | None = None
        self._accesses = 0

    # ---- state ---- #
    @property
    def empty(self) -> bool:
        return self._history is None or self._history.empty

    @property
    def trading_index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def history(self) -> pd.DataFrame | None:
        return self._history

    # ---- accessors ---- #
    def daily(self, field: str) -> pd.DataFrame:
        """Memoized `fundamentals_to_daily(history, field, trading_index)`."""
        self._accesses += 1
        if field not in self._daily:
            self._daily[field] = (
                pd.DataFrame(index=self._index) if self.empty
                else fundamentals_to_daily(self._history, field, self._index))
        return self._daily[field]

    def __call__(self, field: str) -> pd.DataFrame:
        """`FieldGetter` alias, so a `PitFrames` can be passed anywhere `capital.py`
        or `_derived_fields` expects a `daily`-style accessor."""
        return self.daily(field)

    def change(self, field: str, kind: Literal["pct", "diff"] = "pct",
               periods: int | None = None) -> pd.DataFrame:
        """Memoized `fiscal_change_to_daily`. `periods=None` uses this history's own
        filing cadence (`yoy_periods`), which is what a year-over-year change means."""
        n = self.yoy_periods if periods is None else int(periods)
        key = (field, kind, n)
        if key not in self._changes:
            self._changes[key] = (
                pd.DataFrame(index=self._index) if self.empty
                else fiscal_change_to_daily(self._history, field, self._index,
                                            kind=kind, periods=n))
        return self._changes[key]

    def applied(self, field: str, key: str,
                func: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
        """Memoized `fiscal_apply_to_daily`. `key` names the transform, since a
        callable (often a lambda) is not a usable cache key."""
        ck = (field, key)
        if ck not in self._applied:
            self._applied[ck] = (
                pd.DataFrame(index=self._index) if self.empty
                else fiscal_apply_to_daily(self._history, field, self._index, func))
        return self._applied[ck]

    # ---- derived ---- #
    @property
    def market_cap(self) -> pd.DataFrame:
        """Memoized `daily_market_cap(history, close)`. Empty when either input is
        absent, matching `daily_market_cap`'s own empty-frame contract."""
        if self._market_cap is None:
            if self.empty or self._close is None or self._close.empty:
                self._market_cap = pd.DataFrame(index=self._index)
            else:
                self._market_cap = daily_market_cap(self._history, self._close)
        return self._market_cap

    @property
    def yoy_periods(self) -> int:
        """Memoized `infer_yoy_periods(history)`."""
        if self._yoy is None:
            self._yoy = 1 if self.empty else infer_yoy_periods(self._history)
        return self._yoy

    def has(self, field: str) -> bool:
        """True when the field is present in the history AND has at least one value."""
        if self.empty or field not in self._history.columns:
            return False
        return bool(pd.to_numeric(self._history[field], errors="coerce").notna().any())

    # ---- guards + diagnostics ---- #
    def assert_matches(self, trading_index: pd.DatetimeIndex,
                       close: pd.DataFrame | None = None) -> None:
        """Refuse to serve a cache built on a different window. Cheap: compares the
        index and the close frame's shape/columns, not their values."""
        if not self._index.equals(trading_index):
            raise ValueError(
                f"PitFrames was built on a {len(self._index)}-day index "
                f"({self._index.min()}..{self._index.max()}) but is being used with a "
                f"{len(trading_index)}-day one -- build one cache per warm-up window")
        if close is not None and self._close is not None:
            if (self._close.shape != close.shape
                    or not self._close.columns.equals(close.columns)):
                raise ValueError("PitFrames was built on a different `close` frame "
                                 f"({self._close.shape} vs {close.shape})")

    def stats(self) -> dict[str, int]:
        """What the cache actually collapsed, so the sub-step can log it and the tests
        can assert it: distinct fields pivoted vs total accesses."""
        computed = len(self._daily)
        return {"fields": computed, "accesses": self._accesses,
                "hits": max(0, self._accesses - computed),
                "changes": len(self._changes), "applied": len(self._applied),
                "market_cap": int(self._market_cap is not None)}
