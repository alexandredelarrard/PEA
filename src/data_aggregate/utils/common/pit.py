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


def daily_market_cap(fundamentals_history: pd.DataFrame,
                     close_split: pd.DataFrame,
                     *, level_factor: pd.DataFrame | None) -> pd.DataFrame:
    """Historical daily market cap = ffilled `sharesOutstanding` x `close_split` x `S(d)`.

    ⚠ BOTH PRICE LEGS MUST BE SPLIT-ADJUSTED, and the parameter is named for it. The vendor
    back-fills `sharesbas` to today's basis (`sharesbas(d) = real_shares(d) x F(d)`) and
    Yahoo restates `Close` to the same one (`close_split(d) = raw_price(d) / F(d)`), so the
    future-SPLIT factor CANCELS IDENTICALLY in the product.

    Handing it `close_total` instead reintroduces the defect this signature exists to
    prevent: nothing in a share count carries a dividend factor, so `D(d)` survives into the
    product -- median 0.618 in 2003, i.e. market cap 38% too low, and monotone in FUTURE
    dividends. Handing it a de-adjusted share count breaks the cancellation the other way.

    ⚠ `level_factor` IS REQUIRED AND KEYWORD-ONLY, deliberately, because the cancellation
    above holds for splits and FAILS for SPINOFFS. Yahoo back-adjusts `Close` across a
    spinoff and a spinoff does not change the share count, so that factor does not cancel and
    the product comes out LOW by exactly `S(d)`. Measured on FDX 2020-12-17:

        close_split 235.5036 x 265,070,592 shares  = $62.425bn
        x S = 1.241                                = $77.470bn
        Sharadar's own `marketcap`                 = $77.470bn

    GE reads 40% low before 2019 and DD 80% low before 2019 without it. There is no default:
    a caller that has not thought about the basis must get a `TypeError`, not a number that
    is quietly wrong for 80 of 491 tickers. Pass `None` ONLY to mean "S is 1.0 everywhere",
    which is true for a synthetic fixture and for ~89% of real cells but never for the
    universe as a whole.

    Requires a `sharesOutstanding` column (the VENDOR-basis one, not `sharesOutstandingPit`).
    """
    shares = fundamentals_to_daily(fundamentals_history, "sharesOutstanding",
                                   close_split.index)
    if shares.empty:
        return pd.DataFrame(index=close_split.index)

    cols = [c for c in shares.columns if c in close_split.columns]
    if not cols:
        return pd.DataFrame(index=close_split.index)

    mcap = close_split[cols].mul(shares[cols])
    if level_factor is not None and not level_factor.empty:
        # Reindexed rather than assumed aligned: `level_factor` spans the whole universe
        # while `cols` is the intersection with the filing history. `fillna(1.0)` because a
        # missing factor means "no adjustment", never "no market cap" -- the alternative
        # silently NULLs a ticker's entire series.
        mcap = mcap.mul(level_factor.reindex(index=mcap.index, columns=cols).fillna(1.0))
    return mcap.where(mcap > 0)


#: The two growth ratios `kpi_catalogue.CUBE_TIME_COLUMNS` declares are computed HERE rather
#: than by the history build, mapped to the TTM level each is the growth of. Duplicated as a
#: literal instead of imported: `src/data_aggregate/` must not import from
#: `src/data_extract/`, and the catalogue owns the declaration, not the arithmetic.
_CUBE_TIME_GROWTH: dict[str, str] = {
    "revenueGrowth": "totalRevenue",
    "earningsGrowth": "netIncome",
}

#: How far from the 365-day target a filing may sit and still count as "one year ago".
#:
#: The match is NEAREST, not backward, and the difference is measured: AAPL's 2026-07-31 row
#: has filings at 2025-08-01 (target + 1 day) and 2025-05-02 (target - 90). A backward match
#: is forced to skip the one that is one day too late and compare TTM revenue 455 days apart,
#: which is not a year-over-year growth. Nearest picks 2025-08-01.
#:
#: Nearest stays leak-free: the worst-case match is `as_of - 365 + 180 = as_of - 185` days,
#: always strictly before the row's own filing date, so both legs are public on `as_of`.
#: 180 days also still refuses to call a multi-year gap a YoY comparison -- the defect a bare
#: `shift(4)` cannot even detect.
_YOY_TOLERANCE_DAYS = 180


def add_cube_time_growth(fund_hist: pd.DataFrame) -> pd.DataFrame:
    """`fundamentals_history` + the `revenueGrowth` / `earningsGrowth` columns, row-level.

    THE OFFSET IS THE WHOLE POINT, and it is why these two are computed at cube time rather
    than by the history build (`kpi_catalogue.CUBE_TIME_COLUMNS`). Growth is measured against
    the filing NEAREST 365 CALENDAR DAYS back, found per ticker by an as-of match. The
    history build could only take a 4-ROW offset, and under the publication-event grain an
    amendment row makes four rows ~9 months rather than 12 -- so the denominator would be the
    wrong quarter for exactly the names that restate.

    Point-in-time by construction: both legs are filings already public on their own `as_of`,
    and the result is keyed on the LATER of the two, so `fundamentals_to_daily` forward-fills
    it from the day the new filing landed. No look-ahead.

    Returns a COPY. Absent source column, or a history with no `as_of`, leaves the frame
    unchanged rather than emitting an all-NaN column -- an all-NaN column reads downstream as
    "this firm did not grow" instead of "growth was never computed".
    """
    out = fund_hist.copy()
    if "as_of" not in out.columns or "ticker" not in out.columns or out.empty:
        return out

    # ⚠ `[ns]` EXPLICITLY, not just `to_datetime`. Postgres DATE columns come back as
    # `datetime64[s]`, and subtracting a Timedelta below promotes that to `[us]` -- so the
    # two sides of the `merge_asof` end up on DIFFERENT RESOLUTIONS and pandas raises
    # `MergeError: incompatible merge keys`. Reading the same rows out of a parquet or a CSV
    # gives `[ns]` on both sides and hides it entirely, which is why this only reproduces
    # against the live store.
    as_of = pd.to_datetime(out["as_of"], errors="coerce").astype("datetime64[ns]")
    for name, source in _CUBE_TIME_GROWTH.items():
        if source not in out.columns:
            continue
        base = pd.DataFrame({
            "ticker": out["ticker"].astype(str),
            "as_of": as_of,
            "level": pd.to_numeric(out[source], errors="coerce"),
        }).dropna(subset=["ticker", "as_of"])
        if base.empty:
            continue
        # One row per (ticker, as_of): an amendment republishes the same publication date,
        # and merge_asof would otherwise match against whichever duplicate sorted last.
        base = (base.sort_values(["ticker", "as_of"])
                    .drop_duplicates(["ticker", "as_of"], keep="last"))
        right = base.rename(columns={"as_of": "prior_as_of", "level": "prior"})
        left = base.assign(target=base["as_of"] - pd.Timedelta(days=365))

        matched = pd.merge_asof(
            left.sort_values("target"), right.sort_values("prior_as_of"),
            left_on="target", right_on="prior_as_of", by="ticker",
            direction="nearest", tolerance=pd.Timedelta(days=_YOY_TOLERANCE_DAYS),
        )
        # A zero or negative prior makes the ratio meaningless, not infinite: a swing from a
        # loss to a profit has no percentage growth, and dividing by it manufactures a huge
        # number with an arbitrary sign that then dominates every z-score it reaches.
        prior = matched["prior"].where(matched["prior"] > 0)
        growth = (matched["level"] / prior - 1.0).replace([np.inf, -np.inf], np.nan)
        keyed = pd.Series(growth.to_numpy(),
                          index=pd.MultiIndex.from_arrays(
                              [matched["ticker"].to_numpy(), matched["as_of"].to_numpy()]))
        out[name] = pd.MultiIndex.from_arrays(
            [out["ticker"].astype(str), as_of]).map(keyed)
    return out


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
                 close: pd.DataFrame | None = None,
                 level_factor: pd.DataFrame | None = None) -> None:
        self._history = history
        self._index = trading_index
        self._close = close
        #: `S(d)`, forwarded to `daily_market_cap`. Positional-optional here rather than
        #: required as it is there: `PitFrames` is also built for histories with no price at
        #: all, where `market_cap` is never reached and demanding a basis would be noise.
        self._level_factor = level_factor
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
        """Memoized `daily_market_cap(history, close, level_factor=...)`. Empty when either
        input is absent, matching `daily_market_cap`'s own empty-frame contract."""
        if self._market_cap is None:
            if self.empty or self._close is None or self._close.empty:
                self._market_cap = pd.DataFrame(index=self._index)
            else:
                self._market_cap = daily_market_cap(
                    self._history, self._close, level_factor=self._level_factor)
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
