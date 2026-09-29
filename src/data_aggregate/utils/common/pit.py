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

from collections.abc import Callable
from typing import Literal, Protocol

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
def null_invalid_gross_profit_sentinels(fund_hist: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the provider's impossible zero-COGS sentinel nulled.

    Positive revenue, zero cost of revenue, and a gross margin of exactly one is
    not a usable accounting observation.  Null all three mutually dependent
    fields so downstream ratios cannot turn that source sentinel into a perfect
    gross-margin or profitability signal.
    """
    out = fund_hist.copy()
    required = {"totalRevenue", "costOfRevenue", "grossMargins"}
    if not required.issubset(out.columns):
        return out
    revenue = pd.to_numeric(out["totalRevenue"], errors="coerce")
    cost = pd.to_numeric(out["costOfRevenue"], errors="coerce")
    margin = pd.to_numeric(out["grossMargins"], errors="coerce")
    invalid = revenue.gt(0) & cost.eq(0) & margin.eq(1)
    columns = [column for column in ("costOfRevenue", "grossProfit", "grossMargins") if column in out.columns]
    out.loc[invalid, columns] = np.nan
    return out


def fundamentals_to_daily(
    fundamentals_history: pd.DataFrame,
    field: str,
    trading_index: pd.DatetimeIndex,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """
    Turn a (ticker, as_of, <fields>) history into a daily wide frame for one
    field, forward-filled point-in-time: value on date d is the most recent
    as_of <= d. No look-ahead.
    """
    if field not in fundamentals_history.columns:
        return pd.DataFrame(index=trading_index)
    df = fundamentals_history[["ticker", "as_of", field]].copy()
    return _observations_to_daily(df, field, trading_index, max_age_days=max_age_days)


def _observations_to_daily(
    observations: pd.DataFrame,
    value_col: str,
    trading_index: pd.DatetimeIndex,
    *,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Project producing observations to daily rows, optionally expiring their age."""
    df = observations[["ticker", "as_of", value_col]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.dropna(subset=["ticker", "as_of", value_col])
    if df.empty:
        return pd.DataFrame(index=trading_index)
    df = df.sort_values(["ticker", "as_of"]).drop_duplicates(["ticker", "as_of"], keep="last")
    wide = df.pivot(index="as_of", columns="ticker", values=value_col).sort_index()
    union = wide.index.union(trading_index)
    daily = wide.reindex(union).ffill().reindex(trading_index)
    if max_age_days is None:
        return daily

    produced = df.assign(_produced_at=df["as_of"]).pivot(index="as_of", columns="ticker", values="_produced_at")
    produced = produced.sort_index().reindex(union).ffill().reindex(trading_index)
    current = pd.Series(trading_index, index=trading_index)
    horizon = pd.Timedelta(days=int(max_age_days))
    for ticker in daily.columns:
        age = current - produced[ticker]
        daily[ticker] = daily[ticker].where(age <= horizon)
    return daily


def daily_market_cap(
    fundamentals_history: pd.DataFrame,
    close_split: pd.DataFrame,
    *,
    level_factor: pd.DataFrame | None,
    max_age_days: int | None = None,
) -> pd.DataFrame:
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
    shares = fundamentals_to_daily(
        fundamentals_history,
        "sharesOutstanding",
        close_split.index,
        max_age_days=max_age_days,
    )
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


def add_cube_time_growth(fund_hist: pd.DataFrame) -> pd.DataFrame:
    """`fundamentals_history` + the `revenueGrowth` / `earningsGrowth` columns, row-level.

    Growth is measured against the same fiscal period one year earlier (within 45 days),
    using only amendments already public on the current row's `as_of`. Publication-date or
    row-count offsets are not fiscal-period definitions.

    Point-in-time by construction: both legs are filings already public on their own `as_of`,
    and the result is keyed on the LATER of the two, so `fundamentals_to_daily` forward-fills
    it from the day the new filing landed. No look-ahead.

    Returns a COPY. Absent source column, or a history with no `as_of`, leaves the frame
    unchanged rather than emitting an all-NaN column -- an all-NaN column reads downstream as
    "this firm did not grow" instead of "growth was never computed".
    """
    out = fund_hist.copy()
    required = {"ticker", "as_of", "fiscal_end"}
    if not required.issubset(out.columns) or out.empty:
        return out
    positions = fiscal_prior_positions(out, years=1, tolerance_days=45)
    for name, source in _CUBE_TIME_GROWTH.items():
        if source not in out.columns:
            continue
        level = pd.to_numeric(out[source], errors="coerce")
        prior = fiscal_prior_values(
            out,
            level,
            years=1,
            tolerance_days=45,
            positions=positions,
        )
        out[name] = (level / prior.where(prior > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
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


def fiscal_prior_positions(
    fund_hist: pd.DataFrame,
    *,
    years: int = 1,
    tolerance_days: int = 45,
) -> pd.Series:
    """Positional index of each row's point-in-time fiscal predecessor, or -1."""
    required = {"ticker", "as_of", "fiscal_end"}
    if not required.issubset(fund_hist.columns) or fund_hist.empty:
        return pd.Series(-1, index=fund_hist.index, dtype="int64")
    work = pd.DataFrame(
        {
            "ticker": fund_hist["ticker"].astype(str),
            "as_of": pd.to_datetime(fund_hist["as_of"], errors="coerce"),
            "fiscal_end": pd.to_datetime(fund_hist["fiscal_end"], errors="coerce"),
            "position": np.arange(len(fund_hist)),
        }
    )
    result = np.full(len(fund_hist), -1, dtype="int64")
    tolerance_ns = pd.Timedelta(days=int(tolerance_days)).value
    for _, group in work.groupby("ticker", sort=False):
        positions = group["position"].to_numpy(dtype="int64")
        as_of = group["as_of"].to_numpy(dtype="datetime64[ns]")
        fiscal_end = group["fiscal_end"].to_numpy(dtype="datetime64[ns]")
        valid = ~np.isnat(as_of) & ~np.isnat(fiscal_end)
        as_of_ns = as_of.astype("int64")
        fiscal_end_ns = fiscal_end.astype("int64")
        for offset in np.flatnonzero(valid):
            target_ns = (pd.Timestamp(fiscal_end[offset]) - pd.DateOffset(years=int(years))).value
            eligible = np.flatnonzero(valid & (as_of_ns <= as_of_ns[offset]) & (fiscal_end_ns < fiscal_end_ns[offset]))
            if not len(eligible):
                continue
            distance = np.abs(fiscal_end_ns[eligible] - target_ns)
            eligible = eligible[distance <= tolerance_ns]
            distance = distance[distance <= tolerance_ns]
            if not len(eligible):
                continue
            closest = eligible[distance == distance.min()]
            latest = closest[as_of_ns[closest] == as_of_ns[closest].max()]
            result[positions[offset]] = positions[latest].max()
    return pd.Series(result, index=fund_hist.index, dtype="int64")


def fiscal_prior_values(
    fund_hist: pd.DataFrame,
    field: str | pd.Series,
    *,
    years: int = 1,
    tolerance_days: int = 45,
    positions: pd.Series | None = None,
) -> pd.Series:
    """Prior fiscal value aligned to each input row and limited to then-public data.

    Matching is by `(ticker, fiscal_end)` rather than publication-row count. Amendments of
    the selected prior period are eligible only when their `as_of` is no later than the
    current row, and the latest eligible amendment wins.
    """
    required = {"ticker", "as_of", "fiscal_end"}
    if not required.issubset(fund_hist.columns) or fund_hist.empty:
        return pd.Series(np.nan, index=fund_hist.index, dtype="float64")
    if isinstance(field, str):
        values = pd.to_numeric(fund_hist[field], errors="coerce")
    else:
        if len(field) != len(fund_hist):
            raise ValueError("fiscal value series must be row-aligned with fund_hist")
        values = pd.Series(
            pd.to_numeric(field.to_numpy(), errors="coerce"),
            index=fund_hist.index,
        )
    matches = (
        fiscal_prior_positions(
            fund_hist,
            years=years,
            tolerance_days=tolerance_days,
        )
        if positions is None
        else positions
    )
    result = np.full(len(fund_hist), np.nan, dtype="float64")
    match_array = matches.to_numpy(dtype="int64")
    valid = match_array >= 0
    source = values.to_numpy(dtype="float64", na_value=np.nan)
    result[valid] = source[match_array[valid]]
    return pd.Series(result, index=fund_hist.index, dtype="float64")


def fiscal_prior_to_daily(
    fund_hist: pd.DataFrame,
    field: str | pd.Series,
    idx: pd.DatetimeIndex,
    *,
    years: int = 1,
    tolerance_days: int = 45,
    max_age_days: int | None = None,
    positions: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily PIT projection of the row-aligned fiscal predecessor."""
    prior = fiscal_prior_values(
        fund_hist,
        field,
        years=years,
        tolerance_days=tolerance_days,
        positions=positions,
    )
    observations = fund_hist[["ticker", "as_of"]].assign(_prior=prior)
    return _observations_to_daily(
        observations,
        "_prior",
        idx,
        max_age_days=max_age_days,
    )


def fiscal_values_to_daily(
    fund_hist: pd.DataFrame,
    values: pd.Series,
    idx: pd.DatetimeIndex,
    *,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Project row-aligned fiscal values from each row's publication date."""
    if len(values) != len(fund_hist):
        raise ValueError("fiscal value series must be row-aligned with fund_hist")
    observations = fund_hist[["ticker", "as_of"]].assign(_value=pd.to_numeric(values.to_numpy(), errors="coerce"))
    return _observations_to_daily(
        observations,
        "_value",
        idx,
        max_age_days=max_age_days,
    )


def fiscal_change_values(
    fund_hist: pd.DataFrame,
    field: str,
    kind: str = "pct",
    periods: int = 1,
    *,
    years: int | None = None,
    tolerance_days: int = 45,
    positions: pd.Series | None = None,
) -> pd.Series:
    """Row-aligned fiscal change, with a filing-count fallback only without fiscal dates."""
    if field not in fund_hist.columns:
        return pd.Series(np.nan, index=fund_hist.index, dtype="float64")
    current = pd.to_numeric(fund_hist[field], errors="coerce")
    has_fiscal_end = "fiscal_end" in fund_hist.columns and fund_hist["fiscal_end"].notna().any()
    if has_fiscal_end:
        match_years = years
        if match_years is None:
            cadence = infer_yoy_periods(fund_hist)
            match_years = max(1, int(round(int(periods) / cadence)))
        prior = fiscal_prior_values(
            fund_hist,
            current,
            years=match_years,
            tolerance_days=tolerance_days,
            positions=positions,
        )
    else:
        work = pd.DataFrame(
            {
                "ticker": fund_hist["ticker"].astype(str),
                "as_of": pd.to_datetime(fund_hist["as_of"], errors="coerce"),
                "value": current,
                "position": np.arange(len(fund_hist)),
            }
        ).sort_values(["ticker", "as_of", "position"])
        work["prior"] = work.groupby("ticker", sort=False)["value"].shift(periods=int(periods))
        prior_values = np.full(len(fund_hist), np.nan, dtype="float64")
        prior_values[work["position"].to_numpy(dtype="int64")] = work["prior"].to_numpy(dtype="float64", na_value=np.nan)
        prior = pd.Series(prior_values, index=fund_hist.index)
    if kind == "pct":
        result = current / prior.where(prior != 0) - 1.0
    elif kind == "diff":
        result = current - prior
    else:
        raise ValueError("kind must be 'pct' or 'diff'")
    return result.replace([np.inf, -np.inf], np.nan)


def fiscal_change_to_daily(
    fund_hist: pd.DataFrame,
    field: str,
    idx: pd.DatetimeIndex,
    kind: str = "pct",
    periods: int = 1,
    *,
    years: int | None = None,
    tolerance_days: int = 45,
    max_age_days: int | None = None,
    positions: pd.Series | None = None,
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
    if fund_hist.empty:
        return pd.DataFrame(index=idx)
    values = fiscal_change_values(
        fund_hist,
        field,
        kind=kind,
        periods=periods,
        years=years,
        tolerance_days=tolerance_days,
        positions=positions,
    )
    return fiscal_values_to_daily(
        fund_hist,
        values,
        idx,
        max_age_days=max_age_days,
    )


def fiscal_apply_to_daily(
    fund_hist: pd.DataFrame,
    field: str,
    idx: pd.DatetimeIndex,
    func: Callable[[pd.Series], pd.Series],
    *,
    max_age_days: int | None = None,
) -> pd.DataFrame:
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
    df["v"] = df["v"].replace([np.inf, -np.inf], np.nan)
    return _observations_to_daily(df, "v", idx, max_age_days=max_age_days)


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

    def __init__(
        self,
        history: pd.DataFrame | None,
        trading_index: pd.DatetimeIndex,
        close: pd.DataFrame | None = None,
        level_factor: pd.DataFrame | None = None,
    ) -> None:
        self._history = history
        self._index = trading_index
        self._close = close
        #: `S(d)`, forwarded to `daily_market_cap`. Positional-optional here rather than
        #: required as it is there: `PitFrames` is also built for histories with no price at
        #: all, where `market_cap` is never reached and demanding a basis would be noise.
        self._level_factor = level_factor
        self._daily: dict[tuple[str, int | None], pd.DataFrame] = {}
        self._changes: dict[tuple[str, str, int, int | None, int | None], pd.DataFrame] = {}
        self._applied: dict[tuple[str, str, int | None], pd.DataFrame] = {}
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
    def daily(self, field: str, max_age_days: int | None = None) -> pd.DataFrame:
        """Memoized `fundamentals_to_daily(history, field, trading_index)`."""
        self._accesses += 1
        key = (field, max_age_days)
        if key not in self._daily:
            if self.empty:
                self._daily[key] = pd.DataFrame(index=self._index)
            elif max_age_days is None:
                self._daily[key] = fundamentals_to_daily(
                    self._history,
                    field,
                    self._index,
                )
            else:
                self._daily[key] = fundamentals_to_daily(
                    self._history,
                    field,
                    self._index,
                    max_age_days=max_age_days,
                )
        return self._daily[key]

    def __call__(self, field: str) -> pd.DataFrame:
        """`FieldGetter` alias, so a `PitFrames` can be passed anywhere `capital.py`
        or `_derived_fields` expects a `daily`-style accessor."""
        return self.daily(field)

    def change(
        self,
        field: str,
        kind: Literal["pct", "diff"] = "pct",
        periods: int | None = None,
        *,
        years: int | None = None,
        max_age_days: int | None = None,
    ) -> pd.DataFrame:
        """Memoized `fiscal_change_to_daily`. `periods=None` uses this history's own
        filing cadence (`yoy_periods`), which is what a year-over-year change means."""
        n = self.yoy_periods if periods is None else int(periods)
        key = (field, kind, n, years, max_age_days)
        if key not in self._changes:
            self._changes[key] = (
                pd.DataFrame(index=self._index)
                if self.empty
                else fiscal_change_to_daily(
                    self._history,
                    field,
                    self._index,
                    kind=kind,
                    periods=n,
                    years=years,
                    max_age_days=max_age_days,
                )
            )
        return self._changes[key]

    def applied(self, field: str, key: str, func: Callable[[pd.Series], pd.Series], *, max_age_days: int | None = None) -> pd.DataFrame:
        """Memoized `fiscal_apply_to_daily`. `key` names the transform, since a
        callable (often a lambda) is not a usable cache key."""
        ck = (field, key, max_age_days)
        if ck not in self._applied:
            self._applied[ck] = (
                pd.DataFrame(index=self._index)
                if self.empty
                else fiscal_apply_to_daily(
                    self._history,
                    field,
                    self._index,
                    func,
                    max_age_days=max_age_days,
                )
            )
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
                self._market_cap = daily_market_cap(self._history, self._close, level_factor=self._level_factor)
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
    def assert_matches(self, trading_index: pd.DatetimeIndex, close: pd.DataFrame | None = None) -> None:
        """Refuse to serve a cache built on a different window. Cheap: compares the
        index and the close frame's shape/columns, not their values."""
        if not self._index.equals(trading_index):
            raise ValueError(
                f"PitFrames was built on a {len(self._index)}-day index "
                f"({self._index.min()}..{self._index.max()}) but is being used with a "
                f"{len(trading_index)}-day one -- build one cache per warm-up window"
            )
        if close is not None and self._close is not None:
            if self._close.shape != close.shape or not self._close.columns.equals(close.columns):
                raise ValueError(f"PitFrames was built on a different `close` frame ({self._close.shape} vs {close.shape})")

    def stats(self) -> dict[str, int]:
        """What the cache actually collapsed, so the sub-step can log it and the tests
        can assert it: distinct fields pivoted vs total accesses."""
        computed = len(self._daily)
        return {
            "fields": computed,
            "accesses": self._accesses,
            "hits": max(0, self._accesses - computed),
            "changes": len(self._changes),
            "applied": len(self._applied),
            "market_cap": int(self._market_cap is not None),
        }
