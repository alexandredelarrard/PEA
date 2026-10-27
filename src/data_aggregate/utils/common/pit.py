"""Project public observations and fiscal comparisons onto a requested daily calendar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd


class FieldGetter(Protocol):
    """Read a reported field as filing-row values or daily values."""

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
    """Project the last observation per requested session without pivoting older history."""
    df = observations[["ticker", "as_of", value_col]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.dropna(subset=["ticker", "as_of", value_col])
    if df.empty:
        return pd.DataFrame(index=trading_index)
    df = df.sort_values(["ticker", "as_of"]).drop_duplicates(["ticker", "as_of"], keep="last")
    columns = pd.Index(df["ticker"].unique(), name="ticker").sort_values()
    calendar = trading_index.unique().sort_values()
    value_dtype = df[value_col].dtype
    # Keep the dtype promotion of the original sparse history/calendar pivot.
    if len(df) != df["as_of"].nunique() * len(columns) or not trading_index.isin(df["as_of"]).all():
        value_dtype = pd.Series([], dtype=value_dtype).reindex([0]).dtype
    df = df.loc[df["as_of"] <= calendar[-1]].copy() if len(calendar) else df.iloc[:0].copy()
    if df.empty:
        daily = pd.DataFrame({ticker: pd.Series([], dtype=value_dtype) for ticker in columns})
        return daily.reindex(index=trading_index, columns=columns)

    produced_at = df["as_of"].array.copy()
    # Pre-window rows become a seed; between-session publications become visible next session.
    df["as_of"] = calendar.take(calendar.searchsorted(df["as_of"], side="left"))
    visible = ~df.duplicated(["ticker", "as_of"], keep="last")
    wide = df.loc[visible].pivot(index="as_of", columns="ticker", values=value_col)
    daily = wide.reindex(calendar).ffill().reindex(index=trading_index, columns=columns).astype(value_dtype)
    if max_age_days is None:
        return daily

    df["_produced_at"] = produced_at
    produced = df.loc[visible].pivot(index="as_of", columns="ticker", values="_produced_at")
    produced = produced.reindex(calendar).ffill().reindex(index=trading_index, columns=columns).astype(produced_at.dtype)
    current = pd.Series(trading_index, index=trading_index)
    horizon = pd.Timedelta(days=int(max_age_days))
    for ticker in daily.columns:
        age = current - produced[ticker]
        daily[ticker] = daily[ticker].where(age <= horizon)
    return daily


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
    preserve_nulls: bool = False,
) -> pd.DataFrame:
    """Project fiscal values; optionally let invalid new filings clear prior state."""
    if len(values) != len(fund_hist):
        raise ValueError("fiscal value series must be row-aligned with fund_hist")
    observations = fund_hist[["ticker", "as_of"]].assign(_value=pd.to_numeric(values.to_numpy(), errors="coerce"))
    if preserve_nulls:
        observations["as_of"] = pd.to_datetime(observations["as_of"], errors="coerce")
        observations = observations.dropna(subset=["ticker", "as_of"])
        observations = observations.sort_values(["ticker", "as_of"]).drop_duplicates(["ticker", "as_of"], keep="last")
        daily: dict[str, pd.Series] = {}
        for ticker, rows in observations.groupby("ticker", sort=False):
            source = rows.set_index("as_of")["_value"]
            projected = source.reindex(idx, method="ffill")
            if max_age_days is not None:
                produced = pd.Series(source.index, index=source.index).reindex(idx, method="ffill")
                age = pd.Series(idx, index=idx) - produced
                projected = projected.where(age <= pd.Timedelta(days=int(max_age_days)))
            daily[str(ticker)] = projected
        return pd.DataFrame(daily, index=idx).rename_axis(columns="ticker")
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
    """Fiscal change; only legacy pct/diff modes permit a filing-count fallback."""
    if field not in fund_hist.columns:
        return pd.Series(np.nan, index=fund_hist.index, dtype="float64")
    current = pd.to_numeric(fund_hist[field], errors="coerce")
    has_fiscal_end = "fiscal_end" in fund_hist.columns and fund_hist["fiscal_end"].notna().any()
    if kind == "signed_pct" and not has_fiscal_end:
        return pd.Series(np.nan, index=fund_hist.index, dtype="float64")
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
    elif kind == "signed_pct":
        current = current.replace([np.inf, -np.inf], np.nan)
        prior = prior.replace([np.inf, -np.inf], np.nan)
        result = (current - prior) / prior.abs().where(prior != 0)
    elif kind == "diff":
        result = current - prior
    else:
        raise ValueError("kind must be 'pct', 'signed_pct' or 'diff'")
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
