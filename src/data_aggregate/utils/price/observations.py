"""Bind financial share units to the raw observation that supplied each value.

These ephemeral working views change representation units only. They never rebuild
financial economics, overwrite source rows, or infer provenance from saved capital.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.price.capitalization import canonical_source_support
from src.utils.cutover_continuity import PredecessorSeries
from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import vendor_observation_factor
from src.utils.quarters import quarter_ordinal

_KEYS = ["ticker", "as_of", "fiscal_end"]
_COUNT_FIELDS = {"basicShares": "shareswa", "dilutedShares": "shareswadil"}


def _source_windows(df_raw: pd.DataFrame, basis: ShareBasis, predecessors: tuple[PredecessorSeries, ...]) -> pd.DataFrame:
    """Prove coherent original four-quarter windows using the extraction grain/rules."""
    required = {"ticker", "date", "reportperiod", "calendardate", "lastupdated"}
    if not required.issubset(df_raw):
        return pd.DataFrame(columns=_KEYS)
    df_frame = df_raw.loc[df_raw["dimension"].eq("ARQ")].copy() if "dimension" in df_raw else df_raw.copy()
    for field in ("date", "reportperiod", "calendardate", "lastupdated"):
        df_frame[field] = pd.to_datetime(df_frame[field], errors="coerce")
    df_frame = df_frame.sort_values(["ticker", "reportperiod", "date"]).drop_duplicates(["ticker", "reportperiod"], keep="first")
    df_frame = df_frame.sort_values(["ticker", "reportperiod"]).reset_index(drop=True)
    names = df_frame["ticker"]
    observed = vendor_observation_factor(basis, names, df_frame["date"], df_frame["lastupdated"])
    tail = basis.factor(names, df_frame["lastupdated"], "vendor").where(observed.notna())
    owner = canonical_source_support(names, df_frame["date"], predecessors) & canonical_source_support(names, df_frame["reportperiod"], predecessors)
    df_frame["_tail"] = tail.where(owner & np.isfinite(tail) & tail.gt(0))
    grouped = df_frame.groupby("ticker", sort=False)
    ordinal = quarter_ordinal(df_frame["calendardate"])
    span = (df_frame["reportperiod"] - grouped["reportperiod"].shift(3)).dt.days
    whole = span.between(240, 320)
    for lag in range(1, 4):
        whole &= (ordinal - ordinal.groupby(names, sort=False).shift(lag)).eq(lag)
        whole &= grouped["date"].shift(lag).le(df_frame["date"])
    rolling = grouped["_tail"].rolling(4)
    coherent = whole & rolling.count().reset_index(level=0, drop=True).reindex(df_frame.index).eq(4)
    low = rolling.min().reset_index(level=0, drop=True).reindex(df_frame.index)
    high = rolling.max().reset_index(level=0, drop=True).reindex(df_frame.index)
    df_frame["_ttm_tail"] = df_frame["_tail"].where(coherent & low.eq(high))
    for field in ("shareswa", "shareswadil", "dps"):
        values = pd.to_numeric(df_frame.get(field, pd.Series(np.nan, index=df_frame.index)), errors="coerce")
        window = values.groupby(names, sort=False).rolling(4)
        aggregate = window.sum() if field == "dps" else window.mean()
        df_frame[f"_ttm_{field}"] = aggregate.reset_index(level=0, drop=True).reindex(df_frame.index).where(whole)
    return df_frame.rename(columns={"date": "as_of", "reportperiod": "fiscal_end"})


def _matched(values: pd.Series, source_values: pd.Series) -> pd.Series:
    """Permit numerical rounding only, independently for each source value."""
    values = pd.to_numeric(values, errors="coerce")
    source_values = pd.to_numeric(source_values, errors="coerce")
    return pd.Series(np.isfinite(values) & np.isfinite(source_values) & np.isclose(values, source_values, rtol=1e-12, atol=0), index=values.index)


def fundamental_share_observations(
    df_history: pd.DataFrame, df_raw: pd.DataFrame | None, basis: ShareBasis, predecessors: tuple[PredecessorSeries, ...]
) -> pd.DataFrame:
    """Express proven raw share blocks on current vendor units; unsupported fields stay null."""
    df_result = df_history.copy()
    if "sharesOutstanding" in df_result:
        df_result["_source_shares_outstanding"] = df_result["sharesOutstanding"]
    for key in ("as_of", "fiscal_end"):
        if key in df_result:
            df_result[key] = pd.to_datetime(df_result[key], errors="coerce")
    df_source = _source_windows(df_raw, basis, predecessors) if df_raw is not None else pd.DataFrame(columns=_KEYS)
    columns = [*_KEYS, *[name for name in df_source if isinstance(name, str) and (name.startswith("_ttm_") or name in ("_tail", "sharesbas"))]]
    if set(_KEYS).issubset(df_result):
        df_bound = df_result[_KEYS].merge(df_source[columns], on=_KEYS, how="left", validate="many_to_one")
        df_bound.index = df_result.index
    else:
        df_bound = pd.DataFrame(index=df_result.index)
    missing = pd.Series(np.nan, index=df_result.index)
    for field, raw_field in _COUNT_FIELDS.items():
        if field in df_result:
            support = _matched(df_result[field], df_bound.get(f"_ttm_{raw_field}", missing))
            df_result[field] = pd.to_numeric(df_result[field], errors="coerce") * df_bound.get("_ttm_tail", missing).where(support)
    if "epsDiluted" in df_result:
        original = df_history.get("netIncome", missing) / df_history.get("dilutedShares", missing)
        supported = _matched(df_history["epsDiluted"], original) & df_result.get("dilutedShares", missing).notna()
        df_result["epsDiluted"] = pd.to_numeric(df_history["epsDiluted"], errors="coerce") / df_bound.get("_ttm_tail", missing).where(supported)
    if "dividendsPerShare" in df_result:
        supported = _matched(df_history["dividendsPerShare"], df_bound.get("_ttm_dps", missing))
        df_result["dividendsPerShare"] = pd.to_numeric(df_history["dividendsPerShare"], errors="coerce") / df_bound.get("_ttm_tail", missing).where(
            supported
        )
    if "optionOverhang" in df_result:
        df_result["optionOverhang"] = df_result["optionOverhang"].where(
            df_result.get("basicShares", missing).notna() & df_result.get("dilutedShares", missing).notna()
        )
    if "sharesOutstanding" in df_result:
        supported = _matched(df_history["sharesOutstanding"], df_bound.get("sharesbas", missing))
        df_result["sharesOutstanding"] = pd.to_numeric(df_history["sharesOutstanding"], errors="coerce") * df_bound.get("_tail", missing).where(
            supported
        )
    return df_result
