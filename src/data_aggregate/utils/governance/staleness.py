"""Source-aged annual expiry and null-preserving governance measurements."""

from __future__ import annotations

import pandas as pd

from src.constants.constants import GOVERNANCE_ANNUAL_MAX_AGE_DAYS
from src.data_aggregate.utils.common.pit import fiscal_change_values, fundamentals_to_daily
from src.data_aggregate.utils.governance.names import ceo_stint_ids, key_at_last_value

#: Marks a proxy that ends the previous CEO stint's value in `stint_to_daily`; never a real value.
_STINT_ENDED = float("-inf")

# Family classifications retain their APIs; all annual observations share one limit.
GOVERNANCE_EVENT_MAX_AGE_DAYS = GOVERNANCE_ANNUAL_MAX_AGE_DAYS
LEVEL_MAX_AGE_DAYS = GOVERNANCE_ANNUAL_MAX_AGE_DAYS


LEVEL_HORIZON_FIELDS: frozenset[str] = frozenset(
    {
        "ceo_pay_growth",
        "ceo_pay_vs_revenue_growth",
        "ceo_pay_ratio",
        "ceo_equity_pay_pct",
        "ceo_tenure",
        "founder_ceo",
        "pct_independent_directors",
        "pct_female_directors",
        "board_size",
        "avg_board_tenure",
        "insider_ownership_pct",
    }
)

#: ⚠ DEPRECATED NAME, kept because the set itself is still the right set -- only its MEANING
#: changed (see above). Read `LEVEL_HORIZON_FIELDS`; this alias exists so a stale import does
#: not break, and it should go once the tests naming it are next touched.
LEGACY_EXEMPT_FROM_EXPIRY: frozenset[str] = LEVEL_HORIZON_FIELDS


def source_date_column(field: str) -> str:
    """Internal provenance column paired with a governance value."""
    return f"{field}_source_as_of"


def ultimate_source_dates(history: pd.DataFrame, field: str) -> pd.Series:
    """Observation date for each non-null filing-grain value, including carried lineage."""
    fallback = pd.to_datetime(history["as_of"], errors="coerce").where(history[field].notna())
    column = source_date_column(field)
    if column not in history.columns:
        return fallback
    explicit = pd.to_datetime(history[column], errors="coerce").where(history[field].notna())
    return explicit.combine_first(fallback)


def measurement_to_daily(history: pd.DataFrame, field: str, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Project computed observations, preserving rejected measurements as carry barriers.

    A null at a newer observation ends the old value until another valid observation.
    Raw disclosure silence still uses `fundamentals_to_daily` and its source-age limit.
    """
    if field not in history.columns:
        return pd.DataFrame(index=trading_index)
    values = history[["ticker", "as_of", field]].copy()
    values[field] = pd.to_numeric(values[field], errors="coerce").fillna(_STINT_ENDED)
    daily = fundamentals_to_daily(values, field, trading_index)
    return daily.mask(daily == _STINT_ENDED)


def measurement_change_to_daily(
    history: pd.DataFrame, field: str, trading_index: pd.DatetimeIndex, *, kind: str = "diff", periods: int = 1
) -> pd.DataFrame:
    """Calculate a filing-level change without carrying across rejected comparisons."""
    values = history[["ticker", "as_of", field]].copy()
    values[field] = fiscal_change_values(history, field, kind=kind, periods=periods)
    return measurement_to_daily(values, field, trading_index)


def horizon_for(feature: str, default: int = GOVERNANCE_EVENT_MAX_AGE_DAYS) -> int:
    """Return the annual source-age limit, preserving the family override contract."""
    return LEVEL_MAX_AGE_DAYS if feature in LEVEL_HORIZON_FIELDS else default


def expire_stale(daily: pd.DataFrame, history: pd.DataFrame, field: str, max_age_days: int | None = None, feature: str | None = None) -> pd.DataFrame:
    """Expire daily values by their ultimate public source date, never a carried row's date."""
    name = feature if feature is not None else field
    if max_age_days is None:
        max_age_days = horizon_for(name)
    if daily is None or daily.empty or history is None or history.empty:
        return daily
    if field not in history.columns or "as_of" not in history.columns:
        return daily

    # The producing `as_of`, pivoted with the SAME ffill as the value itself rather than a
    # second hand-written one: `_produced_at` is the filing date carried as a payload. Rows
    # where the field is NULL are dropped first, because `fundamentals_to_daily` aggregates
    # with `last` (which skips NaN) -- so a cell's value comes from the last NON-NULL filing,
    # and its age must be measured against that same filing, not against a later empty one.
    h = history[["ticker", "as_of", field]].copy()
    h["_source_as_of"] = ultimate_source_dates(history, field)
    h["as_of"] = pd.to_datetime(h["as_of"], errors="coerce")
    h = h.dropna(subset=["ticker", "as_of", field])
    if h.empty:
        return daily
    h["_produced_at"] = h["_source_as_of"].map(pd.Timestamp.toordinal).astype("float64")

    produced = fundamentals_to_daily(h, "_produced_at", pd.DatetimeIndex(daily.index))
    if produced.empty:
        return daily
    produced = produced.reindex(columns=daily.columns)

    today = pd.Series(daily.index.map(pd.Timestamp.toordinal), index=daily.index, dtype="float64")
    age = produced.rsub(today, axis=0)
    # `> max_age` is False wherever the age is NaN (no filing yet), which is the right
    # answer: those cells are already NaN in `daily` and masking them changes nothing.
    return daily.mask((age < 0) | (age > float(max_age_days)))


def stint_to_daily(history: pd.DataFrame, field: str, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """`fundamentals_to_daily` for a CEO-person field (start year, founder, chair).

    A proxy that names a different CEO stint than the last valued proxy, and has no value of its
    own, ends the previous CEO's value: the daily cell goes blank instead of being bridged. Rows
    with an unknown CEO break no stint, as in `ceo_stint_ids`.
    """
    if field not in history.columns or "ceo_name_proxy" not in history.columns:
        return fundamentals_to_daily(history, field, trading_index)
    h = history[["ticker", "as_of", "ceo_name_proxy", field]].copy()
    h["as_of"] = pd.to_datetime(h["as_of"], errors="coerce")
    h = h.dropna(subset=["ticker", "as_of"]).sort_values(["ticker", "as_of"], kind="stable").reset_index(drop=True)
    value = pd.to_numeric(h[field], errors="coerce")
    stint = ceo_stint_ids(h["ceo_name_proxy"], h["ticker"], h["as_of"])
    current = stint.groupby(h["ticker"], sort=False).ffill()
    source = key_at_last_value(current, value, h["ticker"])
    ends = value.isna() & stint.notna() & source.notna() & (stint != source)
    # A same-day row that carries a value wins over the terminator.
    keys = pd.MultiIndex.from_frame(h[["ticker", "as_of"]])
    ends &= ~keys.isin(keys[value.notna().to_numpy()])
    h[field] = value.mask(ends, _STINT_ENDED)
    daily = fundamentals_to_daily(h, field, trading_index)
    return daily.mask(daily == _STINT_ENDED)


def _expire_family(
    frames: dict[str, pd.DataFrame],
    history: pd.DataFrame,
    selected: frozenset[str] | set[str],
    sources: dict[str, str] | None,
    max_age_days: int,
    skip: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """The shared workhorse of the two wrappers below. `skip` is never expired at all."""
    out: dict[str, pd.DataFrame] = {}
    stats: dict[str, tuple[int, int]] = {}
    for name, frame in frames.items():
        if name not in selected or name in skip:
            out[name] = frame
            continue
        field = (sources or {}).get(name, name)
        before = int(frame.notna().to_numpy().sum())
        capped = expire_stale(frame, history, field, max_age_days=max_age_days, feature=name)
        after = int(capped.notna().to_numpy().sum())
        out[name] = capped
        stats[name] = (before - after, before)
    return out, stats


def expire_event_fields(
    frames: dict[str, pd.DataFrame],
    history: pd.DataFrame,
    event_fields: frozenset[str] | set[str],
    sources: dict[str, str] | None = None,
    max_age_days: int = GOVERNANCE_EVENT_MAX_AGE_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """Expire a family's event fields and return removed/before counts per feature."""
    return _expire_family(frames, history, event_fields, sources, max_age_days, skip=LEVEL_HORIZON_FIELDS)


def expire_level_fields(
    frames: dict[str, pd.DataFrame],
    history: pd.DataFrame,
    level_fields: frozenset[str] | set[str],
    sources: dict[str, str] | None = None,
    max_age_days: int = LEVEL_MAX_AGE_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """Expire a family's level fields and return removed/before counts per feature."""
    return _expire_family(frames, history, level_fields, sources, max_age_days)
