"""Freshness of stored tables against their declared cadence, and the prediction input guard.

A table's tolerance is `DATA_FRESHNESS_MAX_AGE_DAYS[table.freshness]`, measured on
`table.freshness_col`. `key_freshness` is the share of keys whose own latest date is within it;
`table_freshness` is the age of the table-wide latest date. `check_prediction_inputs` raises
`StaleInputsError` when a prediction input is stale per key or the cube lags `prices`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import pandas as pd

from src.constants.constants import DATA_FRESHNESS_MAX_AGE_DAYS
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore

# The tables `predict` refuses to score without; targets and every other source only warn.
PREDICTION_INPUTS: tuple[Table, ...] = (Tables.prices, Tables.sharadar_fundamentals, Tables.fundamentals_history)


class StaleInputsError(RuntimeError):
    """A prediction input is stale, so the latest cube date must not be scored."""


def max_age_days(table: Table) -> int:
    """The age tolerance, in days, of the table's declared freshness cadence."""
    return DATA_FRESHNESS_MAX_AGE_DAYS[cast(str, table.freshness)]


def last_dates_by_key(store: DataStore, table: Table) -> dict[str, pd.Timestamp]:
    """Each key's latest `freshness_col` date, in one grouped read (markers counted)."""
    return store.max_date_by(table, cast(str, table.ticker_col), table.freshness_col)


def fresh_share(last_by_key: Mapping[str, pd.Timestamp], keys: Iterable[str], as_of: pd.Timestamp, max_age: int) -> float:
    """Share of `keys` whose own latest date is at most `max_age` days before `as_of`; 0.0 for no keys."""
    wanted = set(keys)
    if not wanted:
        return 0.0
    day = pd.Timestamp(as_of).normalize()
    fresh = sum(1 for key in wanted if (last := last_by_key.get(key)) is not None and (day - last).days <= max_age)
    return fresh / len(wanted)


def key_freshness(store: DataStore, table: Table, keys: Iterable[str], as_of: pd.Timestamp) -> float:
    """Share of `keys` whose own latest date in `table` is within the table's cadence tolerance."""
    return fresh_share(last_dates_by_key(store, table), keys, as_of, max_age_days(table))


def table_freshness(store: DataStore, table: Table, as_of: pd.Timestamp) -> int | None:
    """Days between `as_of` and the table-wide latest `freshness_col` date; None when the table is empty."""
    latest = store.max_date(table, table.freshness_col)
    return None if latest is None else int((pd.Timestamp(as_of).normalize() - latest).days)


def check_prediction_inputs(store: DataStore, keys: Iterable[str], as_of: pd.Timestamp, min_share: float) -> dict[str, float]:
    """Per-key fresh share of each prediction input; raises `StaleInputsError` when one is below
    `min_share`, or when the cube's latest date is earlier than the latest `prices` date."""
    universe = list(keys)
    shares = {table.name: key_freshness(store, table, universe, as_of) for table in PREDICTION_INPUTS}
    problems = [f"{name} fresh share {share:.3f} < {min_share:.3f}" for name, share in shares.items() if share < min_share]
    cube_last = store.max_date(Tables.cube, "date")
    price_last = store.max_date(Tables.prices, "date")
    if cube_last is not None and price_last is not None and cube_last < price_last:
        problems.append(f"{Tables.cube.name} ends {cube_last.date()} before {Tables.prices.name} ({price_last.date()})")
    if problems:
        raise StaleInputsError(f"stale prediction inputs as of {pd.Timestamp(as_of).date()} over {len(universe)} keys: " + "; ".join(problems))
    return shares
