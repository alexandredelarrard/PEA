"""Freshness of stored tables against their declared cadence, and the prediction input guard.

A table's tolerance, cadence and measurement clock are declared in `data.yml`. `key_freshness` is the share of keys whose own latest date is within it;
`table_freshness` is the age of the table-wide latest date. `check_prediction_inputs` raises
`StaleInputsError` when a prediction input is stale per key or the cube lags `prices`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import ALL, Table, Tables

# The tables `predict` refuses to score without; targets and every other source only warn.
PREDICTION_INPUTS: tuple[Table, ...] = (Tables.prices, Tables.sharadar_fundamentals_facts, Tables.sharadar_fundamentals_history)


class StaleInputsError(RuntimeError):
    """A prediction input is stale, so the latest cube date must not be scored."""


def freshness_tables(config: DictConfig) -> tuple[Table, ...]:
    """Tables explicitly enabled for freshness checks, in registry order."""
    return tuple(table for table in ALL if config.data.tables[table.name].get("freshness") is not None)


def max_age_days(config: DictConfig, table: Table) -> int:
    """The configured tolerance for this table's configured cadence."""
    policy = config.data.tables[table.name].freshness
    if policy is None:
        raise ValueError(f"{table.name}: freshness is not configured as dated")
    return int(config.data.freshness.max_age_days[policy.cadence])


def last_dates_by_key(context: Context, table: Table) -> dict[str, pd.Timestamp]:
    """Each key's latest `freshness_col` date, in one grouped read (markers counted)."""
    return context.store.max_date_by(table, cast(str, table.ticker_col), str(context.config.data.tables[table.name].freshness.clock))


def fresh_share(last_by_key: Mapping[str, pd.Timestamp], keys: Iterable[str], as_of: pd.Timestamp, max_age: int) -> float:
    """Share of `keys` whose own latest date is at most `max_age` days before `as_of`; 0.0 for no keys."""
    wanted = set(keys)
    if not wanted:
        return 0.0
    day = pd.Timestamp(as_of).date()
    fresh = sum(1 for key in wanted if (last := last_by_key.get(key)) is not None and (day - last.date()).days <= max_age)
    return fresh / len(wanted)


def key_freshness(context: Context, table: Table, keys: Iterable[str], as_of: pd.Timestamp) -> float:
    """Share of `keys` whose own latest date in `table` is within the table's cadence tolerance."""
    return fresh_share(last_dates_by_key(context, table), keys, as_of, max_age_days(context.config, table))


def table_freshness(context: Context, table: Table, as_of: pd.Timestamp) -> int | None:
    """Calendar days between `as_of` and the latest `freshness_col` date; None when the table is empty."""
    latest = context.store.max_date(table, str(context.config.data.tables[table.name].freshness.clock))
    return None if latest is None else (pd.Timestamp(as_of).date() - latest.date()).days


def check_prediction_inputs(context: Context, keys: Iterable[str], as_of: pd.Timestamp) -> dict[str, float]:
    """Per-key fresh share of each prediction input; raises `StaleInputsError` when one is below
    `min_share`, or when the cube's latest date is earlier than the universe's latest `prices` date."""
    universe = list(keys)
    min_share = float(context.config.data.freshness.prediction_min_fresh_share)
    shares = {table.name: key_freshness(context, table, universe, as_of) for table in PREDICTION_INPUTS}
    problems = [f"{name} fresh share {share:.3f} < {min_share:.3f}" for name, share in shares.items() if share < min_share]
    cube_last = context.store.max_date(Tables.cube, "date")
    # `prices` also holds the companies' secondary classes; only the universe's own bars set the frontier.
    price_by_key = last_dates_by_key(context, Tables.prices)
    universe_last = [price_by_key[key] for key in universe if key in price_by_key]
    price_last = max(universe_last) if universe_last else None
    if cube_last is not None and price_last is not None and cube_last < price_last:
        problems.append(f"{Tables.cube.name} ends {cube_last.date()} before {Tables.prices.name} ({price_last.date()})")
    if problems:
        raise StaleInputsError(f"stale prediction inputs as of {pd.Timestamp(as_of).date()} over {len(universe)} keys: " + "; ".join(problems))
    return shares
