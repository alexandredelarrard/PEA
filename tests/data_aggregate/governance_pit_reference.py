"""Explicit legacy unitary-count inputs for governance regression controls.

These fixtures preserve older source-only scenarios. Independent saved-count,
publication and unknown-unit controls exercise production interfaces separately.
"""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.clean import GovernanceInputs
from src.data_aggregate.utils.governance.clean import clean_governance_inputs as clean_inputs
from src.data_aggregate.utils.governance.clean import pit_shares_outstanding as saved_counts
from src.data_aggregate.utils.price.frames import PriceFrames


def legacy_count_inputs(df_proxies: pd.DataFrame, df_history: pd.DataFrame | None) -> PriceFrames | None:
    """Saved-count frames built from the fixture's `sharesOutstandingPit` at the scenario's observation dates."""
    if df_history is None or df_history.empty or "sharesOutstandingPit" not in df_history or not {"ticker", "as_of"}.issubset(df_proxies):
        return None
    dates = pd.concat([df_proxies["as_of"], df_history["as_of"]], ignore_index=True)
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
    df_counts = fundamentals_to_daily(df_history, "sharesOutstandingPit", index)
    return PriceFrames(index, tuple(map(str, df_counts.columns)), {}, shares_outstanding_pit=df_counts)


def clean_governance_inputs(raw: GovernanceInputs, fundamentals: pd.DataFrame | None) -> tuple[GovernanceInputs, dict[str, int]]:
    return clean_inputs(raw, legacy_count_inputs(raw.def14a, fundamentals) if raw.def14a is not None else None)


def pit_shares_outstanding(df_proxies: pd.DataFrame, fundamentals: pd.DataFrame | None, tally: dict[str, int] | None = None) -> pd.Series | None:
    return saved_counts(df_proxies, legacy_count_inputs(df_proxies, fundamentals), tally)
