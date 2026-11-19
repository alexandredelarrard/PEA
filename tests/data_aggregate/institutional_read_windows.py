"""Explicit full-history scopes for the source parser and share-basis known-truth fixtures."""

from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.inputs import input_windows
from src.data_store.schema import Table, Tables
from src.utils.window import Window


def full_reference_windows(context: Context) -> dict[Table, Window | None]:
    """Declare full source evidence; finite pushdown is tested by test_input_policy."""
    policy = context.config.data.tables.cube_part_institutionals.warmup
    for dependency in policy:
        policy[dependency] = None
    return input_windows(context, Tables.cube_part_institutionals, Window(context.run.as_of, context.run.as_of))


def reference_sessions(context: Context) -> pd.DatetimeIndex:
    """Literal business sessions covering the source parser fixture dates and publication tails."""
    return pd.bdate_range("2010-01-01", context.run.as_of)
