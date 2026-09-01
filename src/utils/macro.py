"""
macro.py  (src/utils/macro.py)
------------------------------
THE long->wide adapter for `prices_macro`, shared by every consumer: the cube's beta/target
step, the long-book and trend-CTA sleeves, the portfolio benchmark, the L/S diagnostics.

`prices_macro` stores (date, ticker, close) where `ticker` is the series NAME -- so pivoting
it reproduces exactly the column vocabulary the two old wide tables (`macro`,
`macro_asset_prices`) had. That is deliberate and it is what keeps this refactor cheap: each
consumer swapped one `store.load(...)` line for one `load_macro_wide(...)` line and kept all
of its wide logic (the `if "equity_tr" in d.columns` guards, `pct_change`,
`cash_rate/100/252`, the rename maps) untouched.

Lives in src/utils/ because it is read from `data_aggregate`, `modelling`, `strategies` and
`portfolio` alike, and those must not import each other.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.data_store.schema import Tables


def load_macro_wide(store, series: Sequence[str] | None = None,
                    since=None) -> pd.DataFrame | None:
    """`prices_macro` -> wide frame with `date` as a COLUMN and one column per series.

    Returns None (not an empty frame) when the table is missing or empty, which is the
    contract the call sites branch on -- they were written against
    `store.load(..., optional=True)` and their `if df is None` guards are preserved.

    `date` is left as a column rather than the index so this is a drop-in for the read it
    replaced; every consumer does its own `set_index("date")` immediately after.

    `series` narrows the read server-side. Callers that compute CHANGES should NOT pass
    `since`: the diffs and forward-fills need each series' own prior observation, and a
    windowed read silently nulls the first row of every change column.
    """
    where = {"ticker": [str(s) for s in series]} if series else None
    long = store.load(Tables.prices_macro, columns=["date", "ticker", "close"],
                      where=where, since=since, optional=True)
    if long is None or long.empty:
        return None

    long = long.copy()
    long["date"] = pd.to_datetime(long["date"], format="%Y-%m-%d")
    wide = long.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide = wide.sort_index()
    wide.columns.name = None
    return wide.reset_index()


def load_macro_series(store, name: str, since=None) -> pd.Series | None:
    """One macro series as a date-indexed Series, or None when it has no rows.

    The single-series shortcut for the benchmark / calendar / market-reference readers, so
    they neither pivot a frame they will immediately subset nor re-implement the None guard.
    """
    wide = load_macro_wide(store, series=[name], since=since)
    if wide is None or name not in wide.columns:
        return None
    out = wide.set_index("date")[name].astype(float)
    return out if out.notna().any() else None
