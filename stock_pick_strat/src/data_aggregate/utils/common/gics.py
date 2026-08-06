"""
gics.py  (src/data_aggregate/utils/common/gics.py)
-----------------------------------------------
GICS sector / industry_group lookups off `sp500_tickers`, used two ways:

  * `load_gics_maps`      ticker -> group name, for the TARGET's sector neutralization
                          (per-day within-group demeaning of the residual, so sector or
                          industry membership cannot predict the target -- if it could, it
                          would dominate the model).
  * `apply_categorical_codes`  the same membership as INTEGER category codes on the cube,
                          so LightGBM can make native non-linear categorical splits.
                          Stored as ints so they flow through the numeric panel path
                          unchanged; the linear ensemble member ignores them (they are
                          listed under inputs.categoricals, not inputs.columns).

Both are keyed by TICKER only, so they are horizon-independent and can be applied once to
the pre-horizon-merge base rather than per horizon slice.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.context import Context

GICS_COLUMNS = ("sector", "industry_group")
_UNIVERSE_TABLE = "sp500_tickers"


def load_gics_maps(context: Context) -> dict[str, dict[str, str]]:
    """{"sector": {ticker: group}, "industry_group": {...}} -- only the columns present and
    populated in `sp500_tickers`."""
    ref = context.store.load(_UNIVERSE_TABLE)
    maps: dict[str, dict[str, str]] = {}
    for col in GICS_COLUMNS:
        if not ref.empty and col in ref.columns:
            maps[col] = {str(t): str(g) for t, g in zip(ref["ticker"], ref[col])
                         if pd.notna(g) and str(g).strip()}
    return maps


def apply_categorical_codes(df: pd.DataFrame, context: Context,
                            log: logging.Logger | None = None) -> pd.DataFrame:
    """Attach GICS sector / industry_group as INTEGER category codes (deterministic sorted
    mapping; unknown / NaN -> -1)."""
    log = log or logging.getLogger(__name__)
    ref = context.store.load(_UNIVERSE_TABLE)
    for col in GICS_COLUMNS:
        if ref is None or ref.empty or col not in ref.columns:
            log.warning("%s has no '%s' -> categorical skipped", _UNIVERSE_TABLE, col)
            continue
        m = dict(zip(ref["ticker"].astype(str), ref[col].astype("string")))
        cats = df["ticker"].astype(str).map(m).astype("category")
        df[col] = cats.cat.codes.astype("int16")            # unknown / NaN -> -1
        log.info("Added categorical '%s' (%d categories)", col, cats.cat.categories.size)
    return df
