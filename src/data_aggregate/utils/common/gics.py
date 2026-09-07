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
  * `attach_gics_columns`  the membership as the STRING group names, on a filing-grain
                          frame, for `sector_gates.row_gate` -- which compares against the
                          names in `SECTOR_KPI_SCOPE`, not against codes.

All three are keyed by TICKER only, so they are horizon-independent and can be applied once
to the pre-horizon-merge base rather than per horizon slice.

⚠ THIS IS A CURRENT SNAPSHOT PROJECTED BACKWARDS. `sp500_tickers` holds today's GICS
classification with no date axis, so a name that changed sector carries its 2026 label back
to 1995. That is accepted rather than unnoticed: it is the same look-ahead the peer basket
already carries (one static 2026-vintage embedding basket applied over the whole history),
so it adds no new KIND of bias to the panel. A point-in-time GICS history would fix both at
once and neither alone is worth a separate vendor feed.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data_store.schema import Tables
from src.context import Context

GICS_COLUMNS = ("sector", "industry_group")



def load_gics_maps(context: Context) -> dict[str, dict[str, str]]:
    """{"sector": {ticker: group}, "industry_group": {...}} -- only the columns present and
    populated in `sp500_tickers`."""
    ref = context.store.load(Tables.sp500_tickers)
    maps: dict[str, dict[str, str]] = {}
    for col in GICS_COLUMNS:
        if not ref.empty and col in ref.columns:
            maps[col] = {str(t): str(g) for t, g in zip(ref["ticker"], ref[col])
                         if pd.notna(g) and str(g).strip()}
    return maps


def attach_gics_columns(df: pd.DataFrame, context: Context,
                        log: logging.Logger | None = None) -> pd.DataFrame:
    """Attach `sector` / `industry_group` as the STRING group names, keyed on `ticker`.

    What `sector_gates.row_gate` needs and never had. The gate fails CLOSED on a missing
    GICS column -- correctly, so a KPI is never emitted for a name we cannot classify -- but
    `fundamentals_history` has neither column, so every sector KPI was gated off for every
    row of every family. The reference data was in the DB the whole time: `sp500_tickers`
    has both columns populated 500/500.

    Returns a COPY: the frame this runs on is the once-loaded `fundamentals_history` shared
    by five builders, and mutating a caller's input to add a column is how two builders end
    up disagreeing about the frame they were handed.
    """
    log = log or logging.getLogger(__name__)
    maps = load_gics_maps(context)
    out = df.copy()
    tickers = out["ticker"].astype(str)
    for col in GICS_COLUMNS:
        if col not in maps:
            log.warning("%s has no populated '%s' -> the sector KPIs scoped on it stay "
                        "gated off", Tables.sp500_tickers, col)
            continue
        out[col] = tickers.map(maps[col])
        log.info("Attached GICS '%s' to %s: %d/%d rows classified",
                 col, Tables.fundamentals_history, int(out[col].notna().sum()), len(out))
    return out


def apply_categorical_codes(df: pd.DataFrame, context: Context,
                            log: logging.Logger | None = None) -> pd.DataFrame:
    """Attach GICS sector / industry_group as INTEGER category codes (deterministic sorted
    mapping; unknown / NaN -> -1)."""
    log = log or logging.getLogger(__name__)
    ref = context.store.load(Tables.sp500_tickers, optional=True)
    for col in GICS_COLUMNS:
        if ref is None or col not in ref.columns:
            log.warning("%s has no '%s' -> categorical skipped", Tables.sp500_tickers, col)
            continue
        m = dict(zip(ref["ticker"].astype(str), ref[col].astype("string")))
        cats = df["ticker"].astype(str).map(m).astype("category")
        df[col] = cats.cat.codes.astype("int16")            # unknown / NaN -> -1
        log.info("Added categorical '%s' (%d categories)", col, cats.cat.categories.size)
    return df
