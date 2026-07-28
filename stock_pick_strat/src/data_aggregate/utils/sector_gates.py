"""
sector_gates.py  (src/data_aggregate/utils/sector_gates.py)
-----------------------------------------------------------
ONE definition of "which names is this sector KPI defined for", shared by the
row-level KPI layer (`sector_features.compute_sector_kpis`) and the daily
peer-relative layer (`fundamental_features._derived_fields`).

The scope comes from GICS (`SECTOR_KPI_SCOPE` in src/constants/constants.py),
never from "did the filer report tag X". Tag presence was wrong both ways:

  * NOT sector-exclusive -> the KPI leaked into the wrong sector. `InterestIncome-
    ExpenseNet` (the `netInterestIncome` fallback) is reported by 59 non-Financials,
    so `net_interest_margin` / `bank_roa` / `bank_operating_margin` were computed
    for industrials and health-care names; `OperatingLeaseLeaseIncome` (the
    `rentalIncome` fallback) did the same for FFO / AFFO / implied cap rate on
    utilities, IT and industrial lessors.
  * sector-exclusive but RARELY tagged -> the KPI starved. Only 3 of 21 Energy
    names tag `OilAndGasProperty*`, so `ebitdax_margin` / `ddna_intensity` /
    `ebitdax_to_ev` covered 14% of the sector.

Availability gating still applies on top: a KPI whose inputs the filer never
reported stays NaN. The GICS gate only decides "is this metric MEANINGFUL here",
which is a property of the business model, not of the filer's tagging habits.

Three shapes, because the two layers work on different objects:
    row_gate       -> bool Series over `fundamentals` rows  (row-level KPIs)
    family_tickers -> the set of in-scope tickers           (cheap "any?" test)
    mask_columns   -> a daily date x ticker frame with out-of-scope tickers NaN'd
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import SECTOR_KPI_SCOPE

__all__ = ["UnknownKpiFamilyError", "row_gate", "family_tickers", "mask_columns"]


class UnknownKpiFamilyError(KeyError):
    """A KPI family with no GICS scope declared in `SECTOR_KPI_SCOPE`."""


def _scope(family: str) -> tuple[str, tuple[str, ...]]:
    try:
        return SECTOR_KPI_SCOPE[family]
    except KeyError as exc:
        raise UnknownKpiFamilyError(
            f"KPI family {family!r} has no GICS scope; add it to "
            f"SECTOR_KPI_SCOPE (known: {sorted(SECTOR_KPI_SCOPE)})"
        ) from exc


def row_gate(fundamentals: pd.DataFrame, family: str) -> pd.Series:
    """Boolean mask over `fundamentals` ROWS: True where the row's GICS level is in
    the family's scope. All-False when the GICS column is absent (a history built
    before the sector tags existed) -- fail CLOSED, so a KPI is never emitted for a
    name we cannot classify."""
    level, values = _scope(family)
    if level not in fundamentals.columns:
        return pd.Series(False, index=fundamentals.index)
    return fundamentals[level].isin(values).fillna(False)


def family_tickers(fundamentals: pd.DataFrame, family: str) -> set[str]:
    """Tickers inside the family's GICS scope (empty set when unclassifiable)."""
    if "ticker" not in fundamentals.columns:
        return set()
    gate = row_gate(fundamentals, family)
    return set(fundamentals.loc[gate, "ticker"].dropna().unique())


def mask_columns(frame: pd.DataFrame, fundamentals: pd.DataFrame,
                 family: str) -> pd.DataFrame:
    """A daily (date x ticker) `frame` with every ticker OUTSIDE the family's GICS scope
    set to NaN. Sector is a per-ticker constant, so the mask is a column Series
    broadcast down the rows -- no date x ticker boolean frame is materialised, and the
    frame's own columns are preserved (no alignment growth)."""
    if frame is None or frame.empty:
        return frame
    tickers = family_tickers(fundamentals, family)
    out_of_scope = [c for c in frame.columns if c not in tickers]
    if not out_of_scope:
        return frame
    masked = frame.copy()
    masked[out_of_scope] = np.nan
    return masked
