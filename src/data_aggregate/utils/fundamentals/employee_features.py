# ruff: noqa: N806
"""
employee_features.py  (src/data_aggregate/utils/employee_features.py)
---------------------------------------------------------------------
Peer-relative WORKFORCE features built from the `employees_sec` column of
`fundamentals_history` -- the headcount parsed out of each 10-K's body text by
`fundamentals_employees.py` (this used to be a separate `employees_history`
table). Because each count carries its SEC filing date as `as_of`, these are
point-in-time and backtestable (unlike the yfinance snapshot, which only knows
today's headcount):

    revenue_per_employee   TTM revenue / employees   (operational efficiency / moat)
    employee_growth        year-over-year change in headcount (expansion / retrenchment)
    headcount_elasticity   %Δemployees / %Δrevenue   (M&A digestion: <1 = revenue
                           outgrowing the people pool = scale / synergies captured)

Both are applied strictly point-in-time (stepwise from each filing's `as_of`),
and employee_growth compares only past-vs-past headcounts, so there is no
look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import (
    fiscal_change_values,
    fiscal_values_to_daily,
)
from src.data_aggregate.utils.fundamentals.feature_views import build_fundamental_views

_ANNUAL_MAX_AGE_DAYS = 460

#: Headcount is SEC-OWNED in the Sharadar-first merged table: it is parsed out of the 10-K
#: body text, and Sharadar does not deliver it. `merge_history` therefore namespaces it
#: `employees_sec`, so the column NAME carries its provenance and a NULL is attributable to
#: one producer. Reading the bare `employees` returned an empty frame and killed all four
#: features on the first lookup, before revenue was ever read. Live coverage 75.7%.
_HEADCOUNT_FIELD = "employees_sec"


def _employee_fields(
    employees_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
) -> dict:
    """Annual-event workforce values projected point-in-time from each publication."""
    F: dict[str, pd.DataFrame] = {}
    if _HEADCOUNT_FIELD not in employees_hist.columns:
        return F

    columns = ["ticker", "as_of", _HEADCOUNT_FIELD]
    if "fiscal_end" in employees_hist.columns:
        columns.append("fiscal_end")
    elif "period" in employees_hist.columns:
        columns.append("period")
    if "totalRevenue" in employees_hist.columns:
        columns.append("totalRevenue")
    observations = employees_hist[columns].copy()
    observations["as_of"] = pd.to_datetime(observations["as_of"], errors="coerce")
    observations[_HEADCOUNT_FIELD] = pd.to_numeric(observations[_HEADCOUNT_FIELD], errors="coerce")
    observations = observations.dropna(subset=["ticker", "as_of", _HEADCOUNT_FIELD]).reset_index(drop=True)
    if observations.empty:
        return F
    if "fiscal_end" not in observations.columns:
        observations["fiscal_end"] = pd.to_datetime(observations.get("period"), errors="coerce")

    if "totalRevenue" not in observations.columns:
        observations["totalRevenue"] = np.nan
    emp_growth_values = fiscal_change_values(
        observations,
        _HEADCOUNT_FIELD,
        kind="pct",
        periods=1,
        years=1,
    )
    emp_growth = fiscal_values_to_daily(
        observations,
        emp_growth_values,
        idx,
        max_age_days=_ANNUAL_MAX_AGE_DAYS,
    )
    if emp_growth.notna().any().any():
        F["employee_growth"] = emp_growth

    headcount = observations[_HEADCOUNT_FIELD].where(observations[_HEADCOUNT_FIELD] > 0)
    rpe_values = observations["totalRevenue"] / headcount
    rpe_values = rpe_values.replace([np.inf, -np.inf], np.nan)
    rev_per_emp = fiscal_values_to_daily(
        observations,
        rpe_values,
        idx,
        max_age_days=_ANNUAL_MAX_AGE_DAYS,
    )
    if rev_per_emp.notna().any().any():
        F["revenue_per_employee"] = rev_per_emp
        observations["_rpe"] = rpe_values
        rpe_growth_values = fiscal_change_values(
            observations,
            "_rpe",
            kind="pct",
            periods=1,
            years=1,
        )
        rpe_growth = fiscal_values_to_daily(
            observations,
            rpe_growth_values,
            idx,
            max_age_days=_ANNUAL_MAX_AGE_DAYS,
        )
        if rpe_growth.notna().any().any():
            F["revenue_per_employee_growth"] = rpe_growth

    revenue_growth_values = fiscal_change_values(
        observations,
        "totalRevenue",
        kind="pct",
        periods=1,
        years=1,
    )
    elasticity_values = emp_growth_values / revenue_growth_values.where(revenue_growth_values.abs() >= 0.02)
    elasticity_values = elasticity_values.replace([np.inf, -np.inf], np.nan)
    elasticity = fiscal_values_to_daily(
        observations,
        elasticity_values,
        idx,
        max_age_days=_ANNUAL_MAX_AGE_DAYS,
    )
    if elasticity.notna().any().any():
        F["headcount_elasticity"] = elasticity
    return F


def build_employee_feature_panel(
    headcount_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format workforce panel using the approved raw/self-history view contract.
    Empty if the merged fundamentals/employee-count history is unavailable."""
    if headcount_history is None or headcount_history.empty or "as_of" not in headcount_history.columns:
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _employee_fields(headcount_history, trading_index)
    return build_fundamental_views(fields, peer_dict)
