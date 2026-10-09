# ruff: noqa: N806
"""
employee_features.py  (src/data_aggregate/utils/fundamentals/employee_features.py)
-------------------------------------------------------------------------------
Peer-relative WORKFORCE features from `fundamentals_employees` (the 10-K headcount
components and their basis) joined onto `fundamentals_history` (revenue, fiscal dates):

    revenue_per_employee         TTM revenue / headcount proxy
    employee_growth              year-over-year change in the headcount proxy
    revenue_per_employee_growth  year-over-year change in revenue per employee
    headcount_elasticity         %Δheadcount / %Δrevenue (|%Δrevenue| ≥ 2 %)

The headcount proxy is `FT + α·PT` (α = `build_cube.workforce.part_time_weight`), else
`FT + α·(total − FT)`, else the stated total, FTE or full-time count. Every growth-type value
is NaN when the proxy's basis differs from the prior year's.

Point in time: each employee row is attached backward as-of to the history rows filed on or
after it (within `SHARADAR_SEC_ASOF_TOLERANCE_DAYS`), and values project from those rows' `as_of`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SHARADAR_SEC_ASOF_TOLERANCE_DAYS
from src.data_aggregate.utils.common.pit import (
    fiscal_change_values,
    fiscal_values_to_daily,
)
from src.data_aggregate.utils.fundamentals.feature_views import build_fundamental_views

_ANNUAL_MAX_AGE_DAYS = 460
_PART_TIME_WEIGHT_RANGE = (0.5, 1.0)
_COMPONENTS = ("employees_total", "employees_full_time", "employees_part_time")
#: The `fundamentals_employees` projection the cube loads.
EMPLOYEE_COLUMNS = ("ticker", "as_of", *_COMPONENTS, "basis")
#: Bases whose proxy is the stated total, never blended with a full-time count.
_TOTAL_BASES = ("total", "fte", "total_incl_contractors")
_HEADCOUNT = "headcount"
_BASIS = "basis"


def part_time_weight(cube_cfg: DictConfig) -> float:
    """α of the headcount proxy, read from `build_cube.workforce.part_time_weight` and range-checked."""
    return _checked_weight(float(cube_cfg.workforce.part_time_weight))


def _checked_weight(alpha: float) -> float:
    low, high = _PART_TIME_WEIGHT_RANGE
    if not low <= alpha <= high:
        raise ValueError(f"part_time_weight must lie in [{low}, {high}], got {alpha}")
    return alpha


def headcount_proxy(employees: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """One `(ticker, as_of, headcount, basis)` row per employee row with a usable count.

    Rows whose components are all NULL (status-only) are dropped. A row without a stored basis
    takes the basis of the branch that produced its proxy.
    """
    alpha = _checked_weight(alpha)
    frame = employees.loc[employees[list(_COMPONENTS)].notna().any(axis=1), list(EMPLOYEE_COLUMNS)].copy()
    total, full, part = (pd.to_numeric(frame[column], errors="coerce").astype("float64") for column in _COMPONENTS)
    stated_total = frame[_BASIS].isin(_TOTAL_BASES)
    branches = [
        (full.notna() & part.notna() & ~stated_total, full + alpha * part, "full_part"),
        (full.notna() & total.notna() & ~stated_total, full + alpha * (total - full), "full_part"),
        (total.notna(), total, "total"),
        (full.notna(), full, "full_time_only"),
    ]
    frame[_HEADCOUNT] = np.select([mask for mask, _, _ in branches], [value for _, value, _ in branches], default=np.nan)
    branch_basis = np.select([mask for mask, _, _ in branches], [basis for _, _, basis in branches], default="")
    frame[_BASIS] = frame[_BASIS].where(frame[_BASIS].notna(), pd.Series(branch_basis, index=frame.index))
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "as_of", _HEADCOUNT])
    return frame[["ticker", "as_of", _HEADCOUNT, _BASIS]].reset_index(drop=True)


def attach_headcount(
    history: pd.DataFrame,
    proxy: pd.DataFrame,
    *,
    tolerance_days: int = SHARADAR_SEC_ASOF_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """`history` in its own row order with the latest proxy filed on or before each row's `as_of`."""
    left = history.drop(columns=[_HEADCOUNT, _BASIS], errors="ignore").copy()
    left["_row"] = np.arange(len(left))
    left["_key"] = left["ticker"].astype(str)
    left["_on"] = pd.to_datetime(left["as_of"], errors="coerce").astype("datetime64[ns]")
    right = pd.DataFrame(
        {
            "_key": proxy["ticker"].astype(str),
            "_on": pd.to_datetime(proxy["as_of"]).astype("datetime64[ns]"),
            _HEADCOUNT: proxy[_HEADCOUNT],
            _BASIS: proxy[_BASIS],
        }
    )
    dated = left.dropna(subset=["_on"])
    joined = pd.merge_asof(
        dated.sort_values("_on", kind="stable"),
        right.sort_values("_on", kind="stable"),
        on="_on",
        by="_key",
        direction="backward",
        tolerance=pd.Timedelta(days=int(tolerance_days)),
    )
    undated = left[left["_on"].isna()]
    out = pd.concat([joined, undated], ignore_index=True) if len(undated) else joined
    out = out.sort_values("_row", kind="stable").reset_index(drop=True)
    return out.drop(columns=["_row", "_key", "_on"])


def _employee_fields(
    employees_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
) -> dict:
    """Annual-event workforce values projected point-in-time from each history row carrying a proxy."""
    F: dict[str, pd.DataFrame] = {}
    if _HEADCOUNT not in employees_hist.columns:
        return F

    columns = ["ticker", "as_of", _HEADCOUNT, _BASIS]
    if "fiscal_end" in employees_hist.columns:
        columns.append("fiscal_end")
    elif "period" in employees_hist.columns:
        columns.append("period")
    if "totalRevenue" in employees_hist.columns:
        columns.append("totalRevenue")
    observations = employees_hist[columns].copy()
    observations["as_of"] = pd.to_datetime(observations["as_of"], errors="coerce")
    observations[_HEADCOUNT] = pd.to_numeric(observations[_HEADCOUNT], errors="coerce")
    observations = observations.dropna(subset=["ticker", "as_of", _HEADCOUNT]).reset_index(drop=True)
    if observations.empty:
        return F
    if "fiscal_end" not in observations.columns:
        observations["fiscal_end"] = pd.to_datetime(observations.get("period"), errors="coerce")

    if "totalRevenue" not in observations.columns:
        observations["totalRevenue"] = np.nan
    observations["_basis_code"] = pd.Series(pd.factorize(observations[_BASIS])[0], index=observations.index).astype("float64")
    basis_shift = fiscal_change_values(observations, "_basis_code", kind="diff", periods=1, years=1)
    switched = basis_shift.notna() & (basis_shift != 0)
    # A switch row also stops the previous growth value from carrying past it.
    current_basis_ok = fiscal_values_to_daily(
        observations,
        (~switched).astype("float64"),
        idx,
        max_age_days=_ANNUAL_MAX_AGE_DAYS,
    )

    def _growth_daily(values: pd.Series) -> pd.DataFrame:
        daily = fiscal_values_to_daily(observations, values.where(~switched), idx, max_age_days=_ANNUAL_MAX_AGE_DAYS)
        return daily.where(current_basis_ok.reindex(index=daily.index, columns=daily.columns) == 1.0)

    emp_growth_values = fiscal_change_values(
        observations,
        _HEADCOUNT,
        kind="pct",
        periods=1,
        years=1,
    ).where(~switched)
    emp_growth = _growth_daily(emp_growth_values)
    if emp_growth.notna().any().any():
        F["employee_growth"] = emp_growth

    headcount = observations[_HEADCOUNT].where(observations[_HEADCOUNT] > 0)
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
        rpe_growth = _growth_daily(rpe_growth_values)
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
    elasticity = _growth_daily(elasticity_values)
    if elasticity.notna().any().any():
        F["headcount_elasticity"] = elasticity
    return F


def build_employee_feature_panel(
    fundamentals_history: pd.DataFrame | None,
    employees: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    *,
    part_time_weight: float,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Long-format workforce panel using the approved raw/self-history view contract.
    Empty when either the fundamentals history or the employee table is unavailable."""
    empty = pd.DataFrame(columns=["date", "ticker"])
    if fundamentals_history is None or fundamentals_history.empty or "as_of" not in fundamentals_history.columns:
        return empty
    if employees is None or employees.empty:
        return empty

    proxy = headcount_proxy(employees, part_time_weight)
    fields = _employee_fields(attach_headcount(fundamentals_history, proxy), trading_index)
    return build_fundamental_views(
        fields,
        peer_dict,
        history_fields=history_fields,
        output_since=output_since,
    )
