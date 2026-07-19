"""
employee_features.py  (src/data_aggregate/utils/employee_features.py)
---------------------------------------------------------------------
Peer-relative WORKFORCE features built from the genuinely-historical FMP
employee-count archive (fetch_employees). Because each count carries its SEC
filing date as `as_of`, these are point-in-time and backtestable (unlike the
yfinance snapshot, which only knows today's headcount):

    revenue_per_employee   TTM revenue / employees   (operational efficiency / moat)
    employee_growth        year-over-year change in headcount (expansion / retrenchment)
    headcount_elasticity   %Δemployees / %Δrevenue   (M&A digestion: <1 = revenue
                           outgrowing the people pool = scale / synergies captured)

Both are applied strictly point-in-time (stepwise from each filing's `as_of`),
and employee_growth compares only past-vs-past headcounts, so there is no
look-ahead.
"""

from __future__ import annotations
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import _ratio, build_peer_relative_panel

_YOY_TRADING_DAYS = 252   # ~1 year of trading days for the YoY headcount change


def _employee_fields(
    employees_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    fundamentals: pd.DataFrame | None,
) -> dict:
    """Daily wide frames (date x ticker), point-in-time from each filing `as_of`."""
    F: dict[str, pd.DataFrame] = {}

    employees = fundamentals_to_daily(employees_hist, "employees", idx)
    if employees.empty or not employees.notna().any().any():
        return F

    # year-over-year headcount growth (past vs past -> leak-free)
    emp_growth = employees / employees.shift(_YOY_TRADING_DAYS) - 1.0
    if emp_growth.notna().any().any():
        F["employee_growth"] = emp_growth

    # revenue per employee = TTM revenue / employees (both historical, PIT)
    if fundamentals is not None:
        revenue = fundamentals_to_daily(fundamentals, "totalRevenue", idx)
        rev_per_emp = _ratio(revenue, employees, positive_den=True)
        if not rev_per_emp.empty and rev_per_emp.notna().any().any():
            F["revenue_per_employee"] = rev_per_emp
        # headcount elasticity to revenue (M&A DIGESTION #3): %Δemployees / %Δrevenue.
        # <1 = revenue outgrowing the people pool (scale / synergies captured); ~1 =
        # headcount scaling 1:1 with (often acquired) revenue -> integration not landing.
        if "employee_growth" in F:
            rev_growth = revenue / revenue.shift(_YOY_TRADING_DAYS) - 1.0
            el = _ratio(F["employee_growth"], rev_growth.where(rev_growth.abs() >= 0.02))
            if not el.empty and el.notna().any().any():
                F["headcount_elasticity"] = el
    return F


def build_employee_feature_panel(
    employees_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    fundamentals_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format workforce feature panel (`f_<name>_vs_peers`, `f_<name>_xs`).
    Empty if the employee-count history is unavailable."""
    if (employees_history is None or employees_history.empty
            or "as_of" not in employees_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _employee_fields(employees_history, trading_index, fundamentals_history)
    return build_peer_relative_panel(fields, peer_dict)
