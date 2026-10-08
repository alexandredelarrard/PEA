# ruff: noqa: N806
"""Tests for the workforce features (src/data_aggregate/utils/fundamentals/employee_features.py).

Headcount comes from `fundamentals_employees` (components plus basis), joined backward as-of onto the
`fundamentals_history` rows that carry revenue. Known-truth frames check the REQ-007 proxy branches,
basis-consistent growth, point-in-time visibility, the configured part-time weight, and the
existing growth / revenue-per-employee math.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_aggregate.utils.fundamentals.employee_features import (
    EMPLOYEE_COLUMNS,
    _employee_fields,
    attach_headcount,
    build_employee_feature_panel,
    headcount_proxy,
    part_time_weight,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _employees(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """A `fundamentals_employees` projection; missing components are NULL."""
    frame = pd.DataFrame(rows)
    for column in EMPLOYEE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    for column in ("employees_total", "employees_full_time", "employees_part_time"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame[list(EMPLOYEE_COLUMNS)]


def _history(as_of: list[str], fiscal_end: list[str], revenue: list[float], ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "as_of": pd.to_datetime(as_of),
            "fiscal_end": pd.to_datetime(fiscal_end),
            "totalRevenue": revenue,
        }
    )


def _fields(history: pd.DataFrame, employees: pd.DataFrame, idx: pd.DatetimeIndex, alpha: float = 0.5) -> dict:
    return _employee_fields(attach_headcount(history, headcount_proxy(employees, alpha)), idx)


# --------------------------------------------------------------------------- #
# REQ-007 proxy branches                                                       #
# --------------------------------------------------------------------------- #
def test_proxy_full_and_part_time_is_ft_plus_alpha_pt():
    employees = _employees(
        [
            {
                "ticker": "DLT",
                "as_of": "2025-03-20",
                "employees_total": 199_327,
                "employees_full_time": 60_217,
                "employees_part_time": 139_110,
                "basis": "full_part",
            }
        ]
    )
    for alpha, expected in ((0.5, 60_217 + 0.5 * 139_110), (1.0, 199_327.0)):
        proxy = headcount_proxy(employees, alpha)
        assert proxy["headcount"].tolist() == [expected]
        assert proxy["basis"].tolist() == ["full_part"]
    print("\n=== SANITY CHECK: proxy FT + a*PT ===")
    print(f"  DLTR-shape 60,217 FT + 139,110 PT: a=0.5 -> {60_217 + 0.5 * 139_110:,.1f}; a=1 -> 199,327 (the stated total). Validated.")


def test_proxy_total_and_full_time_weights_the_rest_by_alpha():
    employees = _employees([{"ticker": "AAA", "as_of": "2025-03-20", "employees_total": 1_500, "employees_full_time": 1_000, "basis": "full_part"}])
    assert headcount_proxy(employees, 0.5)["headcount"].tolist() == [1_250.0]
    assert headcount_proxy(employees, 1.0)["headcount"].tolist() == [1_500.0]
    assert headcount_proxy(employees, 0.5)["basis"].tolist() == ["full_part"]
    print("\n=== SANITY CHECK: proxy FT + a*(total - FT) ===")
    print("  total 1,500 with FT 1,000: a=0.5 -> 1,250; a=1 -> 1,500; no part-time component is stored. Validated.")


def test_proxy_total_fte_and_full_time_only_are_taken_as_stated():
    employees = _employees(
        [
            {"ticker": "TOT", "as_of": "2025-03-20", "employees_total": 2_000, "basis": "total"},
            {"ticker": "FTE", "as_of": "2025-03-20", "employees_total": 800, "employees_full_time": 700, "basis": "fte"},
            {"ticker": "CON", "as_of": "2025-03-20", "employees_total": 900, "basis": "total_incl_contractors"},
            {"ticker": "FTO", "as_of": "2025-03-20", "employees_full_time": 450, "basis": "full_time_only"},
        ]
    )
    proxy = headcount_proxy(employees, 0.5).set_index("ticker")
    assert proxy.loc["TOT", "headcount"] == 2_000.0 and proxy.loc["TOT", "basis"] == "total"
    assert proxy.loc["FTE", "headcount"] == 800.0 and proxy.loc["FTE", "basis"] == "fte"
    assert proxy.loc["CON", "headcount"] == 900.0 and proxy.loc["CON", "basis"] == "total_incl_contractors"
    assert proxy.loc["FTO", "headcount"] == 450.0 and proxy.loc["FTO", "basis"] == "full_time_only"
    print("\n=== SANITY CHECK: single-figure proxies ===")
    print(
        "  total 2,000 -> 2,000; FTE 800 (FT 700 ignored) -> 800; contractor total 900 -> 900; FT-only 450 -> 450; each keeps its basis. Validated."
    )


def test_status_only_rows_are_ignored_and_do_not_cut_the_carry():
    employees = _employees(
        [
            {"ticker": "AAA", "as_of": "2020-02-03", "employees_total": 1_000, "basis": "total"},
            {"ticker": "AAA", "as_of": "2020-08-03"},  # a 10-K/A stating nothing usable
        ]
    )
    assert len(headcount_proxy(employees, 0.5)) == 1
    history = _history(["2020-02-03", "2020-05-01", "2020-11-02"], ["2019-12-31", "2020-03-31", "2020-09-30"], [1e6, 1e6, 1e6])
    joined = attach_headcount(history, headcount_proxy(employees, 0.5))
    assert joined["headcount"].tolist() == [1_000.0, 1_000.0, 1_000.0]
    print("\n=== SANITY CHECK: status-only rows ===")
    print("  The NULL-component row is dropped; the 1,000 count still reaches the November history row. Validated.")


# --------------------------------------------------------------------------- #
# point in time and the as-of join                                             #
# --------------------------------------------------------------------------- #
def test_employee_row_filed_after_a_history_row_is_invisible_to_it():
    history = _history(["2020-02-01", "2020-05-01"], ["2019-12-31", "2020-03-31"], [1e6, 1e6])
    employees = _employees([{"ticker": "AAA", "as_of": "2020-02-05", "employees_total": 1_000, "basis": "total"}])
    joined = attach_headcount(history, headcount_proxy(employees, 0.5))
    assert pd.isna(joined.loc[0, "headcount"]) and joined.loc[1, "headcount"] == 1_000.0

    idx = pd.date_range("2020-01-15", "2020-06-01")
    rpe = _fields(history, employees, idx)["revenue_per_employee"]["AAA"]
    assert rpe.loc[: pd.Timestamp("2020-04-30")].isna().all()
    assert rpe.loc[pd.Timestamp("2020-05-01")] == 1_000.0
    print("\n=== SANITY CHECK: point in time ===")
    print("  A count filed 2020-02-05 is not attached to the 2020-02-01 history row; it is first visible 2020-05-01. Validated.")


def test_join_tolerance_stops_a_stale_count():
    history = _history(["2020-02-03", "2021-03-01"], ["2019-12-31", "2020-12-31"], [1e6, 1e6])
    employees = _employees([{"ticker": "AAA", "as_of": "2020-02-03", "employees_total": 1_000, "basis": "total"}])
    joined = attach_headcount(history, headcount_proxy(employees, 0.5))
    assert joined["headcount"].iloc[0] == 1_000.0 and pd.isna(joined["headcount"].iloc[1])
    print("\n=== SANITY CHECK: as-of tolerance ===")
    print("  A count 392 days older than the history row (> 370-day tolerance) is not carried. Validated.")


# --------------------------------------------------------------------------- #
# basis-consistent growth                                                      #
# --------------------------------------------------------------------------- #
def test_growth_is_nan_across_a_basis_change_and_computed_within_one():
    history = _history(
        ["2018-02-01", "2019-02-01", "2020-02-03", "2021-02-01"],
        ["2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"],
        [400_000.0, 500_000.0, 600_000.0, 780_000.0],
    )
    employees = _employees(
        [
            {"ticker": "AAA", "as_of": "2018-02-01", "employees_total": 800, "basis": "total"},
            {"ticker": "AAA", "as_of": "2019-02-01", "employees_total": 1_000, "basis": "total"},
            {
                "ticker": "AAA",
                "as_of": "2020-02-03",
                "employees_total": 1_400,
                "employees_full_time": 1_000,
                "employees_part_time": 400,
                "basis": "full_part",
            },
            {
                "ticker": "AAA",
                "as_of": "2021-02-01",
                "employees_total": 1_820,
                "employees_full_time": 1_300,
                "employees_part_time": 520,
                "basis": "full_part",
            },
        ]
    )
    idx = pd.date_range("2018-01-15", "2021-03-01")
    F = _fields(history, employees, idx)
    before, switch, same = pd.Timestamp("2019-02-10"), pd.Timestamp("2020-02-10"), pd.Timestamp("2021-02-10")
    assert np.isclose(F["employee_growth"].loc[before, "AAA"], 0.25)
    assert np.isclose(F["headcount_elasticity"].loc[before, "AAA"], 1.0)
    for name in ("employee_growth", "revenue_per_employee_growth", "headcount_elasticity"):
        assert pd.isna(F[name].loc[switch, "AAA"]), f"{name} crossed the total -> full_part switch (or kept the stale pre-switch value)"
    assert F["revenue_per_employee"].loc[switch, "AAA"] == 600_000.0 / 1_200.0
    assert np.isclose(F["employee_growth"].loc[same, "AAA"], 1_560.0 / 1_200.0 - 1.0)
    assert np.isclose(F["revenue_per_employee_growth"].loc[same, "AAA"], (780_000.0 / 1_560.0) / 500.0 - 1.0)
    assert np.isclose(F["headcount_elasticity"].loc[same, "AAA"], 0.3 / 0.3)
    print("\n=== SANITY CHECK: basis-consistent growth ===")
    print("  total 800 -> 1,000: growth +25%, elasticity 1.0.")
    print(
        "  total 1,000 -> full_part proxy 1,200: growth, rpe growth and elasticity NaN (neither +20% nor the stale +25%); rpe = 500 uses the proxy."
    )
    print("  full_part 1,200 -> 1,560 a year later: growth +30%, rpe growth 0%, elasticity 1.0. Validated.")


# --------------------------------------------------------------------------- #
# the part-time weight comes from config                                       #
# --------------------------------------------------------------------------- #
def test_part_time_weight_is_read_from_build_cube_config_and_range_checked():
    cfg = OmegaConf.load(CONFIG_DIR / "build_cube.yml")
    alpha = part_time_weight(cfg.build_cube)
    assert alpha == float(cfg.build_cube.workforce.part_time_weight)
    assert 0.5 <= alpha <= 1.0
    for bad in (0.3, 1.2):
        with pytest.raises(ValueError, match="part_time_weight"):
            part_time_weight(OmegaConf.create({"workforce": {"part_time_weight": bad}}))
    with pytest.raises(ValueError, match="part_time_weight"):
        headcount_proxy(_employees([{"ticker": "AAA", "as_of": "2020-01-01", "employees_total": 1, "basis": "total"}]), 0.4)
    print("\n=== SANITY CHECK: part-time weight ===")
    print(f"  configs/build_cube.yml build_cube.workforce.part_time_weight = {alpha}; 0.3, 0.4 and 1.2 raise. Validated.")


def test_step_loads_the_projected_employee_table_with_two_years_of_lookback():
    from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
    from src.data_store.schema import Tables

    calls: list[dict[str, Any]] = []

    class _Store:
        def load(self, table: object, columns: object = None, where: object = None, **kwargs: object) -> pd.DataFrame:
            calls.append({"table": table, "columns": columns, "where": where, **kwargs})
            return _employees([{"ticker": "AAA", "as_of": "2020-01-01", "employees_total": 1, "basis": "total"}])

    step = cast(Any, object.__new__(StepCubeFundamentals))
    step._context = SimpleNamespace(store=_Store())
    step._log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    frame = step._load_employees(("AAA", "BBB"), since=pd.Timestamp("2020-06-30"))
    assert frame is not None and len(calls) == 1
    call = calls[0]
    assert call["table"] == Tables.fundamentals_employees
    assert list(call["columns"]) == list(EMPLOYEE_COLUMNS)
    assert call["where"] == {"ticker": ["AAA", "BBB"]}
    assert call["since"] == pd.Timestamp("2018-06-30") and call["optional"] is True
    step._load_employees(("AAA",), since=None)
    assert calls[1]["since"] is None
    print("\n=== SANITY CHECK: employee load ===")
    print(f"  {Tables.fundamentals_employees} projected to {list(EMPLOYEE_COLUMNS)}, universe-scoped, since 2020-06-30 - 2y = 2018-06-30. Validated.")


# --------------------------------------------------------------------------- #
# growth and revenue-per-employee math                                         #
# --------------------------------------------------------------------------- #
def test_employee_fields_pit_growth_and_rev_per_employee():
    # two annual filings a year apart: 1,000 -> 1,200 employees (+20% YoY); fiscal dates read from `period`
    history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")],
            "period": [pd.Timestamp("2018-12-31"), pd.Timestamp("2019-12-31")],
            "totalRevenue": [np.nan, 600_000.0],
        }
    )
    employees = _employees(
        [
            {"ticker": "AAA", "as_of": "2019-02-01", "employees_total": 1_000, "basis": "total"},
            {"ticker": "AAA", "as_of": "2020-02-03", "employees_total": 1_200, "basis": "total"},
        ]
    )
    idx = pd.bdate_range("2018-06-01", "2020-06-01")
    F = _fields(history, employees, idx)

    before_any = pd.Timestamp("2018-11-30")
    after_second = pd.Timestamp("2020-03-02")
    for name, frame in F.items():
        assert np.isnan(frame.loc[before_any, "AAA"]), f"{name} leaked before first as_of"
    growth = F["employee_growth"].loc[after_second, "AAA"]
    assert abs(growth - 0.20) < 0.02, f"expected ~+20% headcount growth, got {growth}"
    assert abs(F["revenue_per_employee"].loc[after_second, "AAA"] - 500.0) < 1e-9

    print("\n=== SANITY CHECK: workforce features (10-K headcount history) ===")
    print(f"  headcount 1,000 -> 1,200: YoY growth = {growth:+.1%} (expected +20%)")
    print(f"  revenue/employee = 600,000/1,200 = {F['revenue_per_employee'].loc[after_second, 'AAA']:.0f}")
    print("  Both NaN before the first filing -> historical & leak-free. Validated.")


def test_revenue_per_employee_growth():
    """rev/employee GROWTH: revenue outgrowing headcount; past-vs-past, leak-free."""
    history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")],
            "totalRevenue": [500_000.0, 720_000.0],  # rev/emp 500 -> 600 (+20%)
        }
    )
    employees = _employees(
        [
            {"ticker": "AAA", "as_of": "2019-02-01", "employees_total": 1_000, "basis": "total"},
            {"ticker": "AAA", "as_of": "2020-02-03", "employees_total": 1_200, "basis": "total"},
        ]
    )
    idx = pd.bdate_range("2018-06-01", "2020-06-01")
    F = _fields(history, employees, idx)
    after = pd.Timestamp("2020-03-02")
    assert "revenue_per_employee_growth" in F
    g = F["revenue_per_employee_growth"].loc[after, "AAA"]
    assert abs(g - 0.20) < 0.03, f"expected ~+20% rev/employee growth, got {g}"
    assert pd.isna(F["revenue_per_employee_growth"].loc[pd.Timestamp("2018-11-30"), "AAA"])
    print("\n=== SANITY CHECK: revenue-per-employee GROWTH ===")
    print(f"  rev/emp 500 -> 600 => growth {g:+.1%} (~+20%, revenue outrunning headcount). Validated.")


def test_rev_per_employee_growth_handles_inf_no_crash():
    """A zero prior-period revenue-per-employee makes the YoY growth inf; it must become NaN without raising."""
    history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "ZZZ", "ZZZ"],
            "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")] * 2,
            "totalRevenue": [500_000.0, 720_000.0, 0.0, 500_000.0],  # AAA 500->600 (+20%); ZZZ 0->5000 (inf)
        }
    )
    employees = _employees(
        [
            {"ticker": ticker, "as_of": as_of, "employees_total": count, "basis": "total"}
            for ticker, as_of, count in (
                ("AAA", "2019-02-01", 1_000),
                ("AAA", "2020-02-03", 1_200),
                ("ZZZ", "2019-02-01", 100),
                ("ZZZ", "2020-02-03", 100),
            )
        ]
    )
    idx = pd.bdate_range("2018-06-01", "2020-06-01")

    F = _fields(history, employees, idx)  # must NOT raise IndexError
    assert "revenue_per_employee_growth" in F
    g = F["revenue_per_employee_growth"]
    after = pd.Timestamp("2020-03-02")
    assert abs(g.loc[after, "AAA"] - 0.20) < 0.03
    assert not np.isinf(g.to_numpy()).any(), "inf not scrubbed -> would poison the z-score"
    assert "ZZZ" not in g.columns or pd.isna(g.loc[after, "ZZZ"])
    print("\n=== SANITY CHECK: rev/employee growth inf handling ===")
    print(f"  AAA finite +{g.loc[after, 'AAA']:.0%}; ZZZ 0->5000 gives +inf -> scrubbed to NaN, no IndexError. Validated.")


def test_employee_features_follow_fiscal_year_and_expire_after_460_days():
    """Annual workforce facts match fiscal years, not publication-date spacing."""
    history = _history(["2020-02-15", "2021-05-10"], ["2019-12-31", "2020-12-31"], [500_000.0, 720_000.0])
    employees = _employees(
        [
            {"ticker": "AAA", "as_of": "2020-02-15", "employees_total": 1_000, "basis": "total"},
            {"ticker": "AAA", "as_of": "2021-05-10", "employees_total": 1_200, "basis": "total"},
        ]
    )
    idx = pd.date_range("2021-05-10", "2022-08-15")

    fields = _fields(history, employees, idx)

    assert np.isclose(fields["employee_growth"].loc[pd.Timestamp("2021-05-10"), "AAA"], 0.2)
    assert np.isclose(fields["revenue_per_employee"].loc[pd.Timestamp("2022-08-13"), "AAA"], 600.0)
    assert pd.isna(fields["revenue_per_employee"].loc[pd.Timestamp("2022-08-14"), "AAA"])
    print("\n=== SANITY CHECK: fiscal workforce alignment and freshness ===")
    print("  Irregular filing dates still match adjacent fiscal years; day 460 is valid and day 461 is null. Validated.")


def test_build_panel_empty_without_history_or_employees():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    history = _history(["2020-02-03"], ["2019-12-31"], [1e6])
    employees = _employees([{"ticker": "AAA", "as_of": "2020-02-03", "employees_total": 1_000, "basis": "total"}])
    for fundamentals, headcount in ((None, employees), (history, None), (history, employees.iloc[0:0])):
        panel = build_employee_feature_panel(fundamentals, headcount, {"AAA": ["BBB"]}, idx, part_time_weight=0.5)
        assert list(panel.columns) == ["date", "ticker"] and panel.empty

    print("\n=== SANITY CHECK: graceful skip ===")
    print("  No history, no employee table, or an empty one -> empty (date,ticker) panel, no crash. Validated.")
