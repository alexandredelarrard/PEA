from __future__ import annotations

import pandas as pd

from src.validate.checks.insider_edge import (
    INSIDER_EDGE_COLUMNS,
    evaluate_insider_edge,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in pd.date_range("2026-09-01", periods=3, freq="B"):
        for index in range(25):
            row: dict[str, object] = {"date": date, "ticker": f"T{index:02d}"}
            row.update({column: float(index + offset) for offset, column in enumerate(INSIDER_EDGE_COLUMNS)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_covered_edge_requires_cross_sectional_spread() -> None:
    frame = _frame()
    result = evaluate_insider_edge(frame, pd.Timestamp("2026-09-30"))

    assert result.passed
    assert not result.failures
    print("sanity: covered insider edge passes only with a populated cross-sectional spread")


def test_uncovered_edge_requires_all_null_and_rejects_confident_zero() -> None:
    frame = _frame()
    frontier = pd.Timestamp("2026-09-01")
    uncovered = pd.to_datetime(frame["date"]) > frontier
    frame.loc[uncovered, list(INSIDER_EDGE_COLUMNS)] = pd.NA

    honest = evaluate_insider_edge(frame, frontier)
    assert honest.passed

    frame.loc[uncovered, INSIDER_EDGE_COLUMNS[0]] = 0.0
    fabricated = evaluate_insider_edge(frame, frontier)
    assert not fabricated.passed
    assert {failure["feature"] for failure in fabricated.failures} == {INSIDER_EDGE_COLUMNS[0]}
    print("sanity: unavailable insider dates accept NULL and reject universe-wide zero")
