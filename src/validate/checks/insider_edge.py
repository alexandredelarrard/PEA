"""Acceptance check for insider-derived features at the prediction edge."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

INSIDER_EDGE_COLUMNS = (
    "f_ic_insider_discretionary_sell_mcap_60d",
    "f_ic_insider_discretionary_sell_mcap_60d_xs",
    "f_ic_insider_planned_sell_mcap_60d",
    "f_ic_insider_planned_sell_mcap_60d_xs",
    "f_ic_insider_cluster_buy_120d",
    "f_ic_insider_distinct_buyers_120d",
)


@dataclass(frozen=True)
class InsiderEdgeResult:
    """Machine-readable edge measurements and their overall verdict."""

    passed: bool
    frontier: pd.Timestamp
    edge_dates: tuple[pd.Timestamp, ...]
    observations: tuple[dict[str, object], ...]
    failures: tuple[dict[str, object], ...]


def evaluate_insider_edge(
    frame: pd.DataFrame,
    complete_through: pd.Timestamp,
    *,
    edge_sessions: int = 5,
    min_tickers: int = 20,
    max_degenerate_distinct: int = 2,
) -> InsiderEdgeResult:
    """Require covered edge cells to vary and uncovered edge cells to be entirely NULL."""
    required = {"date", "ticker", *INSIDER_EDGE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"insider edge frame is missing columns: {missing}")
    if edge_sessions < 1:
        raise ValueError("edge_sessions must be positive")

    data = frame.loc[:, sorted(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", "ticker"])
    dates = tuple(sorted(data["date"].unique())[-edge_sessions:])
    if not dates:
        raise ValueError("insider edge frame contains no dated rows")

    frontier = pd.Timestamp(complete_through).normalize()
    observations: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for date in dates:
        day = data.loc[data["date"] == date]
        covered = pd.Timestamp(date) <= frontier
        for column in INSIDER_EDGE_COLUMNS:
            values = pd.to_numeric(day[column], errors="coerce")
            non_null = int(values.notna().sum())
            distinct = int(values.nunique(dropna=True))
            standard_deviation = float(values.std(ddof=0)) if non_null else None
            if covered:
                acceptable = (
                    non_null >= min_tickers and distinct > max_degenerate_distinct and standard_deviation is not None and standard_deviation > 0.0
                )
                expected = "covered_with_cross_sectional_spread"
            else:
                acceptable = non_null == 0
                expected = "uncovered_and_all_null"
            observation: dict[str, object] = {
                "date": str(pd.Timestamp(date).date()),
                "feature": column,
                "covered": covered,
                "n_tickers": int(day["ticker"].nunique()),
                "n_non_null": non_null,
                "n_distinct": distinct,
                "standard_deviation": standard_deviation,
                "expected": expected,
                "ok": acceptable,
            }
            observations.append(observation)
            if not acceptable:
                failures.append(observation)

    return InsiderEdgeResult(
        passed=not failures,
        frontier=frontier,
        edge_dates=tuple(pd.Timestamp(date) for date in dates),
        observations=tuple(observations),
        failures=tuple(failures),
    )
