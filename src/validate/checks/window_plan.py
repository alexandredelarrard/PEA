"""Independent price-grid and calendar-input checks for actual cube planner traces."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.validate.result import CheckResult, Finding


def _subtract_days(anchor: pd.Timestamp, days: object) -> pd.Timestamp:
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ValueError("Calendar history must be a nonnegative integer")
    return anchor - pd.Timedelta(days=days)


def check_window_plans(context: Context, replay: dict[str, Any]) -> dict[str, CheckResult]:
    """Compare real planner footprints to raw configuration; never call Window arithmetic."""
    sessions = pd.DatetimeIndex(pd.to_datetime(replay["sessions"]))
    trace = replay.get("planning")
    if trace is None or sessions.empty:
        return {Tables.cube.name: CheckResult.abstained("window_plan", Tables.cube.name, "No actual planner trace or observed price dates")}
    if sessions.hasnans or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Planner evidence needs ordered unique observed price dates")
    results = {}
    for name, plan in trace["parts"].items():
        results[name] = _check_part(context, name, plan, trace, sessions)
    plan = trace["assembly"]
    if plan is None:
        results[Tables.cube.name] = CheckResult.abstained("window_plan", Tables.cube.name, "Actual assembly planner returned no window")
        return results
    selected = trace["selected_components"]
    end = min(pd.Timestamp(trace["as_of"]), sessions[-1])
    start = sessions[0]
    if plan["mode"] == "incremental":
        component_start = min(pd.Timestamp(trace["parts"][name]["write"]["start"]) for name in selected)
        catchup = pd.Timestamp(plan["frontier"]) if plan["frontier"] is not None else sessions[0]
        start = max(sessions[0], min(component_start, catchup, end))
    findings = []
    if pd.Timestamp(plan["write"]["start"]) != start:
        findings.append(
            Finding.at(
                8,
                "Assembly did not inherit actual selected component dates and catch-up",
                "Observed price dates, selected component writes and scoped catch-up",
                expected_start=start,
                actual=plan["write"]["start"],
            )
        )
    budget = context.config.data.tables[Tables.cube.name].warmup[Tables.cube_part_prices.name]
    expected = {
        "requested": {"start": sessions[0], "end": end},
        "write": {"start": start, "end": end},
        "read": {"start": sessions[0] if budget is None else _subtract_days(start, budget), "end": end},
    }
    for section, bounds in expected.items():
        if any(pd.Timestamp(plan[section][side]) != day for side, day in bounds.items()):
            findings.append(
                Finding.at(
                    8,
                    f"Assembly {section} differs from independent configured bounds",
                    "Root as-of, selected component dates, scoped catch-up and inclusive end",
                    expected_bounds=bounds,
                    actual=plan[section],
                )
            )
    if set(plan["keys"].get("ticker", [])) != set(trace["tickers"]):
        findings.append(Finding.at(8, "Assembly planner changed the pinned ticker scope", "Exact five-key scope"))
    configured = context.config.data_aggregate.assembly
    if set(context.config.data.tables[Tables.cube.name].warmup) != set([*configured.compulsory_parts, *configured.optional_parts]):
        findings.append(Finding.at(8, "Assembly input map differs from its configured selected components", "Every selected dependency is explicit"))
    results[Tables.cube.name] = CheckResult.measured(
        "window_plan",
        Tables.cube.name,
        findings,
        scope={"tickers": trace["tickers"], "selected_components": selected},
        metrics={"actual": plan, "expected_write_start": start},
        reason="Planning only; separate from finite probe writes and feature lineage.",
    )
    return results


def _check_part(context: Context, name: str, plan: dict[str, Any], trace: dict[str, Any], sessions: pd.DatetimeIndex) -> CheckResult:
    if "reason" in plan:
        return CheckResult.abstained("window_plan", name, plan["reason"])
    item = context.config.data.tables[name]
    requested_start, requested_end = sessions[0], min(pd.Timestamp(trace["as_of"]), sessions[-1])
    start = requested_start
    if plan["mode"] == "incremental":
        frontier = min(pd.Timestamp(plan["frontier"]), requested_end)
        start = max(start, _subtract_days(frontier, item.overlap.days))
    source = Tables.prices if name == Tables.cube_part_prices.name else Tables.cube_part_prices
    if source.name not in item.warmup:
        raise ValueError(f"{name}: missing input history for {source.name}")
    budget = item.warmup[source.name]
    read_start = sessions[0] if budget is None else _subtract_days(start, budget)
    read_end = requested_end
    expected = {
        "requested": {"start": requested_start, "end": requested_end},
        "write": {"start": start, "end": requested_end},
        "read": {"start": read_start, "end": read_end},
    }
    findings = []
    for section, bounds in expected.items():
        if any(pd.Timestamp(plan[section][side]) != day for side, day in bounds.items()):
            findings.append(
                Finding.at(
                    8,
                    f"Actual {section} differs from configured dates/context",
                    "Observed prices, calendar overlap and rewrite-anchored input history",
                    expected_bounds=bounds,
                    actual=plan[section],
                )
            )
    if set(plan["keys"].get("ticker", [])) != set(trace["tickers"]):
        findings.append(Finding.at(8, "Actual planner changed the pinned ticker scope", "Exact five-key scope"))
    return CheckResult.measured(
        "window_plan",
        name,
        findings,
        scope={"tickers": trace["tickers"], "role": "forward outcome labels" if name == Tables.cube_part_targets.name else "feature part"},
        metrics={"actual": plan, "expected": expected, "price_input": source.name, "additional_calendar_days": budget},
        reason="Actual production plan and scoped frontier; no cube rebuild or production write.",
    )
