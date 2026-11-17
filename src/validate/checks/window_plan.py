"""Independent configured-overlap checks for actual read-only cube planner traces."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.validate.result import CheckResult, Finding


def _move(anchor: pd.Timestamp, offset: dict[str, Any], sessions: pd.DatetimeIndex, direction: int = -1) -> pd.Timestamp:
    amount = int(offset["value"])
    if offset["unit"] == "calendar_days":
        return anchor + pd.Timedelta(days=direction * amount)
    if offset["unit"] != "trading_sessions":
        raise ValueError("Cube plan oracle requires declared calendar-day or trading-session units")
    if not amount:
        return anchor
    prior = [day for day in sessions if day < anchor] if direction < 0 else [day for day in sessions if day <= anchor]
    position = len(prior) if direction < 0 else len(prior) - 1
    return sessions[max(0, min(len(sessions) - 1, position + direction * amount))]


def check_window_plans(context: Context, replay: dict[str, Any]) -> dict[str, CheckResult]:
    """Compare real planner footprints to raw configuration; never call Window arithmetic."""
    sessions = pd.DatetimeIndex(pd.to_datetime(replay["sessions"]))
    trace = replay.get("planning")
    if trace is None:
        return {Tables.cube.name: CheckResult.abstained("window_plan", Tables.cube.name, "No actual production planner trace")}
    results = {}
    for name, plan in trace["parts"].items():
        results[name] = _check_part(context, name, plan, trace, sessions)
    plan = trace["assembly"]
    if plan is None:
        results[Tables.cube.name] = CheckResult.abstained("window_plan", Tables.cube.name, "Actual assembly planner returned no window")
        return results
    selected = trace["selected_components"]
    start = max(
        pd.Timestamp(context.config.data.tables[Tables.cube.name].start_date),
        min(pd.Timestamp(trace["parts"][name]["write"]["start"]) for name in selected),
    )
    if plan["mode"] == "incremental":
        catchup = pd.Timestamp(plan["frontier"]) if plan["frontier"] is not None else pd.Timestamp(plan["requested"]["start"])
        start = max(pd.Timestamp(plan["requested"]["start"]), min(start, catchup, pd.Timestamp(plan["requested"]["end"])))
    else:
        start = pd.Timestamp(plan["requested"]["start"])
    findings = []
    if pd.Timestamp(plan["write"]["start"]) != start:
        findings.append(
            Finding.at(
                8,
                "Assembly did not inherit actual selected component dates and catch-up",
                "Compare resolved CD/S dates rather than raw offset numbers",
                expected_start=start,
                actual=plan["write"]["start"],
            )
        )
    item = context.config.data.tables[Tables.cube.name]
    end = min(pd.Timestamp(trace["as_of"]), sessions[-1], pd.Timestamp(item.end_date) if item.end_date is not None else sessions[-1])
    expected = {
        "requested": {"start": pd.Timestamp(item.start_date), "end": end},
        "write": {"start": start, "end": end},
        "read": {"start": start, "end": end},
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
    requested_start = max(
        pd.Timestamp(item.start_date), pd.Timestamp(item.source_start) if item.source_start is not None else pd.Timestamp(item.start_date)
    )
    requested_end = min(pd.Timestamp(trace["as_of"]), sessions[-1], pd.Timestamp(item.end_date) if item.end_date is not None else sessions[-1])
    start = requested_start
    if plan["mode"] == "incremental":
        frontier = min(pd.Timestamp(plan["frontier"]), requested_end)
        start = max(start, _move(frontier, dict(item.reprocess.overlap), sessions))
    grid = item.warmup.grid
    read_start = sessions[0] if grid.mode == "full_history" else min(start, _move(start, {"value": grid.value, "unit": grid.unit}, sessions))
    read_end = requested_end
    if "lookahead" in item:
        read_end = min(
            pd.Timestamp(trace["as_of"]),
            max(
                requested_end,
                _move(requested_end, {"value": max(context.config.data_aggregate.targets.horizons), "unit": "trading_sessions"}, sessions, 1),
            ),
        )
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
                    "Independent inclusive CD/S planning oracle",
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
        metrics={"actual": plan, "expected": expected},
        reason="Actual production plan and scoped frontier; no cube rebuild or production write.",
    )
