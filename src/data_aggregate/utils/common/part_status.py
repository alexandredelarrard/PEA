"""Readiness at the latest universe source-price date; preserve CLI/DAG JSON fields."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.assembly import AssemblyPolicy, part_has_rows
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_store.schema import Tables
from src.utils.universe import load_universe_tickers

_LAG_TOLERANCE_DAYS = 4


def _fmt(value: pd.Timestamp | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None


def _edge_detail(part_max: pd.Timestamp | None, reference: pd.Timestamp | None) -> dict[str, Any]:
    if part_max is None or reference is None:
        return {"status": "missing", "max_date": _fmt(part_max), "delta_days": None}
    delta = int((part_max.normalize() - reference.normalize()).days)
    return {"status": "aligned" if delta == 0 else "behind" if delta < 0 else "ahead", "max_date": _fmt(part_max), "delta_days": delta}


def cube_part_edge_report(context: Context) -> dict[str, Any]:
    store = context.store
    policy = AssemblyPolicy.from_config(context.config.get("build_cube", {}))
    universe = set(load_universe_tickers(context))
    maxima = store.max_date_by(Tables.prices, "ticker")
    reference = max((day for ticker, day in maxima.items() if ticker in universe), default=None)
    details: dict[str, dict[str, Any]] = {}
    for part in CUBE_PARTS:
        present = part_has_rows(store, part)
        detail = _edge_detail(store.max_date(part.table) if present else None, reference)
        detail.update(required=part.name not in policy.optional, skipped=False, expected_date=_fmt(reference), refresh_command=part.command)
        if not present and part.name in policy.optional:
            detail.update(status="skipped", skipped=True, reason="absent or empty")
        details[part.name] = detail
    return {
        "price_max_date": _fmt(store.max_date(Tables.cube_part_prices)),
        "reference_price_date": _fmt(reference),
        "misaligned": [name for name, detail in details.items() if detail["status"] not in ("aligned", "skipped")],
        "parts": details,
    }


def _insider_source_status(context: Context, part_max_date: str | None, tolerance_days: int) -> dict[str, Any]:
    store = context.store
    complete_through = schedule_complete_through(store, Tables.insider_transactions, store.max_date(Tables.cube_part_prices))
    lag_days = int((pd.Timestamp(part_max_date) - complete_through).days) if part_max_date is not None and complete_through is not None else None
    return {
        "complete_through": _fmt(complete_through),
        "part_max_date": part_max_date,
        "lag_days": lag_days,
        "tolerance_days": tolerance_days,
        "ok": bool(lag_days is not None and lag_days <= tolerance_days),
    }


def part_status_report(context: Context, log: logging.Logger | None = None) -> dict[str, Any]:
    log = log or context.log
    store = context.store
    edge = cube_part_edge_report(context)
    parts: dict[str, dict[str, Any]] = {}
    for table in [*[part.table for part in CUBE_PARTS], *TERMINAL_TABLES]:
        exists = store.exists(table)
        info = {"exists": exists, "max_date": _fmt(store.max_date(table)) if exists else None, "rows": store.row_count(table) if exists else None}
        if table.name in edge["parts"]:
            detail = edge["parts"][table.name]
            info.update(detail)
            info.update(edge_status=detail["status"], delta_vs_price_days=detail["delta_days"])
        parts[table.name] = info
    reference = edge["reference_price_date"]
    cube_max = parts[Tables.cube.name]["max_date"]
    cube_detail = _edge_detail(pd.Timestamp(cube_max) if cube_max else None, pd.Timestamp(reference) if reference else None)
    parts[Tables.cube.name].update(
        edge_status=cube_detail["status"], delta_vs_price_days=cube_detail["delta_days"], expected_date=reference, refresh_command="assemble-cube"
    )
    misaligned = list(edge["misaligned"])
    if cube_detail["status"] != "aligned":
        misaligned.append(Tables.cube.name)
    behind = list(misaligned)
    if reference is None:
        behind.append(Tables.prices.name)
    institutional = parts[Tables.cube_part_institutionals.name]
    tolerance = int(context.config.get("source_freshness", {}).get("insider_max_lag_days", _LAG_TOLERANCE_DAYS))
    if institutional["skipped"]:
        insider_status = {"status": "skipped", "reason": "optional institutional part absent or empty", "ok": None}
    else:
        insider_status = _insider_source_status(context, institutional["max_date"], tolerance)
        if not insider_status["ok"]:
            behind.append(f"{Tables.cube_part_institutionals.name}:insider_transactions")
    sources = {
        Tables.prices.name: {
            "max_date": reference,
            "ok": reference is not None,
            "refresh_command": "price-history (seed-universe if universe is absent)",
        },
        Tables.cube_part_institutionals.name: {"insider_transactions": insider_status},
    }
    report = {
        "as_of": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
        "max_date": cube_max,
        "cube_max_date": cube_max,
        "price_max_date": edge["price_max_date"],
        "reference_price_date": reference,
        "misaligned": misaligned,
        "ok": not behind,
        "behind": behind,
        "parts": parts,
        "sources": sources,
    }
    log.info("Cube readiness: source-price=%s cube=%s ok=%s", reference, cube_max, report["ok"])
    for name in misaligned:
        info = parts[name]
        log.warning(
            "Refresh %s: expected=%s actual=%s status=%s; run %s", name, reference, info["max_date"], info["edge_status"], info["refresh_command"]
        )
    if reference is None:
        log.warning("No source-price reference for the analysis universe; run %s", sources[Tables.prices.name]["refresh_command"])
    if not institutional["skipped"] and not insider_status["ok"]:
        log.warning("Institutional insider source incomplete: %s", insider_status)
    return report
