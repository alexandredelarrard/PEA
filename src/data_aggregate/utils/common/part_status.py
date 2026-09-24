"""
part_status.py  (src/data_aggregate/utils/common/part_status.py)
-------------------------------------------------------------
The cube-parts status report: latest date + row count of every part (and of the downstream
cube / predictions tables), so drift is visible.

THE RETURNED SHAPE IS A CONTRACT. `src/dags/dag_data_aggregation.py::_cube_status` runs the
`cube-status` CLI command, parses the last JSON line of stdout, pushes it to XCom and reads
exactly three things:

    report["ok"]                        -> bool, fails the task when False
    report["behind"]                     -> [str], named in the failure message
    report["parts"][name]["max_date"]    -> pushed as XCom key `max_<name>`

so those keys must keep their names and meanings. Only the SET of part names changes with
this refactor, and `_cube_status` iterates it generically.

The part list comes from the `parts.py` registry rather than being hard-coded here, which
is what let the old version report `cube_part_attention` as permanently missing: that group
was commented out of the DAG but never removed from the dict the status walked.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES
from src.data_aggregate.utils.institutionals.insider_sources import (
    as_quarter,
    bulk_complete_through,
)
from src.data_store.schema import Tables

# more than ~one build behind the cube is a gap worth attention
_LAG_TOLERANCE_DAYS = 4


def _fmt(value: pd.Timestamp | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None


def _insider_source_status(
    context: Context,
    part_max_date: str | None,
    tolerance_days: int,
) -> dict[str, object]:
    """Canonical bulk/live completeness versus the institutional part edge."""
    _, latest_quarter = context.store.bounds(Tables.insider_transactions, "quarter")
    latest_period = as_quarter(latest_quarter)
    bulk = None
    live = None
    if latest_period is not None:
        bulk = context.store.load(
            Tables.insider_transactions,
            columns=("accession_number", "filing_date"),
            where={"quarter": latest_quarter},
            optional=True,
        )
        live = context.store.load(
            Tables.insider_transactions_live,
            columns=("accession_number", "filing_date"),
            since=latest_period.start_time.normalize(),
            until=latest_period.end_time.normalize(),
            date_col="filing_date",
            optional=True,
        )
    source_config = getattr(getattr(context, "config", {}), "source_freshness", {})
    bulk_cutover = source_config.get("insider_bulk_authoritative_through")
    bulk_frontier = bulk_complete_through(
        latest_quarter,
        bulk,
        live,
        bulk_authoritative_through=bulk_cutover,
    )

    price_max = context.store.max_date(Tables.cube_part_prices)
    expected = set(
        map(
            str,
            context.store.distinct(
                Tables.cube_part_prices,
                "ticker",
                where={"date": price_max} if price_max is not None else None,
            ),
        )
    )
    coverage = context.store.load(
        Tables.insider_transactions_live_coverage,
        columns=("ticker", "complete_through"),
        where={"ticker": sorted(expected)} if expected else None,
        optional=True,
    )
    live_frontier = None
    covered: set[str] = set()
    if coverage is not None and not coverage.empty:
        coverage = coverage.dropna(subset=["ticker", "complete_through"])
        covered = set(coverage["ticker"].astype(str))
        if expected and expected.issubset(covered):
            live_frontier = pd.to_datetime(coverage.groupby("ticker")["complete_through"].max(), errors="coerce").min()
            live_frontier = live_frontier.normalize() if pd.notna(live_frontier) else None

    candidates = [value for value in (bulk_frontier, live_frontier) if value is not None]
    complete_through = max(candidates) if candidates else None
    lag_days = None
    if part_max_date is not None and complete_through is not None:
        lag_days = int((pd.Timestamp(part_max_date) - complete_through).days)
    ok = bool(part_max_date is not None and complete_through is not None and lag_days is not None and lag_days <= tolerance_days)
    return {
        "bulk_reported_quarter": str(latest_quarter) if latest_quarter is not None else None,
        "bulk_authoritative_through": str(bulk_cutover) if bulk_cutover else None,
        "bulk_complete_through": _fmt(bulk_frontier),
        "live_complete_through": _fmt(live_frontier),
        "complete_through": _fmt(complete_through),
        "part_max_date": part_max_date,
        "lag_days": lag_days,
        "tolerance_days": tolerance_days,
        "covered_tickers": len(covered & expected),
        "expected_tickers": len(expected),
        "ok": ok,
    }


def part_status_report(context: Context, log: logging.Logger | None = None) -> dict:
    """Report every cube part + the downstream tables. See the module docstring for the
    contract on the returned shape."""
    log = log or logging.getLogger(__name__)
    store = context.store

    # the market part is ALWAYS fully replaced by build-prices, so its max date is by
    # construction the prices part's -- reporting it as "behind" would be noise
    # `.name`, not the `Table` object: these become the KEYS of report["parts"], which the DAG
    # pushes as XCom JSON (`max_<name>`) -- a Table key is neither serialisable nor the shape
    # `dag_data_aggregation._cube_status` reads.
    terminal = [t.name for t in TERMINAL_TABLES]
    names = [p.name for p in CUBE_PARTS] + terminal
    never_behind = {p.name for p in CUBE_PARTS if p.kind == "market"} | set(terminal)

    parts: dict[str, dict] = {}
    for name in names:
        info: dict = {"exists": False, "max_date": None, "rows": None}
        if context.store.exists(name):
            mx = store.max_date(name)
            info = {"exists": True, "max_date": mx.strftime("%Y-%m-%d") if mx is not None else None, "rows": store.row_count(name)}
        parts[name] = info

    cube_max = parts.get("cube", {}).get("max_date")
    behind: list[str] = []
    if cube_max is not None:
        cmax = pd.Timestamp(cube_max)
        for name, info in parts.items():
            if name in never_behind:
                continue
            if info["exists"] and info["max_date"] is not None:
                lag = int((cmax - pd.Timestamp(info["max_date"])).days)
                info["lag_vs_cube_days"] = lag
                if lag > _LAG_TOLERANCE_DAYS:
                    behind.append(name)
            elif not info["exists"]:
                behind.append(name)

    source_config = getattr(context.config, "source_freshness", {})
    insider_tolerance = int(source_config.get("insider_max_lag_days", _LAG_TOLERANCE_DAYS))
    insider_status = _insider_source_status(
        context,
        parts.get(Tables.cube_part_institutionals.name, {}).get("max_date"),
        insider_tolerance,
    )
    source_name = f"{Tables.cube_part_institutionals.name}:insider_transactions"
    if not insider_status["ok"] and source_name not in behind:
        behind.append(source_name)
    sources = {
        Tables.cube_part_institutionals.name: {
            "insider_transactions": insider_status,
        }
    }

    report = {
        "as_of": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
        "cube_max_date": cube_max,
        "ok": not behind,
        "behind": behind,
        "parts": parts,
        "sources": sources,
    }

    log.info("=== Cube parts status @ %s (cube max=%s, ok=%s) ===", report["as_of"], cube_max, report["ok"])
    for name, info in parts.items():
        log.info(
            "  %-26s exists=%-5s max=%-11s rows=%-9s lag_vs_cube=%s",
            name,
            info["exists"],
            info["max_date"] or "-",
            info["rows"] if info["rows"] is not None else "-",
            info.get("lag_vs_cube_days", "-"),
        )
    if behind:
        log.warning("Cube parts BEHIND / missing (%d): %s", len(behind), ", ".join(behind))
    log.info("  insider source: %s", insider_status)
    return report
