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
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_store.schema import Tables
from src.data_store.store import DataStore

# more than ~one build behind the cube is a gap worth attention
_LAG_TOLERANCE_DAYS = 4


def _fmt(value: pd.Timestamp | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None and pd.notna(value) else None


def _edge_detail(
    part_max: pd.Timestamp | None,
    price_max: pd.Timestamp | None,
) -> dict[str, object]:
    """Describe one table edge against prices; positive delta means ahead."""
    if part_max is None or price_max is None:
        return {"status": "missing", "max_date": _fmt(part_max), "delta_days": None}
    delta = int((part_max.normalize() - price_max.normalize()).days)
    status = "aligned" if delta == 0 else "behind" if delta < 0 else "ahead"
    return {"status": status, "max_date": _fmt(part_max), "delta_days": delta}


def cube_part_edge_report(store: DataStore) -> dict[str, object]:
    """Exact max-date alignment of every registered part to persisted prices."""
    price_name = Tables.cube_part_prices.name
    price_max = store.max_date(price_name) if store.exists(price_name) else None
    details: dict[str, dict[str, object]] = {}
    misaligned: list[str] = []
    for part in CUBE_PARTS:
        part_max = store.max_date(part.name) if store.exists(part.name) else None
        detail = _edge_detail(part_max, price_max)
        details[part.name] = detail
        if detail["status"] != "aligned":
            misaligned.append(part.name)
    return {
        "price_max_date": _fmt(price_max),
        "misaligned": misaligned,
        "parts": details,
    }


def _insider_source_status(
    context: Context,
    part_max_date: str | None,
    tolerance_days: int,
) -> dict[str, object]:
    """EDGAR completeness of `insider_transactions` versus the institutional part edge.

    The frontier is the cube's DB rule (`schedule_complete_through`), anchored on the last
    `cube_part_prices` session."""
    store = context.store
    complete_through = schedule_complete_through(store, Tables.insider_transactions, store.max_date(Tables.cube_part_prices))
    lag_days = None
    if part_max_date is not None and complete_through is not None:
        lag_days = int((pd.Timestamp(part_max_date) - complete_through).days)
    ok = bool(lag_days is not None and lag_days <= tolerance_days)
    return {
        "complete_through": _fmt(complete_through),
        "part_max_date": part_max_date,
        "lag_days": lag_days,
        "tolerance_days": tolerance_days,
        "ok": ok,
    }


def part_status_report(context: Context, log: logging.Logger | None = None) -> dict:
    """Report every cube part + the downstream tables. See the module docstring for the
    contract on the returned shape."""
    log = log or logging.getLogger(__name__)
    store = context.store

    # `.name`, not the `Table` object: these become the KEYS of report["parts"], which the DAG
    # pushes as XCom JSON (`max_<name>`) -- a Table key is neither serialisable nor the shape
    # `dag_data_aggregation._cube_status` reads.
    terminal = [t.name for t in TERMINAL_TABLES]
    names = [p.name for p in CUBE_PARTS] + terminal
    edge = cube_part_edge_report(store)

    parts: dict[str, dict] = {}
    for name in names:
        info: dict = {"exists": False, "max_date": None, "rows": None}
        if context.store.exists(name):
            mx = store.max_date(name)
            info = {"exists": True, "max_date": mx.strftime("%Y-%m-%d") if mx is not None else None, "rows": store.row_count(name)}
        if name in edge["parts"]:
            detail = edge["parts"][name]
            info.update(
                edge_status=detail["status"],
                delta_vs_price_days=detail["delta_days"],
            )
        parts[name] = info

    cube_max = parts.get("cube", {}).get("max_date")
    price_max = edge["price_max_date"]
    cube_detail = _edge_detail(
        pd.Timestamp(cube_max) if cube_max is not None else None,
        pd.Timestamp(price_max) if price_max is not None else None,
    )
    parts[Tables.cube.name].update(
        edge_status=cube_detail["status"],
        delta_vs_price_days=cube_detail["delta_days"],
    )
    misaligned = list(edge["misaligned"])
    if cube_detail["status"] != "aligned":
        misaligned.append(Tables.cube.name)
    behind = list(misaligned)

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
        "max_date": cube_max,
        "cube_max_date": cube_max,
        "price_max_date": price_max,
        "misaligned": misaligned,
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
