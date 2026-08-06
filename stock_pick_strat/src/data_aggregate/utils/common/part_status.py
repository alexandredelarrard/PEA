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
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import CUBE_PARTS, TERMINAL_TABLES

# more than ~one build behind the cube is a gap worth attention
_LAG_TOLERANCE_DAYS = 4


def part_status_report(context: Context, log: logging.Logger | None = None) -> dict:
    """Report every cube part + the downstream tables. See the module docstring for the
    contract on the returned shape."""
    log = log or logging.getLogger(__name__)
    parts_io = PartStore(context.store, log)

    # the market part is ALWAYS fully replaced by build-prices, so its max date is by
    # construction the prices part's -- reporting it as "behind" would be noise
    names = [p.name for p in CUBE_PARTS] + list(TERMINAL_TABLES)
    never_behind = {p.name for p in CUBE_PARTS if p.kind == "market"} | set(TERMINAL_TABLES)

    parts: dict[str, dict] = {}
    for name in names:
        info: dict = {"exists": False, "max_date": None, "rows": None}
        if context.store.exists(name):
            mx = parts_io.max_date(name)
            info = {"exists": True,
                    "max_date": mx.strftime("%Y-%m-%d") if mx is not None else None,
                    "rows": parts_io.row_count(name)}
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

    report = {"as_of": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
              "cube_max_date": cube_max, "ok": not behind, "behind": behind, "parts": parts}

    log.info("=== Cube parts status @ %s (cube max=%s, ok=%s) ===",
             report["as_of"], cube_max, report["ok"])
    for name, info in parts.items():
        log.info("  %-26s exists=%-5s max=%-11s rows=%-9s lag_vs_cube=%s",
                 name, info["exists"], info["max_date"] or "-",
                 info["rows"] if info["rows"] is not None else "-",
                 info.get("lag_vs_cube_days", "-"))
    if behind:
        log.warning("Cube parts BEHIND / missing (%d): %s", len(behind), ", ".join(behind))
    return report
