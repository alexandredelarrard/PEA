"""
freshness.py  (src/data_extract/utils/common/freshness.py)
----------------------------------------------------------
DATA-FRESHNESS / GAP check for the nightly extraction DAG. The pipeline runs at 01:00 to refresh
inputs for next-day stock prediction, so before triggering aggregation we verify every source is
UP TO DATE for its cadence — a daily source (prices / macro / short interest) must have day -1, a
weekly source must be within a week, and so on up to yearly (10-K / DEF 14A).

For each registered source (`DATA_FRESHNESS_SOURCES`) we read the latest observed date
(`max(date_col)`), compute its age in days, and flag it STALE when the age exceeds the cadence
threshold (`DATA_FRESHNESS_MAX_AGE_DAYS`). Thresholds fold in weekends/holidays (daily) and the
normal filing lag (quarterly/yearly), so only a genuine gap trips the warning. The returned report
(latest date + age + status per source, grouped by cadence, plus an overall `ok`) is what the DAG
pushes to XCom and uses to colour the task RED when anything is not as expected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.constants.constants import (
    DATA_FRESHNESS_CADENCE_ORDER,
    DATA_FRESHNESS_MAX_AGE_DAYS,
    DATA_FRESHNESS_SOURCES,
    FRESHNESS_FUNDAMENTALS_SOURCE,
    FRESHNESS_SNAPSHOT_DIR,
    FUNDAMENTALS_SNAPSHOT_FILE,
)
from src.context import Context

logger = logging.getLogger(__name__)


def _table_exists(conn, table: str) -> bool:
    q = text("SELECT 1 FROM information_schema.tables WHERE table_name = :t LIMIT 1")
    return conn.execute(q, {"t": table}).first() is not None


def _column_exists(conn, table: str, col: str) -> bool:
    q = text("SELECT 1 FROM information_schema.columns "
             "WHERE table_name = :t AND column_name = :c LIMIT 1")
    return conn.execute(q, {"t": table, "c": col}).first() is not None


def _max_date(conn, table: str, col: str) -> pd.Timestamp | None:
    v = conn.execute(text(f'SELECT MAX("{col}") FROM "{table}"')).scalar()
    if v is None:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts).normalize()


# --------------------------------------------------------------------------- #
# New-fundamentals-filings delta (which tickers got new earnings since last run) #
# --------------------------------------------------------------------------- #
def _snapshot_path(context: Context) -> Path:
    d = Path(context.paths["DATA_STORE"]) / FRESHNESS_SNAPSHOT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / FUNDAMENTALS_SNAPSHOT_FILE


def _load_snapshot(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}


def _compute_new_filings(current: dict[str, str], prev: dict[str, str]) -> dict:
    """PURE diff of two {ticker: latest_date} maps. A ticker is NEW when it is absent from `prev`
    or its latest date advanced. On the very first run (`prev` empty) nothing is reported as new —
    it just establishes the baseline. Returns {baseline, new_count, tracked, new{ticker:{from,to}}}."""
    if not prev:
        return {"baseline": True, "new_count": 0, "tracked": len(current), "new": {}}
    new: dict[str, dict] = {}
    for tk, cur in current.items():
        p = prev.get(tk)
        if cur is not None and (p is None or cur > p):
            new[tk] = {"from": p, "to": cur}
    return {"baseline": False, "new_count": len(new), "tracked": len(current), "new": new}


def _fundamentals_latest_by_ticker(context: Context) -> dict[str, str]:
    """{ticker: latest fundamentals date (YYYY-MM-DD)} from the fundamentals table (its freshness
    source's date_col = the reported period end `as_of`)."""
    table, col, _ = DATA_FRESHNESS_SOURCES[FRESHNESS_FUNDAMENTALS_SOURCE]
    df = context.store.load(table, columns=["ticker", col])
    if df is None or df.empty:
        return {}
    s = pd.to_datetime(df[col], errors="coerce")
    out = (pd.DataFrame({"ticker": df["ticker"].astype(str), col: s})
           .dropna(subset=[col]).groupby("ticker")[col].max())
    return {tk: d.strftime("%Y-%m-%d") for tk, d in out.items()}


def _new_fundamentals_filings(context: Context, log: logging.Logger) -> dict:
    """Which tickers got a NEW fundamentals filing (new earnings period) since the last run:
    compare the current per-ticker latest `as_of` to the snapshot from the previous run, then
    re-write the snapshot. Best-effort — any failure returns an `error` note so the freshness
    report still builds."""
    try:
        current = _fundamentals_latest_by_ticker(context)
        path = _snapshot_path(context)
        prev = _load_snapshot(path)
        delta = _compute_new_filings(current, prev)
        path.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
        if delta.get("baseline"):
            log.info("New fundamentals filings: baseline established (%d tickers tracked)",
                     delta["tracked"])
        elif delta["new_count"]:
            log.warning("New fundamentals filings since last run (%d): %s", delta["new_count"],
                        ", ".join(f"{tk} {v['from']}->{v['to']}" for tk, v in delta["new"].items()))
        else:
            log.info("No new fundamentals filings since last run (%d tickers tracked).",
                     delta["tracked"])
        return delta
    except Exception as e:                                     # noqa: BLE001
        log.warning("New-fundamentals-filings delta unavailable: %s", e)
        return {"baseline": False, "new_count": 0, "tracked": 0, "new": {},
                "error": f"{type(e).__name__}: {e}"}


def check_data_freshness(context: Context, today: pd.Timestamp | str | None = None,
                         log: logging.Logger | None = None,
                         track_new_fundamentals: bool = True) -> dict:
    """Build the freshness report: per source the latest observed date, its age (days) and a status
    of `ok` / `stale` / `empty` / `missing_table` / `missing_column` / `error:*`. Returns
    ``{"as_of", "ok", "stale": [labels], "sources": {label: {...}}, "new_fundamentals": {...}}``
    where `ok` is True only when every source is `ok`, and `new_fundamentals` lists which tickers
    got a new earnings filing since the last run. `track_new_fundamentals=False` skips that diff
    (and its snapshot write) so the check stays fully read-only (used in focused tests)."""
    log = log or logger
    today = (pd.Timestamp(today).normalize() if today is not None
             else pd.Timestamp.today().normalize())
    sources: dict[str, dict] = {}
    stale: list[str] = []

    with context.store.engine.connect() as conn:
        for label, (table, col, cadence) in DATA_FRESHNESS_SOURCES.items():
            max_age = DATA_FRESHNESS_MAX_AGE_DAYS[cadence]
            info: dict = {"table": table, "date_col": col, "cadence": cadence,
                          "max_age_days": max_age, "latest": None, "age_days": None,
                          "status": None}
            try:
                if not _table_exists(conn, table):
                    info["status"] = "missing_table"
                elif not _column_exists(conn, table, col):
                    info["status"] = "missing_column"
                else:
                    latest = _max_date(conn, table, col)
                    if latest is None:
                        info["status"] = "empty"
                    else:
                        age = int((today - latest).days)
                        info["latest"] = latest.strftime("%Y-%m-%d")
                        info["age_days"] = age
                        info["status"] = "ok" if age <= max_age else "stale"
            except Exception as e:                                  # noqa: BLE001
                info["status"] = f"error:{type(e).__name__}"
            if info["status"] != "ok":
                stale.append(label)
            sources[label] = info

    report = {"as_of": today.strftime("%Y-%m-%d"), "ok": not stale,
              "stale": stale, "sources": sources}
    if track_new_fundamentals:
        report["new_fundamentals"] = _new_fundamentals_filings(context, log)
    _log_report(report, log)
    return report


def _log_report(report: dict, log: logging.Logger) -> None:
    """Log the report as a table grouped by cadence tier (daily -> yearly), each row flagged
    ok / STALE so a glance shows what is not up to date."""
    log.info("=== Data freshness @ %s === (overall: %s)",
             report["as_of"], "OK" if report["ok"] else "NOT UP TO DATE")
    by_cadence: dict[str, list[str]] = {}
    for label, info in report["sources"].items():
        by_cadence.setdefault(info["cadence"], []).append(label)
    for cadence in DATA_FRESHNESS_CADENCE_ORDER:
        labels = by_cadence.get(cadence)
        if not labels:
            continue
        log.info("--- %s (allowed age <= %d days) ---", cadence.upper(),
                 DATA_FRESHNESS_MAX_AGE_DAYS[cadence])
        for label in labels:
            info = report["sources"][label]
            flag = "ok " if info["status"] == "ok" else ">>>"
            log.info("  %s %-24s latest=%-10s age=%-5s status=%s", flag, label,
                     info["latest"] or "-",
                     "-" if info["age_days"] is None else info["age_days"], info["status"])
    if report["stale"]:
        log.warning("NOT up to date (%d): %s", len(report["stale"]), ", ".join(report["stale"]))
    nf = report.get("new_fundamentals")
    if nf and not nf.get("baseline") and nf.get("new_count"):
        log.info("--- NEW fundamentals filings since last run (%d) ---", nf["new_count"])
        for tk, v in nf["new"].items():
            log.info("  + %-6s %s -> %s", tk, v["from"] or "-", v["to"])
