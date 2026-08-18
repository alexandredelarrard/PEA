"""
data_profile.py  (scripts/dod/data_profile.py)
----------------------------------------------
The DATA Definition-of-Done report: what is actually in the tables you touched, and five gates
that catch the ways an extraction or aggregation change silently loses data.

    "$PY" scripts/dod/data_profile.py --slug fix-q4 --tables fundamentals_history
    "$PY" scripts/dod/data_profile.py --slug fix-q4 --tables fundamentals_history \
          --tickers AAPL,JPM --slug smoke
    "$PY" scripts/dod/data_profile.py --slug cube-rebuild --tables cube --parts --freshness

Gates
    D1  the declared primary key is UNIQUE over the rows profiled
    D2  row count did not DECREASE vs the recorded baseline (unless --declare-shrink)
    D3  no column disappeared vs the baseline
    D4  the date range covers the expected window (--expect-through)
    D5  no per-field null rate got WORSE than the baseline (unless --declare-nulls)

Design notes
  * EVERY READ GOES THROUGH `context.store`. This file must never import `sqlalchemy` or call
    `pd.read_sql` -- `tests/data_store/test_store_boundary.py` enforces that boundary for
    `src/`, and a report generator that broke it would be the worst possible example to set.
  * SCOPE IS A FIRST-CLASS FIELD, NOT A FOOTNOTE. "null rate 4%" means nothing without "over
    which tickers, over which window". `--tickers`/`--since`/`--limit` all land in §1 and in
    the metrics block, and a partial scope is BLOCKED from overwriting the baseline
    (see baseline.is_full_scope) so a two-ticker smoke run cannot neuter D2.
  * THE OUTLIER COUNT REUSES THE AUDIT'S KERNEL (`src/utils/outliers.modified_zscore`), so the
    profiler and `analyze_history` can never disagree about what an outlier is.
  * `store.load` RAISES on an empty read by design, so every read here passes `optional=True`
    and branches on `is None` -- an empty table is a legitimate thing for a profiler to report.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dod.baseline import (                                    # noqa: E402
    is_full_scope, load_profile_baseline, save_profile_baseline, snapshot_from_profile,
)
from scripts.dod.report_common import (                               # noqa: E402
    Gate, announce, metrics_table, repo_root, write_report,
)
from src.context import get_config_context                            # noqa: E402
from src.data_aggregate.utils.common.part_status import part_status_report   # noqa: E402
from src.data_extract.utils.common.freshness import check_data_freshness     # noqa: E402
from src.data_store import schema                                     # noqa: E402
from src.utils.outliers import count_mad_outliers, mad_center_scale    # noqa: E402

GENERATOR = "scripts/dod/data_profile.py@1"
#: Safety rail on an unscoped profile. `sec13f_hr` is ~21.7M rows; reading it whole to compute
#: a median would OOM the box, and a bounded sample answers the same question.
DEFAULT_LIMIT = 500_000
PERCENTILES = (0.01, 0.25, 0.50, 0.75, 0.99)
_LOG = logging.getLogger("dod.data_profile")


# --------------------------------------------------------------------------- #
# Profiling                                                                   #
# --------------------------------------------------------------------------- #
def _field_stats(series: pd.Series) -> dict:
    """dtype / null rate / nunique, plus the numeric distribution when the field is numeric."""
    n = int(len(series))
    stats: dict = {
        "dtype": str(series.dtype),
        "nulls": int(series.isna().sum()),
        "null_rate": (float(series.isna().mean()) if n else None),
        "nunique": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        vals = series.dropna().astype(float)
        if len(vals):
            qs = vals.quantile(list(PERCENTILES))
            centre, scale = mad_center_scale(vals)
            stats.update({
                "mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max()),
                "p01": float(qs.loc[0.01]), "p25": float(qs.loc[0.25]),
                "p50": float(qs.loc[0.50]), "p75": float(qs.loc[0.75]),
                "p99": float(qs.loc[0.99]),
                "mad_center": centre, "mad_scale": scale,
                "mad_outliers": count_mad_outliers(vals),
            })
    return stats


def profile_table(context, table: schema.Table, *, tickers: list[str] | None,
                  since: str | None, limit: int | None) -> dict:
    """Row/column counts (server-side) plus per-field stats over the scoped sample."""
    store = context.store
    name = table.name
    out: dict = {"table": name, "exists": store.exists(name), "kind": table.kind,
                 "pk": list(table.pk), "date_col": table.date_col,
                 "scope": {"tickers": tickers, "since": since, "limit": limit}}
    if not out["exists"]:
        out.update({"rows": 0, "columns": [], "fields": {}, "sampled_rows": 0,
                    "date_min": None, "date_max": None})
        return out

    out["rows"] = store.row_count(name)
    out["columns"] = store.columns(name)

    if table.date_col and table.date_col in out["columns"]:
        lo, hi = store.bounds(name)
        out["date_min"], out["date_max"] = (str(lo) if lo is not None else None,
                                            str(hi) if hi is not None else None)
    else:
        out["date_min"] = out["date_max"] = None

    where: dict | None = None
    if tickers and table.ticker_col and table.ticker_col in out["columns"]:
        where = {table.ticker_col: tickers}

    df = store.load(name, where=where, since=since, limit=limit, optional=True)
    if df is None:
        out.update({"sampled_rows": 0, "fields": {},
                    "note": "no rows matched the requested scope"})
        return out

    out["sampled_rows"] = int(len(df))
    out["fields"] = {col: _field_stats(df[col]) for col in df.columns}

    # The SAMPLE's own date range, reported separately from the table-wide bounds above.
    # Mixing the two in one row is a trap: for `fundamentals_facts` scoped to AAPL+JPM the
    # table-wide bounds are 2011-08-12..2026-08-12 while the sample only covers
    # 2011-10-26..2026-08-06, so a reader comparing a sample null rate against a table-wide
    # date range is silently combining two different populations.
    if table.date_col and table.date_col in df.columns:
        dates = pd.to_datetime(df[table.date_col], errors="coerce").dropna()
        out["sample_date_min"] = str(dates.min().date()) if len(dates) else None
        out["sample_date_max"] = str(dates.max().date()) if len(dates) else None
    else:
        out["sample_date_min"] = out["sample_date_max"] = None

    # PK uniqueness over exactly the rows read, so the claim is never wider than the evidence.
    #
    # A PARTIAL key is NEVER checked. `fundamentals_facts` is the live example: the registry
    # declares `(ticker, accession_number, field, fiscal_year, fiscal_period, duration_type)`
    # but the table has no `field` and no `duration_type` column. Counting duplicates over the
    # 4-column prefix that DOES exist reported "67,282 duplicate rows", which is not a data
    # defect at all -- a prefix of a compound key is expected to repeat. Drift between the
    # declared key and the live columns is itself the finding, so it is reported as one.
    pk_present = [c for c in table.pk if c in df.columns]
    pk_missing = [c for c in table.pk if c not in df.columns]
    out["pk_checked_cols"] = pk_present if not pk_missing else []
    out["pk_missing_cols"] = pk_missing
    out["pk_complete"] = not pk_missing and bool(pk_present)
    out["pk_checked_rows"] = int(len(df))
    if out["pk_complete"]:
        out["pk_duplicate_rows"] = int(len(df) - len(df.drop_duplicates(subset=pk_present)))
    else:
        out["pk_duplicate_rows"] = None
    return out


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #
def build_gates(profiles: dict[str, dict], baseline: dict, *, expect_through: str | None,
                declare_shrink: str, declare_nulls: str) -> list[Gate]:
    gates: list[Gate] = []
    scoped = [n for n, p in profiles.items() if not is_full_scope(p["scope"])]

    # ---- D1: PK uniqueness ------------------------------------------------ #
    # Two distinct failures, never conflated: duplicate rows under a VERIFIABLE key, and a
    # declared key that cannot be verified because the live table lacks its columns. The
    # second is schema drift and fails too -- an unverifiable PK is not a passing PK.
    dupes = [f"{n}: {p['pk_duplicate_rows']:,} duplicate row(s) on ({', '.join(p['pk_checked_cols'])})"
             for n, p in profiles.items() if (p.get("pk_duplicate_rows") or 0) > 0]
    drift = [f"{n}: declared PK names column(s) absent from the live table: "
             f"{', '.join(p['pk_missing_cols'])}"
             for n, p in profiles.items()
             if p.get("exists") and p.get("sampled_rows") and p.get("pk_missing_cols")]
    checked = [n for n, p in profiles.items() if p.get("pk_complete")]
    problems = dupes + drift
    if problems:
        verdict: bool | None = False
    elif checked:
        verdict = True
    else:
        verdict = None
    gates.append(Gate("D1", "declared PK unique over the rows profiled", verdict,
                      "; ".join(problems) if problems
                      else (f"unique across {len(checked)} table(s): {', '.join(checked)}"
                            if checked else "no table had rows to check")))

    # ---- D2: row count not decreased -------------------------------------- #
    declared = {t.strip() for t in declare_shrink.replace(",", " ").split() if t.strip()}
    shrunk, compared = [], 0
    for n, p in profiles.items():
        base = baseline.get(n) or {}
        before = base.get("rows")
        if before is None or not is_full_scope(p["scope"]):
            continue
        compared += 1
        if p["rows"] < before and n not in declared:
            shrunk.append(f"{n}: {before:,} -> {p['rows']:,}")
    if compared == 0:
        gates.append(Gate("D2", "row count not decreased", None,
                          "no full-scope baseline to compare against"
                          + (f" (scoped run: {', '.join(scoped)})" if scoped else "")
                          + " — this run records one"))
    else:
        gates.append(Gate("D2", "row count not decreased", not shrunk,
                          "; ".join(shrunk) if shrunk
                          else f"{compared} table(s) at or above baseline"
                               + (f"; declared shrink: {', '.join(sorted(declared))}"
                                  if declared else "")))

    # ---- D3: no column lost ----------------------------------------------- #
    lost, checked = [], 0
    for n, p in profiles.items():
        base = baseline.get(n) or {}
        before = base.get("columns")
        if not before:
            continue
        checked += 1
        gone = sorted(set(before) - set(p.get("columns") or []))
        if gone:
            lost.append(f"{n}: {', '.join(gone)}")
    gates.append(Gate("D3", "no column lost", None if checked == 0 else not lost,
                      "; ".join(lost) if lost
                      else (f"{checked} table(s) keep every baseline column" if checked
                            else "no baseline columns recorded yet")))

    # ---- D4: date range covers the expected window ------------------------ #
    if not expect_through:
        gates.append(Gate("D4", "date range covers the expected window", None,
                          "no --expect-through given"))
    else:
        want = pd.Timestamp(expect_through)
        short = []
        for n, p in profiles.items():
            if not p.get("date_max"):
                continue
            got = pd.Timestamp(p["date_max"])
            if got < want:
                short.append(f"{n}: max {got.date()} < {want.date()}")
        gates.append(Gate("D4", "date range covers the expected window", not short,
                          "; ".join(short) if short
                          else f"every dated table reaches {want.date()}"))

    # ---- D5: null rates not worse ----------------------------------------- #
    ok_nulls = {t.strip() for t in declare_nulls.replace(",", " ").split() if t.strip()}
    worse, n_fields = [], 0
    for n, p in profiles.items():
        base_nulls = (baseline.get(n) or {}).get("null_rate") or {}
        if not base_nulls or not is_full_scope(p["scope"]):
            continue
        for field, stats in (p.get("fields") or {}).items():
            before, now = base_nulls.get(field), stats.get("null_rate")
            if before is None or now is None:
                continue
            n_fields += 1
            # 0.5pp of slack: a growing table's null rate wobbles without anything regressing
            if now > before + 0.005 and f"{n}.{field}" not in ok_nulls:
                worse.append(f"{n}.{field}: {before:.1%} -> {now:.1%}")
    if n_fields == 0:
        gates.append(Gate("D5", "per-field null rate not worse", None,
                          "no full-scope baseline null rates to compare against"))
    else:
        gates.append(Gate("D5", "per-field null rate not worse", not worse,
                          "; ".join(worse[:6]) if worse
                          else f"{n_fields} field(s) at or below baseline (+0.5pp slack)"))
    return gates


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _table_rows(profiles: dict[str, dict]) -> list[dict]:
    return [{"table": n, "exists": p.get("exists"), "rows": p.get("rows"),
             "sampled": p.get("sampled_rows"), "cols": len(p.get("columns") or []),
             "pk": ",".join(p.get("pk") or []),
             "pk_absent_cols": ",".join(p.get("pk_missing_cols") or []) or None,
             "pk_dupes": p.get("pk_duplicate_rows"),
             "date_min": p.get("date_min"), "date_max": p.get("date_max"),
             "sample_date_min": p.get("sample_date_min"),
             "sample_date_max": p.get("sample_date_max")}
            for n, p in sorted(profiles.items())]


def _field_rows(profiles: dict[str, dict], top: int) -> list[dict]:
    """Per-field rows, worst null rate first -- that is what a reader is looking for."""
    rows = []
    for n, p in sorted(profiles.items()):
        for field, s in (p.get("fields") or {}).items():
            rows.append({"table": n, "field": field, "dtype": s.get("dtype"),
                         "null_%": (None if s.get("null_rate") is None
                                    else round(100 * s["null_rate"], 2)),
                         "nunique": s.get("nunique"), "mean": s.get("mean"),
                         "std": s.get("std"), "min": s.get("min"), "p01": s.get("p01"),
                         "p50": s.get("p50"), "p99": s.get("p99"), "max": s.get("max"),
                         "mad_outliers": s.get("mad_outliers")})
    rows.sort(key=lambda r: (-(r["null_%"] or 0), r["table"], r["field"]))
    return rows[:top]


def _parts_md(report: dict) -> str:
    parts = report.get("parts") or {}
    rows = [{"part": k, "exists": v.get("exists"), "rows": v.get("rows"),
             "max_date": v.get("max_date"), "lag_vs_cube_days": v.get("lag_vs_cube_days")}
            for k, v in sorted(parts.items())]
    behind = report.get("behind") or []
    head = (f"**{len(behind)} part(s) behind the cube:** {', '.join(behind)}" if behind
            else "**No cube part is behind.**")
    return head + "\n\n" + metrics_table(
        rows, ["part", "exists", "rows", "max_date", "lag_vs_cube_days"])


def _freshness_md(report: dict) -> str:
    sources = report.get("sources") or {}
    rows = [{"source": k, "latest": v.get("latest"), "age_days": v.get("age_days"),
             "cadence": v.get("cadence"), "max_age_days": v.get("max_age_days"),
             "status": v.get("status")}
            for k, v in sorted(sources.items())]
    stale = report.get("stale") or []
    head = (f"**{len(stale)} stale source(s):** {', '.join(stale)}" if stale
            else "**Every source is fresh.**")
    return head + "\n\n" + metrics_table(
        rows, ["source", "latest", "age_days", "cadence", "max_age_days", "status"])


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DATA Definition-of-Done report")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--tables", default="", help="comma separated table names (registry names)")
    ap.add_argument("--tickers", default="", help="restrict the sample to these tickers")
    ap.add_argument("--since", default=None, help="restrict the sample to dates >= this")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max rows read per table (default {DEFAULT_LIMIT}; 0 = no limit)")
    ap.add_argument("--expect-through", default=None, help="D4: the date coverage must reach")
    ap.add_argument("--declare-shrink", default="", help="D2: tables whose shrink is intended")
    ap.add_argument("--declare-nulls", default="", help="D5: `table.field` pairs allowed to worsen")
    ap.add_argument("--parts", action="store_true", help="include the cube-part status block")
    ap.add_argument("--freshness", action="store_true", help="include the source freshness block")
    ap.add_argument("--top-fields", type=int, default=60, help="rows in the per-field table")
    ap.add_argument("--update-baseline", action="store_true",
                    help="record this profile as the new baseline (full-scope tables only)")
    ap.add_argument("--config", default="./configs")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = repo_root()
    names = [t.strip() for t in args.tables.replace(",", " ").split() if t.strip()]
    if not names and not (args.parts or args.freshness):
        ap.error("give --tables, and/or --parts / --freshness")

    tickers = [t.strip().upper() for t in args.tickers.replace(",", " ").split() if t.strip()]
    limit = args.limit or None

    _, context = get_config_context(args.config, use_cache=False, save=False)

    profiles: dict[str, dict] = {}
    unknown: list[str] = []
    for name in names:
        try:
            table = schema.resolve(name)
        except (schema.UnknownTableError, KeyError):
            unknown.append(name)
            continue
        _LOG.info("profiling %s ...", name)
        profiles[table.name] = profile_table(context, table, tickers=tickers or None,
                                            since=args.since, limit=limit)

    baseline = load_profile_baseline(root)
    gates = build_gates(profiles, baseline, expect_through=args.expect_through,
                        declare_shrink=args.declare_shrink, declare_nulls=args.declare_nulls)

    parts_report = part_status_report(context, _LOG) if args.parts else None
    fresh_report = (check_data_freshness(context, log=_LOG, track_new_fundamentals=False)
                    if args.freshness else None)

    metrics_parts = [
        "_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are "
        "**table-wide** (server-side); every other number is over the **sample** described "
        "in §1. Do not compare across the two._",
        "**Tables**",
        metrics_table(_table_rows(profiles), ["table", "exists", "rows", "sampled", "cols",
                                             "pk", "pk_absent_cols", "pk_dupes",
                                             "date_min", "date_max",
                                             "sample_date_min", "sample_date_max"]),
    ]
    field_rows = _field_rows(profiles, args.top_fields)
    if field_rows:
        metrics_parts += [
            f"**Fields** (worst null rate first, top {len(field_rows)})",
            metrics_table(field_rows, ["table", "field", "dtype", "null_%", "nunique", "mean",
                                      "std", "min", "p01", "p50", "p99", "max", "mad_outliers"]),
        ]
    if parts_report:
        metrics_parts += ["**Cube parts** (`part_status_report`)", _parts_md(parts_report)]
    if fresh_report:
        metrics_parts += ["**Source freshness** (`check_data_freshness`)",
                          _freshness_md(fresh_report)]

    scope_md = "\n".join([
        "**SAMPLE SCOPE** — a metric without its scope is not a measurement:",
        "",
        f"- tables: {', '.join(sorted(profiles)) if profiles else 'none'}",
        f"- tickers: {', '.join(tickers) if tickers else '**all** (no ticker filter)'}",
        f"- since: {args.since or '**no lower bound**'}",
        f"- row limit per table: {limit:,}" if limit else "- row limit per table: **none**",
        f"- full-scope tables (eligible to set the baseline): "
        f"{', '.join(n for n, p in profiles.items() if is_full_scope(p['scope'])) or 'none'}",
    ] + ([f"- **unknown table name(s) skipped: {', '.join(unknown)}**"] if unknown else []))

    evidence_md = "\n".join(
        [f"- baseline file: `{'reports/baselines/data_profile.json'}` "
         f"({len(baseline)} table(s) recorded)"]
        + [f"- `{n}`: {p.get('rows', 0):,} rows, {len(p.get('columns') or [])} cols, "
           f"{p.get('sampled_rows', 0):,} sampled" for n, p in sorted(profiles.items())]
        + ([f"- cube parts behind: {', '.join(parts_report.get('behind') or []) or 'none'}"]
           if parts_report else [])
        + ([f"- stale sources: {', '.join(fresh_report.get('stale') or []) or 'none'}"]
           if fresh_report else []))

    payload = {
        "scope": {"tables": sorted(profiles), "tickers": tickers, "since": args.since,
                  "limit": limit, "unknown_tables": unknown},
        "metrics": {"tables": profiles,
                    "parts_behind": (parts_report or {}).get("behind"),
                    "stale_sources": (fresh_report or {}).get("stale")},
    }

    path = write_report("DATA", args.slug, generator=GENERATOR, gates=gates,
                        metrics_md="\n\n".join(metrics_parts), evidence_md=evidence_md,
                        payload=payload, scope_md=scope_md, root=root,
                        session_id=args.session_id)

    if args.update_baseline:
        updated, refused = [], []
        for n, p in profiles.items():
            if is_full_scope(p["scope"]) and p.get("exists"):
                baseline[n] = snapshot_from_profile(p)
                updated.append(n)
            else:
                refused.append(n)
        if updated:
            save_profile_baseline(root, baseline)
        print(f"  baseline updated for: {', '.join(updated) or 'nothing'}")
        if refused:
            print(f"  baseline REFUSED (partial scope, would neuter D2/D5): {', '.join(refused)}")

    announce(path, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
