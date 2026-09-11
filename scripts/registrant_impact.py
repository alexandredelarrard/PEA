"""
registrant_impact.py  (scripts/)
--------------------------------------------------------------------------------------------
Freeze, and later re-measure, every table a registrant boundary can truncate -- so phase 9's
re-extraction can be reported as a measured before/after rather than an assertion.

WHY THE CONTENTS AND NOT THE TIMESTAMPS. Phase 8's worst process failure was recording
pre-rebuild JSON *timestamps* but not their *contents*; `run_all` overwrote the files and seven
of nine findings lost their true before/after. A snapshot here writes the numbers themselves,
so a later `--compare` is arithmetic on two frozen files and cannot be invalidated by a rerun.

WHY PER-TICKER AND NOT JUST PER-TABLE. A global row-count delta hides the case this phase is
most likely to produce: one ticker gains everything and forty gain nothing. The per-ticker
first date is also the actual defect signal -- a registrant cutover shows up as a filing
archive that starts years after the price history, and it is repaired when that date moves
back, not when a total rises.

Read-only. Server-side aggregation throughout: a 3.8M x 250 cube part is never materialised in
the client. `scripts/` sits outside the `src/data_store` boundary
(`tests/data_store/test_store_boundary.py` scans `src/` only), and one pass of `GROUP BY` per
table is the cheap way to do this; the equivalent through `store.load` would pull millions of
rows across the wire to count them.

    "$PY" scripts/registrant_impact.py --snapshot baseline
    "$PY" scripts/registrant_impact.py --compare  baseline                 # vs the live DB
    "$PY" scripts/registrant_impact.py --compare  baseline --against after # two frozen files
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context  # noqa: E402

OUT_ROOT = ROOT / "reports/validate/registrant/_out"

#: The tables a registrant boundary can truncate, as `name -> (ticker_col, date_col, cik_col)`.
#: A `None` means the table has no such column and that measurement is skipped rather than
#: faked. `def14a_llm` genuinely carries no `cik` -- the provenance lives on its four child
#: tables -- and `insider_footnotes` is keyed by accession alone, so it is a row count only.
IMPACTED: dict[str, tuple[str | None, str | None, str | None]] = {
    # tier A -- `new_filings` consumers, resolved by ticker, union with the register
    "sec_8k":                   ("ticker", "filing_date", "cik"),
    "sec_8k_votes":             ("ticker", "filing_date", "cik"),
    "sec_13d":                  ("ticker", "filing_date", "cik"),
    "sec_13d_transactions":     ("ticker", "filing_date", "cik"),
    "sec_13g":                  ("ticker", "filing_date", "cik"),
    "sec_filing_text":          ("ticker", "filed", "cik"),
    "sec_def14a":               ("ticker", "filing_date", "cik"),
    # tier B -- dated-split consumers
    "fundamentals_facts":       ("ticker", "filing_date", "cik"),
    "fundamentals_history_sec": ("ticker", "as_of", None),
    "def14a_llm":               ("ticker", "as_of", None),
    "def14a_directors":         ("ticker", "as_of", "cik"),
    "def14a_executive_comp":    ("ticker", "as_of", "cik"),
    "def14a_director_comp":     ("ticker", "as_of", "cik"),
    "def14a_ownership":         ("ticker", "as_of", "cik"),
    # tier C -- bulk datasets, mapped to a ticker by CIK, with no cutover concept before 9b
    "insider_transactions":     ("ticker", "filing_date", "issuer_cik"),
    "insider_footnotes":        (None, None, None),
    "notes_num":                ("ticker", "filed", "cik"),
    "notes_text":               ("ticker", "filed", "cik"),
    "pension_facts":            ("ticker", "filed", "cik"),
    # the merged fundamentals table every cube consumer actually reads
    "fundamentals_history":     ("ticker", "as_of", None),
}

#: The eight cube parts plus the assembled cube. Measured separately because their delta is
#: the only evidence that recovered extraction rows reached the model inputs -- rows in
#: `def14a_llm` that never become a governance feature are worth nothing.
CUBE_TABLES = ["cube_part_prices", "cube_part_momentum", "cube_part_targets", "cube_part_betas",
               "cube_part_fundamentals", "cube_part_governance", "cube_part_text",
               "cube_part_institutionals", "cube"]

#: Reference tables whose own row counts frame every number above (the universe, and the price
#: history the "filings start N years after the price" screen is measured against).
CONTEXT_TABLES = ["sp500_tickers", "prices", "sharadar_tickers", "sharadar_actions"]

#: Non-null counts are taken in column batches so the query plan stays small on a 250-column
#: part. One `count(col)` per column in a single pass would otherwise build a very wide node.
_COUNT_CHUNK = 40


def _existing(conn, names: list[str]) -> list[str]:
    """`names` restricted to tables that actually exist, in the given order.

    A missing table is a *finding*, not an error: `cube_part_text` and
    `cube_part_institutionals` do not exist in the live DB today, and a snapshot that crashed
    on them would hide that rather than record it.
    """
    live = {r[0] for r in conn.execute(text(
        "select tablename from pg_tables where schemaname = 'public'"))}
    return [n for n in names if n in live]


def _columns(conn, table: str) -> list[str]:
    return [r[0] for r in conn.execute(text(
        "select column_name from information_schema.columns "
        "where table_schema = 'public' and table_name = :t order by ordinal_position"),
        {"t": table})]


def _table_summary(conn, table: str, ticker_col: str | None, date_col: str | None,
                   cik_col: str | None) -> dict:
    """Row count, distinct tickers, distinct CIKs and the date span -- one query."""
    sel = ["count(*) as rows"]
    if ticker_col:
        sel.append(f"count(distinct {ticker_col}) as tickers")
    if cik_col:
        sel.append(f"count(distinct {cik_col}) as ciks")
    if date_col:
        sel.append(f"min({date_col})::text as date_min, max({date_col})::text as date_max")
    row = conn.execute(text(f"select {', '.join(sel)} from {table}")).mappings().one()
    return {k: (int(v) if isinstance(v, int) else v) for k, v in row.items()}


def _per_ticker(conn, table: str, ticker_col: str, date_col: str | None,
                cik_col: str | None) -> pd.DataFrame:
    """One row per ticker: rows, first/last date, distinct CIKs.

    `n_ciks` is the D9.3 provenance measurement. Today every tier-A row is stamped with the
    *roster's* CIK, so this is 1 everywhere; after phase 9c every register ticker must show
    >= 2, and that is the single clearest proof the fix reached the data.
    """
    sel = [f"{ticker_col} as ticker", "count(*) as rows"]
    if date_col:
        sel.append(f"min({date_col})::text as first_date, max({date_col})::text as last_date")
    if cik_col:
        sel.append(f"count(distinct {cik_col}) as n_ciks")
    sql = (f"select {', '.join(sel)} from {table} "
           f"where {ticker_col} is not null group by 1 order by 1")
    return pd.DataFrame(conn.execute(text(sql)).mappings().all())


def _non_null(conn, table: str, cols: list[str]) -> dict[str, int]:
    """Per-column non-null count, in batches. This is the cube's real "how filled is it"."""
    out: dict[str, int] = {}
    for i in range(0, len(cols), _COUNT_CHUNK):
        batch = cols[i:i + _COUNT_CHUNK]
        sel = ", ".join(f'count("{c}") as "{c}"' for c in batch)
        row = conn.execute(text(f"select {sel} from {table}")).mappings().one()
        out.update({c: int(row[c]) for c in batch})
    return out


def snapshot(context, label: str) -> Path:
    """Freeze the live DB into `_out/<label>/`. Idempotent: two runs write the same numbers."""
    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []

    with context.store.engine.connect() as conn:
        present = _existing(conn, list(IMPACTED) + CUBE_TABLES + CONTEXT_TABLES)
        missing = [t for t in list(IMPACTED) + CUBE_TABLES + CONTEXT_TABLES if t not in present]

        for table, (tkr, dt, cik) in IMPACTED.items():
            if table not in present:
                continue
            tables[table] = _table_summary(conn, table, tkr, dt, cik)
            if tkr:
                df = _per_ticker(conn, table, tkr, dt, cik)
                df.insert(0, "table", table)
                frames.append(df)

        for table in CONTEXT_TABLES:
            if table in present:
                dt = "date" if table == "prices" else None
                tkr = "ticker" if table in ("sp500_tickers", "prices", "sharadar_tickers",
                                            "sharadar_actions") else None
                tables[table] = _table_summary(conn, table, tkr, dt, None)

        cube: dict[str, dict] = {}
        for table in CUBE_TABLES:
            if table not in present:
                continue
            cols = _columns(conn, table)
            summary = _table_summary(conn, table, "ticker" if "ticker" in cols else None,
                                     "date" if "date" in cols else None, None)
            summary["n_columns"] = len(cols)
            summary["non_null"] = _non_null(conn, table, cols)
            cube[table] = summary

    per_ticker = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    per_ticker.to_csv(out / "per_ticker.csv", index=False)
    (out / "tables.json").write_text(json.dumps(tables, indent=2, sort_keys=True),
                                     encoding="utf-8")
    (out / "cube_parts.json").write_text(json.dumps(cube, indent=2, sort_keys=True),
                                         encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "label": label,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "missing_tables": missing,
        "n_tables": len(tables), "n_cube_tables": len(cube), "n_per_ticker_rows": len(per_ticker),
    }, indent=2), encoding="utf-8")
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                              # pragma: no cover
        return "unknown"


def _load(label: str) -> tuple[dict, dict, pd.DataFrame]:
    d = OUT_ROOT / label
    per = pd.read_csv(d / "per_ticker.csv") if (d / "per_ticker.csv").exists() else pd.DataFrame()
    return (json.loads((d / "tables.json").read_text(encoding="utf-8")),
            json.loads((d / "cube_parts.json").read_text(encoding="utf-8")), per)


def compare(before: str, after: str, tickers: list[str] | None) -> str:
    """Markdown: per-table delta, per-cube-part delta, and per-ticker delta for `tickers`."""
    b_tab, b_cube, b_per = _load(before)
    a_tab, a_cube, a_per = _load(after)
    lines = [f"# registrant impact — `{before}` → `{after}`", ""]

    lines += ["## Tables", "",
              "| table | rows before | rows after | Δ | tickers b/a | CIKs b/a | first b/a |",
              "|---|---:|---:|---:|---|---|---|"]
    for t in sorted(set(b_tab) | set(a_tab)):
        b, a = b_tab.get(t, {}), a_tab.get(t, {})
        d = a.get("rows", 0) - b.get("rows", 0)
        lines.append(
            f"| `{t}` | {b.get('rows', 0):,} | {a.get('rows', 0):,} | {d:+,} | "
            f"{b.get('tickers', '—')}/{a.get('tickers', '—')} | "
            f"{b.get('ciks', '—')}/{a.get('ciks', '—')} | "
            f"{b.get('date_min', '—')}/{a.get('date_min', '—')} |")

    lines += ["", "## Cube parts", "",
              "| part | rows before | rows after | Δ | cols b/a | filled cells b/a |",
              "|---|---:|---:|---:|---|---|"]
    for t in sorted(set(b_cube) | set(a_cube)):
        b, a = b_cube.get(t, {}), a_cube.get(t, {})
        d = a.get("rows", 0) - b.get("rows", 0)
        lines.append(
            f"| `{t}` | {b.get('rows', 0):,} | {a.get('rows', 0):,} | {d:+,} | "
            f"{b.get('n_columns', '—')}/{a.get('n_columns', '—')} | "
            f"{sum(b.get('non_null', {}).values()):,}/{sum(a.get('non_null', {}).values()):,} |")

    if tickers and not b_per.empty and not a_per.empty:
        lines += ["", "## Per ticker (register names)", "",
                  "| table | ticker | rows b→a | first date b→a | n_ciks b→a |",
                  "|---|---|---|---|---|"]
        key = ["table", "ticker"]
        m = b_per.merge(a_per, on=key, how="outer", suffixes=("_b", "_a"))
        m = m[m["ticker"].isin(tickers)]
        moved = m[(m["rows_b"].fillna(0) != m["rows_a"].fillna(0))
                  | (m["first_date_b"].fillna("") != m["first_date_a"].fillna(""))]
        for _, r in moved.sort_values(key).iterrows():
            lines.append(f"| `{r['table']}` | {r['ticker']} | "
                         f"{int(r['rows_b'] or 0):,}→{int(r['rows_a'] or 0):,} | "
                         f"{r.get('first_date_b', '—')}→{r.get('first_date_a', '—')} | "
                         f"{r.get('n_ciks_b', '—')}→{r.get('n_ciks_a', '—')} |")
        if moved.empty:
            lines.append("| — | — | nothing moved | — | — |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="./configs")
    p.add_argument("--snapshot", metavar="LABEL", help="freeze the live DB under this label")
    p.add_argument("--compare", metavar="LABEL", help="baseline label to compare against")
    p.add_argument("--against", metavar="LABEL",
                   help="second frozen label; omit to snapshot the live DB as `_live`")
    p.add_argument("--tickers", default="", help="comma-separated names for the per-ticker table")
    p.add_argument("--out", help="write the comparison markdown here instead of stdout")
    args = p.parse_args(argv)

    if not (args.snapshot or args.compare):
        p.error("one of --snapshot / --compare is required")

    _, context = get_config_context(config_path=args.config, use_cache=False, save=False)

    if args.snapshot:
        out = snapshot(context, args.snapshot)
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        print(f"snapshot `{args.snapshot}` -> {out}")
        print(f"  tables={man['n_tables']} cube={man['n_cube_tables']} "
              f"per_ticker_rows={man['n_per_ticker_rows']} git={man['git_sha']}")
        if man["missing_tables"]:
            print(f"  ⚠ absent from the DB: {', '.join(man['missing_tables'])}")

    if args.compare:
        after = args.against
        if after is None:
            after = "_live"
            snapshot(context, after)
        md = compare(args.compare, after,
                     [t for t in args.tickers.split(",") if t] or None)
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
