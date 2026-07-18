"""
table_health.py — mechanical data-sanity checker for the pipeline's DB tables.

For a random sample of each created table it reports, per column: dtype, null-rate,
distinct count; plus PRIMARY-KEY integrity (nulls / duplicates) and date-column
coverage on the full table. Columns that are all-null (confirmed on the full table)
or above the null-rate threshold are flagged into a TRIAGE worklist.

This script only MEASURES. Deciding whether a gap is expected vs a bug, tracing it
to the source, and fixing it are done by the `data-sanity-checks` skill.

Run from stock_pick_strat/ (so `./configs` and `src` resolve), using the project env:
    poetry run python <skill>/table_health.py                        # all tables
    poetry run python <skill>/table_health.py --tables prices,cube   # a subset
    poetry run python <skill>/table_health.py --sample 300 --flag-threshold 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import text

# run from stock_pick_strat/ — put it on the path so `import src...` works
sys.path.insert(0, os.getcwd())


def _store():
    """Load .env + build the DataStore the same way the pipeline does."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        return ctx.store
    except Exception as e:                       # fall back to a bare engine
        print(f"[warn] get_config_context failed ({e}); using bare engine + env defaults")
        from src.data_store.store import DataStore
        from src.utils.db import get_engine
        return DataStore(get_engine())


def _is_arraylike(s: pd.Series) -> bool:
    v = s.dropna()
    return (not v.empty) and isinstance(v.iloc[0], (list, tuple))


def _sample(engine, name: str, n: int, total: int) -> pd.DataFrame:
    """A random sample of up to n rows. Uses TABLESAMPLE on very large Postgres
    tables to avoid a full random sort; otherwise ORDER BY random()."""
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql" and total > 500_000:
            frac = min(100.0, 100.0 * (n * 5) / total)
            try:
                return pd.read_sql(text(
                    f'SELECT * FROM "{name}" TABLESAMPLE BERNOULLI({frac:.4f}) LIMIT {n}'), conn)
            except Exception:
                pass
        return pd.read_sql(text(f'SELECT * FROM "{name}" ORDER BY random() LIMIT {n}'), conn)


def _scalar(engine, sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _check_table(store, spec, n: int, threshold: float) -> list[str]:
    """Print one table's report; return its TRIAGE lines (col -> reason)."""
    engine = store.engine
    name = spec.name
    total = store.row_count(name)
    print(f"\n{'=' * 78}\nTABLE  {name}   [{spec.kind}]   pk={spec.pk}   rows={total:,}")
    if total == 0:
        print("  >>> EMPTY (table created but no rows) — nothing to sample.")
        return [f"{name}: EMPTY table"]

    df = _sample(engine, name, n, total)
    print(f"  sampled {len(df)} rows  ({len(df.columns)} columns)")
    triage: list[str] = []

    # ---- PRIMARY KEY integrity (full table, cheap aggregates) ----
    pk_cols = list(spec.pk)
    if all(c in df.columns for c in pk_cols):
        null_pred = " OR ".join(f'"{c}" IS NULL' for c in pk_cols)
        pk_nulls = _scalar(engine, f'SELECT count(*) FROM "{name}" WHERE {null_pred}')
        grp = ", ".join(f'"{c}"' for c in pk_cols)
        dups = _scalar(engine, f'SELECT count(*) FROM (SELECT 1 FROM "{name}" '
                               f'GROUP BY {grp} HAVING count(*) > 1 LIMIT 10000) s')
        flag = "  <<< PK BROKEN" if (pk_nulls or dups) else ""
        print(f"  PK check: null-in-pk={pk_nulls}  duplicate-pk-groups={dups}{flag}")
        if pk_nulls:
            triage.append(f"{name}: {pk_nulls} rows with NULL in PK {pk_cols}")
        if dups:
            triage.append(f"{name}: {dups} duplicate PK groups (upsert/merge bug)")
    else:
        missing = [c for c in pk_cols if c not in df.columns]
        print(f"  PK check: PK columns absent from table: {missing}  <<< SCHEMA BUG")
        triage.append(f"{name}: PK columns absent {missing}")

    # ---- date-column coverage (full table) ----
    if spec.date_col and spec.date_col in df.columns:
        lo = _scalar(engine, f'SELECT min("{spec.date_col}") FROM "{name}"')
        hi = _scalar(engine, f'SELECT max("{spec.date_col}") FROM "{name}"')
        dn = _scalar(engine, f'SELECT count(*) FROM "{name}" WHERE "{spec.date_col}" IS NULL')
        print(f"  date_col '{spec.date_col}': {lo} .. {hi}   nulls={dn}")
        if dn:
            triage.append(f"{name}: {dn} NULL in date_col '{spec.date_col}'")

    # ---- per-column null-rate on the sample ----
    rows = []
    for c in df.columns:
        s = df[c]
        if _is_arraylike(s):
            rows.append((c, "array", float(s.isna().mean()), s.notna().sum(), "vector"))
            continue
        null_rate = float(s.isna().mean())
        rows.append((c, str(s.dtype), null_rate, int(s.nunique(dropna=True)), ""))
    rows.sort(key=lambda r: r[2], reverse=True)

    fully_ok = sum(1 for r in rows if r[2] == 0.0)
    print(f"  columns fully populated in sample: {fully_ok}/{len(rows)}")
    print(f"  {'column':<34}{'dtype':<12}{'null%':>7}  {'ndistinct':>9}")
    for col, dtype, nr, ndist, note in rows:
        if nr == 0.0 and not note:
            continue                              # only surface columns with gaps
        mark = ""
        if nr >= 1.0:                             # all-null in sample — confirm on full table
            nonnull_full = _scalar(engine, f'SELECT count("{col}") FROM "{name}"')
            if nonnull_full == 0:
                mark = "  <<< ALL-NULL (confirmed full table)"
                triage.append(f"{name}.{col}: ALL-NULL on full table")
            else:
                mark = f"  <<< all-null in sample (full table has {nonnull_full} non-null)"
                triage.append(f"{name}.{col}: all-null in sample only — resample/investigate")
        elif nr >= threshold:
            mark = "  <<< HIGH null-rate"
            triage.append(f"{name}.{col}: null-rate {nr:.0%} in sample (>= {threshold:.0%})")
        print(f"  {col:<34}{dtype:<12}{nr*100:>6.1f}%  {ndist:>9}{mark}")
    return triage


def main() -> None:
    ap = argparse.ArgumentParser(description="DB table sanity checker")
    ap.add_argument("--tables", default="", help="comma-separated table names (default: all created)")
    ap.add_argument("--sample", type=int, default=200, help="rows to sample per table")
    ap.add_argument("--flag-threshold", type=float, default=0.6,
                    help="sample null-rate at/above which a column is flagged (0-1)")
    args = ap.parse_args()

    from src.data_store.schema_registry import ALL_TABLES, BY_NAME
    store = _store()

    if args.tables:
        specs = [BY_NAME[t.strip()] for t in args.tables.split(",") if t.strip()]
    else:
        specs = [s for s in ALL_TABLES if store.exists(s.name)]

    print(f"Checking {len(specs)} table(s); sample={args.sample}, "
          f"flag-threshold={args.flag_threshold:.0%}")
    all_triage: list[str] = []
    for spec in specs:
        if not store.exists(spec.name):
            print(f"\n{'=' * 78}\nTABLE  {spec.name}: not created yet — skipped.")
            continue
        all_triage += _check_table(store, spec, args.sample, args.flag_threshold)

    print(f"\n{'#' * 78}\nTRIAGE WORKLIST ({len(all_triage)} item(s)) -- "
          f"investigate each: data issue (absent at source) vs extraction bug")
    if not all_triage:
        print("  none — every sampled column populated within expectations.")
    for line in all_triage:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
