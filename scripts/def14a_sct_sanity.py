"""
def14a_sct_sanity.py  (scripts/)
--------------------------------------------------------------------------------------------
Apply the CEO-pay sanity step (`validate.sanity_check_exec_comp`) to the stored `def14a_llm`
rows and, on request, write the settled values back.

WHY THIS IS A SCRIPT AND NOT PART OF THE WRITE PATH. Half of the step is within-row and DOES
run on every extraction -- `flatten._flatten` calls `validate.repair_pay_ratio`. The other half
scores a suspect total against the same CEO's OTHER filings, and `flatten._result_frames`
flattens ONE filing at a time, so that reference does not exist there. This runs it as a batch.

It is therefore RE-RUNNABLE, not a one-off backfill: run it after any DEF 14A extraction that
added filings. The step is idempotent -- a row it has already settled satisfies the identities
and is not tested a second time -- so a repeat run over unchanged data writes nothing.

DRY RUN IS THE DEFAULT. `--write` is required to touch the table, because this is the only
script in the repo that edits an extracted source column. Compare the tally against the numbers
in `reports/planning/active-tasks/2026-09-08-governance-fixes/phase-2-sct-sanity.md` first.

    "$PY" scripts/def14a_sct_sanity.py [-c ./configs] [-t AAPL,JPM] [--show 60]
    "$PY" scripts/def14a_sct_sanity.py --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_extract.utils.structure.def14a.validate import (
    DEF14A_SCT_PART_COLS, sanity_check_exec_comp, sct_reference,
)
from src.data_store.schema import Tables

#: Every column the step reads or writes, plus the primary key. Loading a narrow frame is not
#: only cheaper -- `store.save` upserts exactly the columns it is handed, so a narrow frame is
#: what keeps this from rewriting `def14a_json` (the largest column in the database) on every
#: run just to change one float.
_PK = ["ticker", "accession_number"]
_COLS = _PK + ["as_of", "ceo_name_proxy", "ceo_total_comp", "ceo_pay_ratio",
               "median_employee_pay", *DEF14A_SCT_PART_COLS]
#: The two columns the step is allowed to change. `ceo_salary` and the other components are
#: read-only inputs: a negative component is the filer's own number (see `validate`).
_WRITTEN = ["ceo_total_comp", "ceo_pay_ratio"]


def _decisions(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """One row per cell the step changed, with the evidence that drove it."""
    rows = []
    ref = sct_reference(before)
    parts = before[[c for c in DEF14A_SCT_PART_COLS if c in before.columns]]
    parts = parts.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    for col in _WRITTEN:
        b = pd.to_numeric(before[col], errors="coerce")
        a = pd.to_numeric(after[col], errors="coerce")
        changed = (b.notna() & a.isna()) | (b.notna() & a.notna() & (b != a))
        for i in before.index[changed]:
            rows.append({
                "ticker": before.at[i, "ticker"],
                "as_of": pd.to_datetime(before.at[i, "as_of"]).date(),
                "ceo": before.at[i, "ceo_name_proxy"],
                "column": col,
                "before": b.at[i],
                "after": a.at[i],
                "action": "NULLED" if pd.isna(a.at[i]) else "REWRITTEN",
                "sum_parts": parts.at[i] if col == "ceo_total_comp" else float("nan"),
                "ceo_reference": ref.at[i] if col == "ceo_total_comp" else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("-t", "--tickers", default=None,
                    help="comma-separated subset. ⚠ A SUBSET CHANGES THE ANSWER: the neighbour "
                         "reference is per (ticker, CEO), so restricting tickers is safe, but "
                         "the tally will not match the full-table numbers in the plan.")
    ap.add_argument("--show", type=int, default=80, help="how many decisions to print")
    ap.add_argument("--write", action="store_true",
                    help="actually upsert the settled values. Omitted = dry run.")
    args = ap.parse_args()

    _, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store
    if not store.exists(Tables.def14a_llm):
        print("def14a_llm does not exist — nothing to do.")
        return 1

    live = set(store.columns(Tables.def14a_llm))
    missing = [c for c in _COLS if c not in live]
    if missing:
        print(f"def14a_llm is missing expected columns: {missing}")
        return 1
    before = store.load(Tables.def14a_llm, columns=_COLS)
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        before = before[before["ticker"].isin(wanted)].copy()
    print(f"loaded {len(before):,} rows / {before['ticker'].nunique()} tickers"
          f"{' (SUBSET)' if args.tickers else ''}")

    after, tally = sanity_check_exec_comp(before)
    print("\n--- tally ---")
    for key, value in tally.items():
        print(f"  {key:44} {value:7,}")

    decisions = _decisions(before, after)
    print(f"\n--- {len(decisions)} cell decision(s) ---")
    if not decisions.empty:
        shown = decisions.sort_values(["column", "ticker", "as_of"]).head(args.show)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(shown.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        if len(decisions) > args.show:
            print(f"  ... {len(decisions) - args.show} more (raise --show)")

    if not args.write:
        print("\nDRY RUN — nothing written. Re-run with --write to apply.")
        return 0
    if decisions.empty:
        print("\nnothing to write.")
        return 0
    payload = after[_PK + _WRITTEN]
    n = store.save(Tables.def14a_llm, payload, pk=_PK)
    print(f"\nwrote {n:,} row(s) ({len(decisions)} changed cell(s)) to def14a_llm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
