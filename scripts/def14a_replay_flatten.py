"""
def14a_replay_flatten.py  (scripts/)
--------------------------------------------------------------------------------------------
Replay the flatten + the four child-table row builders over the ALREADY-STORED `def14a_json`
blobs. **Zero LLM calls** -- the tokens were paid when the filings were first extracted.

This is how Phase 3's schema change is verified before a single new token is spent. Every
derived column and every child row is a pure function of `Def14AExtract`, so re-running them
over the Phase-0 parquet snapshot exercises the whole flatten against 445 real filings.

What it prints, and why each number is the actual check:
  * rows produced per child table -- the recall floor
  * the `reconciles` rate on both comp tables -- the D10 flag's failure rate, measurable rather
    than hidden by a repair
  * max value per numeric column -- the research expects **2 rows above $1e9 out of 34,741**
    (versus 109 on the retired edgar path). More than that means the FLATTEN has a bug, not the
    LLM, and this is where it shows up.
  * the `gender_basis` distribution -- how much of `gender` is document evidence versus a
    first-name prior

Read-only. Reads the Phase-0 parquet, writes nothing unless `--out` is given.

    "$PY" scripts/def14a_replay_flatten.py [--baseline DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_extract.utils.structure.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.fetch_def14a_llm import (
    _CHILD_SPEC, _child_frames, _flatten,
)

DEFAULT_BASELINE = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline"
#: A component above this is arithmetically impossible for an S&P 500 executive.
IMPLAUSIBLE_USD = 1e9
#: Which `Def14AExtract` array each child table is built from -- used to tell "the builder
#: produced nothing" apart from "the stored blobs predate this array".
_SOURCE_ARRAY = {
    "def14a_executive_comp": "compensation",
    "def14a_director_comp": "director_compensation",
    "def14a_ownership": "ownership_holders",
    "def14a_directors": "directors",
}


def replay(baseline: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Re-run the flatten over every stored `def14a_json`. Returns (parent, children, stats)."""
    src = pd.read_parquet(baseline / "def14a_llm.parquet")
    parents, children = [], {name: [] for name in _CHILD_SPEC}
    arrays_present: set[str] = set()
    n_ok = n_bad = 0

    for _, r in src.iterrows():
        blob = r.get("def14a_json")
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            extract = Def14AExtract.model_validate_json(blob)
        except Exception as e:
            # A stored blob that no longer validates is EXPECTED and informative: the schema
            # dropped `n_technology_directors` / `technology_committee` and added fields. Pydantic
            # ignores unknown keys by default, so a failure here means something structural.
            n_bad += 1
            if n_bad <= 3:
                print(f"  ! {r['ticker']} {r['accession_number']}: {type(e).__name__}: {e}")
            continue
        n_ok += 1
        arrays_present.update(k for k, v in json.loads(blob).items() if isinstance(v, list) and v)
        filing = pd.Series({"filing_date": r["as_of"], "accession_number": r["accession_number"],
                            "period_of_report": r.get("period"), "cik": r.get("cik")})
        parents.append(_flatten(r["ticker"], filing, extract))
        for name, rows in _child_frames(r["ticker"], filing, extract).items():
            children[name].extend(rows)

    return (pd.DataFrame(parents),
            {k: pd.DataFrame(v) for k, v in children.items()},
            {"blobs_replayed": n_ok, "blobs_failed": n_bad, "source_rows": len(src),
             "arrays_present": sorted(arrays_present)})


def report(parent: pd.DataFrame, children: dict[str, pd.DataFrame], stats: dict) -> bool:
    """Print the replay conclusion. Returns True when every check passes."""
    ok = True
    print(f"\n=== SANITY CHECK: flatten replay over stored def14a_json (0 LLM calls) ===")
    print(f"  {stats['blobs_replayed']} of {stats['source_rows']} blobs replayed"
          f" ({stats['blobs_failed']} failed to validate)")

    print(f"\n  {'table':<26}{'rows':>8}{'filings':>9}{'tickers':>9}")
    print(f"  {'def14a_llm':<26}{len(parent):>8}{parent['accession_number'].nunique():>9}"
          f"{parent['ticker'].nunique():>9}")
    for name, df in children.items():
        if df.empty:
            # An empty table is only a FAILURE when the source blobs actually carry the array
            # it is built from. `director_compensation` and `ownership_holders` are NEW in this
            # phase's schema, so a replay of blobs extracted under the old contract cannot
            # produce them -- they populate on the first NEW extraction, and their builders are
            # covered by the unit tests instead.
            expected = _SOURCE_ARRAY.get(name) in stats["arrays_present"]
            note = "<-- EMPTY (builder produced nothing)" if expected else \
                   "<-- expected: the stored blobs predate this array"
            print(f"  {name:<26}{0:>8}{0:>9}{0:>9}   {note}")
            ok = ok and not expected
            continue
        print(f"  {name:<26}{len(df):>8}{df['accession_number'].nunique():>9}"
              f"{df['ticker'].nunique():>9}")

    # ---- the $1e9 check: the research's expectation is 2 in 34,741 ----
    print(f"\n  implausible values (> ${IMPLAUSIBLE_USD:,.0f}):")
    for name, df in children.items():
        if df.empty:
            continue
        numeric = [c for c in _CHILD_SPEC[name][0] if c in df.columns and c != "reconciles"]
        vals = df[numeric].apply(pd.to_numeric, errors="coerce")
        bad = vals.abs().gt(IMPLAUSIBLE_USD).any(axis=1)
        n = int(bad.sum())
        print(f"    {name:<26}{n:>6} of {len(df):>6} rows"
              f"   (edgar path: 109 in executive_comp)")
        if n:
            for _, r in df[bad].head(3).iterrows():
                worst = max((abs(pd.to_numeric(r[c], errors="coerce") or 0), c) for c in numeric)
                print(f"        {r['ticker']} {r['accession_number']} "
                      f"{r.get('name', r.get('holder_name', ''))}: {worst[1]}={worst[0]:,.0f}")

    # ---- the reconciles flag: a measured failure rate, not a repair ----
    print(f"\n  `reconciles` rate (components sum to total within $10):")
    for name in ("def14a_executive_comp", "def14a_director_comp"):
        df = children.get(name, pd.DataFrame())
        if df.empty or "reconciles" not in df.columns:
            continue
        v = pd.to_numeric(df["reconciles"], errors="coerce")
        computable = v.notna().sum()
        print(f"    {name:<26}{v.mean():.1%} of {computable} computable rows"
              f"  ({len(df) - computable} have no `total`)")

    # ---- gender provenance ----
    dirs = children.get("def14a_directors", pd.DataFrame())
    if not dirs.empty and "gender_basis" in dirs.columns:
        g = dirs[dirs["gender"].notna()]
        print(f"\n  gender: {len(g)} of {len(dirs)} director rows have a value")
        if len(g):
            basis_ok = g["gender_basis"].notna().all()
            print(f"    gender_basis populated on every non-null gender: {basis_ok}")
            ok = ok and bool(basis_ok)
            for b, c in g["gender_basis"].value_counts(dropna=False).items():
                print(f"      {str(b):<12}{c:>7}  {100 * c / len(g):>5.1f}%")

    # ---- multi-year SCT: n_neos must count NAMES, sct_years must show the 3 years ----
    if "sct_years" in parent.columns:
        sy = pd.to_numeric(parent["sct_years"], errors="coerce").dropna()
        nn = pd.to_numeric(parent["n_neos"], errors="coerce").dropna()
        print(f"\n  SCT shape: sct_years mean {sy.mean():.2f} (Item 402(c) requires 3)")
        print(f"             n_neos mean {nn.mean():.2f}, == 1 on {(nn == 1).mean():.1%} of rows")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the DEF 14A flatten over stored JSON.")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out", default=None, help="write the replayed frames as parquet here")
    args = ap.parse_args()

    parent, children, stats = replay(Path(args.baseline))
    ok = report(parent, children, stats)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        parent.to_parquet(out / "def14a_llm.parquet", index=False)
        for name, df in children.items():
            df.to_parquet(out / f"{name}.parquet", index=False)
        (out / "replay_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\n  written -> {out}")

    print(f"\n  {'PASS' if ok else 'FAIL'}: the flatten is a pure replay of paid tokens; "
          f"no LLM call was made.")


if __name__ == "__main__":
    main()
