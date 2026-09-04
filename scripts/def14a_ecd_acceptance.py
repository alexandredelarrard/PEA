"""
def14a_ecd_acceptance.py  (scripts/)
--------------------------------------------------------------------------------------------
Acceptance sheet for `def14a_ecd.py`: run the new ECD reader over real proxies and print what
it recovers, per column and per gate case.

This is the measurement the Phase-4 design rests on. Four filings proved the *mechanism*
(`scripts/def14a_ecd_probe.py`); this proves the mechanism holds across a roster rather than on
the examples it was written from -- a rule that works on its own four fixtures is not a rule.

Zero LLM. One `filing.xbrl()` per filing, cached by edgartools after the first read.

    "$PY" scripts/def14a_ecd_acceptance.py [-c ./configs] [--tickers BA,NKE] [--since 2023]
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
from src.data_extract.utils.structure.def14a.ecd import ecd_facts, ecd_row, has_ecd_block

BASELINE = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline"

#: Cases with a known right answer, from the Phase-4 measurement. `None` means "no row".
GATES: dict[tuple[str, int], dict] = {
    ("BA", 2025): {"n_peos": 2.0, "names_contain": ("Ortberg", "Calhoun")},
    ("NKE", 2025): {"n_peos": 2.0, "names_contain": ("Hill", "Donahoe")},
    ("SBUX", 2026): {"peo_total_nonzero": True, "names_contain": ("Niccol",)},
    ("AAPL", 2026): {"names_contain": ("Cook",)},
    ("AAPL", 2023): {"no_row": True},
}


def _tickers(args) -> list[str]:
    if args.tickers:
        return args.tickers.split(",")
    idx = pd.read_parquet(BASELINE / "filings.parquet")
    return sorted(idx["ticker"].dropna().unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Acceptance sheet for the ECD reader.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--since", type=int, default=2023, help="earliest filing YEAR to read")
    ap.add_argument("--out", default=None, help="write the per-filing rows as parquet")
    args = ap.parse_args()

    _config, context = get_config_context(args.config, use_cache=False, save=False)
    context.ensure_edgar_identity()
    from edgar import Company                              # noqa: E402

    rows, skipped = [], []
    for ticker in _tickers(args):
        try:
            filings = Company(ticker).get_filings(form=["DEF 14A", "DEF 14C"])
        except Exception as e:                             # noqa: BLE001
            print(f"  {ticker}: listing failed ({type(e).__name__}: {e})")
            continue
        for f in filings:
            if f.filing_date.year < args.since:
                continue
            try:
                facts = ecd_facts(f)
                if not has_ecd_block(facts):
                    skipped.append((ticker, f.filing_date.year, f.accession_no))
                    continue
                r = ecd_row(facts)
            except Exception as e:                         # noqa: BLE001
                print(f"  {ticker} {f.filing_date}: FAILED ({type(e).__name__}: {e})")
                continue
            r.update(ticker=ticker, filing_year=f.filing_date.year,
                     accession_number=f.accession_no, filing_date=f.filing_date)
            rows.append(r)
            print(f"  {ticker:<6}{f.filing_date}  n_peos={r['n_peos']}  "
                  f"total={r['peo_total_comp']:,.0f}  cap={r['peo_actually_paid_comp']:,.0f}  "
                  f"{r['peo_name']}")

    df = pd.DataFrame(rows)
    ok = report(df, skipped)
    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"\n  written -> {args.out}")
    sys.exit(0 if ok else 1)


def report(df: pd.DataFrame, skipped: list) -> bool:
    print("\n=== SANITY: the ECD reader on real proxies ===")
    if df.empty:
        print("  no rows produced -- the concept names or the dimension rule are wrong.")
        return False
    ok = True
    print(f"  {len(df)} rows written, {len(skipped)} filings correctly skipped "
          f"(no ecd: facts -> pre-402(v) fiscal year)")

    print("\n  fill per column:")
    for c in sorted(df.columns):
        if c in ("ticker", "accession_number", "filing_date", "filing_year"):
            continue
        fill = df[c].notna().mean()
        print(f"    {c:<38}{100 * fill:>6.1f}%")

    # a PEO total of exactly 0 is the SBUX matrix leaking through
    zeros = int((pd.to_numeric(df["peo_total_comp"], errors="coerce") == 0).sum())
    print(f"\n  peo_total_comp == 0 exactly: {zeros} (must be 0 -- that value is a "
          f"'not applicable this year' matrix cell, never a disclosure)")
    ok &= zeros == 0

    # the sign must survive; a run with zero negatives means an abs() crept in
    cap = pd.to_numeric(df["peo_actually_paid_comp"], errors="coerce")
    n_neg = int((cap < 0).sum())
    print(f"  negative peo_actually_paid_comp: {n_neg} of {int(cap.notna().sum())} "
          f"({100 * n_neg / max(int(cap.notna().sum()), 1):.1f}%) -- research measured 11.83%; "
          f"0 would mean the sign was destroyed")
    ok &= n_neg > 0

    n = pd.to_numeric(df["n_peos"], errors="coerce")
    print(f"  n_peos distribution: {n.value_counts().sort_index().to_dict()}")
    ok &= bool((n >= 1).all())

    print("\n  gate cases:")
    for (ticker, year), want in GATES.items():
        sub = df[(df["ticker"] == ticker) & (df["filing_year"] == year)]
        if want.get("no_row"):
            good = sub.empty
            print(f"    {ticker} {year}: no row -> {good}  (pre-402(v) fiscal year)")
            ok &= good
            continue
        if sub.empty:
            print(f"    {ticker} {year}: MISSING -- expected a row")
            ok = False
            continue
        r = sub.iloc[0]
        bits = []
        if "n_peos" in want:
            good = float(r["n_peos"]) == want["n_peos"]
            bits.append(f"n_peos={r['n_peos']} (want {want['n_peos']}) {'OK' if good else 'BAD'}")
            ok &= good
        if want.get("peo_total_nonzero"):
            good = bool(pd.notna(r["peo_total_comp"])) and float(r["peo_total_comp"]) != 0.0
            bits.append(f"total={r['peo_total_comp']:,.0f} {'OK' if good else 'BAD'}")
            ok &= good
        names = f"{r['peo_name']}|{r['peo_names_all']}"
        for token in want.get("names_contain", ()):
            good = token in names
            bits.append(f"{token}{'' if good else ' MISSING'}")
            ok &= good
        print(f"    {ticker} {year}: " + "; ".join(bits))

    print(f"\n  {'PASS' if ok else 'FAIL'}: {len(df)} proxies read straight from filer-tagged "
          f"XBRL; no LLM, no HTML parsing.")
    return ok


if __name__ == "__main__":
    main()
