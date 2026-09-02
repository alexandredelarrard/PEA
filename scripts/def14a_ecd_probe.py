"""
def14a_ecd_probe.py  (scripts/)
--------------------------------------------------------------------------------------------
Phase-4 measurement, run BEFORE any ECD code is written: what does the inline-XBRL facts frame
of a real proxy actually look like?

The plan says to filter `dim_ecd_ExecutiveCategoryAxis == 'ecd:PeoMember'`. That string is a
guess until measured -- edgartools builds the dimension column names and member values from the
instance, and the repo's own rule is to never trust a tag name. So this script prints, for each
probed filing:

  * every column of `filing.xbrl().facts.to_dataframe()` whose name mentions a dimension
  * every distinct value those columns take on `ecd:` concepts
  * the `ecd:PeoName` facts with their dimensions, which is where the library's bug lives
    (26 `ecd:PeoName` facts on AAPL, only 5 of them PEO)
  * the per-individual PEO totals, so the co-PEO cases (BA: Ortberg + Calhoun, NKE: Hill +
    Donahoe) and the SBUX zero-matrix are visible as data rather than as claims

No LLM. `filing.xbrl()` is one archive fetch per filing, cached by edgartools thereafter.

    "$PY" scripts/def14a_ecd_probe.py [-c ./configs] [--tickers BA,NKE,SBUX,AAPL] [--out FILE]
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

#: (ticker, filing year) -> the claim this filing is here to check.
DEFAULT_TARGETS = (
    ("BA", 2026, "co-PEO: Ortberg 18,388,629 kept, Calhoun 15,050,812 dropped by the library"),
    ("NKE", 2026, "co-PEO: Hill + Donahoe"),
    ("SBUX", 2026, "individual x year matrix with 0.0 in non-applicable cells"),
    ("AAPL", 2026, "26 ecd:PeoName facts, only 5 PeoMember-dimensioned"),
    ("AAPL", 2023, "0 ecd tags: FY2022 ended before the 2022-12-16 threshold -> no row"),
)


def _facts(filing) -> pd.DataFrame | None:
    """The same frame `ProxyStatement._facts_dataframe` uses -- nothing extra downloaded."""
    xbrl = filing.xbrl()
    if xbrl is None:
        return None
    df = xbrl.facts.to_dataframe()
    return None if df is None or df.empty else df


def describe(ticker: str, filing, out: list[str]) -> None:
    def say(s: str = "") -> None:
        print(s)
        out.append(s)

    say(f"\n{'=' * 92}")
    say(f"{ticker}  {filing.filing_date}  {filing.accession_no}  form={filing.form}")
    facts = _facts(filing)
    if facts is None:
        say("  filing.xbrl() -> None or empty  =>  NO ROW should be written for this filing")
        return

    ecd = facts[facts["concept"].astype(str).str.startswith("ecd:")]
    say(f"  facts: {len(facts):,} total, {len(ecd):,} on ecd: concepts")
    if ecd.empty:
        say("  no ecd: concepts  =>  pre-threshold proxy, NO ROW")
        return

    dim_cols = [c for c in facts.columns if "dim" in c.lower() or "axis" in c.lower()]
    say(f"  dimension-ish columns ({len(dim_cols)}): {dim_cols}")
    for c in dim_cols:
        vals = ecd[c].dropna().unique().tolist()
        if vals:
            say(f"    {c}: {vals[:8]}{' …' if len(vals) > 8 else ''}")

    # Full concept inventory. Necessary because the plan names `ecd:PeoTotalComp`, and a concept
    # that does not exist under the name you expect is indistinguishable from a filer omission.
    say("  ecd concepts (n facts / n with an ExecutiveCategoryAxis / n with an IndividualAxis):")
    cat_col = "dim_ecd_ExecutiveCategoryAxis"
    ind_col = "dim_ecd_IndividualAxis"
    for concept, sub in sorted(ecd.groupby("concept"), key=lambda kv: -len(kv[1])):
        n_cat = int(sub[cat_col].notna().sum()) if cat_col in sub.columns else 0
        n_ind = int(sub[ind_col].notna().sum()) if ind_col in sub.columns else 0
        say(f"    {concept:<46}{len(sub):>4}{n_cat:>6}{n_ind:>6}")

    # the bug's epicentre: how many PeoName facts, and how are they discriminated?
    names = ecd[ecd["concept"] == "ecd:PeoName"]
    say(f"  ecd:PeoName facts: {len(names)}")
    if not names.empty:
        keep = [c for c in ("value", "period_end", *dim_cols) if c in names.columns]
        say(names[keep].to_string(max_rows=40))

    # per-individual PEO comp, which is what the storage-shape decision rests on.
    # NOTE the `Amt` suffix: the plan writes `ecd:PeoTotalComp`, which does not exist.
    for concept in ("ecd:PeoTotalCompAmt", "ecd:PeoActuallyPaidCompAmt"):
        sub = ecd[ecd["concept"] == concept]
        if sub.empty:
            say(f"  {concept}: absent")
            continue
        keep = [c for c in ("value", "period_end", *dim_cols) if c in sub.columns]
        say(f"  {concept}: {len(sub)} facts")
        say(sub[keep].to_string(max_rows=40))


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the ECD dimensions on real proxies.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--targets", default=None, help="comma-separated TICKER:YEAR")
    ap.add_argument("--out", default=None, help="write the transcript here")
    args = ap.parse_args()

    _config, context = get_config_context(args.config, use_cache=False, save=False)
    context.ensure_edgar_identity()          # SEC blocks a request without a descriptive UA
    from edgar import Company                              # noqa: E402  (needs the identity set)

    targets = ([(t.split(":")[0], int(t.split(":")[1]), "requested")
                for t in args.targets.split(",")] if args.targets else list(DEFAULT_TARGETS))

    out: list[str] = []
    for ticker, year, why in targets:
        print(f"\n>>> {ticker} {year}: {why}")
        try:
            filings = Company(ticker).get_filings(form=["DEF 14A", "DEF 14C"])
            picked = [f for f in filings if f.filing_date.year == year]
            if not picked:
                print(f"    no {year} proxy found")
                continue
            describe(ticker, picked[0], out)
        except Exception as e:
            print(f"    FAILED ({type(e).__name__}: {e})")

    if args.out:
        Path(args.out).write_text("\n".join(out), encoding="utf-8")
        print(f"\n  written -> {args.out}")


if __name__ == "__main__":
    main()
