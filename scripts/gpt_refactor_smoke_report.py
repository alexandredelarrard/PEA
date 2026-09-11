"""
gpt_refactor_smoke_report.py  (scripts/)
--------------------------------------------------------------------------------------------
What the refactored `gpt_extract` path actually wrote, for a human to accept or reject.

Read-only. Compares the five live DEF 14A tables against the pre-run snapshot
(`gpt_refactor_smoke_prep.py` wrote it) and prints the named spot checks the plan asks for.

Three sections:

  1. **Shape and fill.** Rows, columns and per-column non-null rate for `def14a_llm` and the
     four child tables, restricted to the 15 smoke tickers. A permanently-NULL column reads
     downstream as "this company does not disclose it" rather than "we never extracted it",
     which is exactly how the retired edgar table's `auditor_name` sat at 2.05% fill while the
     firm name is present in 98% of documents -- so the fill rate is the finding, not a stat.

  2. **Old vs new, same accession.** The snapshot's 292 rows against the re-extracted ones.
     Both paths saw the same filings, so a per-column fill delta is a like-for-like comparison
     of the extraction, not of the corpus.

  3. **Named spot checks**, each pinned to a filing geometry the plan chose its tickers for.

    "$PY" scripts/gpt_refactor_smoke_report.py [-c ./configs] [--snapshot DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_store.schema import Tables

from scripts.gpt_refactor_smoke_prep import SMOKE_TICKERS

SNAPSHOT = ROOT / "reports/planning/active-tasks/2026-09-02-gpt-extract-refactor/baseline"

TABLES = {
    "def14a_llm": Tables.def14a_llm,
    "def14a_executive_comp": Tables.def14a_executive_comp,
    "def14a_director_comp": Tables.def14a_director_comp,
    "def14a_ownership": Tables.def14a_ownership,
    "def14a_directors": Tables.def14a_directors,
}

#: A Summary Compensation Table title that means "this person was a principal executive
#: officer during the year". Two such rows in ONE (accession, fiscal_year) is the co-PEO
#: geometry BA and NKE are in the smoke set for.
_PEO_TITLE = re.compile(r"chief executive|\bCEO\b|president and chief", re.I)

#: TRI-STATE by contract, so a low or zero fill rate is the DESIGNED answer and not a gap:
#: null means the proxy was silent. Inferring FALSE from silence made `poison_pill` true in
#: 0.1% of rows and flipped `majority_voting` 21.2% year-over-year on a bylaw that does not
#: change, which is why the prompt forbids it.
_TRI_STATE = {"poison_pill", "majority_voting"}


def _fill(df: pd.DataFrame) -> pd.Series:
    """Non-null rate per column, as a percentage."""
    return (df.notna().mean() * 100).round(1)


def section_shape(live: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 78)
    print("1. SHAPE AND FILL — the 15 smoke tickers")
    print("=" * 78)
    print(f"\n  {'table':<26}{'rows':>8}{'cols':>7}{'tickers':>9}{'accessions':>12}")
    for name, df in live.items():
        if df.empty:
            print(f"  {name:<26}{'ABSENT or EMPTY':>36}")
            continue
        print(f"  {name:<26}{len(df):>8}{df.shape[1]:>7}"
              f"{df['ticker'].nunique():>9}{df['accession_number'].nunique():>12}")

    for name, df in live.items():
        if df.empty:
            continue
        fill = _fill(df).sort_values()
        print(f"\n  {name} — per-column non-null %  ({len(df)} rows)")
        for col, pct in fill.items():
            if col in _TRI_STATE:
                mark = "  (tri-state: null == the proxy is SILENT, so this is by contract)"
            elif pct == 0.0:
                mark = "  <-- NEVER WRITTEN"
            else:
                mark = ""
            print(f"    {col:<34}{pct:>6.1f}%{mark}")
        empty = [c for c, p in fill.items() if p == 0.0 and c not in _TRI_STATE]
        if empty:
            print(f"    {len(empty)} column(s) at 0% with no tri-state excuse: {empty}")
            print(f"      each reads downstream as a disclosure absence, not an extraction gap")


def section_old_vs_new(old: pd.DataFrame, new: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2. OLD vs NEW — same accessions, both paths")
    print("=" * 78)

    # Only a ticker that HAS been re-extracted can have lost an accession. Comparing against
    # tickers the run has not reached yet reports the whole backlog as a recall regression.
    done_tickers = sorted(set(new["ticker"]))
    old = old[old["ticker"].isin(done_tickers)]
    print(f"\n  {len(done_tickers)} of {len(SMOKE_TICKERS)} smoke ticker(s) re-extracted: "
          f"{', '.join(done_tickers)}")

    old_acc, new_acc = set(old["accession_number"]), set(new["accession_number"])
    both = old_acc & new_acc
    print(f"  snapshot accessions {len(old_acc)}, re-extracted {len(new_acc)}, "
          f"in both {len(both)}")
    lost = sorted(old_acc - new_acc)
    gained = sorted(new_acc - old_acc)
    if lost:
        # a filing the OLD path produced a row for and the new one did not: a recall REGRESSION,
        # and the only number in this report that can invalidate the refactor on its own
        print(f"  ! {len(lost)} accession(s) the old path had and the new one does NOT:")
        for a in lost[:10]:
            r = old[old["accession_number"] == a].iloc[0]
            print(f"      {r['ticker']:<7}{str(r['as_of'])[:10]}  {a}")
    if gained:
        print(f"  + {len(gained)} accession(s) NEW to this path (the old run never stored them)")
        for a in gained[:10]:
            r = new[new["accession_number"] == a].iloc[0]
            print(f"      {r['ticker']:<7}{str(r['as_of'])[:10]}  {a}")

    o = old[old["accession_number"].isin(both)]
    n = new[new["accession_number"].isin(both)]
    shared = [c for c in o.columns if c in n.columns and c != "def14a_json"]
    of, nf = _fill(o[shared]), _fill(n[shared])
    delta = (nf - of).sort_values()

    print(f"\n  per-column non-null % on the {len(both)} shared accessions "
          f"({len(shared)} shared columns)")
    print(f"    {'column':<34}{'old':>8}{'new':>8}{'delta':>9}")
    for col in delta.index:
        if col in _TRI_STATE:
            # A DROP here is the intended change, not a loss: the old path inferred FALSE from
            # silence, this one returns null. That inference is what made `poison_pill` true in
            # 0.1% of rows and flipped `majority_voting` 21.2% year-over-year.
            flag = "  <-- tri-state now; the old fill was FALSE-from-silence"
        elif delta[col] <= -5:
            flag = "  <-- FILL LOST"
        else:
            flag = ""
        print(f"    {col:<34}{of[col]:>7.1f}%{nf[col]:>7.1f}%{delta[col]:>+8.1f}{flag}")

    real_losses = [c for c in delta.index if delta[c] <= -5 and c not in _TRI_STATE]
    print(f"\n  {len(real_losses)} column(s) lost >=5 points of fill with no contract change "
          f"to explain it: {real_losses or 'none'}")
    print("    These are findings to HAND OVER, not to fix by re-tuning the prompt — prompt")
    print("    quality is out of this refactor's scope and every re-run costs tokens.")

    only_new = [c for c in n.columns if c not in o.columns]
    if only_new:
        nfn = _fill(n[only_new]).sort_values(ascending=False)
        print(f"\n  {len(only_new)} column(s) the old path did not have at all:")
        for col, pct in nfn.items():
            print(f"    {col:<34}{pct:>6.1f}%")


def _spot_co_peo(exe: pd.DataFrame) -> None:
    print("\n  [BA / NKE] co-PEO years — two PEO-titled rows in one (accession, fiscal_year)")
    sub = exe[exe["ticker"].isin(["BA", "NKE"])]
    if sub.empty:
        print("    no executive-comp rows for BA/NKE")
        return
    peo = sub[sub["title"].fillna("").str.contains(_PEO_TITLE)]
    grp = peo.groupby(["ticker", "accession_number", "fiscal_year"]).size()
    multi = grp[grp > 1]
    print(f"    {len(peo)} PEO-titled row(s) over {sub['accession_number'].nunique()} filing(s); "
          f"{len(multi)} (ticker, filing, year) group(s) carry more than one")
    for (t, acc, fy), n in multi.head(8).items():
        names = peo[(peo["accession_number"] == acc) & (peo["fiscal_year"] == fy)]
        print(f"      {t} FY{int(fy)} {acc}: {n} — "
              + "; ".join(f"{r['name']} ({r['title']})" for _, r in names.iterrows()))
    if multi.empty:
        print("      NONE — either the co-PEO years are outside the window or the SCT titles "
              "did not survive; check a known co-PEO filing by hand before accepting")


def _spot_dual_class(own: pd.DataFrame) -> None:
    print("\n  [GOOGL / BRK-B] percent_of_class must be a CLASS percentage, not voting power")
    for t in ("GOOGL", "BRK-B"):
        sub = own[own["ticker"] == t]
        if sub.empty:
            print(f"    {t}: no ownership rows")
            continue
        pct = pd.to_numeric(sub["percent_of_class"], errors="coerce")
        latest = sub[sub["as_of"] == sub["as_of"].max()]
        print(f"    {t}: {len(sub)} row(s) over {sub['accession_number'].nunique()} filing(s), "
              f"percent_of_class present on {pct.notna().mean():.0%}, "
              f"max {pct.max() if pct.notna().any() else float('nan'):.4f}")
        # a value above 1.0 means a PERCENT was stored where a FRACTION was contracted; a
        # per-filing sum far above 1.0 means the voting-power column leaked in
        for _, r in latest.sort_values("percent_of_class", ascending=False).head(6).iterrows():
            print(f"        {str(r['as_of'])[:10]}  {r['holder_name'][:38]:<38}"
                  f"{r['holder_type']:<18}{r['percent_of_class']}")


def _spot_pre_2001(parent: pd.DataFrame) -> None:
    print("\n  [A] pre-2001 proxies — the `primaryDocument == \"\"` .txt fallback")
    sub = parent[(parent["ticker"] == "A") &
                 (pd.to_datetime(parent["as_of"]) < pd.Timestamp("2001-01-01"))]
    if sub.empty:
        print("    no pre-2001 rows for A — the filings are either outside the window or "
              "produced no row at all (which is the failure this ticker is here to catch)")
        return
    body = [c for c in sub.columns if c not in ("ticker", "as_of", "period",
                                                "accession_number", "def14a_json")]
    filled = sub[body].notna().sum(axis=1)
    print(f"    {len(sub)} row(s); non-null fields per row: min {filled.min()}, "
          f"median {int(filled.median())}, max {filled.max()} (of {len(body)})")
    for (_, r), n in zip(sub.iterrows(), filled):
        print(f"      {str(r['as_of'])[:10]}  {r['accession_number']}  {n:>2} fields  "
              f"ceo={r.get('ceo_name_proxy')}  directors={r.get('n_directors')}")


def _spot_reconciles(exe: pd.DataFrame, dirc: pd.DataFrame) -> None:
    print("\n  [all] `reconciles` — components sum to `total` within $10 (a FLAG, not a repair)")
    for name, df in (("def14a_executive_comp", exe), ("def14a_director_comp", dirc)):
        if df.empty or "reconciles" not in df.columns:
            print(f"    {name}: no rows")
            continue
        v = pd.to_numeric(df["reconciles"], errors="coerce")
        print(f"    {name:<24}{v.mean():.1%} of {v.notna().sum()} computable row(s)"
              f"  ({v.isna().sum()} have no `total`)")


def _spot_pvp(store) -> None:
    print("\n  [SBUX] Pay-versus-Performance / negative `peo_actually_paid_comp`")
    print("    NOT on this path. PVP is an XBRL ECD block read by `fetch_def14a_edgar`")
    print("    (`ecd.py` -> `sec_def14a`), which this refactor did not touch — it is not an")
    print("    LLM path. `def14a_llm` has no `peo_*` column, so there is nothing to check here.")
    print(f"    sec_def14a on this deployment: "
          f"{'EXISTS' if store.exists('sec_def14a') else 'ABSENT (dropped at an earlier cutover)'}")


def section_spot_checks(live: dict[str, pd.DataFrame], store) -> None:
    print("\n" + "=" * 78)
    print("3. NAMED SPOT CHECKS")
    print("=" * 78)
    _spot_co_peo(live["def14a_executive_comp"])
    _spot_dual_class(live["def14a_ownership"])
    _spot_pre_2001(live["def14a_llm"])
    _spot_reconciles(live["def14a_executive_comp"], live["def14a_director_comp"])
    _spot_pvp(store)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the Phase-7 smoke run for inspection.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    args = ap.parse_args()

    _, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store

    live: dict[str, pd.DataFrame] = {}
    for name, table in TABLES.items():
        df = store.load(table, where={"ticker": list(SMOKE_TICKERS)}, optional=True)
        live[name] = pd.DataFrame() if df is None else df

    section_shape(live)

    snap = Path(args.snapshot) / "def14a_llm.parquet"
    if snap.exists() and not live["def14a_llm"].empty:
        old = pd.read_parquet(snap)
        old = old[old["ticker"].isin(SMOKE_TICKERS)]
        section_old_vs_new(old, live["def14a_llm"])
    else:
        print(f"\n(no snapshot at {snap} — skipping the old/new comparison)")

    section_spot_checks(live, store)


if __name__ == "__main__":
    main()
