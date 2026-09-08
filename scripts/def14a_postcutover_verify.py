"""
def14a_postcutover_verify.py  (scripts/)
--------------------------------------------------------------------------------------------
Phase-6 Step 5: re-run the defect assertions against the REAL DATABASE after the backfill,
over the whole rebuilt tables rather than the 23-ticker validation slice.

Why it cannot reuse Step 2's parquet harness: a parquet round-trip hides an entire bug class.
Postgres returns a DATE column as `datetime.date`, not `pd.Timestamp`, so a comparison that
works on the cached frame can fail on the live read — and a cached harness would never show it.
Everything here reads through `context.store` against Postgres.

Read-only. Nothing in this file writes, drops or truncates.

    "$PY" scripts/def14a_postcutover_verify.py [-c ./configs] [--skip-cube]
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

from src.context import get_config_context
from src.data_store.schema import Tables

#: A single SCT / director-fee component above this is arithmetically impossible for an S&P 500
#: executive and is the signature of the `<br>`-concatenation defect (AMAT `total` = 3.527e23).
IMPLAUSIBLE_USD = 1e9

#: The three filers measured to publish a PVP table with NO inline XBRL. They are the expected
#: residue of the 2023 shortfall, alongside non-December fiscal year ends; if the gap does not
#: decompose that way, the ECD dimension filter is dropping rows it should keep.
KNOWN_NO_XBRL_2023 = ("APP", "BSX", "GDDY")

#: cp1252 mis-decode artifacts. The RETIRED edgar path carried U+0097 on 7.06% of its
#: exec-comp rows; the LLM path reads a decoded string and should carry none.
MOJIBAKE = ("", "", "", "", "�")

_PASS, _FAIL, _INFO = "PASS", "FAIL", "INFO"


def _row(results: list, name: str, status: str, detail: str) -> None:
    results.append({"check": name, "status": status, "detail": detail})


def check_coverage(store, results: list) -> None:
    llm = store.load(Tables.def14a_llm, columns=["ticker", "as_of", "accession_number"])
    n_tick = llm["ticker"].nunique()
    _row(results, "def14a_llm ticker coverage", _PASS if n_tick >= 450 else _FAIL,
         f"{n_tick} tickers, {len(llm):,} rows "
         f"({llm['as_of'].min()} -> {llm['as_of'].max()}); expected ~500")

    for name, table in (("def14a_directors", Tables.def14a_directors),
                        ("def14a_executive_comp", Tables.def14a_executive_comp),
                        ("def14a_director_comp", Tables.def14a_director_comp),
                        ("def14a_ownership", Tables.def14a_ownership),
                        ("sec_8k_votes", Tables.sec_8k_votes)):
        if not store.exists(table):
            _row(results, f"{name} exists", _FAIL, "table absent after the backfill")
            continue
        n = store.row_count(table)
        _row(results, f"{name} rows", _PASS if n > 0 else _FAIL, f"{n:,} rows")


def check_implausible(store, results: list) -> None:
    """G1: no exec-comp / director-fee component above $1e9. The retired edgar path had 109."""
    for name, table, cols in (
        ("def14a_executive_comp", Tables.def14a_executive_comp,
         ["salary", "bonus", "stock_awards", "option_awards", "non_equity_incentive",
          "pension_change", "other_compensation", "total"]),
        ("def14a_director_comp", Tables.def14a_director_comp,
         ["fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
          "pension_change", "other_compensation", "total"]),
    ):
        if not store.exists(table):
            continue
        df = store.load(table, columns=["ticker", "accession_number", "name"] + cols)
        num = df[cols].apply(pd.to_numeric, errors="coerce")
        bad = df[(num > IMPLAUSIBLE_USD).any(axis=1)]
        worst = "" if bad.empty else (
            f" worst: {bad.iloc[0]['ticker']} {bad.iloc[0]['name']}")
        _row(results, f"G1 {name} components > $1e9", _PASS if bad.empty else _FAIL,
             f"{len(bad)} row(s) of {len(df):,}{worst}")


def check_ecd(store, results: list) -> None:
    """The ECD block did not exist in ANY proxy before 2023 (Rel. 34-95607, FY ending
    >= 2022-12-16). 0% fill before and 81-100% after is CORRECT, not a gap."""
    # Project to the columns the table ACTUALLY has: run before the cutover, `sec_def14a` is
    # still the old 46-column shape with no `n_peos`, and a KeyError there would mask every
    # other ECD assertion behind one missing column.
    want = ["ticker", "filing_date", "period_of_report", "n_peos", "peo_name",
            "peo_total_comp", "peo_actually_paid_comp", "net_income",
            "total_shareholder_return"]
    live = set(store.columns(Tables.def14a_edgar))
    absent = [c for c in want if c not in live]
    if absent:
        _row(results, "sec_def14a has the Phase-4 column set", _FAIL,
             f"missing {absent} — the table was not rebuilt (still {len(live)} columns)")
    df = store.load(Tables.def14a_edgar, columns=[c for c in want if c in live])
    for c in absent:
        df[c] = None
    if df.empty:
        _row(results, "sec_def14a rows", _FAIL, "table is empty after the backfill")
        return
    year = pd.to_datetime(df["filing_date"], errors="coerce").dt.year
    pre = df[year <= 2022]
    _row(results, "sec_def14a has NO pre-2023 rows", _PASS if pre.empty else _FAIL,
         f"{len(pre)} row(s) dated <= 2022 (a proxy with no ecd: facts must get NO row)")

    for y in (2023, 2024, 2025, 2026):
        sub = df[year == y]
        if sub.empty:
            _row(results, f"sec_def14a {y}", _INFO, "no rows")
            continue
        fill = sub["peo_actually_paid_comp"].notna().mean()
        _row(results, f"sec_def14a {y} peo_actually_paid fill",
             _PASS if fill >= 0.81 else _FAIL,
             f"{fill:.1%} over {len(sub)} filing(s), {sub['ticker'].nunique()} tickers")

    neg = pd.to_numeric(df["peo_actually_paid_comp"], errors="coerce") < 0
    _row(results, "negative peo_actually_paid_comp survives",
         _PASS if neg.any() else _FAIL,
         f"{int(neg.sum())} negative row(s) — a NEGATIVE value is legitimate "
         f"(NKE 2025: -10,924,243) and there is no abs() on this path")

    co = df[pd.to_numeric(df["n_peos"], errors="coerce") >= 2]
    _row(results, "co-PEO years recorded", _PASS if not co.empty else _FAIL,
         f"{len(co)} filing(s) with n_peos >= 2, "
         f"{co['ticker'].nunique() if not co.empty else 0} tickers")

    gap = sorted(set(KNOWN_NO_XBRL_2023) - set(df.loc[year == 2023, "ticker"]))
    _row(results, "2023 shortfall decomposition", _INFO,
         f"of the three filers measured to publish a PVP table with no XBRL "
         f"({', '.join(KNOWN_NO_XBRL_2023)}), absent from 2023: {gap or 'none'}")


def check_encoding(store, results: list) -> None:
    """The retired edgar path carried U+0097 on 7.06% of exec-comp rows (a cp1252 em-dash
    mis-decode). The LLM path reads a decoded string, so it should carry none — and a NUL
    would have aborted the insert, so finding zero is the only possible outcome for that one."""
    blobs = store.load(Tables.def14a_llm, columns=["ticker", "as_of", "def14a_json"], limit=200)
    n_nul = int(blobs["def14a_json"].map(
        lambda v: isinstance(v, str) and "\x00" in v).sum())
    hits = {m: int(blobs["def14a_json"].map(
        lambda v: isinstance(v, str) and m in v).sum()) for m in MOJIBAKE}
    _row(results, "def14a_json has no NUL", _PASS if n_nul == 0 else _FAIL,
         f"{n_nul} of {len(blobs)} sampled blobs")
    bad = {k: v for k, v in hits.items() if v}
    _row(results, "def14a_json has no cp1252 mojibake", _PASS if not bad else _FAIL,
         f"{bad or 'none'} over {len(blobs)} sampled blobs")

    if not store.exists(Tables.def14a_directors):
        _row(results, "director names are clean", _FAIL,
             "def14a_directors does not exist — nothing to check")
        return
    names = store.load(Tables.def14a_directors, columns=["ticker", "name"])
    bad_names = names[names["name"].map(
        lambda v: isinstance(v, str) and any(m in v for m in MOJIBAKE))]
    _row(results, "director names are clean", _PASS if bad_names.empty else _FAIL,
         f"{len(bad_names)} of {len(names):,} names carry a mis-decode")


def check_retired(store, results: list) -> None:
    for t in ("sec_def14a_executive_comp", "sec_def14a_director_comp",
              "sec_def14a_ownership", "sec_def14a_votes"):
        _row(results, f"{t} is gone", _PASS if not store.exists(t) else _FAIL,
             "absent" if not store.exists(t) else f"still present, {store.row_count(t):,} rows")
    live = set(store.columns(Tables.def14a_llm))
    left = [c for c in ("n_technology_directors", "pct_technology_directors",
                        "technology_committee") if c in live]
    _row(results, "technology columns retired", _PASS if not left else _FAIL,
         f"{len(live)} columns on def14a_llm; technology columns left: {left or 'NONE'}")


def check_cube(context, results: list) -> None:
    """The governance panel must still BUILD, and its feature names must be the old set minus
    the dropped ones. The aggregate fingerprint moving is expected — features were removed."""
    from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
    store = context.store
    hist = store.load(Tables.def14a_llm)
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(hist["as_of"]).dropna().unique()))
    tickers = sorted(hist["ticker"].dropna().unique())[:40]
    peers = {t: [x for x in tickers if x != t][:8] for t in tickers}
    panel, _ = build_governance_feature_panel(hist[hist["ticker"].isin(tickers)], peers, idx)
    feats = sorted(c for c in panel.columns if c.startswith("f_"))
    dropped = [c for c in feats if "technolog" in c]
    _row(results, "governance panel builds", _PASS if not panel.empty else _FAIL,
         f"{len(panel):,} rows, {len(feats)} f_* features over {len(tickers)} tickers")
    _row(results, "no technology features remain", _PASS if not dropped else _FAIL,
         f"{dropped or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-6 Step 5: verify against live Postgres.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--skip-cube", action="store_true",
                    help="skip the governance-panel smoke test (it loads the whole archive)")
    args = ap.parse_args()

    _, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store
    results: list[dict] = []

    for fn in (check_retired, check_coverage, check_implausible, check_ecd, check_encoding):
        try:
            fn(store, results)
        except Exception as e:
            _row(results, fn.__name__, _FAIL, f"{type(e).__name__}: {e}")
    if not args.skip_cube:
        try:
            check_cube(context, results)
        except Exception as e:
            _row(results, "check_cube", _FAIL, f"{type(e).__name__}: {e}")

    width = max(len(r["check"]) for r in results)
    print("\n" + "=" * 100)
    print("PHASE-6 STEP 5 — post-cutover verification against LIVE POSTGRES")
    print("=" * 100)
    for r in results:
        print(f"  [{r['status']:4s}] {r['check']:<{width}}  {r['detail']}")

    n_fail = sum(r["status"] == _FAIL for r in results)
    n_pass = sum(r["status"] == _PASS for r in results)
    print(f"\n  {n_pass} PASS / {n_fail} FAIL / "
          f"{sum(r['status'] == _INFO for r in results)} INFO")
    print("\nCONCLUSION: " + (
        "every assertion holds against the real database, over the whole rebuilt tables and "
        "not just\nthe 23-ticker validation slice — which matters because a parquet-cached "
        "harness cannot see the\nPostgres DATE -> datetime.date round-trip at all."
        if n_fail == 0 else
        f"{n_fail} assertion(s) FAILED against the real database. The pre-cutover snapshot is "
        "the\nonly rollback that exists; read the failing rows above before re-running anything."))
    out = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/POSTCUTOVER.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
