"""
test_impute_exec_comp.py  (tests/data_aggregate/test_impute_exec_comp.py)
------------------------------------------------------------------------
`impute_exec_comp` (D22) — the NEO-total repair on `def14a_executive_comp`.

The three properties that make it safe to feed a pay-slice denominator:

  * it FILLS a NULL total from the seven Item 402(c) components;
  * it never turns an all-NULL row into a $0 package (`min_count=1`), which is the one way a
    row sum can fabricate a fact;
  * it never overwrites a filer-stated total — the same non-destructive invariant
    `test_impute_real_data_nondestructive` pins for the parent grain.

Plus the number the exact top-5 CPS actually depends on: how many filings carry five NEO
totals before and after the repair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.def14a_impute import EXEC_COMPONENTS, impute_exec_comp


def _row(ticker: str, name: str, fy: int, total=np.nan, **comp) -> dict:
    r = {"ticker": ticker, "accession_number": f"{ticker}-{fy}", "as_of": f"{fy}-04-01",
         "name": name, "fiscal_year": fy, "total": total}
    r.update({c: comp.get(c, np.nan) for c in EXEC_COMPONENTS})
    return r


def test_exec_comp_rules_synthetic():
    """Fill from components, keep an all-NULL row NULL, never overwrite a stated total."""
    df = pd.DataFrame([
        # 1. every component present, total NULL -> filled with the sum (7 x 1000 = 7000)
        _row("AAA", "Alice", 2020, **{c: 1000.0 for c in EXEC_COMPONENTS}),
        # 2. a PARTIAL row: three components, total NULL -> filled with 600 (min_count=1)
        _row("AAA", "Bob", 2020, salary=100.0, bonus=200.0, stock_awards=300.0),
        # 3. every component NULL, total NULL -> STAYS NULL. Never 0.0.
        _row("AAA", "Carol", 2020),
        # 4. a stated total that DISAGREES with its components -> preserved, untouched
        _row("BBB", "Dave", 2021, total=999_999.0, salary=1.0, bonus=2.0),
    ])
    out, stats = impute_exec_comp(df)

    assert out.loc[0, "total"] == pytest.approx(7000.0)
    assert out.loc[1, "total"] == pytest.approx(600.0)
    assert pd.isna(out.loc[2, "total"]), "an all-NULL row became a $0 package"
    assert out.loc[3, "total"] == pytest.approx(999_999.0), "a filer-stated total was overwritten"

    # provenance is set on exactly the two rows this function wrote, and nowhere else
    assert out["total_imputed"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert stats["total = sum(components)"] == 2
    assert stats["total still NULL (no component at all)"] == 1

    print("\n=== SANITY CHECK: impute_exec_comp (synthetic) ===")
    print(f"  stats: {stats}")
    print("  7 components -> 7000; partial 3 components -> 600 (min_count=1 keeps a")
    print("  fully-NULL row NULL rather than $0); a stated 999,999 that disagrees with its")
    print("  own components is PRESERVED; total_imputed set on exactly the 2 written rows.")


def test_exec_comp_real_data_and_neo_bar():
    """Live table: non-destructive, and the filing counts the exact top-5 CPS depends on."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_executive_comp")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_executive_comp not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_executive_comp empty")

    out, stats = impute_exec_comp(raw)

    # --- non-destructive: not one stated total moved ---
    present = raw["total"].notna()
    overwritten = int((~np.isclose(raw.loc[present, "total"].to_numpy(),
                                   out.loc[present, "total"].to_numpy(),
                                   equal_nan=True)).sum())
    assert overwritten == 0, f"{overwritten} filer-stated totals overwritten"
    assert not (out.loc[present, "total_imputed"] > 0).any(), \
        "total_imputed stamped on a row whose total the filer stated"

    # --- the identity's credibility where it is checkable ---
    rec = pd.to_numeric(raw.get("reconciles"), errors="coerce")
    checkable = rec.notna()
    rec_rate = float(rec[checkable].mean()) if checkable.any() else float("nan")

    # --- the bar the exact top-5 CPS has to clear ---
    #
    # ⚠ THE GRAIN MATTERS AND THE PLAN'S NUMBERS ARE ON THE LOOSER ONE. A Summary
    # Compensation Table discloses THREE fiscal years per NEO (10,454 of 11,316 accessions
    # carry 3, 461 carry 2, 374 carry 1), so "5 distinct NEOs in this accession" can be
    # satisfied by five people who are not all present in the same YEAR. A pay slice divides
    # one CEO's package by the top-5 total OF THAT FISCAL YEAR, so the denominator's real
    # population is (ticker, accession, fiscal_year). Both are reported: the first
    # reproduces the plan, the second is what phase 4 must actually build on.
    acc = ["ticker", "accession_number"]
    fy = ["ticker", "accession_number", "fiscal_year"]

    def bar(df: pd.DataFrame, key: list[str], n: int, distinct: bool) -> int:
        d = df[df["total"].notna()]
        sized = d.groupby(key)["name"].nunique() if distinct else d.groupby(key).size()
        return int((sized >= n).sum())

    b5, a5 = bar(raw, acc, 5, True), bar(out, acc, 5, True)
    b3, a3 = bar(raw, acc, 3, True), bar(out, acc, 3, True)
    fb5, fa5 = bar(raw, fy, 5, True), bar(out, fy, 5, True)
    fb3, fa3 = bar(raw, fy, 3, True), bar(out, fy, 3, True)
    n_filings, n_fy = raw.groupby(acc).ngroups, raw.groupby(fy).ngroups

    assert a5 > b5, "the repair did not widen the 5-NEO population at all"
    assert fa5 > fb5, "the repair did not widen the 5-NEO population on the fiscal-year grain"

    print("\n=== SANITY CHECK: impute_exec_comp (real def14a_executive_comp) ===")
    print(f"  rows={len(raw)}  tickers={raw['ticker'].nunique()}  filings={n_filings}"
          f"  filing-years={n_fy}")
    print(f"  total stated by the filer : {stats['total stated by the filer']}")
    print(f"  total = sum(components)   : {stats['total = sum(components)']} filled")
    print(f"  still NULL (no component) : {stats['total still NULL (no component at all)']}")
    print(f"  `reconciles`==1 where both sides present: {rec_rate:.1%} "
          f"({int(checkable.sum())} rows checkable)")
    print(f"  present totals overwritten: {overwritten}  <- the non-destructive invariant")
    print(f"  per ACCESSION (the plan's grain), distinct NEOs:")
    print(f"     >=5 totals: {b5} -> {a5}  ({a5 - b5:+d}, {(a5 / b5 - 1) * 100:+.0f}%)"
          f"   [plan: 8,026 -> 10,695]")
    print(f"     >=3 totals: {b3} -> {a3}  ({a3 - b3:+d}) of {n_filings}"
          f"   [plan: 8,422 -> 11,232]")
    print(f"  per ACCESSION x FISCAL YEAR (what a pay slice actually divides by):")
    print(f"     >=5 totals: {fb5} -> {fa5}  ({fa5 - fb5:+d}, {(fa5 / fb5 - 1) * 100:+.0f}%)")
    print(f"     >=3 totals: {fb3} -> {fa3}  ({fa3 - fb3:+d}) of {n_fy}")
    print("  CONCLUSION: the exact top-5 pay slice becomes computable on a materially wider")
    print("  population on BOTH grains, every repaired denominator is flagged, and no stated")
    print("  total moved. (!) Phase 4 must key the denominator on the fiscal-year grain: an SCT")
    print("  shows 3 years per NEO, so the accession grain can assemble a 'top 5' out of")
    print("  people who never appeared in the same year.")
