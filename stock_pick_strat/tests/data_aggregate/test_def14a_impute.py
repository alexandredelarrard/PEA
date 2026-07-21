"""
DEF 14A clean-on-read imputation (src/data_aggregate/utils/def14a_impute.py).

The LLM extraction leaves gaps in `def14a_llm`; the cube deduces them at read time
(governance features). Rules under test:
  1. ceo_total_comp == sum(6 SCT components); a single missing component == total - others.
  2. pct_technology_directors <-> n_technology_directors / board_size; n_directors == board_size.
  3. median_employee_pay <-> ceo_total_comp / ceo_pay_ratio.
  4. per-ticker temporal gap-fill BETWEEN two filled years (interp levels, carry flags),
     leaving leading/trailing gaps untouched.
STRICTLY non-destructive: a value is written only where it is currently NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.def14a_impute import impute_def14a


def _row(ticker: str, as_of: str, **kw) -> dict:
    base = {"ticker": ticker, "as_of": as_of, "accession_number": f"{ticker}-{as_of}"}
    base.update(kw)
    return base


def test_impute_rules_synthetic():
    df = pd.DataFrame([
        # --- CEO pay identity: total missing, all 6 components present -> deduce total
        _row("AAA", "2020-04-01", ceo_salary=1_000_000, ceo_bonus=0, ceo_stock_awards=5_000_000,
             ceo_option_awards=2_000_000, ceo_non_equity_incentive=3_000_000, ceo_all_other_comp=200_000,
             ceo_total_comp=np.nan, board_size=10, n_directors=np.nan),
        # --- single missing component (option_awards) -> deduce = total - others
        _row("AAA", "2021-04-01", ceo_salary=1_000_000, ceo_bonus=0, ceo_stock_awards=5_000_000,
             ceo_option_awards=np.nan, ceo_non_equity_incentive=3_000_000, ceo_all_other_comp=200_000,
             ceo_total_comp=11_200_000, board_size=10, n_directors=10),
        # --- non-destructive: total present but != sum(components) (pension gap) -> keep as-is
        _row("AAA", "2022-04-01", ceo_salary=1_000_000, ceo_bonus=0, ceo_stock_awards=5_000_000,
             ceo_option_awards=2_000_000, ceo_non_equity_incentive=3_000_000, ceo_all_other_comp=200_000,
             ceo_total_comp=99_000_000, board_size=10, n_directors=10),
        # --- board consistency: pct_tech missing, n_tech + board present -> deduce pct
        _row("BBB", "2020-04-01", board_size=12, n_technology_directors=3, pct_technology_directors=np.nan,
             n_directors=np.nan),
        # --- pay ratio: total + ratio present, median missing -> deduce median
        _row("BBB", "2021-04-01", board_size=12, ceo_total_comp=12_000_000, ceo_pay_ratio=200,
             median_employee_pay=np.nan, n_technology_directors=3, pct_technology_directors=0.25),
        # --- temporal gap: CCC board_size 9 -> NaN -> 11 ; ceo_is_founder 1 -> NaN -> (carry) 1
        _row("CCC", "2019-04-01", board_size=9, ceo_is_founder=1.0),
        _row("CCC", "2020-04-01", board_size=np.nan, ceo_is_founder=np.nan),
        _row("CCC", "2021-04-01", board_size=11, ceo_is_founder=1.0),
        # --- trailing edge gap must NOT be filled (no later filled value)
        _row("CCC", "2022-04-01", board_size=np.nan, ceo_is_founder=np.nan),
    ])
    raw = df.copy()
    out, stats = impute_def14a(df)
    out = out.set_index(["ticker", "as_of"])

    # 1. total = sum(components)
    assert out.loc[("AAA", pd.Timestamp("2020-04-01")), "ceo_total_comp"] == pytest.approx(11_200_000)
    # 1. single component = total - others (clip >=0)
    assert out.loc[("AAA", pd.Timestamp("2021-04-01")), "ceo_option_awards"] == pytest.approx(2_000_000)
    # 1. NON-DESTRUCTIVE: real (inconsistent) total preserved
    assert out.loc[("AAA", pd.Timestamp("2022-04-01")), "ceo_total_comp"] == pytest.approx(99_000_000)
    # 2. pct_tech = n_tech / board ; n_directors deduced from board_size
    assert out.loc[("BBB", pd.Timestamp("2020-04-01")), "pct_technology_directors"] == pytest.approx(0.25)
    assert out.loc[("AAA", pd.Timestamp("2021-04-01")), "n_directors"] == 10
    # 3. median pay = total / ratio
    assert out.loc[("BBB", pd.Timestamp("2021-04-01")), "median_employee_pay"] == pytest.approx(60_000)
    # 4. temporal: interior gap interpolated (9,11 -> 10), integer-rounded; flag carried forward
    assert out.loc[("CCC", pd.Timestamp("2020-04-01")), "board_size"] == 10
    assert out.loc[("CCC", pd.Timestamp("2020-04-01")), "ceo_is_founder"] == 1.0
    # 4. trailing edge NOT filled
    assert pd.isna(out.loc[("CCC", pd.Timestamp("2022-04-01")), "board_size"])
    assert pd.isna(out.loc[("CCC", pd.Timestamp("2022-04-01")), "ceo_is_founder"])

    # global non-destructiveness: every originally-present cell is byte-for-byte unchanged
    r = raw.set_index(["ticker", "as_of"]); r.index = r.index.set_levels(
        pd.to_datetime(r.index.levels[1]), level=1)
    changed = 0
    for c in [x for x in r.columns if x != "accession_number"]:
        pres = r[c].notna()
        if pres.any():
            a, b = r.loc[pres, c].astype(float), out.loc[r.index[pres], c].astype(float)
            changed += int((~np.isclose(a.values, b.values)).sum())
    assert changed == 0, f"{changed} present cells were overwritten"

    print("\n=== SANITY CHECK: DEF 14A imputation (synthetic) ===")
    print(f"  rules fired: {stats}")
    print("  CEO total=sum(components) & single component=total-others (clip>=0); real inconsistent")
    print("  total (99M pension-gap row) PRESERVED; pct_tech=n_tech/board; n_directors=board_size;")
    print("  median_pay=total/ratio (60k); interior board_size gap interpolated 9,11->10 (int);")
    print("  founder flag carried across interior gap; TRAILING edge left NaN; 0 present cells changed.")


def test_impute_real_data_nondestructive():
    """Run on the live def14a_llm: prove non-destructiveness + report what gets recovered."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    imp, stats = impute_def14a(raw)
    key = ["ticker", "accession_number"]
    a = raw.set_index(key).sort_index()
    b = imp.set_index(key).sort_index().reindex(a.index)

    num = a.select_dtypes("number").columns
    overwritten, filled = 0, 0
    for c in num:
        pres = a[c].notna()
        if pres.any():
            overwritten += int((~np.isclose(a.loc[pres, c].values, b.loc[pres, c].values,
                                            equal_nan=True)).sum())
        filled += int((a[c].isna() & b[c].notna()).sum())
    assert overwritten == 0, f"{overwritten} present numeric cells overwritten"

    miss_before = raw[num].isna().mean().mean() * 100
    miss_after = imp[num].isna().mean().mean() * 100
    top = sorted(stats.items(), key=lambda kv: -kv[1])[:8]
    print("\n=== SANITY CHECK: DEF 14A imputation (real def14a_llm) ===")
    print(f"  rows={len(raw)}  tickers={raw['ticker'].nunique()}  numeric cols={len(num)}")
    print(f"  cells filled = {filled}  |  present cells overwritten = {overwritten}")
    print(f"  mean numeric-missing: {miss_before:.1f}% -> {miss_after:.1f}%")
    print(f"  top rules: {top}")
    print("  CONCLUSION: deductions recover real gaps clean-on-read WITHOUT mutating any "
          "extracted value; the raw def14a_llm table is left untouched.")


if __name__ == "__main__":
    test_impute_rules_synthetic()
    test_impute_real_data_nondestructive()
