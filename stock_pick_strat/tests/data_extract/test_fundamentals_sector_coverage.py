"""
Sector / edge-case coverage of fundamentals_history extraction. Each fix below was
proven data-vs-extraction against real cached companyfacts, then unit-tested here
with synthetic known-truth inputs.

Fixes covered:
  * base is the UNION of core period-ends (revenue + income + cash flow + assets +
    equity + liabilities), so a revenue-tag gap (banks ~2013, utilities ~2014) no
    longer truncates the whole ticker (NEE was cut off at 2013).
  * CHARGE_FLOWS (impairment, restructuring, acquisitions, buybacks, …) are 0-filled
    within the reporting span, so their TTM is a usable feature (0 in normal years)
    instead of NaN (they are reported only in event periods / as YTD cumulatives).
  * coalesced tag variants: longTermDebt += LongTermDebtAndCapitalLeaseObligations
    (HD, MU pre-2020); changeInReceivables/Payables += generic / combined variants.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    build_ticker_history, CHARGE_FLOWS, EXTRA_FLOW_TAGS, STOCK_TAGS)


def _q(end: str, start: str, val: float) -> dict:
    return {"end": end, "start": start,
            "filed": (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat(),
            "form": "10-Q", "fp": "Q1", "val": val}


def _year(val: float, years: list[int]) -> list[dict]:
    rows = []
    for y in years:
        for s, e in [("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")]:
            rows.append(_q(f"{y}-{e}", f"{y}-{s}", val))
    return rows


def _inst(val: float, years: list[int]) -> list[dict]:
    return [{"end": f"{y}-12-31", "start": None, "filed": f"{y + 1}-02-10",
             "form": "10-K", "fp": "FY", "val": val} for y in years]


def _build(gaap: dict, ticker: str = "T") -> pd.DataFrame:
    return build_ticker_history(ticker, {"facts": {"us-gaap": gaap, "dei": {}}})


def test_revenue_gap_does_not_truncate_history():
    """Revenue tag stops after 2017 but the balance sheet continues to 2019 — the
    ticker's history must extend to 2019 (base = union of ends), not stop at 2017."""
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2016, 2017])}},          # revenue GAP after 2017
        "Assets": {"units": {"USD": _inst(5000.0, [2016, 2017, 2018, 2019])}},
        "StockholdersEquity": {"units": {"USD": _inst(2000.0, [2016, 2017, 2018, 2019])}},
    }
    fe = _build(gaap)
    assert fe["fiscal_end"].max() >= "2019-12-31", "history truncated at the revenue gap"
    ye19 = fe[fe["fiscal_end"] == "2019-12-31"].iloc[0]
    assert ye19["totalAssets"] == pytest.approx(5000.0)      # balance sheet survives the gap
    assert pd.isna(ye19["totalRevenue"])                     # revenue-derived stays NaN there

    print("\n=== SANITY CHECK: revenue gap does not truncate ===")
    print(f"  revenue tag ends 2017 but history reaches {fe['fiscal_end'].max()} "
          f"(totalAssets@2019={ye19['totalAssets']:.0f}, totalRevenue@2019=NaN). Validated.")


def test_charge_flow_zero_filled():
    """A sporadic impairment (one quarter) makes the TTM impairment column 0 in
    normal quarters (usable feature) and >=the charge in the trailing window —
    never NaN across the reporting span (after TTM warmup)."""
    assert "impairment" in CHARGE_FLOWS
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        # single impairment event in 2019-Q4
        "GoodwillImpairmentLoss": {"units": {"USD": [_q("2019-12-31", "2019-10-01", 500.0)]}},
    }
    fe = _build(gaap).sort_values("fiscal_end")
    imp = fe.set_index("fiscal_end")["impairment"]
    assert imp.loc["2019-09-30"] == pytest.approx(0.0)       # normal quarter -> 0, not NaN
    assert imp.loc["2019-12-31"] == pytest.approx(500.0)     # trailing window includes the charge
    warm = fe.iloc[3:]                                        # after 4-quarter TTM warmup
    assert warm["impairment"].notna().all(), "charge flow should never be NaN post-warmup"

    print("\n=== SANITY CHECK: charge-flow 0-fill (impairment) ===")
    print(f"  TTM impairment: normal-quarter=0.0, charge-quarter={imp.loc['2019-12-31']:.0f}, "
          f"no NaN after warmup ({warm['impairment'].notna().mean():.0%} populated). Validated.")


def test_longtermdebt_capital_lease_coalesced():
    """A filer that tags LTD only under LongTermDebtAndCapitalLeaseObligations
    (HD / MU pre-2020) still populates longTermDebt."""
    assert "LongTermDebtAndCapitalLeaseObligations" in STOCK_TAGS["longTermDebt"]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019])}},
        "LongTermDebtAndCapitalLeaseObligations": {"units": {"USD": _inst(3000.0, [2018, 2019])}},
    }
    fe = _build(gaap)
    ye = fe[fe["fiscal_end"] == "2019-12-31"].iloc[0]
    assert ye["longTermDebt"] == pytest.approx(3000.0)

    print("\n=== SANITY CHECK: longTermDebt capital-lease coalesce ===")
    print(f"  longTermDebt@2019={ye['longTermDebt']:.0f} from LongTermDebtAndCapitalLeaseObligations. Validated.")


def test_change_in_receivables_variant_coalesced():
    """changeInReceivables recovers from the generic IncreaseDecreaseInReceivables tag."""
    assert "IncreaseDecreaseInReceivables" in EXTRA_FLOW_TAGS["changeInReceivables"]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "IncreaseDecreaseInReceivables": {"units": {"USD": _year(50.0, [2018, 2019, 2020])}},
    }
    fe = _build(gaap).sort_values("fiscal_end")
    ye = fe[fe["fiscal_end"] == "2020-12-31"].iloc[0]
    assert ye["changeInReceivables"] == pytest.approx(200.0)   # TTM = 4 x 50

    print("\n=== SANITY CHECK: changeInReceivables variant coalesce ===")
    print(f"  changeInReceivables(TTM)@2020={ye['changeInReceivables']:.0f} (4x50) from generic tag. Validated.")


def test_nee_not_truncated_real():
    """Real cached companyfacts: NEE (utility) used `Revenues` to 2013 then
    `RegulatedAndUnregulatedOperatingRevenue` — history must now reach recent years
    and its regulatoryAssets (disclosed from 2015) must populate."""
    try:
        from src.context import get_config_context
        from src.data_extract.utils.common.sec_utils import load_cik_mapping
        _, ctx = get_config_context("./configs", use_cache=True, save=False)
        cik = load_cik_mapping(ctx)
    except Exception as e:                                      # noqa: BLE001
        pytest.skip(f"DB/context not reachable: {e}")
    row = cik[cik["ticker"] == "NEE"]
    if row.empty:
        pytest.skip("NEE not in cik_mapping")
    p = ctx.paths["SEC_BULK_CACHE_DIR"] / f"companyfacts_CIK{row.iloc[0]['cik']}.json"
    if not p.exists():
        pytest.skip("NEE companyfacts not cached")

    fe = build_ticker_history("NEE", json.loads(p.read_text(encoding="utf-8")),
                              row.iloc[0].get("sector"), row.iloc[0].get("industry_group"))
    assert fe["fiscal_end"].max() >= "2024-01-01", "NEE history still truncated"
    reg_recent = fe[fe["fiscal_end"] >= "2016-12-31"]["regulatoryAssets"]
    assert reg_recent.notna().mean() > 0.5, "regulatoryAssets should populate post-2016"

    print("\n=== SANITY CHECK: NEE real-data anti-truncation ===")
    print(f"  NEE history reaches {fe['fiscal_end'].max()} (was 2013); "
          f"regulatoryAssets populated {reg_recent.notna().mean():.0%} of post-2016 rows. Validated.")
