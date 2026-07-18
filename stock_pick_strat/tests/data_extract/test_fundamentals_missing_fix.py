"""
Recovery of previously-missing fundamentals_history columns (data-vs-extraction
audit of grossMargins / operatingMargins / ebitda / totalLiabilities …).

Two proven extraction fixes in fetch_fundamentals.py:
  1. costOfRevenue now coalesces the OLD `CostOfGoodsSold` / `CostOfServices`
     tags, so filers that tagged cost only under the pre-ASC-606 element (e.g.
     LLY before ~2018) recover gross profit -> grossMargins -> (cascades to)
     operatingMargins + ebitda for their early history.
  2. totalLiabilities is DERIVED via the accounting identity Assets - Equity
     when a filer never tags `Liabilities` as a single element (e.g. LLY, AMD).

test_missing_cogs_and_liabilities_recovered — synthetic known-truth for the math
    (incl. a guard that a REAL `Liabilities` value is never overwritten).
test_lly_missing_fundamentals_recovered_real — real cached companyfacts coverage
    check: LLY's previously-sparse columns now populate and are economically sane.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    _extract_concept, build_ticker_history, FLOW_TAGS)


def _q_obs(end: str, start: str, val: float, form: str = "10-Q") -> dict:
    return {"end": end, "start": start,
            "filed": (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat(),
            "form": form, "fp": "Q1", "val": val}


def _discrete_year(val: float, years: list[int]) -> list[dict]:
    """Discrete ~quarterly duration facts across `years`."""
    rows = []
    for y in years:
        for s, e in [("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")]:
            rows.append(_q_obs(f"{y}-{e}", f"{y}-{s}", val))
    return rows


def _instant(val: float, years: list[int]) -> list[dict]:
    return [{"end": f"{y}-12-31", "start": None, "filed": f"{y + 1}-02-10",
             "form": "10-K", "fp": "FY", "val": val} for y in years]


def test_missing_cogs_and_liabilities_recovered():
    years = [2018, 2019, 2020]
    gaap = {
        "Revenues": {"units": {"USD": _discrete_year(1000.0, years)}},
        # ONLY the old cost tag — no GrossProfit, no CostOfGoodsAndServicesSold
        "CostOfGoodsSold": {"units": {"USD": _discrete_year(400.0, years)}},
        # balance sheet: Assets + Equity present, `Liabilities` ABSENT
        "Assets": {"units": {"USD": _instant(1000.0, years)}},
        "StockholdersEquity": {"units": {"USD": _instant(600.0, years)}},
    }

    # (1) the old cost tag is among the coalesced costOfRevenue candidates
    assert "CostOfGoodsSold" in FLOW_TAGS["costOfRevenue"]
    assert not _extract_concept(gaap, FLOW_TAGS["costOfRevenue"]).empty, \
        "CostOfGoodsSold not picked up by costOfRevenue coalescing"

    fe = build_ticker_history("COGSFIX", {"facts": {"us-gaap": gaap, "dei": {}}})
    ye = fe[fe["fiscal_end"] == "2020-12-31"].iloc[0]

    # (1) gross margin recovered: (rev - cogs)/rev = (4000 - 1600)/4000 = 0.6;
    #     operating margin cascades (OI derived = gross profit - SG&A - R&D)
    assert ye["grossMargins"] == pytest.approx(0.6, abs=1e-6)
    assert ye["operatingMargins"] == pytest.approx(0.6, abs=1e-6)
    assert ye["ebitda"] == pytest.approx(2400.0, abs=1e-3)
    # (2) total liabilities derived = Assets - Equity = 1000 - 600 = 400
    assert ye["totalAssets"] == pytest.approx(1000.0)
    assert ye["totalLiabilities"] == pytest.approx(400.0, abs=1e-6)

    # (2-guard) a REAL `Liabilities` value must NOT be overwritten by the derivation
    gaap2 = {**gaap, "Liabilities": {"units": {"USD": _instant(370.0, years)}}}
    fe2 = build_ticker_history("HASLIAB", {"facts": {"us-gaap": gaap2, "dei": {}}})
    ye2 = fe2[fe2["fiscal_end"] == "2020-12-31"].iloc[0]
    assert ye2["totalLiabilities"] == pytest.approx(370.0), \
        "reported Liabilities must win over the Assets-Equity derivation"

    print("\n=== SANITY CHECK: recovered COGS gross margin + derived liabilities ===")
    print(f"  grossMargins={ye['grossMargins']:.3f}, operatingMargins={ye['operatingMargins']:.3f} "
          f"(recovered via old CostOfGoodsSold tag; OI cascade)")
    print(f"  totalLiabilities={ye['totalLiabilities']:.0f} = Assets {ye['totalAssets']:.0f} - Equity 600 "
          f"(no `Liabilities` tag); reported-liab guard keeps {ye2['totalLiabilities']:.0f} not 400.")
    print("  -> both fixes correct; derivation never overwrites a reported value. Validated.")


def test_lly_missing_fundamentals_recovered_real():
    """Real cached companyfacts: LLY's grossMargins (was ~53% null) and
    totalLiabilities (was 100% null) now populate, are economically sane, and the
    derived liabilities satisfy Assets - Equity."""
    try:
        from src.context import get_config_context
        from src.data_extract.utils.common.sec_utils import load_cik_mapping
        _, ctx = get_config_context("./configs", use_cache=True, save=False)
        cik = load_cik_mapping(ctx)
    except Exception as e:                                          # noqa: BLE001
        pytest.skip(f"DB/context not reachable: {e}")

    row = cik[cik["ticker"] == "LLY"]
    if row.empty:
        pytest.skip("LLY not in cik_mapping")
    p = ctx.paths["SEC_BULK_CACHE_DIR"] / f"companyfacts_CIK{row.iloc[0]['cik']}.json"
    if not p.exists():
        pytest.skip("LLY companyfacts not cached")

    facts = json.loads(p.read_text(encoding="utf-8"))
    fe = build_ticker_history("LLY", facts, row.iloc[0].get("sector"),
                              row.iloc[0].get("industry_group"))
    gm, liab = fe["grossMargins"], fe["totalLiabilities"]

    # coverage: previously 53% / 100% null -> now only TTM-warmup sparsity
    assert gm.isna().mean() < 0.15, f"grossMargins still sparse ({gm.isna().mean():.0%})"
    assert liab.isna().mean() < 0.15, f"totalLiabilities still sparse ({liab.isna().mean():.0%})"
    # economic sanity: LLY (pharma) gross margin ~0.70-0.85
    assert 0.65 <= gm.median() <= 0.90, f"LLY gross margin unrealistic ({gm.median():.2f})"
    # derived-liabilities identity holds (LLY never tags `Liabilities`)
    m = fe.dropna(subset=["totalAssets", "stockholdersEquity", "totalLiabilities"])
    ident_ok = ((m["totalLiabilities"] - (m["totalAssets"] - m["stockholdersEquity"])).abs() < 1.0)
    assert ident_ok.mean() > 0.8, "derived liabilities != Assets - Equity"

    print("\n=== SANITY CHECK: LLY real-data recovery ===")
    print(f"  grossMargins null {gm.isna().mean():.0%} (was 53%), median {gm.median():.2f} (pharma-sane)")
    print(f"  totalLiabilities null {liab.isna().mean():.0%} (was 100%), "
          f"Assets-Equity identity holds on {ident_ok.mean():.0%} of rows. Validated.")
