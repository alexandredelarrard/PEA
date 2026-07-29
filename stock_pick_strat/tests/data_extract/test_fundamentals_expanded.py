"""
Expanded SEC fundamentals extraction + supporting infra:

test_build_ticker_history_emits_expanded_columns — universal + sector raw tags
    (NII, premiums, inventory, buybacks, diluted shares) come through as TTM /
    level columns with the right values, plus the stamped sector/industry_group.
test_ensure_columns_evolves_schema — DataStore adds new columns instead of
    silently dropping them (needs Postgres).
test_load_cik_mapping_attaches_sector — the CIK mapping carries GICS sector /
    industry_group joined from sp500_tickers (needs Postgres, seeded universe).
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals import build_ticker_history


def _q(end, start, val, form="10-Q"):
    filed = (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat()
    return {"end": end, "start": start, "filed": filed, "form": form, "fp": "Q1", "val": val}


def _ladder(val):
    """Eight discrete quarterly facts (2022-2023) each worth `val`."""
    rows = []
    for y in (2022, 2023):
        for s, e in [("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")]:
            rows.append(_q(f"{y}-{e}", f"{y}-{s}", val))
    return rows


def test_build_ticker_history_emits_expanded_columns():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": _ladder(1000.0)}},
        "NetIncomeLoss": {"units": {"USD": _ladder(100.0)}},
        "InterestIncomeExpenseNet": {"units": {"USD": _ladder(50.0)}},      # bank NII
        "PremiumsEarnedNet": {"units": {"USD": _ladder(200.0)}},            # insurance
        "PaymentsForRepurchaseOfCommonStock": {"units": {"USD": _ladder(30.0)}},  # buybacks
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": _ladder(1_000_000.0)}},
        "InventoryNet": {"units": {"USD": [
            {"end": "2023-12-31", "start": None, "filed": "2024-02-09",
             "form": "10-K", "fp": "FY", "val": 300.0}]}},                  # instant level
    }}}

    out = build_ticker_history("XCO", facts, sector="Financials", industry_group="Banks")
    assert not out.empty
    last = out.iloc[-1]

    # TTM sums (4 most recent quarters): flows
    assert last["netInterestIncome"] == pytest.approx(200.0)   # 4 x 50
    assert last["premiumsEarned"] == pytest.approx(800.0)      # 4 x 200
    assert last["buybacks"] == pytest.approx(120.0)            # 4 x 30
    # instant level + as-of diluted shares
    assert last["inventory"] == pytest.approx(300.0)
    assert last["dilutedShares"] == pytest.approx(1_000_000.0)
    # sector stamped for downstream KPI gating / neutralization
    assert last["sector"] == "Financials"
    assert last["industry_group"] == "Banks"

    print("\n=== SANITY CHECK: expanded extraction ===")
    print(f"  NII(TTM)={last['netInterestIncome']:.0f} premiums(TTM)={last['premiumsEarned']:.0f} "
          f"buybacks(TTM)={last['buybacks']:.0f} inventory={last['inventory']:.0f} "
          f"diluted={last['dilutedShares']:.0f}")
    print(f"  sector={last['sector']} industry_group={last['industry_group']}. Validated.")


@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs Postgres")
def test_ensure_columns_evolves_schema():
    from sqlalchemy import text
    from src.utils.db import get_engine
    from src.data_store.store import DataStore, ensure_columns

    eng = get_engine()
    with eng.begin() as c:
        c.execute(text('DROP TABLE IF EXISTS _evolve_test'))
        c.execute(text('CREATE TABLE _evolve_test ("ticker" text primary key, "a" double precision)'))
    try:
        st = DataStore(eng)
        df = pd.DataFrame({"ticker": ["X"], "a": [1.0], "loss_ratio": [0.7], "sector": ["Fin"]})
        added = ensure_columns(eng, "_evolve_test", df)
        assert set(added) == {"loss_ratio", "sector"}
        st.save("_evolve_test", df, pk=["ticker"])
        back = st.load("_evolve_test")
        assert {"loss_ratio", "sector"} <= set(back.columns)
        assert back.iloc[0]["loss_ratio"] == pytest.approx(0.7)
        print("\n=== SANITY CHECK: schema evolution ===")
        print(f"  added columns {added}; new values persisted (loss_ratio=0.7). Validated.")
    finally:
        with eng.begin() as c:
            c.execute(text('DROP TABLE IF EXISTS _evolve_test'))


@pytest.mark.skipif(not os.getenv("DATABASE_URL"),
                    reason="DATABASE_URL not set — needs Postgres")
def test_load_cik_mapping_attaches_sector():
    from src.context import get_config_context
    from src.data_extract.utils.common.sec_utils import load_cik_mapping

    try:
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        if ctx.store.load("sp500_tickers").empty or ctx.store.load("cik_mapping").empty:
            pytest.skip("sp500_tickers / cik_mapping not seeded")
    except Exception as e:                                     # noqa: BLE001
        pytest.skip(f"DB not reachable: {e}")

    m = load_cik_mapping(ctx)
    assert "sector" in m.columns and "industry_group" in m.columns
    aapl = m[m["ticker"] == "AAPL"]
    assert not aapl.empty and pd.notna(aapl.iloc[0]["sector"])
    print("\n=== SANITY CHECK: cik_mapping sector enrichment ===")
    print(f"  columns include sector/industry_group; AAPL sector="
          f"{aapl.iloc[0]['sector']!r} industry_group={aapl.iloc[0]['industry_group']!r}. Validated.")


# --------------------------------------------------------------------------- #
# Regression: coalesce candidate tags (ASC-606 revenue split + NCI net-income   #
# split) — the CVX / AVGO / CAT / JNJ / UNH "missing margins & ROE" bug.        #
# --------------------------------------------------------------------------- #
def _q_obs(end, start, val, form="10-Q"):
    return {"end": end, "start": start,
            "filed": (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat(),
            "form": form, "fp": "Q1", "val": val}


def _discrete_year(tag_val, years):
    """discrete ~quarterly observations for a duration concept across `years`."""
    rows = []
    for y in years:
        for s, e in [("01-01", "03-31"), ("04-01", "06-30"),
                     ("07-01", "09-30"), ("10-01", "12-31")]:
            rows.append(_q_obs(f"{y}-{e}", f"{y}-{s}", tag_val))
    return rows


def _instant(tag_val, years):
    return [{"end": f"{y}-12-31", "start": None,
             "filed": (pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=40)).date().isoformat(),
             "form": "10-K", "fp": "FY", "val": tag_val} for y in years]


def test_extract_concept_coalesces_split_tags():
    """A filer that reports revenue under the OLD tag (Revenues) for 2016-2017
    then the ASC-606 tag from 2018, and net income under ProfitLoss (full) but
    NetIncomeLoss only recently, must yield a CONTINUOUS history with margins &
    ROE populated across the whole span (previously truncated to the first tag)."""
    from src.data_extract.utils.fundamentals.fetch_fundamentals import (
        _extract_concept, build_ticker_history)

    old_years, new_years = [2016, 2017], [2018, 2019, 2020]
    gaap = {
        # revenue split across two tags by era (ASC-606 adoption in 2018)
        "Revenues": {"units": {"USD": _discrete_year(1000, old_years)}},
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            {"units": {"USD": _discrete_year(1200, new_years)}},
        # net income: NetIncomeLoss (first candidate) only recent, ProfitLoss full
        "NetIncomeLoss": {"units": {"USD": _discrete_year(150, new_years)}},
        "ProfitLoss": {"units": {"USD": _discrete_year(100, old_years + new_years)}},
        # equity split: StockholdersEquity only recent, IncludingNCI full
        "StockholdersEquity": {"units": {"USD": _instant(2000, new_years)}},
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest":
            {"units": {"USD": _instant(1800, old_years + new_years)}},
    }

    # (1) _extract_concept coalesces revenue across both era tags
    rev = _extract_concept(gaap, ["RevenueFromContractWithCustomerExcludingAssessedTax",
                                  "Revenues", "SalesRevenueNet"])
    span = rev.dropna(subset=["end"])["end"]
    assert span.min().year == 2016 and span.max().year == 2020, "revenue history truncated"

    facts = {"facts": {"us-gaap": gaap, "dei": {}}}
    fe = build_ticker_history("SPLIT", facts)

    # (2) margins & ROE populated across the FULL span (not just the recent tag)
    prof = fe[fe["profitMargins"].notna()]
    roe = fe[fe["returnOnEquity"].notna()]
    assert prof["fiscal_end"].min() <= "2017-12-31", "profitMargins truncated to recent tag"
    assert roe["fiscal_end"].min() <= "2017-12-31", "ROE truncated to recent tag"
    # net income prefers NetIncomeLoss where present (150), else ProfitLoss (100)
    last = fe.sort_values("fiscal_end").iloc[-1]
    assert last["netIncome"] == pytest.approx(150 * 4)          # TTM of the recent quarters

    print("\n=== SANITY CHECK: candidate-tag coalescing (CVX/AVGO regression) ===")
    print(f"  revenue span {span.min().date()}..{span.max().date()} (both era tags merged)")
    print(f"  profitMargins from {prof['fiscal_end'].min()}, ROE from {roe['fiscal_end'].min()} "
          f"(full history, not just the first tag). Validated.")
