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
            {"end": "2023-12-31", "start": None, "filed": "2024-02-10",
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
