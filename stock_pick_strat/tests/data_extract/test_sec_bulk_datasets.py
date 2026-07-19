"""Tests for the SEC bulk quarterly extractors (insider transactions + pension
facts from the Financial Statement Data Sets).

The parse/join/filter functions are PURE and tested on both hand-built inputs and
the REAL cached 2024q1 zips (skipped if not downloaded). The incremental-state
query is tested against a throwaway SQLite DB.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_store.store import DataStore
from src.data_extract.utils.prices import fetch_insider_transactions as ins
from src.data_extract.utils.fundamentals import fetch_financial_statements as fin

REPO = Path(__file__).resolve().parents[2]
INSIDER_ZIP = REPO / "data" / "sec_insider_transactions" / "2024q1_form345.zip"
FINSTMT_ZIP = REPO / "data" / "sec_financial_statements" / "2024q1.zip"


# --------------------------------------------------------------------------- #
# Insider transactions                                                          #
# --------------------------------------------------------------------------- #
def test_insider_quarters_are_deterministic_and_bounded():
    qs = ins._quarters(3, today=pd.Timestamp("2024-05-01"))
    assert qs == ["2021q1", "2021q2", "2021q3", "2021q4", "2022q1", "2022q2",
                  "2022q3", "2022q4", "2023q1", "2023q2", "2023q3", "2023q4",
                  "2024q1", "2024q2", "2024q3", "2024q4"]
    # never emits before the data set exists
    assert all(int(q[:4]) >= ins.SEC_INSIDER_FIRST_YEAR
               for q in ins._quarters(50, today=pd.Timestamp("2024-05-01")))
    print("\n=== SANITY: insider _quarters bounded to data-set era ===")
    print(f"  years_history=3 @2024 -> {len(qs)} quarters 2021q1..2024q4. Validated.")


def test_insider_parse_and_universe_filter_synthetic():
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"],
        "ISSUERCIK": ["320193", "999999"],
        "ISSUERNAME": ["APPLE INC", "OFFUNIVERSE CO"],
        "ISSUERTRADINGSYMBOL": ["AAPL", "ZZZZ"],
        "DOCUMENT_TYPE": ["4", "4"],
        "FILING_DATE": ["31-JAN-2024", "31-JAN-2024"],
        "PERIOD_OF_REPORT": ["29-JAN-2024", "29-JAN-2024"],
    })
    own = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"],
        "RPTOWNERCIK": ["111", "222"],
        "RPTOWNERNAME": ["COOK TIMOTHY", "DOE JOHN"],
        "RPTOWNER_RELATIONSHIP": ["Officer", "Director"],
        "RPTOWNER_TITLE": ["CEO", ""],
    })
    nd = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2"],
        "NONDERIV_TRANS_SK": ["1", "1"],
        "SECURITY_TITLE": ["Common", "Common"],
        "TRANS_DATE": ["29-JAN-2024", "29-JAN-2024"],
        "TRANS_CODE": ["P", "S"],
        "TRANS_SHARES": ["1000", "500"],
        "TRANS_PRICEPERSHARE": ["150", "20"],
        "TRANS_ACQUIRED_DISP_CD": ["A", "D"],
        "SHRS_OWND_FOLWNG_TRANS": ["5000", "100"],
        "DIRECT_INDIRECT_OWNERSHIP": ["D", "D"],
    })
    out = ins._parse_insider(sub, own, nd, pd.DataFrame())
    assert set(out["accession_number"]) == {"a1", "a2"}
    a1 = out[out["accession_number"] == "a1"].iloc[0]
    assert a1["ticker"] == "AAPL" and a1["is_officer"] == 1.0 and a1["transaction_code"] == "P"
    assert abs(a1["value_usd"] - 150000.0) < 1e-6            # 1000 * 150

    filt = ins._filter_universe(out, {"AAPL"}, {"0000999999": "ZZZZ"})
    assert set(filt["ticker"]) == {"AAPL"}                    # ZZZZ mapped but not in universe
    print("\n=== SANITY: insider parse + universe filter ===")
    print(f"  a1 AAPL officer PURCHASE 1000@150 = $150k; universe filter kept AAPL, dropped ZZZZ. Validated.")


@pytest.mark.skipif(not INSIDER_ZIP.exists(), reason="cached insider 2024q1 zip absent")
def test_insider_parse_real_zip():
    tables = ins._read_tables(INSIDER_ZIP)
    assert tables is not None
    df = ins._parse_insider(*tables)
    assert not df.empty and df["transaction_sk"].notna().all()
    assert set(df["security_type"]) <= {"nonderiv", "deriv"}
    codes = df["transaction_code"].value_counts()
    aapl = df[df["ticker"] == "AAPL"]
    print("\n=== SANITY: insider REAL 2024q1 zip ===")
    print(f"  {len(df):,} transactions, {df['ticker'].nunique():,} issuers; "
          f"top codes: {codes.head(4).to_dict()}")
    print(f"  AAPL rows={len(aapl)}; sample value_usd nonnull={aapl['value_usd'].notna().mean():.0%}. Validated.")
    assert len(df) > 10000 and df["ticker"].nunique() > 1000


def test_insider_incremental_state_converges(tmp_path):
    """Quarter-skip comes from the DB; the re-parse-on-new-ticker decision compares
    the CURRENT universe to the PROCESSED-universe sidecar (so it converges instead
    of re-parsing every run just because some names never file that quarter)."""
    from src.data_extract.utils.common.sec_utils import (
        bulk_ingested_quarters, load_processed_universe, save_processed_universe)
    ds = DataStore(create_engine(f"sqlite:///{tmp_path/'t.db'}"))
    ds.save("insider_transactions", pd.DataFrame([{
        "accession_number": "a1", "security_type": "nonderiv", "transaction_sk": "1",
        "ticker": "AAPL", "quarter": "2024q1"}]))
    assert bulk_ingested_quarters(ds, "insider_transactions") == {"2024q1"}

    save_processed_universe(tmp_path, "insider_transactions", {"AAPL", "MSFT"})
    assert load_processed_universe(tmp_path, "insider_transactions") == {"AAPL", "MSFT"}
    # unchanged universe -> nothing new -> done quarters are skipped (converged)
    assert {"AAPL", "MSFT"} - load_processed_universe(tmp_path, "insider_transactions") == set()
    # grown universe -> only the genuinely new name triggers a cached-zip re-parse
    assert {"AAPL", "MSFT", "NVDA"} - load_processed_universe(tmp_path, "insider_transactions") == {"NVDA"}

    print("\n=== SANITY: incremental state converges ===")
    print("  2024q1 ingested -> skipped next run; unchanged universe -> no re-parse; "
          "adding NVDA -> only NVDA flagged for back-fill. Validated.")


# --------------------------------------------------------------------------- #
# Pension facts (Financial Statement Data Sets)                                  #
# --------------------------------------------------------------------------- #
def test_pension_join_filters_segments_and_coreg():
    num = pd.DataFrame({
        "adsh": ["x", "x", "x", "x"],
        "tag": ["PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"] * 3
               + ["SomethingElse"],
        "ddate": ["20231231", "20231231", "20231231", "20231231"],
        "qtrs": ["0", "0", "0", "0"],
        "uom": ["USD"] * 4,
        "segments": ["", "PlanNameAxis=USPlan", "", ""],   # 2nd row = dimensional member
        "coreg": ["", "", "SubCo", ""],                    # 3rd row = co-registrant
        "value": ["1000", "600", "400", "50"],
    })
    sub = pd.DataFrame({"adsh": ["x"], "cik": ["320193"], "form": ["10-K"],
                        "fy": ["2023"], "fp": ["FY"], "filed": ["20240201"]})
    out = fin._join_pension(num, sub)
    # only the consolidated pension row survives (segment + coreg + non-pension dropped)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["tag"].endswith("LiabilitiesNoncurrent") and r["value"] == 1000.0
    assert r["cik"] == "0000320193" and r["qtrs"] == 0.0
    print("\n=== SANITY: pension join (consolidated only) ===")
    print("  kept the 1 consolidated pension fact ($1000), dropped the plan-segment / co-registrant / non-pension rows. Validated.")


@pytest.mark.skipif(not FINSTMT_ZIP.exists(), reason="cached financial-statement 2024q1 zip absent")
def test_pension_parse_real_zip():
    facts = fin._read_pension_facts(FINSTMT_ZIP)
    assert facts is not None and not facts.empty
    net_liab = facts[facts["tag"] ==
                     "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"]
    assert not net_liab.empty
    assert (facts["value"] > 0).mean() > 0.5           # liabilities are positive
    print("\n=== SANITY: pension REAL 2024q1 zip ===")
    print(f"  {len(facts):,} pension facts, {facts['cik'].nunique():,} companies; "
          f"net-liability rows={len(net_liab):,}, median ${net_liab['value'].median():,.0f}")
    print(f"  tags: {facts['tag'].value_counts().head(5).to_dict()}. Validated.")
    assert net_liab["cik"].nunique() > 100      # ~244 filers report a net DB deficit in 2024q1
