"""CUSIP history (`universe_cusip`, phase 4c): second share classes kept as CUSIP rows of their company (X2), EQR -> VMRK
(AC-009), on the phase-0 fixture store with the 4b lineage the parity run wrote. Lines start at their first FTD line
(identity-security-grain Q3: no 13F back-extension).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.security_lines import co_registrant_ciks, load_security_manual
from src.data_extract.utils.identity.cusip_history import (
    EVIDENCE_COLUMNS,
    OBSERVATION_COLUMNS,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.support.cusip_build import derive_cusip_history

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = str(ROOT / "configs")
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"
BUILT_AT = pd.Timestamp("2026-11-04")
MAPPED = ("canonical_current", "canonical_predecessor", "secondary_class")


def _lineage(ticker: str, cik: str, symbols: list[tuple[str, str]]) -> pd.DataFrame:
    """One company: its roster window and its symbol intervals `(symbol, valid_from)`, open."""
    rows = [{"company_id": f"E{cik}", "canonical_ticker": ticker, "cik": cik, "role": "cik_window", "symbol": "", "valid_from": pd.Timestamp("1900-01-01"),
             "valid_to": pd.NaT, "status": "curated", "sources": "roster"}]  # fmt: skip
    rows += [{"company_id": f"E{cik}", "canonical_ticker": ticker, "cik": cik, "role": "symbol", "symbol": s, "valid_from": pd.Timestamp(start),
              "valid_to": pd.NaT, "status": "corroborated", "sources": "form345,roster"} for s, start in symbols]  # fmt: skip
    return pd.DataFrame(rows)


def _obs(cusip: str, symbol: str, description: str, start: str, end: str) -> pd.DataFrame:
    days = pd.bdate_range(start, end, freq="W-WED")
    period = [f"{d.year}{d.month:02d}{'a' if d.day <= 15 else 'b'}" for d in days]
    return pd.DataFrame({"date": days, "cusip": cusip, "source_symbol": symbol, "description": description, "price": 10.0, "period": period})


def _line(rows: pd.DataFrame, cusip: str) -> pd.DataFrame:
    return rows[rows["cusip"].eq(cusip)].sort_values("valid_from")


# --------------------------------------------------------------------------- #
# known truth: the 13F rules                                                     #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# real data: the named companies on the fixture store                            #
# --------------------------------------------------------------------------- #
def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        if (cache / "fixture_store" / "identity.sqlite").exists():
            return cache
    pytest.skip("the run's fixture store is not reachable from this checkout")


@pytest.fixture(scope="module")
def fixture_build():
    """The 4c compute over the fixture store (read-only) with the 4b lineage."""
    cache = _run_cache()
    lineage_path = cache / "parity" / "lineage_4b_new.parquet"
    if not lineage_path.exists():
        pytest.skip(f"no 4b lineage at {lineage_path} (run _scripts/parity_check.py --phase 4b first)")
    path = cache / "fixture_store" / "identity.sqlite"
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    lineage = pd.read_parquet(lineage_path)
    observations = store.load(Tables.sec_fails_to_deliver_security, columns=list(OBSERVATION_COLUMNS))
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik"])
    evidence = store.load(Tables.symbol_tenure, columns=list(EVIDENCE_COLUMNS), where={"source": ["form345", "dei"]})
    presence = store.load(Tables.sec_short_volume_security, columns=["source_symbol", "date"])
    sec_tickers = store.load(Tables.sec_company_tickers, project=True)
    roster = dict(zip(universe["ticker"], universe["cik"].astype(str).str.zfill(10), strict=True))
    manual = load_security_manual(CONFIG_DIR)
    return derive_cusip_history(
        observations,
        lineage,
        roster,
        manual,
        built_at=BUILT_AT,
        sec_tickers=sec_tickers,
        finra_presence=presence,
        co_registrant_ciks=co_registrant_ciks(lineage, evidence, manual.co_registrants),
    )


def _company(rows: pd.DataFrame, cusip: str) -> set[str]:
    return set(rows.loc[rows["cusip"].eq(cusip) & rows["lineage_role"].isin(MAPPED), "canonical_company"])


def _roles(rows: pd.DataFrame, cusip: str) -> set[str]:
    return set(rows.loc[rows["cusip"].eq(cusip), "lineage_role"])


def _start(rows: pd.DataFrame, cusip: str) -> pd.Timestamp:
    return pd.Timestamp(rows.loc[rows["cusip"].eq(cusip) & rows["lineage_role"].isin(MAPPED), "valid_from"].min())


def test_vmrk_holds_29476l107_over_the_whole_eqr_history(fixture_build) -> None:
    """AC-009: EQR's CUSIP is VMRK's line from its first FTD line, with no row under another company."""
    rows = fixture_build.rows
    line = _line(rows, "29476L107")
    assert set(line["canonical_company"]) == {"VMRK"} and _roles(rows, "29476L107") == {"canonical_current"}, line
    print("\n=== SANITY CHECK: VMRK 29476L107 (AC-009) ===")
    print(f"  29476L107 -> VMRK canonical from {_start(rows, '29476L107').date()}; no EQR company row. Validated.")


def test_second_share_classes_are_cusip_rows_of_their_company(fixture_build) -> None:
    """X2: GOOG, BRK-A, FOX keep their CUSIP rows as secondary classes of GOOGL, BRK-B, FOXA (combined at company level);
    Google's class A predecessor C38259P508 is GOOGL's canonical line before Alphabet's 02079K305."""
    rows = fixture_build.rows
    cases = {
        "38259P508": ("GOOGL", {"canonical_predecessor"}),
        "02079K305": ("GOOGL", {"canonical_current"}),
        "02079K107": ("GOOGL", {"secondary_class"}),
        "084670702": ("BRK-B", {"canonical_current"}),
        "084670108": ("BRK-B", {"secondary_class"}),
        "35137L105": ("FOXA", {"canonical_current"}),
        "35137L204": ("FOXA", {"secondary_class"}),
    }
    print("\n=== SANITY CHECK: second share classes (X2) ===")
    for cusip, (company, roles) in cases.items():
        assert _company(rows, cusip) == {company}, (cusip, _line(rows, cusip)[["canonical_company", "lineage_role", "valid_from", "valid_to"]])
        assert roles <= _roles(rows, cusip), (cusip, _roles(rows, cusip))
        print(f"  {cusip} -> {company} {sorted(_roles(rows, cusip))} from {_start(rows, cusip).date()}")
    print("  OK: each class CUSIP maps to its company; class-level outputs are not added")


def test_wbd_dd_xom_tt_lines(fixture_build) -> None:
    """WBD holds the DISCA/DISCB/DISCK CUSIPs; DD's DowDuPont/DuPont CUSIP is DD's; XOM's 30231G102 stays canonical over the
    2026 holdco seam; TT holds the Ingersoll-Rand plc CUSIP, never Ingersoll Rand Inc's 45687V106."""
    rows = fixture_build.rows
    print("\n=== SANITY CHECK: WBD, DD, XOM, TT ===")
    for cusip in ("25470F104", "25470F203", "25470F302", "934423104"):
        assert _company(rows, cusip) == {"WBD"}, (cusip, _line(rows, cusip)[["canonical_company", "lineage_role", "valid_from"]])
        print(f"  {cusip} -> WBD {sorted(_roles(rows, cusip))} from {_start(rows, cusip).date()}")
    assert _company(rows, "26614N102") == {"DD"}, _line(rows, "26614N102")
    xom = _line(rows, "30231G102")
    holdco = _line(rows, "30233Q108")
    assert _company(rows, "30231G102") == {"XOM"}, xom
    assert xom["valid_to"].max() <= pd.Timestamp("2026-07-31") and _company(rows, "30233Q108") == {"XOM"} and holdco["valid_to"].isna().any(), holdco
    assert _company(rows, "G8994E103") == {"TT"} and "TT" not in _company(rows, "45687V106")
    print(
        f"  26614N102 -> DD; 30231G102 -> XOM from {_start(rows, '30231G102').date()}, open; G8994E103 -> TT; 45687V106 -> {sorted(_company(rows, '45687V106'))}"
    )
    print("  OK: the lineage decides the company of each CUSIP line, dated")
