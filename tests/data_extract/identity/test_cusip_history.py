"""CUSIP history (`universe_cusip`, phase 4c): lines back to each CUSIP's first confirmed 13F period (X3), a name never
confirming a foreign CUSIP-6 (X4), second share classes kept as CUSIP rows of their company (X2), EQR -> VMRK (AC-009).

Known-truth fixtures for the 13F rules; the named companies (VMRK, GOOGL, BRK-B, FOXA, WBD, DD, XOM, TT) on the phase-0
fixture store, with the 4b lineage the parity run wrote and the fixture's 13F holder spellings.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.security_lines import SecurityManual, co_registrant_ciks, load_security_manual, tape_intervals
from src.data_extract.utils.identity.cusip_history import (
    EVIDENCE_COLUMNS,
    OBSERVATION_COLUMNS,
    PROFILE_NAME_COLUMNS,
    SHARADAR_CUSIP_COLUMNS,
    company_names,
    lineage_view,
    sharadar_cusips,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.support.cusip_build import NAME_COLUMNS, derive_cusip_history

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


def _holdings(cusip: str, name: str, start: str, end: str, n_rows: int = 5) -> pd.DataFrame:
    periods = pd.date_range(start, end, freq="QE")
    return pd.DataFrame({"cusip": cusip, "period": periods, "issuer_name": name, "n_rows": n_rows})


def _names(ticker: str, name: str) -> pd.DataFrame:
    """The `company_names` rows of one universe ticker's own (undated) name."""
    view = lineage_view(_lineage("NONE", "0000000001", []), {}, pd.DataFrame(columns=["key", "cik", "valid_from", "valid_to"]))
    return company_names(view, pd.DataFrame({"ticker": [ticker], "name": [name]}), None, None)[NAME_COLUMNS]


def _line(rows: pd.DataFrame, cusip: str) -> pd.DataFrame:
    return rows[rows["cusip"].eq(cusip)].sort_values("valid_from")


# --------------------------------------------------------------------------- #
# known truth: the 13F rules                                                     #
# --------------------------------------------------------------------------- #
XYZ = _lineage("XYZ", "0000000100", [("XYZ", "2006-01-03")])
XYZ_OBS = _obs("123456101", "XYZ", "XYZ CORP COM", "2009-07-01", "2010-06-30")


def test_a_pre_2009_cusip_runs_back_to_its_first_confirmed_13f_period() -> None:
    """X3: the FTD line starts 2009; the 13F holders name the company from 1998-12-31 (a 1995 period is before EDGAR 13F,
    a 1998-09-30 period has one holder only, a 1998-06-30 spelling names another company): the line starts 1998-12-31."""
    holdings = pd.concat(
        [
            _holdings("123456101", "XYZ CORP", "1998-12-31", "2009-06-30"),
            _holdings("123456101", "XYZ CORP", "1995-12-31", "1995-12-31"),
            _holdings("123456101", "XYZ CORP", "1998-09-30", "1998-09-30", n_rows=1),
            _holdings("123456101", "ACME WIDGETS INC", "1998-06-30", "1998-06-30"),
        ],
        ignore_index=True,
    )
    names = _names("XYZ", "XYZ CORPORATION")
    plain = derive_cusip_history(XYZ_OBS, XYZ, {"XYZ": "0000000100"}, SecurityManual.empty(), built_at=BUILT_AT)
    build = derive_cusip_history(XYZ_OBS, XYZ, {"XYZ": "0000000100"}, SecurityManual.empty(), built_at=BUILT_AT, holdings=holdings, names=names)

    before, after = _line(plain.rows, "123456101"), _line(build.rows, "123456101")
    assert before["valid_from"].min() > pd.Timestamp("2009-06-01"), before
    assert after["valid_from"].min() == pd.Timestamp("1998-12-31") and after["lineage_role"].eq("canonical_current").all(), after
    assert after["evidence"].str.contains("13F from 1998-12-31 (13F name 'XYZ CORP')", regex=False).all()
    print("\n=== SANITY CHECK: 13F back-extension (X3) ===")
    print(f"  FTD-only line from {before['valid_from'].min().date()}; with the 13F holders from {after['valid_from'].min().date()}")
    print("  the 1995 period (pre-EDGAR), the one-holder 1998-09 period and the foreign 1998-06 spelling are not evidence. Validated.")


def test_a_name_match_is_refused_on_a_foreign_cusip6_qqq_vs_ivz() -> None:
    """X4: 'INVESCO QQQ TR' (46090E103) matches Invesco Ltd by name, but IVZ owns no line of CUSIP-6 46090E: refused,
    no row; IVZ's own G491BT108 is confirmed by its name and runs back."""
    lineage = _lineage("IVZ", "0000914208", [("IVZ", "2007-12-04")])
    obs = _obs("G491BT108", "IVZ", "INVESCO LTD SHS", "2009-07-01", "2010-06-30")
    holdings = pd.concat(
        [
            _holdings("G491BT108", "INVESCO LTD", "2008-03-31", "2009-06-30"),
            _holdings("46090E103", "INVESCO QQQ TR", "1999-03-31", "2009-06-30", n_rows=50),
        ],
        ignore_index=True,
    )
    names = _names("IVZ", "INVESCO LTD")
    build = derive_cusip_history(obs, lineage, {"IVZ": "0000914208"}, SecurityManual.empty(), built_at=BUILT_AT, holdings=holdings, names=names)

    refused = build.thirteenf.refused
    assert refused[["cusip", "company"]].values.tolist() == [["46090E103", "IVZ"]], refused
    assert "46090E103" not in set(build.rows["cusip"]) and "46090E103" not in build.thirteenf.starts
    assert _line(build.rows, "G491BT108")["valid_from"].min() == pd.Timestamp("2008-03-31")
    print("\n=== SANITY CHECK: X4 name match on a foreign CUSIP-6 ===")
    print(
        f"  refused: {refused[['cusip', 'company', 'issuer_name']].values.tolist()}; IVZ G491BT108 runs back to 2008-03-31 on its own name. Validated."
    )


def test_dated_former_names_confirm_only_their_era() -> None:
    """The SEC profile dates each name: 'EQUITY RESIDENTIAL' (2002-2026) and '...PROPERTIES TRUST' (1994-2002) confirm
    the 1999 holders of 29476L107; the 2026 name 'VIVMARK' confirms nothing years before it was adopted."""
    lineage = _lineage("VMRK", "0000906107", [("EQR", "2006-01-05"), ("VMRK", "2026-08-18")])
    profile = pd.DataFrame(
        {
            "cik": ["0000906107"],
            "name": ["VIVMARK RESIDENTIAL"],
            "former_names": [
                json.dumps(
                    [
                        {"name": "EQUITY RESIDENTIAL", "from": "2002-11-13", "to": "2026-08-12"},
                        {"name": "EQUITY RESIDENTIAL PROPERTIES TRUST", "from": "1994-02-10", "to": "2002-10-30"},
                    ]
                )
            ],
        }
    )
    view = lineage_view(lineage, {"VMRK": "0000906107"}, tape_intervals(lineage))
    names = company_names(view, pd.DataFrame({"ticker": ["VMRK"]}), None, profile)
    obs = _obs("29476L107", "EQR", "EQUITY RESIDENTIAL SH BEN INT", "2009-07-01", "2010-06-30")
    old = _holdings("29476L107", "EQUITY RESIDENTIAL", "1999-03-31", "2009-06-30")
    fake = _holdings("29476L107", "VIVMARK RESIDENTIAL", "1998-03-31", "1998-12-31")
    build = derive_cusip_history(
        obs, lineage, {"VMRK": "0000906107"}, SecurityManual.empty(), built_at=BUILT_AT, holdings=pd.concat([old, fake]), names=names
    )

    first, how = build.thirteenf.starts["29476L107"]
    assert (first, how) == (pd.Timestamp("1999-03-31"), "13F name 'EQUITY RESIDENTIAL'"), (first, how)
    assert set(_line(build.rows, "29476L107")["canonical_company"]) == {"VMRK"}
    print("\n=== SANITY CHECK: dated former names ===")
    print(
        f"  names {sorted(set(names['name']))}; 29476L107 confirmed from {first.date()} by {how}; the 1998 'VIVMARK' spelling is out of its era. Validated."
    )


def test_a_run_back_never_crosses_an_earlier_cusip_of_its_symbol() -> None:
    """Two CUSIPs of one symbol: the older one runs back to its 13F start; the newer one only to the older line's end."""
    lineage = _lineage("ABC", "0000000777", [("ABC", "2006-01-03")])
    obs = pd.concat(
        [
            _obs("000111101", "ABC", "ABC CORP COM", "2009-07-01", "2011-12-28"),
            _obs("000111202", "ABC", "ABC CORP COM NEW", "2012-03-07", "2013-06-26"),
        ]
    )
    holdings = pd.concat(
        [_holdings("000111101", "ABC CORP", "2001-03-31", "2011-12-31"), _holdings("000111202", "ABC CORP", "2005-03-31", "2013-06-30")]
    )
    names = _names("ABC", "ABC CORP")
    build = derive_cusip_history(obs, lineage, {"ABC": "0000000777"}, SecurityManual.empty(), built_at=BUILT_AT, holdings=holdings, names=names)

    old, new = _line(build.rows, "000111101"), _line(build.rows, "000111202")
    assert old["valid_from"].min() == pd.Timestamp("2001-03-31"), old
    assert new["valid_from"].min() >= old["valid_to"].max(), (old[["valid_from", "valid_to"]], new[["valid_from", "valid_to"]])
    print("\n=== SANITY CHECK: run back bounded by the symbol's earlier CUSIP ===")
    print(
        f"  old 000111101 {old['valid_from'].min().date()}..{old['valid_to'].max().date()}; new 000111202 from {new['valid_from'].min().date()}; no shared day. Validated."
    )


# --------------------------------------------------------------------------- #
# real data: the named companies on the fixture store                            #
# --------------------------------------------------------------------------- #
def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        if (cache / "fixture_store" / "identity.sqlite").exists():
            return cache
    pytest.skip("the run's fixture store is not reachable from this checkout")


def _fixture_holdings(path: Path, cusips: list[str]) -> pd.DataFrame:
    """The fixture's 13F aggregates in the production read's shape (`_scripts/parity_check.fixture_holdings`)."""
    marks = ",".join("?" * len(cusips))
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as con:
        period = pd.read_sql(
            f"select cusip, period, sum(n_rows) as n_rows from sec13f_cusip_period where cusip in ({marks}) group by cusip, period",
            con,
            params=cusips,
        )
        names = pd.read_sql(
            f"select cusip, issuer_name, min(first_period) as first, max(last_period) as last from sec13f_cusip_issuer where cusip in ({marks}) group by cusip, issuer_name",
            con,
            params=cusips,
        )
    rows = period.merge(names, on="cusip")
    rows = rows[(rows["period"] >= rows["first"]) & (rows["period"] <= rows["last"])].sort_values(["cusip", "period", "issuer_name"])
    return rows.assign(n_rows=rows["n_rows"].where(~rows.duplicated(["cusip", "period"]), 0))[["cusip", "period", "issuer_name", "n_rows"]]


@pytest.fixture(scope="module")
def fixture_build():
    """The 4c compute over the fixture store (read-only) with the 4b lineage and the fixture's 13F holder spellings."""
    cache = _run_cache()
    lineage_path = cache / "parity" / "lineage_4b_new.parquet"
    if not lineage_path.exists():
        pytest.skip(f"no 4b lineage at {lineage_path} (run _scripts/parity_check.py --phase 4b first)")
    path = cache / "fixture_store" / "identity.sqlite"
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    lineage = pd.read_parquet(lineage_path)
    observations = store.load(Tables.sec_fails_to_deliver_security, columns=list(OBSERVATION_COLUMNS))
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik", "name", "sharadar_name", "permaticker"])
    sharadar = store.load(Tables.sharadar_tickers, columns=list(SHARADAR_CUSIP_COLUMNS))
    profile = store.load(
        Tables.sec_cik_profile, columns=list(PROFILE_NAME_COLUMNS), where={"cik": sorted(set(lineage["cik"].astype(str).str.zfill(10)))}
    )
    evidence = store.load(Tables.symbol_tenure, columns=list(EVIDENCE_COLUMNS), where={"source": ["form345", "dei"]})
    presence = store.load(Tables.sec_short_volume_security, columns=["source_symbol", "date"])
    sec_tickers = store.load(Tables.sec_company_tickers, project=True)
    roster = dict(zip(universe["ticker"], universe["cik"].astype(str).str.zfill(10), strict=True))
    manual = load_security_manual(CONFIG_DIR)
    holdings = _fixture_holdings(path, sorted(set(observations["cusip"])))
    names = company_names(lineage_view(lineage, roster, tape_intervals(lineage)), universe, sharadar, profile)
    return derive_cusip_history(
        observations,
        lineage,
        roster,
        manual,
        built_at=BUILT_AT,
        sec_tickers=sec_tickers,
        finra_presence=presence,
        co_registrant_ciks=co_registrant_ciks(lineage, evidence, manual.co_registrants),
        holdings=holdings,
        names=names,
        listed=sharadar_cusips(universe, sharadar),
        as_of=BUILT_AT,
    )


def _company(rows: pd.DataFrame, cusip: str) -> set[str]:
    return set(rows.loc[rows["cusip"].eq(cusip) & rows["lineage_role"].isin(MAPPED), "canonical_company"])


def _roles(rows: pd.DataFrame, cusip: str) -> set[str]:
    return set(rows.loc[rows["cusip"].eq(cusip), "lineage_role"])


def _start(rows: pd.DataFrame, cusip: str) -> pd.Timestamp:
    return pd.Timestamp(rows.loc[rows["cusip"].eq(cusip) & rows["lineage_role"].isin(MAPPED), "valid_from"].min())


def test_vmrk_holds_29476l107_over_the_whole_eqr_history(fixture_build) -> None:
    """AC-009: EQR's CUSIP is VMRK's line from its first 13F period, with no row under another company."""
    rows = fixture_build.rows
    line = _line(rows, "29476L107")
    assert set(line["canonical_company"]) == {"VMRK"} and _roles(rows, "29476L107") == {"canonical_current"}, line
    assert _start(rows, "29476L107") < pd.Timestamp("2000-01-01"), line[["valid_from", "valid_to", "evidence"]]
    print("\n=== SANITY CHECK: VMRK 29476L107 (AC-009) ===")
    print(
        f"  29476L107 -> VMRK canonical from {_start(rows, '29476L107').date()} ({fixture_build.thirteenf.starts['29476L107'][1]}); no EQR company row. Validated."
    )


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
    assert _start(rows, "38259P508") < pd.Timestamp("2009-06-01"), "Google's class A runs back before the FTD era"
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
    assert _company(rows, "30231G102") == {"XOM"} and _start(rows, "30231G102") < pd.Timestamp("2001-01-01"), xom
    assert xom["valid_to"].max() <= pd.Timestamp("2026-07-31") and _company(rows, "30233Q108") == {"XOM"} and holdco["valid_to"].isna().any(), holdco
    assert _company(rows, "G8994E103") == {"TT"} and "TT" not in _company(rows, "45687V106")
    print(
        f"  26614N102 -> DD; 30231G102 -> XOM from {_start(rows, '30231G102').date()}, open; G8994E103 -> TT; 45687V106 -> {sorted(_company(rows, '45687V106'))}"
    )
    print("  OK: the lineage decides the company of each CUSIP line, dated")
