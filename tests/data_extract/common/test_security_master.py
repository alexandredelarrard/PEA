"""Security master: issuer per CUSIP-9, class by positive evidence, lineage role, dated conversion ratio (Q2a).

Known-truth fixtures shaped like the measured FTD lines (GOOGL, BRK, MRK, CB, BAC, APTV, EXE, XOM), the lineage
rows that vote for their issuer, and the real `configs/sec` JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.common.sec_tickers import parse_company_tickers_exchange
from src.data_store.schema import Tables
from src.utils.cutover_continuity import load_vendor_exceptions
from tests.data_extract.fake_context import extract_config

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
BUILT_AT = pd.Timestamp("2026-10-04 12:00:00")
LINEAGE_COLUMNS = ("entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources")


# --------------------------------------------------------------------------- fixtures


def _row(ticker: str, entity_cik: str, cik: str, role: str, symbol: str = "", vf: str | None = None, vt: str | None = None, **kw: str) -> dict:
    return {
        "entity_id": f"E{entity_cik}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": role,
        "symbol": symbol,
        "valid_from": pd.Timestamp(vf or "1900-01-01"),
        "valid_to": pd.Timestamp(vt) if vt else pd.NaT,
        "status": kw.get("status", "corroborated" if role == "symbol" else "curated"),
        "sources": kw.get("sources", "form345" if role == "symbol" else "register"),
    }


def _lineage(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(LINEAGE_COLUMNS))


def _period(day: pd.Timestamp) -> str:
    return f"{day.year}{day.month:02d}{'a' if day.day <= 15 else 'b'}"


def _obs(cusip: str, symbol: str, desc: str, start: str, end: str, price: float = 10.0, freq: str = "W-WED") -> pd.DataFrame:
    days = pd.date_range(start, end, freq=freq)
    if len(days) == 0 or days[0] != pd.Timestamp(start):
        days = pd.DatetimeIndex([pd.Timestamp(start), *days])
    if days[-1] != pd.Timestamp(end):
        days = pd.DatetimeIndex([*days, pd.Timestamp(end)])
    days = days[days.dayofweek < 5].unique()
    return pd.DataFrame(
        {
            "date": days,
            "cusip": cusip,
            "source_symbol": symbol,
            "description": desc,
            "price": price,
            "fails_quantity": 1000.0,
            "period": [_period(day) for day in days],
        }
    )


def _roster(*pairs: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["ticker", "cik"])


def _derive(observations: pd.DataFrame, lineage: pd.DataFrame, roster: pd.DataFrame, manual: sm.SecurityManual | None = None, **kw) -> sm.MasterBuild:
    return sm.derive_security_master(observations, lineage, roster, manual or sm.SecurityManual.empty(), built_at=BUILT_AT, **kw)


def _rows(build: sm.MasterBuild, cusip: str, symbol: str | None = None) -> pd.DataFrame:
    rows = build.rows[build.rows["cusip"].eq(cusip)]
    return rows if symbol is None else rows[rows["source_symbol"].eq(symbol)]


def _canonical_overlaps(build: sm.MasterBuild, ticker: str) -> list[tuple]:
    rows = build.rows[build.rows["canonical_company"].eq(ticker) & build.rows["lineage_role"].isin(sm.CANONICAL_ROLES)]
    spans = sorted(
        (r.valid_from, r.valid_to if pd.notna(r.valid_to) else pd.Timestamp("2262-01-01"), r.cusip, r.source_symbol) for r in rows.itertuples()
    )
    return [(a, b) for a, b in zip(spans, spans[1:], strict=False) if b[0] < a[1]]


GOOGL = _lineage(
    _row("GOOGL", "0001288776", "0001288776", "cik_window", vt="2015-10-02"),
    _row("GOOGL", "0001288776", "0001652044", "cik_window", vf="2015-10-02"),
    _row("GOOGL", "0001288776", "0001288776", "symbol", "GOOG", "2006-01-04", "2015-10-30"),
    _row("GOOGL", "0001288776", "0001652044", "symbol", "GOOG", "2015-10-08"),
    _row("GOOGL", "0001288776", "0001652044", "symbol", "GOOGL", "2015-10-29", sources="form345,roster"),
)
GOOGL_OBS = pd.concat(
    [
        _obs("38259P508", "GOOG", "GOOGLE INC;COM USD0.001 CL'A'", "2013-01-02", "2014-04-03"),
        _obs("38259P508", "GOOGL", "GOOGLE INC;COM USD0.001 CL'A'", "2014-04-07", "2015-10-05"),
        _obs("38259P508", "GOOGLXXXX", "GOOGLE INC;COM USD0.001 CL'A'", "2015-10-06", "2015-10-06"),
        _obs("38259P706", "GOOG", "GOOGLE INC CLASS C", "2014-04-04", "2015-10-05"),
        _obs("02079K305", "GOOGL", "ALPHABET INC CAP STK CL A", "2015-10-06", "2016-06-29"),
        _obs("02079K107", "GOOG", "ALPHABET INC CAP STK CL C", "2015-10-06", "2016-06-29"),
    ],
    ignore_index=True,
)

BRK = _lineage(
    _row("BRK-B", "0001067983", "0001067983", "cik_window", sources="roster"),
    _row("BRK-B", "0001067983", "0001067983", "symbol", "BRK-A", "2006-01-03", sources="dei,form345"),
    _row("BRK-B", "0001067983", "0001067983", "symbol", "BRK-B", "2006-09-28", sources="dei,form345,roster"),
    _row("BRK-B", "0001067983", "0001067983", "symbol", "BRKA", "2009-12-30", "2020-05-20"),
)
BRK_OBS = pd.concat(
    [
        _obs("084670207", "BRKB", "BERKSHIRE HATHWY INC(HLDG CO)B", "2009-07-02", "2010-01-20", price=3300.0),
        _obs("084670702", "BRKBZZZZ", "BERKSHIRE HATHWY INC(HLDG CO)B", "2010-01-21", "2010-01-21", price=0.01),
        _obs("084670702", "BRKB", "BERKSHIRE HATHWY INC(HLDG CO)B", "2010-01-22", "2011-06-29", price=70.0),
        _obs("084670108", "BRKA", "BERKSHIRE HATHWY INC(HLDG CO)A", "2009-07-01", "2010-01-20", price=99000.0),
        _obs("084670108", "BRKA", "BERKSHIRE HATHWY INC(HLDG CO)A", "2010-01-27", "2011-06-29", price=105000.0),
    ],
    ignore_index=True,
)
BRK_MANUAL = sm.parse_security_manual(
    {
        "conversion_ratios": [
            {"ticker": "BRK-B", "cusip": "084670108", "ratio": 30, "valid_from": None, "valid_to": "2010-01-21", "source": "0001157523-10-000241"},
            {"ticker": "BRK-B", "cusip": "084670108", "ratio": 1500, "valid_from": "2010-01-21", "valid_to": None, "source": "0001157523-10-000241"},
        ]
    }
)

MRK = _lineage(
    _row("MRK", "0000064978", "0000064978", "cik_window", vt="2009-11-03"),
    _row("MRK", "0000064978", "0000310158", "cik_window", vf="2009-11-03"),
    _row("MRK", "0000064978", "0000064978", "symbol", "MRK", "2006-01-04", "2011-01-21", status="single_source"),
    _row("MRK", "0000064978", "0000310158", "symbol", "MRK", "2009-11-05"),
    _row("MRK", "0000064978", "0000310158", "symbol", "SGP", "2006-01-04", "2009-11-05", status="single_source"),
)
MRK_OBS = pd.concat(
    [
        _obs("589331107", "MRK", "MERCK & CO INC;COM USD0.01", "2009-07-01", "2009-11-04"),
        _obs("58933Y105", "MRK", "MERCK & CO INC;COM USD0.01", "2009-11-05", "2010-12-29"),
        _obs("58933Y204", "MRKPRB", "MERCK & CO INC 6% PFD CONV", "2009-11-05", "2010-08-16"),
        _obs("806605101", "SGP", "SCHERING PLOUGH CORP", "2009-07-01", "2009-11-04"),
    ],
    ignore_index=True,
)

CB = _lineage(
    _row("CB", "0000020171", "0000896159", "cik_window", sources="roster"),
    _row("CB", "0000020171", "0000020171", "cik_event", sources="form345"),
    _row("CB", "0000020171", "0000020171", "symbol", "CB", "2006-01-04", "2016-02-04"),
    _row("CB", "0000020171", "0000896159", "symbol", "ACE", "2006-01-06", "2015-12-30"),
    _row("CB", "0000020171", "0000896159", "symbol", "CB", "2016-01-19", sources="dei,form345,roster"),
)
CB_OBS = pd.concat(
    [
        _obs("H0023R105", "ACE", "ACE LIMITED (SWITZERLAND)", "2015-06-03", "2016-01-20"),
        _obs("H1467J104", "CB", "CHUBB LTD COM", "2016-01-19", "2016-12-28"),
        _obs("171232101", "CB", "CHUBB CORPORATION", "2015-06-03", "2016-01-07"),
        _obs("171232101", "CBXXXX", "CHUBB CORPORATION", "2016-01-19", "2016-02-01"),
    ],
    ignore_index=True,
)

BAC = _lineage(
    _row("BAC", "0000070858", "0000070858", "cik_window", sources="roster"),
    _row("BAC", "0000070858", "0000070858", "symbol", "BAC", "2006-01-04", sources="dei,form345,roster"),
)
BAC_OBS = pd.concat(
    [
        _obs("060505104", "BAC", "BANK OF AMERICA CORPORATION", "2010-01-06", "2010-02-24"),
        _obs("060505104", "BAC", "BANK OF AMERICA CORPORATION", "2010-06-02", "2010-07-28"),
        _obs("060505682", "BACPRL", "BANK OF AMER CORP 7.25%CNV PFD L", "2010-01-06", "2010-07-28"),
        _obs("060505583", "BACPRE", "BANK OF AMERICA CORP", "2010-01-06", "2010-07-28"),
        _obs("060505146", "BACWSA", "BANK OF AMERICA CORP WT EXP", "2010-01-06", "2010-07-28"),
        _obs("060505559", "BACCU", "BANK OF AMERICA CORP CORP UNIT", "2010-01-06", "2010-07-28"),
        _obs("060505617", "BACXN", "BANK OF AMERICA CORPORATION 6.", "2010-01-06", "2010-07-28"),
    ],
    ignore_index=True,
)


# --------------------------------------------------------------------------- class evidence


@pytest.mark.parametrize(
    ("desc", "symbol", "expected"),
    [
        ("BANK OF AMER CORP 7.25%CNV PFD L", "BACPRL", "preferred"),
        ("BERKLEY W R CORP", "WRBPRE", "preferred"),
        ("GENERAL MTRS CO WT EXP 071019", "GMWSA", "warrant"),
        ("DOMINION RES INC CORP UNIT", "DCUA", "unit"),
        ("ENERGY TRANSFER LP COM UNIT LTD", "ET", None),
        ("UNITED STATES STL CORP NEW", "X", None),
        ("JPMORGAN CHASE & CO NOTES 2028", "JPMXN", "debt"),
        ("BANK OF AMERICA CORPORATION 6.", "BACXN", None),
    ],
)
def test_non_common_kind_needs_positive_evidence(desc, symbol, expected):
    assert sm.non_common_kind(desc, symbol) == expected
    print(f"\n=== SANITY CHECK: E12/E13 {symbol!r} {desc!r} -> {expected} ===\n  OK: word-bounded evidence; COM UNIT and UNITED stay common")


@pytest.mark.parametrize(
    ("desc", "letter"),
    [
        ("GOOGLE INC;COM USD0.001 CL'A'", "A"),
        ("GOOGLE INC CLASS C", "C"),
        ("BERKSHIRE HATHWY INC(HLDG CO)B", "B"),
        ("DISCOVERY COMM INC SER C COM S", "C"),
        ("LENNAR CORPORATION CL-B", "B"),
        ("CLOROX CO DEL", None),
        ("DISCOVERY COMMUNICATIONS INC S", None),
    ],
)
def test_class_letter_from_description(desc, letter):
    assert sm.class_letter(desc) == letter
    print(f"\n=== SANITY CHECK: class letter {desc!r} -> {letter} ===\n  OK")


def test_trade_dates_follow_the_settlement_cycle():
    settle = pd.Series(pd.to_datetime(["2017-09-06", "2017-09-07", "2024-05-28", "2024-05-29", "2010-01-20"]))
    got = sm.trade_dates(settle).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2017-09-01", "2017-09-05", "2024-05-24", "2024-05-28", "2010-01-15"], got
    print(f"\n=== SANITY CHECK: T+3 / T+2 / T+1 eras ===\n  {got}\n  OK: settlement dates convert to trade dates by the cycle in force")


# --------------------------------------------------------------------------- derivation fixtures


def test_e1_goog_class_split_keeps_one_canonical_line():
    build = _derive(GOOGL_OBS, GOOGL, _roster(("GOOGL", "0001652044")))
    old_a = _rows(build, "38259P508")
    assert set(old_a.loc[old_a["source_symbol"].ne("GOOGLXXXX"), "lineage_role"]) == {"canonical_predecessor"}
    assert set(old_a["issuer_cik"]) == {"0001288776"}
    assert _rows(build, "38259P508", "GOOGLXXXX")["lineage_reason"].tolist() == ["transition_placeholder"]
    c_old = _rows(build, "38259P706")
    assert c_old[["lineage_role", "security_class"]].drop_duplicates().values.tolist() == [["secondary_class", "class_C"]]
    assert _rows(build, "02079K305")["lineage_role"].tolist() == ["canonical_current"]
    assert set(_rows(build, "02079K107")["lineage_role"]) == {"secondary_class"}
    assert _canonical_overlaps(build, "GOOGL") == []
    print("\n=== SANITY CHECK: E1 GOOG class split ===")
    print("  38259P508 canonical under GOOG then GOOGL; 38259P706 class C secondary; Alphabet CUSIPs continue; no canonical overlap")


def test_e2_brk_split_cusip_change_and_dated_ratio():
    build = _derive(BRK_OBS, BRK, _roster(("BRK-B", "0001067983")), BRK_MANUAL)
    for cusip in ("084670207", "084670702"):
        rows = _rows(build, cusip, "BRKB")
        assert rows["lineage_role"].tolist() == ["canonical_current"] and rows["conversion_ratio"].tolist() == [1.0]
    a_rows = _rows(build, "084670108").sort_values("valid_from")
    assert a_rows["lineage_role"].tolist() == ["secondary_class", "secondary_class"]
    assert a_rows["security_class"].tolist() == ["class_A", "class_A"]
    assert a_rows["conversion_ratio"].tolist() == [30.0, 1500.0]
    assert a_rows["valid_to"].iloc[0] == pd.Timestamp("2010-01-21") == a_rows["valid_from"].iloc[1]
    assert _canonical_overlaps(build, "BRK-B") == []
    print("\n=== SANITY CHECK: E2 BRK ===")
    print("  B: 084670207 -> 084670702 one canonical line; A secondary, ratio 30 then 1,500 from 2010-01-21")


def test_ac106_b_equivalent_shares_and_ratio_on_the_identity():
    build = _derive(BRK_OBS, BRK, _roster(("BRK-B", "0001067983")), BRK_MANUAL)
    identity = _identity(BRK, ("BRK-B", "0001067983"), build.rows)
    before = identity.security_on(cusip="084670108", source="ftd", day=pd.Timestamp("2009-12-01"))
    after = identity.security_on(cusip="084670108", source="ftd", day=pd.Timestamp("2010-06-01"))
    b_line = identity.security_on(cusip="084670702", source="ftd", day=pd.Timestamp("2010-06-01"))
    assert before is not None and after is not None and b_line is not None
    assert (before.conversion_ratio, after.conversion_ratio, b_line.conversion_ratio) == (30.0, 1500.0, 1.0)
    assert before.canonical_company == after.canonical_company == "BRK-B" and before.lineage_role == "secondary_class"
    b_eq_before = 100 * b_line.conversion_ratio + 2 * before.conversion_ratio
    b_eq_after = 100 * b_line.conversion_ratio + 2 * after.conversion_ratio
    assert (b_eq_before, b_eq_after) == (160.0, 3100.0)
    print("\n=== SANITY CHECK: AC-106 B-equivalent shares ===")
    print(f"  100 B + 2 A = {b_eq_before:.0f} before 2010-01-21, {b_eq_after:.0f} after; A is never raw-summed")


def test_e3_ratio_step_flags_a_split_and_accepts_the_configured_brk_step():
    brk = _derive(BRK_OBS, BRK, _roster(("BRK-B", "0001067983")), BRK_MANUAL)
    assert brk.flags[brk.flags["kind"].eq("security_ratio_step")].empty, brk.flags
    xyz = _lineage(
        _row("XYZ", "0000000111", "0000000111", "cik_window", sources="roster"),
        _row("XYZ", "0000000111", "0000000111", "symbol", "XYZ", "2006-01-04"),
    )
    obs = pd.concat(
        [
            _obs("98765A101", "XYZ", "XYZ CORP CL A", "2015-01-05", "2015-12-28", price=100.0),
            _obs("98765A200", "XYZB", "XYZ CORP CL B", "2015-01-05", "2015-06-24", price=50.0),
            _obs("98765A200", "XYZB", "XYZ CORP CL B", "2015-07-01", "2015-12-28", price=25.0),
        ],
        ignore_index=True,
    )
    build = _derive(obs, xyz, _roster(("XYZ", "0000000111")))
    steps = build.flags[build.flags["kind"].eq("security_ratio_step")]
    assert steps["cusip"].tolist() == ["98765A200"], build.flags
    print("\n=== SANITY CHECK: E3 ratio step check ===")
    print(f"  synthetic 2:1 split in class B flagged once ({steps['detail'].iloc[0]}); BRK's configured 30 -> 1,500 step accepted")


def test_e4_price_level_difference_with_ratio_one_is_not_flagged():
    len_lineage = _lineage(
        _row("LEN", "0000920760", "0000920760", "cik_window", sources="roster"),
        _row("LEN", "0000920760", "0000920760", "symbol", "LEN", "2006-01-05"),
        _row("LEN", "0000920760", "0000920760", "symbol", "LEN-B", "2006-01-05"),
    )
    obs = pd.concat(
        [
            _obs("526057104", "LEN", "LENNAR CORP CL A COMMON", "2015-01-05", "2015-12-28", price=50.0),
            _obs("526057302", "LENB", "LENNAR CORPORATION CL-B", "2015-01-05", "2015-12-28", price=40.6),
        ],
        ignore_index=True,
    )
    build = _derive(obs, len_lineage, _roster(("LEN", "0000920760")))
    assert build.flags[build.flags["kind"].eq("security_ratio_step")].empty
    lenb = _rows(build, "526057302")
    assert lenb[["lineage_role", "security_class", "conversion_ratio"]].values.tolist() == [["secondary_class", "class_B", 1.0]]
    print("\n=== SANITY CHECK: E4 LEN-B at 0.81x LEN with ratio 1 ===\n  OK: the level is not checked, only steps")


def test_e5_e16_e24_mrk_reorganisation_cusips_and_the_acquired_symbol():
    build = _derive(MRK_OBS, MRK, _roster(("MRK", "0000310158")))
    old = _rows(build, "589331107")
    new = _rows(build, "58933Y105")
    assert old[["lineage_role", "issuer_cik"]].values.tolist() == [["canonical_predecessor", "0000064978"]]
    assert pd.notna(old["valid_to"].iloc[0]) and old["valid_to"].iloc[0] <= pd.Timestamp("2009-11-03")
    assert new[["lineage_role", "issuer_cik"]].values.tolist() == [["canonical_current", "0000310158"]]
    sgp = _rows(build, "806605101")
    assert sgp[["lineage_role", "issuer_cik", "lineage_reason"]].values.tolist() == [["acquired_constituent", "0000310158", "outside_window"]]
    pref = _rows(build, "58933Y204")
    assert pref[["lineage_role", "security_class", "lineage_reason"]].values.tolist() == [["excluded", "preferred", "preferred"]]
    assert _canonical_overlaps(build, "MRK") == []
    print("\n=== SANITY CHECK: E5/E16/E24 MRK ===")
    print("  589331107 old Merck canonical_predecessor, closed; 58933Y105 voted by both CIKs -> the window owner 0000310158")
    print("  SGP acquired_constituent; MRKPRB inherits the CUSIP-6 issuer and is excluded as preferred")


def test_e5_ace_to_chubb_successor_continues_and_old_chubb_is_acquired():
    build = _derive(CB_OBS, CB, _roster(("CB", "0000896159")))
    ace = _rows(build, "H0023R105").sort_values("valid_from")
    assert ace["lineage_role"].iloc[0] == "canonical_current"
    assert set(ace["lineage_role"].iloc[1:]) <= {"excluded"} and set(ace["lineage_reason"].iloc[1:]) <= {"superseded"}
    assert _rows(build, "H1467J104")["lineage_role"].tolist() == ["canonical_current"]
    old = _rows(build, "171232101", "CB")
    assert old[["lineage_role", "issuer_cik", "lineage_reason"]].values.tolist() == [["acquired_constituent", "0000020171", "event_only_cik"]]
    assert _canonical_overlaps(build, "CB") == []
    print("\n=== SANITY CHECK: E5 ACE -> Chubb Ltd ===")
    print("  ACE canonical until H1467J104 takes over (tail superseded, never double-counted); old Chubb CB acquired_constituent")


def test_e12_e13_e26_non_common_lines_unclassified_flag_and_gap_bridging():
    build = _derive(BAC_OBS, BAC, _roster(("BAC", "0000070858")))
    roles = {cusip: (role, reason) for cusip, role, reason in build.rows[["cusip", "lineage_role", "lineage_reason"]].itertuples(index=False)}
    assert roles["060505104"] == ("canonical_current", "ticker_symbol")
    assert roles["060505682"] == ("excluded", "preferred")
    assert roles["060505583"] == ("excluded", "preferred")
    assert roles["060505146"] == ("excluded", "warrant")
    assert roles["060505559"] == ("excluded", "unit")
    assert roles["060505617"] == ("excluded", "unclassified")
    unclassified = build.flags[build.flags["kind"].eq("security_unclassified")]
    assert unclassified["cusip"].tolist() == ["060505617"]
    assert len(_rows(build, "060505104")) == 1
    identity = _identity(BAC, ("BAC", "0000070858"), build.rows)
    hit = identity.security_on(cusip="060505104", source="ftd", day=pd.Timestamp("2010-04-01"))
    assert hit is not None and hit.lineage_role == "canonical_current"
    print("\n=== SANITY CHECK: E12/E13/E26 ===")
    print("  preferred (PFD, PR symbol), warrant, unit excluded with reasons; a truncated note is unclassified and flagged")
    print("  a 3-month fail-free gap inside one CUSIP stays covered (no resolution hole)")


def test_e24_straddle_and_two_company_votes_are_flagged_not_attributed():
    lineage = pd.concat(
        [
            MRK,
            _lineage(
                _row("ZZ", "0000000222", "0000000222", "cik_window", sources="roster"),
                _row("ZZ", "0000000222", "0000000222", "symbol", "ZZOLD", "2006-01-04"),
            ),
        ],
        ignore_index=True,
    )
    obs = pd.concat(
        [
            _obs("58933Y998", "MRK", "MERCK STRADDLE COM", "2009-08-05", "2010-03-03"),
            _obs("777777107", "ZZOLD", "SHARED CUSIP CO", "2011-01-05", "2011-03-02"),
            _obs("777777107", "MRK", "SHARED CUSIP CO", "2011-03-09", "2011-05-04"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, lineage, _roster(("MRK", "0000310158"), ("ZZ", "0000000222")))
    kinds = build.flags.groupby("kind")["cusip"].agg(lambda s: sorted(set(s))).to_dict()
    assert kinds.get("security_straddle") == ["58933Y998"], kinds
    assert kinds.get("security_issuer_conflict") == ["777777107"], kinds
    assert build.rows[build.rows["cusip"].eq("777777107")].empty
    straddle = _rows(build, "58933Y998").sort_values("valid_from")
    assert straddle[["lineage_role", "issuer_cik"]].values.tolist() == [["canonical_predecessor", "0000064978"], ["canonical_current", "0000310158"]]
    assert straddle["valid_to"].iloc[0] == pd.Timestamp("2009-11-03") == straddle["valid_from"].iloc[1]
    print("\n=== SANITY CHECK: E24 per-CUSIP-9 dated attribution ===")
    print("  a CUSIP crossing both MRK windows is flagged and dated by window owner (old Merck, then 0000310158)")
    print("  a CUSIP voted for by two companies is a conflict and is not attributed")


def test_ac103_one_symbol_two_cusips_on_one_date_is_a_conflict():
    obs = pd.concat([BAC_OBS, _obs("060505999", "BAC", "BANK OF AMERICA CORP", "2010-01-13", "2010-01-13")], ignore_index=True)
    build = _derive(obs, BAC, _roster(("BAC", "0000070858")))
    conflicts = build.flags[build.flags["kind"].eq("security_symbol_conflict")]
    assert conflicts["source_symbol"].tolist() == ["BAC"] and "060505999" in conflicts["detail"].iloc[0]
    identity = _identity(BAC, ("BAC", "0000070858"), build.rows)
    day = sm.trade_dates(pd.Series([pd.Timestamp("2010-01-13")])).iloc[0]
    assert identity.security_on(symbol="BAC", source="ftd", day=day) is None
    assert identity.security_on(cusip="060505104", source="ftd", day=day) is not None
    print("\n=== SANITY CHECK: AC-103 ===")
    print(f"  FTD symbol BAC under two CUSIPs on {day.date()}: flagged; symbol lookup unresolved, CUSIP lookup still exact")


def test_manual_boundaries_dlph_exe_xom_from_the_real_config():
    manual = sm.load_security_manual(str(CONFIG_DIR))
    aptv = _lineage(
        _row("APTV", "0001521332", "0001521332", "cik_window", sources="roster"),
        _row("APTV", "0001521332", "0001521332", "symbol", "APTV", "2017-12-06"),
        _row("APTV", "0001521332", "0001521332", "symbol", "DLPH", "2011-11-16", "2017-11-30", status="conflict"),
    )
    obs = pd.concat(
        [
            _obs("G27823106", "DLPH", "DELPHI AUTOMOTIVE PLC", "2011-11-22", "2017-12-05"),
            _obs("G27823106", "3106PS", "DELPHI AUTOMOTIVE PLC", "2017-12-06", "2017-12-08"),
            _obs("G6095L109", "APTV", "APTIV PLC SHS (JEY)", "2017-12-06", "2018-06-27"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, aptv, _roster(("APTV", "0001521332")), manual)
    dlph = _rows(build, "G27823106", "DLPH")
    assert dlph[["lineage_role", "lineage_reason"]].values.tolist() == [["canonical_current", "manual_boundary"]]
    assert dlph["valid_to"].iloc[0] == pd.Timestamp("2017-12-05") and dlph["source_accession"].iloc[0]
    aptv_rows = _rows(build, "G6095L109").sort_values("valid_from")
    assert aptv_rows["lineage_role"].iloc[-1] == "canonical_current" and aptv_rows["valid_from"].iloc[-1] == pd.Timestamp("2017-12-05")
    assert _canonical_overlaps(build, "APTV") == []

    exe = _lineage(
        _row("EXE", "0000895126", "0000895126", "cik_window", sources="roster"),
        _row("EXE", "0000895126", "0000895126", "symbol", "CHK", "2021-02-10", "2024-10-02", status="curated", sources="manual"),
        _row("EXE", "0000895126", "0000895126", "symbol", "CHKAQ", "2020-08-10", "2021-01-20"),
        _row("EXE", "0000895126", "0000895126", "symbol", "EXE", "2024-10-02", status="curated", sources="manual,roster"),
    )
    obs = pd.concat(
        [
            _obs("165167107", "CHK", "CHESAPEAKE ENERGY CORP", "2009-07-01", "2020-04-15", freq="ME"),
            _obs("165167743", "CHKAQ", "CHESAPEAKE ENERGY CORP COM", "2020-07-02", "2021-02-19"),
            _obs("165167735", "CHK", "CHESAPEAKE ENERGY CORP COM", "2021-02-12", "2021-06-30"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, exe, _roster(("EXE", "0000895126")), manual)
    chk_old = _rows(build, "165167107")
    assert chk_old["lineage_role"].tolist() == ["canonical_current"] and chk_old["valid_from"].iloc[0] == pd.Timestamp("2009-06-26")
    chkaq = _rows(build, "165167743").sort_values("valid_from")
    assert chkaq["lineage_role"].tolist() == ["canonical_current", "excluded"]
    assert chkaq["lineage_reason"].tolist()[1] == "cancelled_security" and chkaq["valid_from"].iloc[1] == pd.Timestamp("2021-02-10")
    assert _canonical_overlaps(build, "EXE") == []
    print("\n=== SANITY CHECK: manual market boundaries (real config) ===")
    print("  DLPH G27823106 canonical [first obs, 2017-12-05); CHK 165167107 open start back to the first FTD row")
    print("  CHKAQ 165167743 ends at the 2021-02-09 cancellation and is never summed with 165167735")


def test_ac101_ac102_deterministic_and_evidenced():
    obs = pd.concat([GOOGL_OBS, MRK_OBS, CB_OBS], ignore_index=True)
    lineage = pd.concat([GOOGL, MRK, CB], ignore_index=True)
    roster = _roster(("GOOGL", "0001652044"), ("MRK", "0000310158"), ("CB", "0000896159"))
    first = _derive(obs, lineage, roster).rows
    second = _derive(obs.sample(frac=1.0, random_state=7), lineage.iloc[::-1], roster).rows
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(sm.TABLE_COLUMNS)
    assert first["evidence"].str.len().gt(0).all()
    assert first.groupby(["cusip", "source_symbol"])["n_observations"].sum().gt(0).all()
    assert set(first["issuer_cik"]) <= set(lineage["cik"])
    assert first.duplicated(["security_id", "source", "source_symbol", "valid_from"]).sum() == 0
    assert set(first["scope_changed_at"]) == {BUILT_AT}
    print("\n=== SANITY CHECK: AC-101/AC-102 ===")
    print(f"  {len(first)} rows, identical under shuffled inputs with a pinned build time; every row has evidence and a lineage issuer")


def test_scope_changed_at_is_carried_for_unchanged_companies():
    obs = pd.concat([GOOGL_OBS, MRK_OBS], ignore_index=True)
    lineage = pd.concat([GOOGL, MRK], ignore_index=True)
    roster = _roster(("GOOGL", "0001652044"), ("MRK", "0000310158"))
    first = _derive(obs, lineage, roster).rows
    later = pd.Timestamp("2026-10-05 12:00:00")
    grown = pd.concat([obs, _obs("58933Y105", "MRK", "MERCK & CO INC;COM USD0.01", "2011-01-05", "2011-02-02")], ignore_index=True)
    second = sm.derive_security_master(grown, lineage, roster, sm.SecurityManual.empty(), built_at=later, existing=first).rows
    stamps = second.groupby("canonical_company")["scope_changed_at"].max().to_dict()
    assert stamps == {"GOOGL": BUILT_AT, "MRK": later}, stamps
    print("\n=== SANITY CHECK: scope_changed_at ===\n  GOOGL unchanged keeps its stamp; MRK's master rows changed and carry the new build time")


def test_co_registrant_issuer_lines_are_not_stored():
    lineage = _lineage(
        _row("DLR", "0001297996", "0001297996", "cik_window", sources="roster"),
        _row("DLR", "0001297996", "0001494877", "cik_event", sources="form345"),
        _row("DLR", "0001297996", "0001494877", "symbol", "DLRLP", "2010-01-04"),
        _row("DLR", "0001297996", "0001297996", "symbol", "DLR", "2006-01-04"),
    )
    obs = pd.concat(
        [
            _obs("253868103", "DLR", "DIGITAL RLTY TR INC COM", "2015-01-07", "2015-06-24"),
            _obs("25389JAA4", "DLRLP", "DIGITAL REALTY LP UNIT", "2015-01-07", "2015-06-24"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, lineage, _roster(("DLR", "0001297996")), co_registrant_ciks=frozenset({"0001494877"}))
    assert set(build.rows["cusip"]) == {"253868103"}
    print("\n=== SANITY CHECK: D-Q2-1 co-registrants ===\n  the operating partnership's line is not stored; the REIT's common is canonical")


# --------------------------------------------------------------------------- identity accessor


def _identity(lineage: pd.DataFrame, roster_pair: tuple[str, str], master: pd.DataFrame):
    tenure = pd.DataFrame(
        {
            "symbol": [roster_pair[0]],
            "issuer_cik": [roster_pair[1]],
            "valid_from": [pd.Timestamp("2006-01-04")],
            "valid_to": [pd.NaT],
            "n_filings": [1],
            "source": ["form345"],
        }
    )
    return build_identity(lineage=lineage, tenure=tenure, roster=_roster(roster_pair), master=master)


def test_security_on_answers_by_cusip_or_symbol_and_none_outside():
    build = _derive(GOOGL_OBS, GOOGL, _roster(("GOOGL", "0001652044")))
    identity = _identity(GOOGL, ("GOOGL", "0001652044"), build.rows)
    hit = identity.security_on(symbol="GOOG", source="ftd", day=pd.Timestamp("2013-06-03"))
    assert hit is not None and (hit.security_id, hit.lineage_role, hit.canonical_company) == ("C38259P508", "canonical_predecessor", "GOOGL")
    later = identity.security_on(symbol="GOOG", source="ftd", day=pd.Timestamp("2014-06-02"))
    assert later is not None and (later.security_id, later.security_class) == ("C38259P706", "class_C")
    assert identity.security_on(cusip="38259P508", source="ftd", day=pd.Timestamp("2001-01-02")) is None
    assert identity.security_on(symbol="NOPE", source="ftd", day=pd.Timestamp("2013-06-03")) is None
    print("\n=== SANITY CHECK: Identity.security_on ===")
    print("  GOOG -> class A 38259P508 in 2013, class C 38259P706 in 2014; None before the first observation or for an unknown symbol")


# --------------------------------------------------------------------------- configs and snapshot


def test_manual_entries_require_a_source():
    with pytest.raises(ValueError, match="source"):
        sm.parse_security_manual({"market_boundaries": [{"ticker": "X", "cusip": "123456789", "issuer_cik": "1", "role": "canonical_current"}]})
    print("\n=== SANITY CHECK: manual config ===\n  an entry with no URL or accession is refused at load")


def test_real_manual_config_is_evidenced_and_holds_the_brk_ratio():
    manual = sm.load_security_manual(str(CONFIG_DIR))
    brk = manual.ratios[manual.ratios["cusip"].eq("084670108")].sort_values("ratio")
    assert brk["ratio"].tolist() == [30.0, 1500.0]
    assert brk["valid_to"].iloc[0] == pd.Timestamp("2010-01-21") == brk["valid_from"].iloc[1]
    boundaries = manual.boundaries.set_index("cusip")
    assert {"G27823106", "65249B109", "30231G102", "30233Q108", "35137L105", "38259P508", "165167107", "165167743", "165167735"} <= set(
        boundaries.index
    )
    mergers = {entry["ticker"]: entry for entry in manual.mergers}
    assert mergers["MRK"]["acquired_symbol"] == "SGP" and "PLD" not in mergers  # PLD: the CIK window decides (traded view)
    assert {(x.ticker, x.ratio) for x in manual.exchanges} == {("LIN", 1.0), ("EVRG", 1.0), ("BKR", 1.0), ("STE", 1.0), ("JCI", 1.0)}
    assert [(s.ticker, s.vendor_ticker) for s in manual.vendor_series] == [("JCI", "TYC")]
    for frame in (manual.ratios, manual.boundaries):
        assert frame["source"].astype(str).str.len().gt(0).all()
    assert all(entry.get("source") for entry in manual.mergers)
    print("\n=== SANITY CHECK: configs/sec/security_master_manual.json ===")
    print(f"  BRK-A 30 -> 1,500 at 2010-01-21; {len(manual.boundaries)} market boundaries; every entry sourced")
    print(
        "  MRK merger metadata only (PLD's left with the traded-security view); exchange ratios LIN/EVRG/BKR/STE/JCI (PLD/DD removed); vendor series JCI <- TYC"
    )


def test_vendor_exceptions_and_expected_changes_configs():
    rows = json.loads((CONFIG_DIR / "sec" / "vendor_coverage_exceptions.json").read_text(encoding="utf-8"))["exceptions"]
    assert [(r["ticker"], r["quarter"], r["accession"]) for r in rows] == [
        (e.ticker, e.quarter, e.accession) for e in load_vendor_exceptions(CONFIG_DIR)
    ]
    expected = json.loads((CONFIG_DIR / "sec" / "expected_lineage_changes.json").read_text(encoding="utf-8"))["hypotheses"]
    ids = {row["id"] for row in expected}
    assert {"dd_predecessor_periods", "mrvl_predecessor_periods", "ferg_predecessor_periods", "tyco_window_filter", "seam_rule_set_aside"} <= ids
    assert all(row.get("status") == "hypothesis" and row.get("source") for row in expected)
    print("\n=== SANITY CHECK: Q2g/Q2h configs created ===")
    print(f"  {len(rows)} vendor exceptions read back by the shared loader; {len(expected)} lineage-change hypotheses recorded")


def test_company_tickers_exchange_snapshot_parses():
    payload = {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [
            [1067983, "BERKSHIRE HATHAWAY INC", "BRK-B", "NYSE"],
            [1067983, "BERKSHIRE HATHAWAY INC", "BRK-A", "NYSE"],
            [320193, "Apple Inc.", "AAPL", "Nasdaq"],
        ],
    }
    frame = parse_company_tickers_exchange(payload, fetched_at=BUILT_AT)
    assert frame.columns.tolist() == ["cik", "ticker", "name", "exchange", "fetched_at"]
    assert frame[["cik", "ticker", "exchange"]].values.tolist() == [
        ["0000320193", "AAPL", "Nasdaq"],
        ["0001067983", "BRK-A", "NYSE"],
        ["0001067983", "BRK-B", "NYSE"],
    ]
    print("\n=== SANITY CHECK: sec_company_tickers snapshot ===\n  CIK padded, one row per (cik, ticker), sorted, stamped")


# --------------------------------------------------------------------------- cases found on the real build


def test_seam_lag_vote_on_a_reused_symbol_is_ignored():
    hwm = _lineage(
        _row("HWM", "0000004281", "0000004281", "cik_window", sources="roster"),
        _row("HWM", "0000004281", "0000004281", "symbol", "AA", "2006-01-04", "2016-11-01", status="curated", sources="manual"),
        _row("HWM", "0000004281", "0000004281", "symbol", "ARNC", "2016-11-01", status="curated", sources="manual"),
    )
    obs = pd.concat(
        [
            _obs("013817101", "AA", "ALCOA INC", "2015-01-07", "2016-10-26"),
            _obs("03965L100", "ARNC", "ARCONIC INC", "2016-11-02", "2018-06-27"),
            _obs("013872106", "AA", "ALCOA CORP", "2016-11-02", "2018-06-27"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, hwm, _roster(("HWM", "0000004281")))
    assert _rows(build, "013872106").empty, "new Alcoa's AA is not HWM's"
    assert build.flags.loc[build.flags["kind"].eq("security_weak_vote"), "cusip"].tolist() == ["013872106"]
    assert _canonical_overlaps(build, "HWM") == []
    print("\n=== SANITY CHECK: seam-lag vote ===")
    print("  HWM's AA interval votes only on the first days of the spun-off Alcoa CUSIP: ignored and flagged, not attributed")


def test_company_line_before_its_register_window_stays_canonical():
    mdt = _lineage(
        _row("MDT", "0000064670", "0000064670", "cik_window", vt="2015-02-28"),
        _row("MDT", "0000064670", "0001613103", "cik_window", vf="2015-02-28"),
        _row("MDT", "0000064670", "0000064670", "symbol", "MDT", "2006-01-03", "2015-01-30", status="single_source"),
        _row("MDT", "0000064670", "0001613103", "symbol", "MDT", "2014-07-14"),
    )
    obs = pd.concat(
        [
            _obs("585055106", "MDT", "MEDTRONIC INC", "2014-06-04", "2015-01-28"),
            _obs("G5960L103", "MDT", "MEDTRONIC PLC SHS", "2015-01-28", "2016-06-29"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, mdt, _roster(("MDT", "0001613103")))
    plc = _rows(build, "G5960L103").sort_values("valid_from")
    canonical = plc[plc["lineage_role"].eq("canonical_current")]
    assert canonical["valid_from"].min() < pd.Timestamp("2015-02-28"), plc
    assert set(plc["lineage_role"]) <= {"canonical_current", "acquired_constituent"}
    assert _rows(build, "585055106")["lineage_role"].tolist() == ["canonical_predecessor"]
    assert _canonical_overlaps(build, "MDT") == []
    print("\n=== SANITY CHECK: register window after the market seam ===")
    print(f"  Medtronic plc canonical from {canonical['valid_from'].min().date()}, before its 2015-02-28 register window; no overlap with old MDT")


def test_a_one_day_new_line_at_the_data_end_does_not_supersede_the_current_line():
    oke = _lineage(
        _row("OKE", "0001039684", "0001039684", "cik_window", sources="roster"),
        _row("OKE", "0001039684", "0001039684", "symbol", "OKE", "2006-01-10", sources="dei,form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("682680103", "OKE", "ONEOK INC NEW", "2015-01-07", "2016-06-29"),
            _obs("30609A109", "OKE", "FALCON TO ONEOK", "2016-06-29", "2016-06-29"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, oke, _roster(("OKE", "0001039684")))
    main = _rows(build, "682680103").sort_values("valid_from")
    assert main["lineage_role"].iloc[0] == "canonical_current" and main["n_observations"].iloc[0] > 50
    print("\n=== SANITY CHECK: open-ended spans ===\n  the sibling test measures overlap up to the last observation, not to an open end")


def test_a_former_listing_under_the_company_ticker_is_not_a_co_registrant():
    lineage = _lineage(
        _row("GM", "0000040730", "0001467858", "cik_window", sources="roster"),
        _row("GM", "0000040730", "0000040730", "cik_event", sources="form345"),
        _row("GM", "0000040730", "0000040730", "symbol", "GM", "2006-01-03", "2009-06-06", status="single_source"),
        _row("GM", "0000040730", "0001467858", "symbol", "GM", "2010-11-24"),
        _row("PCG", "0001004980", "0001004980", "cik_window", sources="roster"),
        _row("PCG", "0001004980", "0000075488", "cik_event", sources="form345"),
        _row("PCG", "0001004980", "0001004980", "symbol", "PCG", "2006-01-03"),
    )
    lineage["oracle"] = ["roster", "owner_overlap", "owner_overlap", "roster", "roster", "owner_overlap", "roster"]
    evidence = pd.DataFrame(
        [
            ("0000040730", "form345", "2006-01-03", "2019-06-05"),
            ("0001467858", "form345", "2010-11-24", None),
            ("0000075488", "form345", "2006-01-03", None),
            ("0001004980", "form345", "2006-01-03", None),
        ],
        columns=["issuer_cik", "source", "valid_from", "valid_to"],
    )
    assert sm.co_registrant_ciks(lineage, evidence) == frozenset({"0000075488"})
    print(
        "\n=== SANITY CHECK: co-registrants ===\n  the PG&E utility is a co-registrant (not stored); old GM traded as GM, so it stays an acquired constituent"
    )


def test_a_class_suffixed_lineage_symbol_is_class_evidence():
    cmg = _lineage(
        _row("CMG", "0001058090", "0001058090", "cik_window", sources="roster"),
        _row("CMG", "0001058090", "0001058090", "symbol", "CMG", "2006-01-25", sources="dei,form345,roster"),
        _row("CMG", "0001058090", "0001058090", "symbol", "CMG-B", "2006-10-10", "2009-12-17", status="single_source"),
    )
    obs = pd.concat(
        [
            _obs("169656105", "CMG", "CHIPOLTE MEXICAN GRILL, INC. C", "2009-07-01", "2009-12-23"),
            _obs("169656204", "CMGB", "CHIPOTLE MEXICAN GRILL, INC. C", "2009-07-07", "2009-12-16"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, cmg, _roster(("CMG", "0001058090")))
    assert _rows(build, "169656204")[["lineage_role", "security_class", "lineage_reason"]].values.tolist() == [
        ["secondary_class", "class_B", "lineage_class_symbol"]
    ]
    print(
        "\n=== SANITY CHECK: class from the Form 3/4/5 symbol ===\n  CMGB's truncated description names no class; the filed symbol CMG-B does: secondary class B"
    )


# --------------------------------------------------------------------------- Q2d: trading evidence and the WBD seam

STZ = _lineage(
    _row("STZ", "0000016918", "0000016918", "cik_window", sources="roster"),
    _row("STZ", "0000016918", "0000016918", "symbol", "STZ", "2006-01-03", sources="dei,form345,roster"),
)
STZ_OBS = pd.concat(
    [
        _obs("21036P108", "STZ", "CONSTELLATION BRANDS INC CL A", "2019-01-02", "2022-12-28"),
        _obs("21036P207", "STZB", "CONSTELLATION BRANDS INC CL B", "2019-01-09", "2020-02-07"),
    ],
    ignore_index=True,
)


def _finra(symbol: str, start: str, end: str, freq: str = "W-FRI") -> pd.DataFrame:
    return pd.DataFrame({"source_symbol": symbol, "date": pd.date_range(start, end, freq=freq)})


def test_a_thin_secondary_class_runs_over_its_finra_trading_days():
    presence = pd.concat(
        [_finra("STZ/B", "2020-02-07", "2022-10-14"), _finra("STZ/B", "2023-06-02", "2024-06-28"), _finra("STZ", "2019-01-04", "2022-12-30")],
        ignore_index=True,
    )
    without = _rows(_derive(STZ_OBS, STZ, _roster(("STZ", "0000016918"))), "21036P207")
    build = _derive(STZ_OBS, STZ, _roster(("STZ", "0000016918")), finra_presence=presence)
    stzb = _rows(build, "21036P207")
    assert without["valid_to"].max() == pd.Timestamp("2020-02-06")
    assert stzb[["lineage_role", "security_class"]].drop_duplicates().values.tolist() == [["secondary_class", "class_B"]]
    assert stzb["valid_to"].max() == pd.Timestamp("2022-10-15"), stzb
    assert stzb["evidence"].str.contains("FINRA to 2022-10-14").all()
    assert _canonical_overlaps(build, "STZ") == []
    print("\n=== SANITY CHECK: thin secondary class (STZ/B) ===")
    print("  last FTD fail 2020-02-05 -> the class line runs to its last FINRA day before an 8-month gap (2022-10-14)")


def test_trading_evidence_never_extends_an_unclassed_line_or_crosses_the_next_cusip():
    hwm = _lineage(
        _row("HWM", "0000004281", "0000004281", "cik_window", sources="roster"),
        _row("HWM", "0000004281", "0000004281", "symbol", "AA", "2006-01-03", "2016-11-01"),
        _row("HWM", "0000004281", "0000004281", "symbol", "HWM", "2020-04-01", sources="dei,form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("013817101", "AA", "ALCOA INC", "2016-01-06", "2016-10-26"),
            _obs("013817101", "ARNC", "ARCONIC INC", "2016-11-02", "2020-03-25"),
            _obs("443201108", "HWM", "HOWMET AEROSPACE INC", "2020-04-08", "2022-12-28"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, hwm, _roster(("HWM", "0000004281")), finra_presence=_finra("AA", "2016-11-04", "2022-12-30"))
    aa = _rows(build, "013817101", "AA")
    assert aa["valid_to"].max() <= pd.Timestamp("2016-11-02"), aa
    goog = _derive(GOOGL_OBS, GOOGL, _roster(("GOOGL", "0001652044")), finra_presence=_finra("GOOG", "2013-01-04", "2016-06-24", freq="B"))
    class_c = _rows(goog, "38259P706")
    successor = _rows(goog, "02079K107")["valid_from"].min()
    assert class_c["valid_to"].max() <= successor, class_c
    print("\n=== SANITY CHECK: trading evidence is bounded ===")
    print("  AA (no class evidence) is not extended over new Alcoa's FINRA days; Google class C stops at Alphabet's CUSIP")


def test_a_current_sec_listing_keeps_a_class_line_open():
    brk = BRK.copy()
    obs = pd.concat([BRK_OBS, _obs("084670702", "BRKB", "BERKSHIRE HATHWY INC(HLDG CO)B", "2011-07-06", "2013-12-30", price=110.0)])
    listing = pd.DataFrame({"cik": ["0001067983", "0001067983"], "ticker": ["BRK-A", "BRK-B"], "exchange": ["NYSE", "NYSE"]})
    build = _derive(obs, brk, _roster(("BRK-B", "0001067983")), BRK_MANUAL, sec_tickers=listing)
    brka = _rows(build, "084670108").sort_values("valid_from")
    assert brka["lineage_role"].eq("secondary_class").all() and pd.isna(brka["valid_to"].iloc[-1]), brka
    assert brka["market_symbol"].iloc[-1] == "BRK-A"
    print("\n=== SANITY CHECK: listed class ===\n  BRK-A is in the SEC current-tickers snapshot, so its line stays open past its last FTD fail")


def test_a_current_sec_listing_keeps_only_the_latest_cusip_of_its_symbol_open():
    obs = pd.concat([BRK_OBS, _obs("084670702", "BRKB", "BERKSHIRE HATHWY INC(HLDG CO)B", "2011-07-06", "2013-12-30", price=110.0)])
    listing = pd.DataFrame({"cik": ["0001067983", "0001067983"], "ticker": ["BRK-A", "BRK-B"], "exchange": ["NYSE", "NYSE"]})
    build = _derive(obs, BRK, _roster(("BRK-B", "0001067983")), BRK_MANUAL, sec_tickers=listing)
    old = _rows(build, "084670207", "BRKB")
    new_start = _rows(build, "084670702", "BRKB")["valid_from"].min()
    assert old["valid_to"].notna().all() and old["valid_to"].max() <= new_start, old
    assert old["lineage_role"].eq("canonical_current").all(), old
    assert _canonical_overlaps(build, "BRK-B") == []
    print("\n=== SANITY CHECK: listing opens the latest line only ===")
    print("  BRK-B is listed today, so its current CUSIP stays open; the pre-split CUSIP of the same symbol ends where the new one starts")


def test_a_superseded_cusip_that_still_fails_or_is_listed_does_not_stay_open():
    aon = _lineage(
        _row("AON", "0000315293", "0000315293", "cik_window", sources="roster"),
        _row("AON", "0000315293", "0000315293", "symbol", "AON", "2006-01-03", sources="form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("G0408V102", "AON", "AON PLC CL A", "2012-04-04", "2020-04-01"),  # fails on two days past the new CUSIP's first
            _obs("G0403H108", "AON", "AON PLC CL A", "2020-03-30", "2026-09-30"),
        ],
        ignore_index=True,
    )
    listing = pd.DataFrame({"cik": ["0000315293"], "ticker": ["AON"], "exchange": ["NYSE"]})
    for kw in ({"sec_tickers": listing}, {}):
        build = _derive(obs, aon, _roster(("AON", "0000315293")), **kw)
        old = _rows(build, "G0408V102", "AON")
        assert old["valid_to"].notna().all(), (kw, old)
        open_rows = build.rows[build.rows["source_symbol"].eq("AON") & build.rows["valid_to"].isna()]
        assert open_rows["cusip"].tolist() == ["G0403H108"], open_rows
    # still failing in the latest periods: the old CUSIP of a symbol changed in the last period ends too
    late = pd.concat(
        [_obs("682680103", "OKE", "ONEOK INC", "2009-07-01", "2026-09-30"), _obs("30609A109", "OKE", "ONEOK INC", "2026-09-16", "2026-09-30")],
        ignore_index=True,
    )
    oke = _lineage(
        _row("OKE", "0001039684", "0001039684", "cik_window", sources="roster"),
        _row("OKE", "0001039684", "0001039684", "symbol", "OKE", "2006-01-03"),
    )
    build = _derive(late, oke, _roster(("OKE", "0001039684")))
    assert build.rows[build.rows["source_symbol"].eq("OKE") & build.rows["valid_to"].isna()]["cusip"].tolist() == ["30609A109"]
    print("\n=== SANITY CHECK: one open line per symbol ===")
    print(
        "  AON's pre-redomicile CUSIP (fails two days past the new one's first, symbol listed) and ONEOK's pre-change CUSIP (still failing) end; only the latest CUSIP stays open"
    )


def test_wbd_seam_discovery_stays_canonical_until_wbd_first_trades():
    manual = sm.load_security_manual(str(CONFIG_DIR))
    wbd = _lineage(
        _row("WBD", "0001437107", "0001437107", "cik_window", sources="roster"),
        _row("WBD", "0001437107", "0001437107", "symbol", "DISCA", "2008-09-18", "2022-04-09"),
        _row("WBD", "0001437107", "0001437107", "symbol", "DISCK", "2008-09-18", "2022-04-09"),
        _row("WBD", "0001437107", "0001437107", "symbol", "WBD", "2022-04-11", sources="dei,form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("25470F104", "DISCA", "DISCOVERY INC COM SER A", "2021-06-02", "2022-04-12", freq="B"),
            _obs("25470F302", "DISCK", "DISCOVERY INC COM SER C", "2021-06-02", "2022-04-12", freq="B"),
            _obs("934423104", "WBD", "WARNER BROS DISCOVERY INC COM", "2022-04-11", "2022-12-28", freq="B"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, wbd, _roster(("WBD", "0001437107")), manual)
    canonical = build.rows[build.rows["lineage_role"].eq("canonical_current")].sort_values("valid_from")
    disca = canonical[canonical["cusip"].eq("25470F104")]
    wbd_rows = _rows(build, "934423104").sort_values("valid_from")
    assert disca["valid_to"].max() == pd.Timestamp("2022-04-11"), canonical
    assert wbd_rows[wbd_rows["lineage_role"].eq("canonical_current")]["valid_from"].min() == pd.Timestamp("2022-04-11"), wbd_rows
    assert wbd_rows[wbd_rows["valid_from"].lt(pd.Timestamp("2022-04-11"))]["lineage_role"].eq("excluded").all()
    assert "0001193125-22-103051" in set(disca["source_accession"]) | set(wbd_rows["source_accession"])
    assert _canonical_overlaps(build, "WBD") == []
    print("\n=== SANITY CHECK: WBD seam (8-K 0001193125-22-103051) ===")
    print("  DISCA canonical to 2022-04-11 (it traded through 2022-04-08); WBD canonical from its first trading day 2022-04-11")


def test_build_reads_finra_trading_days_through_the_store(sqlite_store, tmp_path):
    obs = STZ_OBS.assign(trade_date=sm.trade_dates(STZ_OBS["date"]), fails_quantity=1000.0)
    sqlite_store.save(
        Tables.sec_fails_to_deliver_security,
        obs[["date", "trade_date", "cusip", "source_symbol", "description", "price", "period", "fails_quantity"]],
    )
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["STZ"], "cik": ["0000016918"]}))
    finra = _finra("STZ/B", "2020-02-07", "2022-10-14").assign(market="N", short_volume=1.0, short_exempt_volume=0.0, total_volume=2.0)
    sqlite_store.save(Tables.sec_short_volume_security, finra)
    context = SimpleNamespace(
        store=sqlite_store, paths={"DATA_STORE": tmp_path}, log=logging.getLogger("test.master"), config_dir=str(CONFIG_DIR), config=extract_config()
    )
    rows = sm.build_security_master(context, STZ, str(CONFIG_DIR), built_at=BUILT_AT)
    stzb = rows[rows["cusip"].eq("21036P207")]
    assert stzb["valid_to"].max() == pd.Timestamp("2022-10-15"), stzb
    print("\n=== SANITY CHECK: build wiring ===\n  the stored FINRA rows of STZ/B reach the derivation: the class line runs to 2022-10-14")


def test_trading_evidence_stops_at_a_successor_cusip_seen_on_the_same_day():
    aon = _lineage(
        _row("AON", "0000315293", "0000315293", "cik_window", sources="roster"),
        _row("AON", "0000315293", "0000315293", "symbol", "AON", "2006-01-03", sources="dei,form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("G0408V102", "AON", "AON PLC CL A", "2019-01-02", "2020-04-01"),
            _obs("G0403H108", "AON", "AON PLC CL A", "2020-04-01", "2021-06-30"),
        ],
        ignore_index=True,
    )
    build = _derive(obs, aon, _roster(("AON", "0000315293")), finra_presence=_finra("AON", "2019-01-04", "2021-06-25"))
    old = _rows(build, "G0408V102")
    assert old["valid_to"].max() <= pd.Timestamp("2020-04-01"), old
    assert not old["evidence"].str.contains("FINRA to").any()
    assert _canonical_overlaps(build, "AON") == []
    print(
        "\n=== SANITY CHECK: successor on the seam day ===\n  the old Aon line is not extended over FINRA days its successor CUSIP already trades on"
    )


def test_a_renamed_class_line_starts_at_its_first_finra_day():
    psky = _lineage(
        _row("PSKY", "0000813828", "0000813828", "cik_window", sources="roster"),
        _row("PSKY", "0000813828", "0000813828", "symbol", "VIACA", "2019-12-05", "2022-02-16"),
        _row("PSKY", "0000813828", "0000813828", "symbol", "PARAA", "2022-02-16"),
        _row("PSKY", "0000813828", "0000813828", "symbol", "PARA", "2022-02-16", sources="dei,form345,roster"),
        _row("PSKY", "0000813828", "0000813828", "symbol", "VIAC", "2019-12-05", "2022-02-16"),
    )
    obs = pd.concat(
        [
            _obs("92556H107", "VIACA", "VIACOMCBS INC CL A", "2020-01-08", "2020-03-11"),
            _obs("92556H107", "PARAA", "PARAMOUNT GLOBAL CL A", "2023-07-12", "2023-09-13"),
            _obs("92556H206", "VIAC", "VIACOMCBS INC CL B", "2020-01-08", "2022-02-14", freq="B"),
            _obs("92556H206", "PARA", "PARAMOUNT GLOBAL CL B", "2022-02-16", "2024-06-26", freq="B"),
        ],
        ignore_index=True,
    )
    presence = pd.concat([_finra("VIACA", "2020-01-10", "2022-02-11"), _finra("PARAA", "2022-02-18", "2024-06-21")], ignore_index=True)
    build = _derive(obs, psky, _roster(("PSKY", "0000813828")), finra_presence=presence)
    paraa = _rows(build, "92556H107", "PARAA")
    viaca = _rows(build, "92556H107", "VIACA")
    assert paraa["valid_from"].min() == pd.Timestamp("2022-02-18") and paraa["valid_to"].max() == pd.Timestamp("2024-06-22"), paraa
    assert viaca["valid_to"].max() == pd.Timestamp("2022-02-12"), viaca
    print("\n=== SANITY CHECK: renamed class line ===")
    print("  VIACA runs to its last FINRA day; PARAA starts at its first FINRA day, not at its first fail 17 months later")


def test_an_sec_listing_is_class_evidence_for_a_class_spelling_only():
    lineage = _lineage(
        _row("FITB", "0000035527", "0000035527", "cik_window", sources="roster"),
        _row("FITB", "0000035527", "0000035527", "symbol", "FITB", "2006-01-03", sources="dei,form345,roster"),
        _row("MKC", "0000063754", "0000063754", "cik_window", sources="roster"),
        _row("MKC", "0000063754", "0000063754", "symbol", "MKC", "2006-01-03", sources="dei,form345,roster"),
    )
    obs = pd.concat(
        [
            _obs("316773100", "FITB", "FIFTH THIRD BANCORP", "2024-01-03", "2026-09-30"),
            _obs("316773852", "FITBP", "FIFTH THIRD BANCORP", "2024-01-10", "2024-03-27"),
            _obs("579780206", "MKC", "MCCORMICK & CO INC", "2024-01-03", "2026-09-30"),
            _obs("579780107", "MKCV", "MCCORMICK & CO INC", "2024-01-10", "2024-03-27"),
        ],
        ignore_index=True,
    )
    listing = pd.DataFrame(
        {"cik": ["0000035527", "0000035527", "0000063754", "0000063754"], "ticker": ["FITB", "FITBP", "MKC", "MKC-V"], "exchange": "NYSE"}
    )
    build = _derive(obs, lineage, _roster(("FITB", "0000035527"), ("MKC", "0000063754")), sec_tickers=listing)
    fitbp, mkcv = _rows(build, "316773852"), _rows(build, "579780107")
    assert set(fitbp["lineage_role"]) == {"excluded"} and set(fitbp["lineage_reason"]) == {"unclassified"}, fitbp
    assert mkcv[["lineage_role", "lineage_reason"]].drop_duplicates().values.tolist() == [["secondary_class", "sec_tickers_listing"]], mkcv
    assert mkcv["market_symbol"].iloc[0] == "MKC-V"
    print("\n=== SANITY CHECK: SEC listing as class evidence ===")
    print("  MKC-V (class spelling) is a secondary class; FITBP (a preferred the SEC file lists without a marker) stays excluded")


# --------------------------------------------------------------------------- Q2n: common lines run back to their listing

VRT = _lineage(
    _row("VRT", "0001674101", "0001674101", "cik_window", sources="roster"),
    _row("VRT", "0001674101", "0001674101", "symbol", "GSAH", "2018-06-07", "2019-11-06", sources="dei,form345"),
    _row("VRT", "0001674101", "0001674101", "symbol", "VRT", "2020-02-11", sources="dei,form345,roster"),
)
VRT_OBS = pd.concat(
    [
        _obs("36255F102", "GSAH", "GS ACQUISITION HLDGS CORP", "2018-08-01", "2020-01-31"),
        _obs("92537N108", "VRT", "VERTIV HLDG CO CL A (DE)", "2021-01-22", "2022-12-28"),
    ],
    ignore_index=True,
)
EG = _lineage(
    _row("EG", "0001095073", "0001095073", "cik_window", sources="roster"),
    _row("EG", "0001095073", "0001095073", "symbol", "RE", "2006-01-03", "2023-07-10", sources="manual"),
    _row("EG", "0001095073", "0001095073", "symbol", "EG", "2023-07-10", sources="manual,roster"),
)
EG_OBS = pd.concat(
    [
        _obs("G3223R108", "RE", "EVEREST RE GP LTD(BERM)HLDG CO", "2022-01-05", "2023-07-03"),
        _obs("G3223R108", "EG", "EVEREST GROUP LTD SHS (BMU)", "2023-07-21", "2024-06-26"),
    ],
    ignore_index=True,
)


def test_a_canonical_line_runs_back_over_its_finra_days_to_its_listing():
    presence = pd.concat(
        [_finra("VRT", "2020-02-07", "2022-12-30", freq="B"), _finra("GSAH", "2018-06-11", "2020-02-06", freq="B")], ignore_index=True
    )
    build = _derive(VRT_OBS, VRT, _roster(("VRT", "0001674101")), finra_presence=presence)
    vrt = _rows(build, "92537N108").sort_values("valid_from")
    assert vrt["lineage_role"].eq("canonical_current").all(), vrt
    assert vrt["valid_from"].min() == pd.Timestamp("2020-02-07"), vrt
    assert vrt["evidence"].str.contains("FINRA from 2020-02-07").all()
    assert _canonical_overlaps(build, "VRT") == []
    eg = _derive(
        EG_OBS,
        EG,
        _roster(("EG", "0001095073")),
        finra_presence=pd.concat([_finra("RE", "2022-01-03", "2023-07-07", freq="B"), _finra("EG", "2023-07-10", "2024-06-28", freq="B")]),
    )
    renamed = _rows(eg, "G3223R108", "EG")
    assert renamed["lineage_role"].eq("canonical_current").all() and renamed["valid_from"].min() == pd.Timestamp("2023-07-10"), renamed
    assert _canonical_overlaps(eg, "EG") == []
    print("\n=== SANITY CHECK: canonical lines from their listing ===")
    print("  VRT (first fail 2021-01-20) is canonical from its first FINRA day 2020-02-07; EG (renamed RE, first fail 2023-07-19) from 2023-07-10")


def test_a_canonical_line_never_runs_back_into_another_use_of_its_symbol():
    alcoa = _lineage(
        _row("AA", "0001675149", "0001675149", "cik_window", sources="roster"),
        _row("AA", "0001675149", "0001675149", "symbol", "AA", "2016-11-01", sources="dei,form345,roster"),
    )
    presence = _finra("AA", "2016-01-04", "2017-12-29", freq="B")
    old_issuer = _obs("013817101", "AA", "ALCOA INC", "2016-01-06", "2016-10-26")  # old Alcoa: another issuer, not in the lineage
    build = _derive(
        pd.concat([old_issuer, _obs("013872106", "AA", "ALCOA CORP", "2016-11-16", "2017-12-27")]),
        alcoa,
        _roster(("AA", "0001675149")),
        finra_presence=presence,
    )
    assert _rows(build, "013872106")["valid_from"].min() == pd.Timestamp("2016-11-01"), _rows(build, "013872106")
    seam = pd.concat(
        [_obs("013817101", "AA", "ALCOA INC", "2016-01-06", "2016-11-16"), _obs("013872106", "AA", "ALCOA CORP", "2016-11-17", "2017-12-27")]
    )
    tight = _rows(_derive(seam, alcoa, _roster(("AA", "0001675149")), finra_presence=presence), "013872106")
    assert tight["valid_from"].min() == pd.Timestamp("2016-11-14") and not tight["evidence"].str.contains("FINRA from").any(), tight
    no_interval = _derive(EG_OBS, EG.iloc[:2], _roster(("EG", "0001095073")), finra_presence=_finra("EG", "2023-07-10", "2024-06-28", freq="B"))
    assert _rows(no_interval, "G3223R108", "EG")["valid_from"].min() == pd.Timestamp("2023-07-19")
    manual = sm.parse_security_manual(
        {
            "market_boundaries": [
                {
                    "ticker": "VRT",
                    "cusip": "92537N108",
                    "issuer_cik": "0001674101",
                    "role": "canonical_current",
                    "valid_from": "2020-06-01",
                    "source": "test",
                }
            ]
        }
    )
    bounded = _rows(
        _derive(VRT_OBS, VRT, _roster(("VRT", "0001674101")), manual, finra_presence=_finra("VRT", "2020-02-07", "2022-12-30", freq="B")), "92537N108"
    )
    assert bounded["valid_from"].min() == pd.Timestamp("2020-06-01"), bounded
    print("\n=== SANITY CHECK: the run back is bounded ===")
    print(
        "  new Alcoa stops at its own AA interval (2016-11-01), not over old Alcoa's last days; no run back when the old CUSIP fails the day before;"
        " none without a lineage interval of the symbol; a manual start bounds it"
    )


def test_build_reads_common_symbol_days_through_the_store(sqlite_store, tmp_path):
    obs = EG_OBS.assign(trade_date=sm.trade_dates(EG_OBS["date"]), fails_quantity=1000.0)
    sqlite_store.save(
        Tables.sec_fails_to_deliver_security,
        obs[["date", "trade_date", "cusip", "source_symbol", "description", "price", "period", "fails_quantity"]],
    )
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["EG"], "cik": ["0001095073"]}))
    finra = _finra("EG", "2023-07-10", "2024-06-28", freq="B").assign(market="N", short_volume=1.0, short_exempt_volume=0.0, total_volume=2.0)
    sqlite_store.save(Tables.sec_short_volume_security, finra)
    context = SimpleNamespace(
        store=sqlite_store, paths={"DATA_STORE": tmp_path}, log=logging.getLogger("test.master"), config_dir=str(CONFIG_DIR), config=extract_config()
    )
    rows = sm.build_security_master(context, EG, str(CONFIG_DIR), built_at=BUILT_AT)
    assert rows[rows["source_symbol"].eq("EG")]["valid_from"].min() == pd.Timestamp("2023-07-10")
    print("\n=== SANITY CHECK: build wiring ===\n  the stored FINRA rows of a common symbol (EG) reach the derivation: the line starts at 2023-07-10")


def test_f107_a_line_run_back_ends_the_bridge_of_the_earlier_cusip_of_its_symbol():
    """STE's redomicile: the old CUSIP's last fail is bridged to the new CUSIP's first; the new line runs back over the
    FINRA days after the old CUSIP's last fail, so the bridge ends where the run back starts (no day under two lines)."""
    cik = "0001757898"
    steris = _lineage(
        _row("STE", cik, cik, "cik_window", sources="roster"), _row("STE", cik, cik, "symbol", "STE", "2015-01-02", sources="dei,form345,roster")
    )
    obs = pd.concat(
        [
            _obs("G84720104", "STE", "STERIS PLC ORD SHS", "2018-06-06", "2019-03-06"),
            _obs("G8473T100", "STE", "STERIS PLC ORD SHS", "2019-04-10", "2020-06-24"),
        ]
    )
    build = _derive(obs, steris, _roster(("STE", cik)), finra_presence=_finra("STE", "2018-06-01", "2020-06-30", freq="B"))
    rows = build.rows[build.rows["source_symbol"].eq("STE")].assign(vt=lambda f: pd.to_datetime(f["valid_to"]).fillna(pd.Timestamp("2262-01-01")))
    old, new = rows[rows["cusip"].eq("G84720104")], rows[rows["cusip"].eq("G8473T100")]
    assert new["valid_from"].min() < pd.Timestamp("2019-04-05"), "the new line runs back over the FINRA days"
    assert old["vt"].max() == new["valid_from"].min(), (old[["valid_from", "valid_to"]], new[["valid_from", "valid_to"]])
    overlaps = [(a, b) for a in old.itertuples() for b in new.itertuples() if max(a.valid_from, b.valid_from) < min(a.vt, b.vt)]
    assert overlaps == []
    print("\n=== SANITY CHECK: F-107 no overlap under one symbol ===")
    print(
        f"  old CUSIP ends {old['vt'].max().date()}, the new line starts {new['valid_from'].min().date()} (run back over FINRA days); no shared day"
    )


def test_reverse_acquisitions_declare_pld_and_jci_only():
    """REQ-014: the survivor's post-seam comparatives are the accounting acquirer's; DD (DowDuPont is a new registrant) is not declared."""
    manual = sm.load_security_manual(str(CONFIG_DIR))
    by_ticker = {entry.ticker: entry for entry in manual.reverse_acquisitions}
    assert set(by_ticker) == {"PLD", "JCI"}, sorted(by_ticker)
    assert all(isinstance(entry, sm.ReverseAcquisition) and entry.source for entry in by_ticker.values())
    pld, jci = by_ticker["PLD"], by_ticker["JCI"]
    assert (pld.seam_date, pld.accounting_acquirer_cik, pld.legal_acquirer_cik) == (pd.Timestamp("2011-06-03"), "0000899881", "0001045609")
    assert (jci.seam_date, jci.accounting_acquirer_cik, jci.legal_acquirer_cik) == (pd.Timestamp("2016-09-02"), "0000053669", "0000833444")
    with pytest.raises(ValueError, match="source"):
        sm.parse_security_manual(
            {"reverse_acquisitions": [{"ticker": "X", "seam_date": "2020-01-01", "accounting_acquirer_cik": "1", "legal_acquirer_cik": "2"}]}
        )
    assert sm.SecurityManual.empty().reverse_acquisitions == ()
    print("\n=== SANITY CHECK: reverse_acquisitions ===")
    print("  PLD 2011-06-03 (accounting acquirer old ProLogis 0000899881, survivor AMB 0001045609)")
    print("  JCI 2016-09-02 (accounting acquirer old JCI 0000053669, survivor Tyco 0000833444); DD undeclared; an unsourced entry is refused")
