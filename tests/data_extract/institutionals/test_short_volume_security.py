"""Short volume per security (Q2c): raw FINRA RegSHO rows stamped through `security_master`, the ticker table rebuilt.

The master rows are real (the Q2a build over the whole FTD cache) for BRK-B, BAC, GOOGL, LEN and DOC; the FINRA
lines copy the real symbology of the CNMSshvol files (`BRK/A`, `BACpB`, `LEN/B`, fractional volumes, `Market`).
Symbols never seen in FTD (E25) use a hand-written lineage with a roster CIK, a predecessor window CIK and an
event-only CIK.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.institutionals import fetch_short_interest as si
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

REPO = Path(__file__).resolve().parents[3]
BUILT = pd.Timestamp("2026-10-04 12:00:00")
#: The run date: LATER falls inside the 7-day re-check window before it, BUILT does not.
RUN_DATE = pd.Timestamp("2026-10-12")
LATER = pd.Timestamp("2026-10-06 09:00:00")
HEADER = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
UNIVERSE = ["BRK-B", "BAC", "GOOGL", "LEN", "DOC", "AAA"]

# Real `security_master` rows (Q2a build): cusip, company, issuer CIK, symbol, class, ratio, role, valid_from, valid_to, reason.
MASTER_ROWS = [
    ("084670108", "BRK-B", "0001067983", "BRKA", "class_A", 30.0, "secondary_class", "2009-06-26", "2010-01-21", "class_description"),
    ("084670108", "BRK-B", "0001067983", "BRKA", "class_A", 1500.0, "secondary_class", "2010-01-21", None, "class_description"),
    ("084670702", "BRK-B", "0001067983", "BRKB", "class_B", 1.0, "canonical_current", "2010-01-19", None, "ticker_symbol"),
    ("060505104", "BAC", "0000070858", "BAC", "common", 1.0, "canonical_current", "2009-06-26", None, "ticker_symbol"),
    ("060505229", "BAC", "0000070858", "BACPRB", "preferred", 1.0, "excluded", "2018-05-16", None, "preferred"),
    ("02079K107", "GOOGL", "0001652044", "GOOG", "class_C", 1.0, "secondary_class", "2015-10-01", None, "class_description"),
    ("02079K305", "GOOGL", "0001652044", "GOOGL", "class_A", 1.0, "canonical_current", "2015-10-01", None, "ticker_symbol"),
    ("526057104", "LEN", "0000920760", "LEN", "class_A", 1.0, "canonical_current", "2009-06-26", None, "ticker_symbol"),
    ("526057302", "LEN", "0000920760", "LENB", "class_B", 1.0, "secondary_class", "2009-06-26", None, "class_description"),
    ("42250P103", "DOC", "0000765880", "PEAK", "common", 1.0, "canonical_current", "2019-11-01", "2024-03-01", "ticker_symbol"),
    ("42250P103", "DOC", "0000765880", "DOC", "common", 1.0, "canonical_current", "2024-03-01", None, "ticker_symbol"),
    ("71943U104", "DOC", "0001574540", "DOC", "common", 1.0, "acquired_constituent", "2013-07-19", "2024-03-01", "event_only_cik"),
]
ROSTER = {
    "BRK-B": "0001067983",
    "BAC": "0000070858",
    "GOOGL": "0001652044",
    "LEN": "0000920760",
    "DOC": "0000765880",
    "AAA": "0000000100",
}
# ticker, cik, role, symbol, valid_from, valid_to (E25: AAA's predecessor window CIK 99 and event-only CIK 98).
LINEAGE_ROWS = [
    ("DOC", "0001574540", "cik_event", "", None, None),
    ("DOC", "0001574540", "symbol", "DOC", "2013-07-19", "2024-03-02"),
    ("DOC", "0000765880", "symbol", "PEAK", "2019-11-05", "2024-03-02"),
    ("DOC", "0000765880", "symbol", "DOC", "2024-03-04", None),
    ("AAA", "0000000099", "cik_window", "", None, "2021-01-01"),
    ("AAA", "0000000100", "cik_window", "", "2021-01-01", None),
    ("AAA", "0000000098", "cik_event", "", None, None),
    ("AAA", "0000000099", "symbol", "OLDA", "2015-01-02", "2021-01-08"),
    ("AAA", "0000000100", "symbol", "AAA", "2021-01-04", None),
    ("AAA", "0000000098", "symbol", "ACQ", "2015-01-02", None),
]


def _master(rows: list[tuple] = MASTER_ROWS, stamp: pd.Timestamp = BUILT) -> pd.DataFrame:
    columns = ["cusip", "canonical_company", "issuer_cik", "source_symbol", "security_class", "conversion_ratio", "lineage_role"]
    frame = pd.DataFrame(rows, columns=[*columns, "valid_from", "valid_to", "lineage_reason"])
    frame = frame.assign(
        security_id="C" + frame["cusip"],
        source=sm.SOURCE_FTD,
        market_symbol=frame["source_symbol"],
        exchange=None,
        source_accession=None,
        evidence="fixture",
        n_observations=1,
        scope_changed_at=stamp,
    )
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame[list(sm.TABLE_COLUMNS)]


def _lineage() -> pd.DataFrame:
    rows = [(t, c, "cik_window", "", None, None) for t, c in ROSTER.items() if t not in ("AAA",)] + LINEAGE_ROWS
    entity = {"DOC": "E0000765880", "AAA": "E0000000100"}
    return pd.DataFrame(
        [
            {
                "entity_id": entity.get(ticker, f"E{ROSTER[ticker]}"),
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": symbol,
                "valid_from": pd.Timestamp(start or "1900-01-01"),
                "valid_to": pd.Timestamp(end) if end else pd.NaT,
                "status": "corroborated" if role == "symbol" else "curated",
                "sources": "form345,roster" if role == "symbol" else "register",
                "scope_changed_at": BUILT,
            }
            for ticker, cik, role, symbol, start, end in rows
        ]
    )


def _identity(master: pd.DataFrame | None = None) -> Any:
    tenure = pd.DataFrame(
        [
            {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2006-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    return build_identity(_lineage(), tenure, roster, master=_master() if master is None else master)


def _file(day: str, rows: list[tuple[str, float, float, float, str]]) -> str:
    return "\n".join([HEADER, *(f"{day}|{s}|{sv}|{ex}|{tv}|{m}" for s, sv, ex, tv, m in rows)]) + "\n"


def _lines(day: str, rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return si._parse_regsho(_file(day, [(s, sv, 0, tv, "Q,N") for s, sv, tv in rows]))


def _context(store, tmp_path: Path) -> Any:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger("test.short_volume"),
        config=extract_config(data_extract={"years_history": 15}),
    )


# --------------------------------------------------------------------------- parse and symbology


def test_parse_keeps_the_raw_symbol_case_market_exempt_and_fractional_volumes():
    text = (
        _file(
            "20260701",
            [
                ("BRK/A", 87.5, 0, 412.25, "B,Q,N"),
                ("BACpB", 1200, 3, 5100, "Q,N"),
                ("BACPB", 10, 0, 20, "Q"),
                ("GOOGL", 438584.394961, 673, 733402.726526, "B,Q,N"),
            ],
        )
        + "7814\n"
    )
    parsed = si._parse_regsho(text)
    assert list(parsed.columns) == list(si.RAW_COLUMNS)
    assert parsed["source_symbol"].tolist() == ["BRK/A", "BACpB", "BACPB", "GOOGL"], "case and separators as filed; trailer dropped"
    googl = parsed[parsed["source_symbol"].eq("GOOGL")].iloc[0]
    assert googl["short_volume"] == pytest.approx(438584.394961) and googl["total_volume"] == pytest.approx(733402.726526)
    assert googl["short_exempt_volume"] == 673 and googl["market"] == "B,Q,N"
    assert parsed["date"].eq(pd.Timestamp("2026-07-01")).all()
    print("\n=== SANITY CHECK: RegSHO raw parse ===")
    print("  BRK/A, BACpB and BACPB kept as filed; fractional 2026 volumes, ShortExemptVolume and Market kept; record-count trailer dropped")


@pytest.mark.parametrize(
    ("symbol", "key", "kind"),
    [
        ("BRK/A", "BRKA", None),
        ("LEN/B", "LENB", None),
        ("BACpB", "BACPRB", "preferred"),
        ("BACPB", "BACPB", None),
        ("BACpY/CL", "BACPRYCL", "preferred"),
        ("BAC/WS/A", "BACWSA", "warrant"),
        ("AAC/U", "AACU", "unit"),
        ("ABCr", "ABCRT", "right"),
        ("ABCw", "ABCWI", "when_issued"),
    ],
)
def test_finra_symbology_is_read_before_upper_casing(symbol, key, kind):
    assert si.finra_key(symbol) == key
    assert si.finra_marker_kind(symbol) == kind
    print(f"\n=== SANITY CHECK: FINRA symbology {symbol} ===")
    print(f"  key {key} (the FTD spelling), marker kind {kind}")


# --------------------------------------------------------------------------- resolution through security_on


def test_classes_are_separated_summed_by_ratio_and_preferreds_never_summed():
    lines = _lines(
        "20200102",
        [
            ("BRK/A", 2.0, 10.0),
            ("BRK/B", 1000.0, 4000.0),
            ("BACpB", 50.0, 100.0),
            ("BAC", 7000.0, 30000.0),
            ("GOOG", 300.0, 900.0),
            ("GOOGL", 400.0, 1000.0),
            ("LEN/B", 5.0, 20.0),
            ("LEN", 60.0, 200.0),
            ("ZZZZ", 1.0, 2.0),
        ],
    )
    stamped = si.stamp_short_volume(lines, _identity(), UNIVERSE)
    roles = dict(zip(stamped["source_symbol"], zip(stamped["ticker"], stamped["lineage_role"], stamped["security_class"], strict=True), strict=True))
    assert roles["BRK/A"] == ("BRK-B", "secondary_class", "class_A") and roles["BRK/B"] == ("BRK-B", "canonical_current", "class_B")
    assert roles["BACpB"] == ("BAC", "excluded", "preferred")
    assert roles["GOOG"] == ("GOOGL", "secondary_class", "class_C") and roles["LEN/B"] == ("LEN", "secondary_class", "class_B")
    assert "ZZZZ" not in roles, "a symbol of no universe security is not kept"
    grain = si.ticker_rows(stamped).set_index("ticker")
    assert grain.loc["BRK-B", "short_volume"] == 1000.0 + 1500 * 2.0 and grain.loc["BRK-B", "total_volume"] == 4000.0 + 1500 * 10.0
    assert grain.loc["BAC", "short_volume"] == 7000.0, "the BACpB preferred contributes 0"
    assert grain.loc["GOOGL", "short_volume"] == 700.0 and grain.loc["LEN", "total_volume"] == 220.0
    assert list(si.ticker_rows(stamped).columns) == list(si.TICKER_COLUMNS)
    print("\n=== SANITY CHECK: class sum ===")
    print("  BRK-B = B + 1,500 x A; GOOGL = A + C; LEN = A + B; BACpB stored as an excluded preferred and never summed")


def test_acquired_constituent_is_kept_labelled_and_never_summed():
    lines = pd.concat([_lines("20230103", [("DOC", 900.0, 3000.0), ("PEAK", 400.0, 1500.0)]), _lines("20240305", [("DOC", 800.0, 2000.0)])])
    stamped = si.stamp_short_volume(lines, _identity(), UNIVERSE)
    by_day = {
        (s, str(d.date())): (r, i)
        for s, d, r, i in zip(stamped["source_symbol"], stamped["date"], stamped["lineage_role"], stamped["security_id"], strict=True)
    }
    assert by_day[("DOC", "2023-01-03")] == ("acquired_constituent", "C71943U104"), "Physicians Realty's DOC"
    assert by_day[("DOC", "2024-03-05")] == ("canonical_current", "C42250P103"), "Healthpeak's DOC after the seam"
    grain = {(t, str(d.date())): v for t, d, v in zip(*si.ticker_rows(stamped)[["ticker", "date", "short_volume"]].T.values, strict=True)}
    assert grain == {("DOC", "2023-01-03"): 400.0, ("DOC", "2024-03-05"): 800.0}
    print("\n=== SANITY CHECK: acquired constituent ===")
    print("  Physicians Realty's DOC (2023) stored as acquired_constituent, 0 in DOC; PEAK counts; DOC counts after 2024-03-01")


def test_e14_case_sensitive_keys_differ_and_a_same_key_collision_is_a_conflict():
    lines = _lines("20200102", [("BACpB", 50.0, 100.0), ("BACPB", 5.0, 10.0)])
    stamped = si.stamp_short_volume(lines, _identity(), UNIVERSE)
    assert stamped["source_symbol"].tolist() == ["BACpB"], "BACPB is another key (no master security): not kept"
    clash = _lines("20200102", [("BACpB", 50.0, 100.0), ("BACPRB", 5.0, 10.0)])
    stamped = si.stamp_short_volume(clash, _identity(), UNIVERSE)
    assert sorted(stamped["source_symbol"]) == ["BACPRB", "BACpB"]
    assert stamped["security_id"].isna().all() and stamped["lineage_role"].isna().all(), "both unresolved on that date"
    assert si.ticker_rows(stamped).empty
    print("\n=== SANITY CHECK: E14 FINRA case sensitivity ===")
    print("  BACpB -> BACPRB and BACPB stay different keys; BACpB with a BACPRB on one date is a conflict: kept unresolved, never summed")


def test_e25_a_symbol_never_seen_in_ftd_maps_only_through_a_window_cik_inside_its_window():
    lines = pd.concat(
        [
            _lines("20200601", [("OLDA", 10.0, 100.0), ("ACQ", 20.0, 200.0)]),
            _lines("20210105", [("OLDA", 11.0, 110.0), ("AAA", 30.0, 300.0), ("AAApA", 1.0, 2.0)]),
        ]
    )
    stamped = si.stamp_short_volume(lines, _identity(), UNIVERSE)
    got = {
        (s, str(d.date())): (i, r)
        for s, d, i, r in zip(stamped["source_symbol"], stamped["date"], stamped["security_id"], stamped["lineage_role"], strict=True)
    }
    assert got == {
        ("OLDA", "2020-06-01"): ("S0000000099:OLDA", "canonical_predecessor"),
        ("AAA", "2021-01-05"): ("S0000000100:AAA", "canonical_current"),
    }, got
    print("\n=== SANITY CHECK: E25 P21 fallback ===")
    print("  OLDA inside its window CIK's window and AAA on the roster CIK map as S securities;")
    print("  OLDA after the window, the event-only ACQ and the AAApA preferred are not kept")


def test_a_known_key_outside_every_master_interval_is_kept_unresolved():
    stamped = si.stamp_short_volume(_lines("20150105", [("GOOG", 3.0, 4.0)]), _identity(), UNIVERSE)
    assert stamped["source_symbol"].tolist() == ["GOOG"] and stamped["lineage_role"].isna().all() and stamped["ticker"].isna().all()
    assert si.ticker_rows(stamped).empty
    print("\n=== SANITY CHECK: known key, no interval ===")
    print("  GOOG in 2015 (before Alphabet's class C line): kept with NULL stamps for a later re-stamp, never summed")


# --------------------------------------------------------------------------- consumer schema


def test_the_ticker_table_keeps_its_consumer_schema_and_the_raw_table_is_spliced():
    ddl = (REPO / "sql" / "schema.sql").read_text(encoding="utf-8")
    block = re.search(r'CREATE TABLE IF NOT EXISTS "sec_short_interest" \((.*?)\);', ddl, re.S)
    assert block is not None
    assert re.findall(r'^\s*"(\w+)"', block.group(1), re.M) == ["date", "ticker", "short_volume", "total_volume"] == list(si.TICKER_COLUMNS)
    assert Tables.short_interest.name == "sec_short_interest" and Tables.short_interest.pk == ("ticker", "date")
    assert Tables.short_interest.read_columns == ("date", "ticker", "short_volume", "total_volume", "short_interest", "avg_daily_volume")
    raw = re.search(r'CREATE TABLE IF NOT EXISTS "sec_short_volume_security" \((.*?)\);', ddl, re.S)
    assert raw is not None
    assert re.findall(r'^\s*"(\w+)"', raw.group(1), re.M) == list(si.SECURITY_COLUMNS)
    assert Tables.sec_short_volume_security.pk == ("source_symbol", "date")
    print("\n=== SANITY CHECK: consumer schema ===")
    print("  sec_short_interest keeps PK (ticker, date) and its four columns; sec_short_volume_security is spliced with PK (source_symbol, date)")


# --------------------------------------------------------------------------- re-stamp (AC-118, E31, E32)


def _stamped_store(store, tmp_path: Path) -> Any:
    store.replace(Tables.security_master, _master())
    lines = pd.concat(
        [
            _lines("20200102", [("BRK/A", 2.0, 10.0), ("BRK/B", 1000.0, 4000.0), ("LEN", 60.0, 200.0), ("LEN/B", 5.0, 20.0)]),
            _lines("20200103", [("LEN", 70.0, 210.0), ("LEN/B", 6.0, 22.0)]),
        ]
    )
    stamped = si.stamp_short_volume(lines, _identity(), UNIVERSE)
    store.save(Tables.sec_short_volume_security, stamped[list(si.SECURITY_COLUMNS)])
    store.save(Tables.short_interest, si.ticker_rows(stamped))
    context = _context(store, tmp_path)
    return context


def test_e31_a_moved_master_boundary_restamps_exactly_the_affected_rows(sqlite_store, tmp_path):
    context = _stamped_store(sqlite_store, tmp_path)
    master = _master()
    lenb = master["cusip"].eq("526057302")
    master.loc[lenb, "valid_from"] = pd.Timestamp("2020-01-03")  # LEN-B's line now starts one day later
    master.loc[master["canonical_company"].eq("LEN"), "scope_changed_at"] = LATER
    sqlite_store.replace(Tables.security_master, master)

    records = si.restamp_short_volume(context, None, UNIVERSE, identity=_identity(master), as_of=RUN_DATE)

    raw = sqlite_store.load(Tables.sec_short_volume_security)
    raw["day"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    lenb = raw[raw["source_symbol"].eq("LEN/B")]
    lenb_rows = {day: (None if pd.isna(role) else role) for day, role in zip(lenb["day"], lenb["lineage_role"], strict=True)}
    assert lenb_rows == {"2020-01-02": None, "2020-01-03": "secondary_class"}
    grain = sqlite_store.load(Tables.short_interest)
    grain = {(t, str(pd.Timestamp(d).date())): v for t, d, v in zip(grain["ticker"], grain["date"], grain["short_volume"], strict=True)}
    assert grain[("LEN", "2020-01-02")] == 60.0 and grain[("LEN", "2020-01-03")] == 76.0 and grain[("BRK-B", "2020-01-02")] == 4000.0
    assert records == []
    print("\n=== SANITY CHECK: E31 master boundary moved ===")
    print("  LEN-B's line starts a day later: its 2020-01-02 row loses its stamp and LEN 2020-01-02 drops to 60; BRK-B untouched")


def test_e32_an_unchanged_master_restamps_nothing(sqlite_store, tmp_path, monkeypatch):
    context = _stamped_store(sqlite_store, tmp_path)
    for name in ("save", "delete", "replace", "load"):
        monkeypatch.setattr(sqlite_store, name, lambda *a, _n=name, **k: pytest.fail(f"unchanged master must not {_n}"))
    assert (
        si.restamp_short_volume(context, None, UNIVERSE, identity=_identity(), stamps=si.change_stamps(_master(), _identity()), as_of=RUN_DATE) == []
    )
    print("\n=== SANITY CHECK: E32 unchanged master ===")
    print("  every company's stamp predates the last run: no read, no re-stamp, no write")
