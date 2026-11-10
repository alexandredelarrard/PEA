"""Short volume per security (P5): FINRA RegSHO lines stored as filed, source columns only; the scope a line is kept by.

The FINRA lines copy the real symbology of the CNMSshvol files (`BRK/A`, `BACpB`, fractional volumes, `Market`). The
cube maps them to companies (`tests/data_aggregate/test_short_volume_series.py`, which reads the fixtures below).

The master rows are real (the Q2a build over the whole FTD cache) for BRK-B, BAC, GOOGL, LEN and DOC. Symbols never seen
in FTD (E25) use a hand-written lineage with a roster CIK, a predecessor window CIK and an event-only CIK.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import security_lines as sl
from src.data_extract.utils.common.identity import build_identity, load_source_identity
from src.data_extract.utils.identity import cusip_history as sm
from src.data_extract.utils.institutionals import fetch_short_interest as si
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.canonical_schema import create_canonical
from tests.support.universe import as_roster

REPO = Path(__file__).resolve().parents[3]
BUILT = pd.Timestamp("2026-10-04 12:00:00")
HEADER = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
UNIVERSE = ["BRK-B", "BAC", "GOOGL", "LEN", "DOC", "AAA"]

# Real `universe_cusip` rows (Q2a build): cusip, company, issuer CIK, symbol, class, ratio, role, valid_from, valid_to, reason.
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
        source=sl.SOURCE_FTD,
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
                "company_id": entity.get(ticker, f"E{ROSTER[ticker]}"),
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


def _identity(master: pd.DataFrame | None = None, *, store: DataStore | None = None) -> Any:
    tenure = pd.DataFrame(
        [
            {
                "symbol": t,
                "issuer_cik": c,
                "valid_from": pd.Timestamp("2006-01-01"),
                "valid_to": None,
                "n_filings": 5,
                "source": "form345",
                "evidence_period": "2006q1",
            }
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    resolver = build_identity(_lineage(), tenure, as_roster(roster), master=_master() if master is None else master)
    if store is not None:
        create_canonical(store.engine, list(resolver.source_cohort))
        for table, frame in resolver.source_cohort.items():
            if not frame.empty and store.load(table, optional=True) is None:
                store.save(table, frame)
        resolver.source_cohort.update(load_source_identity(store))
    return resolver


def _file(day: str, rows: list[tuple[str, float, float, float, str]]) -> str:
    return "\n".join([HEADER, *(f"{day}|{s}|{sv}|{ex}|{tv}|{m}" for s, sv, ex, tv, m in rows), str(len(rows))]) + "\n"


def _berkshire_lineage() -> pd.DataFrame:
    """Berkshire's lineage: its listed class symbols, a Notes-only bond symbol and a noise symbol."""
    rows = [
        ("0001067983", "cik_window", "", "register", "curated"),
        ("0001067983", "symbol", "BRK-B", "form345,roster", "corroborated"),
        ("0001067983", "symbol", "BRK-A", "form345", "corroborated"),
        ("0001067983", "symbol", "BRK59", "dei", "single_source"),
        ("0001067983", "symbol", "TRUE", "form345", "noise"),
    ]
    frame = pd.DataFrame(rows, columns=["cik", "role", "symbol", "sources", "status"])
    return frame.assign(company_id="E0001067983", canonical_ticker="BRK-B", valid_from=pd.Timestamp("1900-01-01"), valid_to=pd.NaT)


def test_parse_keeps_the_raw_symbol_case_market_exempt_and_fractional_volumes():
    text = _file(
        "20260701",
        [
            ("BRK/A", 87.5, 0, 412.25, "B,Q,N"),
            ("BACpB", 1200, 3, 5100, "Q,N"),
            ("BACPB", 10, 0, 20, "Q"),
            ("GOOGL", 438584.394961, 673, 733402.726526, "B,Q,N"),
        ],
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


def test_the_scope_is_the_stored_ftd_symbols_the_tickers_and_the_equity_lineage_symbols():
    keys, symbols = si.short_scope(_berkshire_lineage(), ["BRKA", "BRKB", "BACPRB", "", "JPM"], ["BRK-B", "AAPL"])
    assert keys == {"BRKA", "BRKB", "BACPRB", "JPM"}, "every stored FTD symbol, preferreds included"
    assert symbols == {"BRK-B", "BRK-A", "AAPL"}, "the Notes-only bond BRK59 and the noise TRUE are no tape symbols"
    lines = si._parse_regsho(_file("20260701", [(s, 1, 0, 2, "Q") for s in ("BRK/A", "BACpB", "BRK59", "TRUE", "AAPL", "ZZZZ")]))
    kept = lines["source_symbol"].map(si.finra_key).isin(keys) | lines["source_symbol"].map(si.yahoo_symbol).isin(symbols)
    assert lines.loc[kept, "source_symbol"].tolist() == ["BRK/A", "BACpB", "AAPL"]
    print("\n=== SANITY CHECK: RegSHO scope ===")
    print("  BRK/A by its FTD spelling BRKA, BACpB by BACPRB (a stored FTD symbol), AAPL a ticker; BRK59 (Notes only), TRUE (noise), ZZZZ out")


def test_the_raw_table_stores_source_columns_only_and_the_ticker_table_is_gone():
    ddl = (REPO / "sql" / "schema.sql").read_text(encoding="utf-8")
    raw = re.search(r'CREATE TABLE IF NOT EXISTS "sec_short_volume_security" \((.*?)\);', ddl, re.S)
    assert raw is not None
    assert re.findall(r'^\s*"(\w+)"', raw.group(1), re.M) == list(si.RAW_COLUMNS)
    assert "sec_short_interest" not in ddl and not hasattr(Tables, "short_interest")
    table = Tables.sec_short_volume_security
    assert table.pk == ("source_symbol", "date") and table.ticker_col is None and table.read_columns == si.RAW_COLUMNS
    print("\n=== SANITY CHECK: as-filed schema ===")
    print("  sec_short_volume_security = the six day-file columns, PK (source_symbol, date), no ticker; sec_short_interest removed")
