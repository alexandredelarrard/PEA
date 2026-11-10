"""The CUSIP history reads its FTD evidence from `sec_fails_to_deliver_security` (P4, REQ-002): no zip is opened, and
the lines kept are those the zips gave (same scope, same tape vote)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.common.security_lines import SecurityManual
from src.data_extract.utils.identity import cusip_history as sm
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.support.cusip_build import write_ftd_zips

_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
_TEXTS = {
    "201501a": _HEADER + "20150105|000111CCC|ABCWS|7|ABC CORP WT|1.0\n20150105|999999ZZZ|ZZZ|9|OTHER CO|3.0\n",
    "201501b": _HEADER + "20150120|000111AAA|ABC|100|ABC CORP|10.0\n20150120|000111BBB|ABCPRA|5|ABC CORP PFD|25.0\n",
}


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(sqlite_store.engine, [Tables.sec_fails_to_deliver_security])
    return sqlite_store


def _lineage() -> pd.DataFrame:
    rows = [
        ("E0000000777", "ABC", "0000000777", "cik_window", "", "1900-01-01", None, "single_source", "roster"),
        ("E0000000777", "ABC", "0000000777", "symbol", "ABC", "2006-01-04", None, "corroborated", "form345,roster"),
    ]
    columns = ["company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.assign(valid_from=pd.to_datetime(frame["valid_from"]), valid_to=pd.to_datetime(frame["valid_to"]))


def test_cusip_history_reads_the_stored_lines_and_opens_no_zip(sqlite_store, tmp_path, monkeypatch):
    """Every line of both files is stored (the ingest scope is wider); the evidence keeps the lineage symbol ABC and the
    CUSIP-6 000111 it votes for, drops ZZZ, and no zip reader runs."""
    cache = write_ftd_zips(tmp_path / "ftd", _TEXTS)
    lines = pd.concat([ftd._parse_ftd_lines(text).assign(period=period) for period, text in _TEXTS.items()], ignore_index=True)
    sqlite_store.save(Tables.sec_fails_to_deliver_security, lines)

    def _refuse(*args, **kwargs):
        raise AssertionError("the CUSIP history opened an FTD zip")

    monkeypatch.setattr(ftd, "read_zip_text", _refuse)
    monkeypatch.setattr(ftd, "read_ftd_period", _refuse)
    scope = sm.ftd_scope(_lineage(), {}, SecurityManual.empty())
    assert any(cache.iterdir()), "the zips are cached: reading them would raise"

    evidence = sm.read_ftd_evidence(sqlite_store, scope).sort_values("cusip", ignore_index=True)

    assert evidence["cusip"].tolist() == ["000111AAA", "000111BBB", "000111CCC"]
    assert evidence.columns.tolist() == list(sm.OBSERVATION_COLUMNS)
    print(f"\nSANITY: {len(lines)} stored lines -> {len(evidence)} evidence lines {evidence['cusip'].tolist()} (ABC + voted CUSIP-6 000111);")
    print("  ZZZ left out; no zip reader was called.")
