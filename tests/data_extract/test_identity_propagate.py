"""`identity-propagate`: a lineage contraction purges rows whose filer CIK (or tape symbol) left the
ticker's entity, an expansion re-parses the bulk families for that ticker only, and the derived
history of a purged ticker is rebuilt. Offline: a real `DataStore` on SQLite, a real `Identity` over
hand-written lineage rows, the bulk re-parsers and the merged-history build stubbed where noted.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract import identity_propagate as prop
from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.fundamentals import build_history
from src.data_extract.utils.fundamentals.build_history import TickerHistory
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.institutionals import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from src.utils import filer_tables
from tests.data_extract.fake_context import extract_config

SENTINEL = "1900-01-01"
LAST_RUN = pd.Timestamp("2026-01-10")
BEFORE, AFTER = pd.Timestamp("2026-01-05 09:00"), pd.Timestamp("2026-01-12 09:00")
ALB, AB, MSFT, MSFT_FOREIGN = "0000915913", "0000825313", "0000789019", "0000000666"
TMUS, TMO_USA, SPRINT = "0001283699", "0001097609", "0001727074"
ROSTER = {"ALB": ALB, "MSFT": MSFT, "TMUS": TMUS}
LOGGER = "test.identity_propagate"


def _row(ticker: str, cik: str, role: str, *, symbol: str = "", start: str = SENTINEL, end: str | None = None, stamp=BEFORE, **extra) -> dict:
    return {
        "entity_id": f"E{ROSTER[ticker]}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": role,
        "symbol": symbol,
        "valid_from": start,
        "valid_to": end,
        "status": extra.get("status", "curated"),
        "sources": extra.get("sources", "form345"),
        "oracle": "fixture",
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": stamp,
    }


def _identity(stamps: dict[str, pd.Timestamp], symbol_rows: list[dict] = ()) -> Any:
    """ALB and MSFT each one open window; TMUS a window plus its T-Mobile USA co-registrant (event only)."""
    rows = [
        _row("ALB", ALB, "cik_window", stamp=stamps.get("ALB", BEFORE)),
        _row("MSFT", MSFT, "cik_window", stamp=stamps.get("MSFT", BEFORE)),
        _row("TMUS", TMUS, "cik_window", stamp=stamps.get("TMUS", BEFORE)),
        _row("TMUS", TMO_USA, "cik_event", stamp=stamps.get("TMUS", BEFORE)),
        *symbol_rows,
    ]
    tenure = pd.DataFrame(
        [
            {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}
            for t, c in ROSTER.items()
        ]
    )
    return build_identity(pd.DataFrame(rows), tenure, pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()]))


def _context(store, tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger(LOGGER),
        config=extract_config(data_extract={"years_history": 15, "manifest_full_rescan_days": 30}),
        config_dir="./configs",
    )


def _filings(table, ticker: str, rows: list[tuple[str, str, str]], **extra_cols) -> pd.DataFrame:
    """(accession, filer cik, filed) -> two rows per filing in `table`'s shape."""
    if table == Tables.sec_8k:
        out = [
            {"ticker": ticker, "accession_number": a, "item": item, "cik": c, "filing_date": pd.Timestamp(f)}
            for a, c, f in rows
            for item in ("2.02", "9.01")
        ]
    elif table == Tables.fundamentals_facts:
        out = [
            {
                "ticker": ticker,
                "accession_number": a,
                "field": field,
                "duration_type": "quarterly",
                "period_end": pd.Timestamp(f),
                "cik": c,
                "filing_date": pd.Timestamp(f),
                "value": 1.0,
            }
            for a, c, f in rows
            for field in ("totalRevenue", "totalAssets")
        ]
    elif table == Tables.notes_num:
        out = [
            {"ticker": ticker, "adsh": a, "tag": tag, "ddate": pd.Timestamp(f), "qtrs": 0, "cik": c, "filed": pd.Timestamp(f)}
            for a, c, f in rows
            for tag in ("T1", "T2")
        ]
    elif table == Tables.insider_transactions:
        out = [
            {
                "ticker": ticker,
                "accession_number": a,
                "security_type": "nonderiv",
                "row_sequence": seq,
                "issuer_cik": c,
                "filing_date": pd.Timestamp(f),
            }
            for a, c, f in rows
            for seq in (1, 2)
        ]
    else:
        raise AssertionError(table)
    frame = pd.DataFrame(out)
    if table == Tables.fundamentals_facts:
        frame = frame.assign(**{column: None for column in build_history.FACT_COLUMNS if column not in frame.columns})
    return frame


PURGED_FAMILIES = (Tables.sec_8k, Tables.fundamentals_facts, Tables.notes_num, Tables.insider_transactions)


def _seed(store) -> None:
    """ALB: two own filings and two AllianceBernstein (AB) filings in each family; MSFT: one own, one foreign."""
    for table in PURGED_FAMILIES:
        store.save(
            table,
            _filings(
                table, "ALB", [("alb-1", ALB, "2024-02-01"), ("alb-2", ALB, "2024-05-01"), ("ab-1", AB, "2019-03-01"), ("ab-2", AB, "2021-08-01")]
            ),
        )
        store.save(table, _filings(table, "MSFT", [("msft-1", MSFT, "2024-02-01"), ("msft-x", MSFT_FOREIGN, "2024-03-01")]))


def _record_all(context) -> None:
    for table in (*prop.PURGE_TABLES_BY_NAME.values(), Tables.sec_fails_to_deliver, Tables.fundamentals_history_sec):
        record_run(context, getattr(table, "table", table), ticker_count=3, rows_added=0, run_date=LAST_RUN)


@pytest.fixture
def stubs(monkeypatch) -> dict[str, list]:
    """Bulk re-parsers, the FTD re-stamp and the derived rebuilds recorded instead of run."""
    calls: dict[str, list] = {"notes": [], "pension": [], "insider": [], "ftd": [], "short": [], "history": [], "merged": []}
    monkeypatch.setattr(prop, "reparse_financial_notes", lambda context, tickers: calls["notes"].append(sorted(tickers)) or 0)
    monkeypatch.setattr(prop, "reparse_financial_statements", lambda context, tickers: calls["pension"].append(sorted(tickers)) or 0)
    monkeypatch.setattr(prop, "reparse_insider_transactions", lambda context, tickers: calls["insider"].append(sorted(tickers)) or 0)
    monkeypatch.setattr(
        prop,
        "restamp_fails",
        lambda context, companies, tickers, master=None, dry_run=False: calls["ftd"].append(sorted(companies)) or [],
    )
    monkeypatch.setattr(
        prop,
        "restamp_short_volume",
        lambda context, companies, tickers, identity=None, stamps=None, dry_run=False: calls["short"].append(sorted(companies)) or [],
    )
    monkeypatch.setattr(
        prop,
        "build_fundamentals_history",
        lambda context, tickers, rebuild_history=False: calls["history"].append((sorted(tickers), rebuild_history)),
    )
    monkeypatch.setattr(
        prop, "build_merged_history", lambda context, tickers, full=False, config_dir=None: calls["merged"].append((sorted(tickers), full))
    )
    return calls


def _keys(store, table) -> set[tuple[str, str]]:
    key = prop.PURGE_TABLES_BY_NAME[table.name].key_col
    frame = store.load(table, columns=["ticker", key])
    return set(zip(frame["ticker"], frame[key], strict=True))


def test_a_contraction_purges_the_removed_cik_in_every_family_and_warns_once_per_table(sqlite_store, tmp_path, monkeypatch, caplog, stubs):
    """AC-031/AC-034: ALB's scope changed after each table's last run; AB's rows go, ALB's own stay, MSFT is not read."""
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"ALB": AFTER}))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = prop.propagate_identity(context, list(ROSTER))

    for table in PURGED_FAMILIES:
        keys = _keys(sqlite_store, table)
        assert ("ALB", "ab-1") not in keys and ("ALB", "ab-2") not in keys, table.name
        assert {("ALB", "alb-1"), ("ALB", "alb-2"), ("MSFT", "msft-1"), ("MSFT", "msft-x")} <= keys, table.name
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "identity-propagate: purged" in r.getMessage()]
    assert len(warnings) == len(PURGED_FAMILIES), warnings
    for table in PURGED_FAMILIES:
        line = next(w for w in warnings if f"'{table.name}'" in w)
        assert f"ALB cik {AB} 2019-03-01..2021-08-01 4 row(s)" in line, line
    removed = result.removals
    assert set(removed["table"]) == {t.name for t in PURGED_FAMILIES} and set(removed["cik"]) == {AB} and set(removed["rows"]) == {4}
    assert stubs["history"] == [(["ALB"], True)] and stubs["merged"] == [(["ALB"], True)]
    print("\n=== SANITY CHECK: contraction purge ===")
    print(
        f"  4 families: AB's 2 filings (4 rows each) purged under ALB; ALB's own and MSFT's (scope unchanged) rows kept; {len(warnings)} WARNING lines"
    )
    print("  ALB's history rebuilt (delete + recompute) and its merged history rebuilt in full. Validated.")


def test_a_sibling_cik_of_the_entity_survives_the_purge(sqlite_store, tmp_path, monkeypatch, stubs):
    """AC-022: T-Mobile USA's rows under TMUS are the entity's own; only the foreign filer's go."""
    context = _context(sqlite_store, tmp_path)
    for table in (Tables.sec_8k, Tables.fundamentals_facts):
        sqlite_store.save(
            table, _filings(table, "TMUS", [("t-1", TMUS, "2024-02-01"), ("t-usa", TMO_USA, "2016-02-01"), ("s-1", SPRINT, "2019-07-01")])
        )
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"TMUS": AFTER}))

    prop.propagate_identity(context, list(ROSTER))

    for table in (Tables.sec_8k, Tables.fundamentals_facts):
        rows = sqlite_store.load(table, columns=["ticker", "cik"])
        counts = rows.groupby("cik").size().to_dict()
        assert counts == {TMUS: 2, TMO_USA: 2}, (table.name, counts)
    print("\n=== SANITY CHECK: sibling CIK (AC-022) ===")
    print("  TMUS: T-Mobile USA (event CIK of the entity) 2 rows kept per table; the foreign 0001727074 filing purged")


def test_a_cik_with_no_digit_is_never_judged_as_the_validator_rules(sqlite_store, tmp_path, monkeypatch, stubs):
    """D-7: the purge's pre-filter is the validator's. A junk CIK string ('N/A', '-') is not a filer, so its rows stay."""
    context = _context(sqlite_store, tmp_path)
    for table in (Tables.sec_8k, Tables.fundamentals_facts):
        filings = [("alb-1", ALB, "2024-02-01"), ("junk-1", "N/A", "2023-02-01"), ("junk-2", " - ", "2023-03-01"), ("ab-1", AB, "2019-03-01")]
        sqlite_store.save(table, _filings(table, "ALB", filings))
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"ALB": AFTER}))

    result = prop.propagate_identity(context, list(ROSTER))

    for table in (Tables.sec_8k, Tables.fundamentals_facts):
        keys = _keys(sqlite_store, table)
        assert {("ALB", "alb-1"), ("ALB", "junk-1"), ("ALB", "junk-2")} <= keys, (table.name, keys)
        assert ("ALB", "ab-1") not in keys, table.name
    assert set(result.removals["cik"]) == {AB}, result.removals
    assert filer_tables.judged_cik_mask(pd.Series(["N/A", " - ", "", None, "789019", 320193.0], dtype=object)).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    print("\n=== SANITY CHECK: junk-CIK pre-filter (D-7) ===")
    print("  'N/A' and ' - ' rows kept in sec_8k and fundamentals_facts; only AB's foreign filing purged")
    print("  OK: a CIK with no digit is never judged, by the purge exactly as by the validator.")


def test_the_dry_run_returns_the_same_removals_and_deletes_nothing(sqlite_store, tmp_path, monkeypatch, stubs):
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"ALB": AFTER}))
    before = {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES}

    dry = prop.propagate_identity(context, list(ROSTER), dry_run=True)

    assert {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES} == before
    assert stubs["notes"] == stubs["pension"] == stubs["insider"] == stubs["history"] == stubs["merged"] == []
    wet = prop.propagate_identity(context, list(ROSTER))
    columns = list(prop.REMOVAL_COLUMNS)
    assert dry.removals[columns].reset_index(drop=True).equals(wet.removals[columns].reset_index(drop=True))
    assert len(dry.removals) == len(PURGED_FAMILIES)
    print("\n=== SANITY CHECK: dry run ===")
    print(f"  dry run listed {len(dry.removals)} (table, ticker, cik) removals, deleted 0 rows; the real run removed exactly that set")


def test_an_unchanged_lineage_neither_purges_nor_reparses(sqlite_store, tmp_path, monkeypatch, stubs):
    """AC-032: every scope stamp is older than every table's last run."""
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({}))
    before = {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES}

    result = prop.propagate_identity(context, list(ROSTER))

    assert result.removals.empty
    assert {table.name: sqlite_store.row_count(table) for table in PURGED_FAMILIES} == before
    assert all(not calls for calls in stubs.values()), stubs
    every = prop.propagate_identity(context, list(ROSTER), dry_run=True, every_ticker=True).removals
    assert set(zip(every["ticker"], every["cik"], strict=True)) == {("ALB", AB), ("MSFT", MSFT_FOREIGN)}
    print("\n=== SANITY CHECK: unchanged lineage (AC-032) ===")
    print("  no purge, no bulk re-parse, no FTD re-stamp, no rebuild; the validator's every-ticker dry run still lists both foreign filers")


def test_an_expansion_reparses_the_bulk_families_for_that_ticker_only(sqlite_store, tmp_path, monkeypatch, stubs):
    """AC-030: MSFT's scope changed; ALB and TMUS stay on their incremental frontier."""
    context = _context(sqlite_store, tmp_path)
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"MSFT": AFTER}))

    prop.propagate_identity(context, list(ROSTER))

    assert stubs["notes"] == stubs["pension"] == stubs["insider"] == [["MSFT"]]
    assert stubs["ftd"] == []  # no stored FTD lines, no master
    print("\n=== SANITY CHECK: expansion (AC-030) ===")
    print("  notes, pension and insider bulk re-parsed for MSFT only; FTD untouched (no stored lines)")


def test_short_volume_is_restamped_from_stored_rows_not_purged(sqlite_store, tmp_path, monkeypatch, stubs):
    """Short volume is no purge table: companies whose master rows changed since its last run are re-stamped from stored rows."""
    context = _context(sqlite_store, tmp_path)
    sqlite_store.replace(Tables.security_master, _master("MSFT", {"ALB": AFTER}))
    _record_all(context)
    record_run(context, Tables.short_interest, ticker_count=3, rows_added=0, run_date=LAST_RUN)
    sqlite_store.save(
        Tables.sec_short_volume_security,
        pd.DataFrame({"date": pd.to_datetime(["2024-02-01"]), "source_symbol": ["ALB"], "short_volume": [1.0], "ticker": ["ALB"]}),
    )
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({}))

    prop.propagate_identity(context, list(ROSTER))

    assert Tables.short_interest.name not in prop.PURGE_TABLES_BY_NAME
    assert stubs["short"] == [["ALB"]], stubs["short"]
    print("\n=== SANITY CHECK: short volume re-stamp ===")
    print("  ALB's master rows changed after the last run: its stored short-volume rows are re-stamped; nothing is purged by CIK")


def _master(owner_of_old: str, stamps: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """One canonical CUSIP per ticker, plus `OLD` (2010-2016) held by `owner_of_old`."""
    rows = [
        ("000000AL1", "ALB", "ALB", "2000-01-01", None),
        ("000000MS1", "MSFT", "MSFT", "2000-01-01", None),
        ("000000OL1", owner_of_old, "OLD", "2010-01-01", "2016-01-01"),
    ]
    frame = pd.DataFrame(rows, columns=["cusip", "canonical_company", "source_symbol", "valid_from", "valid_to"])
    frame = frame.assign(
        security_id="C" + frame["cusip"],
        issuer_cik="0000000001",
        source=sm.SOURCE_FTD,
        market_symbol=frame["source_symbol"],
        exchange=None,
        security_class="common",
        conversion_ratio=1.0,
        lineage_role=sm.CANONICAL_CURRENT,
        lineage_reason="fixture",
        source_accession=None,
        evidence="fixture",
        n_observations=1,
    )
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    frame["scope_changed_at"] = [stamps.get(t, BEFORE) for t in frame["canonical_company"]]
    return frame[list(sm.TABLE_COLUMNS)]


def _ftd_raw(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    lines = ["SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"]
    lines += [f"{day}|{cusip}|{symbol}|{qty}|X|1.0" for day, cusip, symbol, qty in rows]
    return ftd._parse_ftd_lines("\n".join(lines) + "\n")


def test_a_moved_master_security_moves_its_fails_rows(sqlite_store, tmp_path, monkeypatch, caplog):
    """FTD contraction and expansion: the `OLD` security moves from ALB to MSFT in the master; ALB's OLD-era
    ticker row goes (WARNING), MSFT gains it, re-stamped from the stored raw lines with no zip read."""
    context = _context(sqlite_store, tmp_path)
    raw = _ftd_raw([("20120301", "000000OL1", "OLD", 10), ("20240201", "000000AL1", "ALB", 5), ("20240201", "000000MS1", "MSFT", 7)])
    raw = raw.assign(
        period=["201203a", "202402a", "202402a"],
        security_id="C" + raw["cusip"],
        ticker=["ALB", "ALB", "MSFT"],
        lineage_role=sm.CANONICAL_CURRENT,
        security_class="common",
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])
    stored = pd.DataFrame(
        {
            "ticker": ["ALB", "ALB", "MSFT"],
            "date": pd.to_datetime(["2012-03-01", "2024-02-01", "2024-02-01"]),
            "fails_quantity": [10.0, 5.0, 7.0],
            "fails_value": [10.0, 5.0, 7.0],
            "period": ["201203a", "202402a", "202402a"],
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver, stored)
    sqlite_store.replace(Tables.security_master, _master("MSFT", {"ALB": AFTER, "MSFT": AFTER}))
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({}))
    monkeypatch.setattr(ftd, "read_zip_text", lambda *a, **k: pytest.fail("the re-stamp reads no zip"))
    monkeypatch.setattr(prop, "reparse_financial_notes", lambda context, tickers: 0)
    monkeypatch.setattr(prop, "reparse_financial_statements", lambda context, tickers: 0)
    monkeypatch.setattr(prop, "reparse_insider_transactions", lambda context, tickers: 0)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        prop.propagate_identity(context, list(ROSTER))

    after = sqlite_store.load(Tables.sec_fails_to_deliver, columns=["ticker", "date", "fails_quantity"])
    got = sorted((t, str(pd.Timestamp(d).date()), q) for t, d, q in after.itertuples(index=False))
    assert got == [("ALB", "2024-02-01", 5.0), ("MSFT", "2012-03-01", 10.0), ("MSFT", "2024-02-01", 7.0)], got
    restamped = sqlite_store.load(Tables.sec_fails_to_deliver_security, where={"cusip": "000000OL1"})
    assert restamped["ticker"].tolist() == ["MSFT"]
    assert any(
        "'sec_fails_to_deliver'" in r.getMessage() and "ALB" in r.getMessage() and "2012-03-01..2012-03-01 1 row(s)" in r.getMessage()
        for r in caplog.records
    )
    print("\n=== SANITY CHECK: FTD re-stamp ===")
    print("  OLD moved from ALB to MSFT in the master: ALB's 2012 row purged (WARNING), MSFT gains it; ALB's own 2024 row kept")


def test_the_master_stamp_alone_drives_the_fails_restamp(sqlite_store, tmp_path, monkeypatch, stubs):
    """Only companies whose master rows changed since the table's last run are re-stamped; no CIK family is re-parsed."""
    context = _context(sqlite_store, tmp_path)
    raw = _ftd_raw([("20240201", "000000AL1", "ALB", 5)]).assign(
        period="202402a", security_id="C000000AL1", ticker="ALB", lineage_role=sm.CANONICAL_CURRENT, security_class="common"
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])
    sqlite_store.replace(Tables.security_master, _master("MSFT", {"ALB": AFTER, "MSFT": AFTER}))
    _record_all(context)
    identity = _identity({})
    monkeypatch.setattr(prop, "load_identity", lambda context: identity)

    prop.propagate_identity(context, list(ROSTER))

    assert stubs["ftd"] == [["ALB", "MSFT"]]
    assert stubs["notes"] == stubs["insider"] == []
    print("\n=== SANITY CHECK: master stamp ===")
    print("  security_master rows carry their own change stamp: FTD re-stamped for ALB and MSFT, no CIK family re-parsed")


def _history(ticker: str, dates: list, revenue: float) -> TickerHistory:
    """A stub replay result carrying every history column (the build reads them back by projection)."""
    columns = list(load_catalogue("./configs").history_columns)
    frame = pd.DataFrame({column: [None] * len(dates) for column in columns}).assign(ticker=ticker, as_of=list(dates), totalRevenue=revenue)
    return TickerHistory(frame, pd.DataFrame(columns=["ticker", "as_of"]))


def test_the_history_of_a_purged_ticker_is_rebuilt_without_the_foreign_filing(sqlite_store, tmp_path, monkeypatch):
    """Published history rows built from a foreign filing are replaced once its facts are purged."""
    context = _context(sqlite_store, tmp_path)
    sqlite_store.save(
        Tables.fundamentals_facts, _filings(Tables.fundamentals_facts, "ALB", [("alb-1", ALB, "2024-02-01"), ("ab-1", AB, "2021-08-01")])
    )

    def one_row_per_filing(ticker, facts, **kwargs):
        dates = sorted(pd.to_datetime(facts["filing_date"]).unique())
        return _history(ticker, dates, float(len(dates)))

    monkeypatch.setattr(build_history, "build_ticker", one_row_per_filing)
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": BEFORE}))
    build_history.build_fundamentals_history(context, ["ALB"])
    assert sqlite_store.row_count(Tables.fundamentals_history_sec) == 2
    _record_all(context)
    monkeypatch.setattr(prop, "load_identity", lambda context: _identity({"ALB": AFTER}))
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": AFTER}))
    for name in ("reparse_financial_notes", "reparse_financial_statements", "reparse_insider_transactions"):
        monkeypatch.setattr(prop, name, lambda context, tickers: 0)
    merged: list = []
    monkeypatch.setattr(prop, "build_merged_history", lambda context, tickers, full=False, config_dir=None: merged.append((list(tickers), full)))

    prop.propagate_identity(context, list(ROSTER))

    history = sqlite_store.load(Tables.fundamentals_history_sec, columns=["ticker", "as_of", "totalRevenue"])
    assert [str(pd.Timestamp(d).date()) for d in history["as_of"]] == ["2024-02-01"] and list(history["totalRevenue"]) == [1.0]
    assert merged == [(["ALB"], True)]
    print("\n=== SANITY CHECK: history rebuild ===")
    print("  2 published rows (one from AB's filing) -> 1 row after the purge and rebuild; merged history rebuilt in full")


def test_the_history_build_rebuilds_a_ticker_whose_scope_changed_since_its_last_run(sqlite_store, tmp_path, monkeypatch):
    """Expansion: predecessor facts change published rows; the replay rebuilds instead of refusing."""
    context = _context(sqlite_store, tmp_path)
    sqlite_store.save(Tables.fundamentals_facts, _filings(Tables.fundamentals_facts, "ALB", [("alb-1", ALB, "2024-02-01")]))

    def count_rows(ticker, facts, **kwargs):
        return _history(ticker, [pd.Timestamp("2024-02-01")], float(len(facts)))

    monkeypatch.setattr(build_history, "build_ticker", count_rows)
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": BEFORE}))
    build_history.build_fundamentals_history(context, ["ALB"])
    record_run(context, Tables.fundamentals_history_sec, ticker_count=1, rows_added=1, run_date=LAST_RUN)
    sqlite_store.save(Tables.fundamentals_facts, _filings(Tables.fundamentals_facts, "ALB", [("pred-1", ALB, "2010-02-01")]))

    with pytest.raises(ValueError, match="append-only"):
        build_history.build_fundamentals_history(context, ["ALB"])
    monkeypatch.setattr(build_history, "load_identity", lambda context: _identity({"ALB": AFTER}))
    build_history.build_fundamentals_history(context, ["ALB"])

    history = sqlite_store.load(Tables.fundamentals_history_sec, columns=["as_of", "totalRevenue"])
    assert list(history["totalRevenue"]) == [4.0]
    print("\n=== SANITY CHECK: history expansion rebuild ===")
    print("  unchanged scope: a changed published row still refuses; scope changed after the last run: ALB rebuilt (2 -> 4 fact rows)")
