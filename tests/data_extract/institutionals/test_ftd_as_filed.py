"""FTD ingest as filed: the scope comes from the lineage, Sharadar and curated CUSIP-6, never from `universe_cusip`; a
run that stores a new period re-reads every other cached period with the current scope, a run that stores none reads no
file, and a killed run commits nothing."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.bulk_cache import read_zip_text
from src.data_extract.utils.common.security_lines import SecurityManual, ftd_scope, parse_security_manual
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.support.cusip_build import write_ftd_zips
from tests.support.universe import seed_universe

_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
RUN_DATE = pd.Timestamp("2026-11-10")
#: The stored source columns of `sec_fails_to_deliver_security`.
SOURCE = ["date", "trade_date", "cusip", "source_symbol", "description", "price", "fails_quantity", "fails_value", "period"]


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(
        sqlite_store.engine, [Tables.universe_cik_symbol, Tables.symbol_tenure, Tables.universe_cusip, Tables.sec_fails_to_deliver_security]
    )
    return sqlite_store


def _context(store, tmp_path, tickers: tuple[str, ...]) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.ftd_as_filed"),
        paths={"DATA_STORE": tmp_path},
        config_dir=str(tmp_path / "configs"),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(fails_deliver="ftd"))),
        universe=seed_universe(store, tickers),
    )


def _lineage(store, rows: list[tuple[str, str, str, pd.Timestamp]]) -> None:
    """One cik_window and one symbol row per (ticker, cik, symbol, scope_changed_at)."""
    frame = pd.DataFrame(
        [
            {
                "company_id": f"E{cik}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": symbol if role == "symbol" else "",
                "valid_from": pd.Timestamp("2000-01-01"),
                "valid_to": None,
                "status": "corroborated",
                "sources": "form345,roster",
                "scope_changed_at": stamp,
            }
            for ticker, cik, symbol, stamp in rows
            for role in ("cik_window", "symbol")
        ]
    ).drop_duplicates(["cik", "role", "symbol"])
    store.save(Tables.universe_cik_symbol, frame)


def _stored(store) -> pd.DataFrame:
    rows = store.load(Tables.sec_fails_to_deliver_security, columns=SOURCE, optional=True)
    return pd.DataFrame(columns=SOURCE) if rows is None else rows.sort_values(["period", "cusip"], ignore_index=True)


def test_ingest_keeps_the_class_c_line_under_the_alphabet_prefix_with_no_universe_cusip(sqlite_store, tmp_path, caplog):
    """GOOGL is the lineage symbol; the GOOG line shares its CUSIP-6, so it is stored as filed. No `universe_cusip` exists."""
    context = _context(sqlite_store, tmp_path, ("GOOGL",))
    _lineage(sqlite_store, [("GOOGL", "0001652044", "GOOGL", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER
            + "20260105|02079K305|GOOGL|100|ALPHABET INC CL A|300.0\n"
            + "20260105|02079K107|GOOG|40|ALPHABET INC CL C|301.0\n"
            + "20260105|999999ZZZ|ZZZ|9|OTHER CO|3.0\n"
        },
    )
    assert sqlite_store.row_count(Tables.universe_cusip) == 0

    with caplog.at_level(logging.INFO):
        ftd.fetch_fails_to_deliver(context)

    stored = _stored(sqlite_store)
    assert stored[["cusip", "source_symbol", "fails_quantity"]].values.tolist() == [["02079K107", "GOOG", 40.0], ["02079K305", "GOOGL", 100.0]]
    print("\nSANITY (no universe_cusip): the GOOGL lineage symbol keeps its line and puts CUSIP-6 02079K in scope, so the GOOG")
    print(f"  class C line is stored as filed: {stored[['cusip', 'source_symbol', 'fails_quantity']].values.tolist()}; ZZZ is not.")


def test_the_ftd_scope_keeps_equity_symbols_not_notes_debt_or_noise_symbols() -> None:
    """ABBV is a filed equity symbol; ABBV28B is a Notes cover-page (`dei`-only) bond symbol and TRUE a `dei` noise row:
    only ABBV is an FTD scope symbol."""
    rows = [("ABBV", "form345,roster", "corroborated"), ("ABBV28B", "dei", "single_source"), ("TRUE", "dei", "noise")]
    lineage = pd.DataFrame(
        [
            {
                "canonical_ticker": "ABBV",
                "cik": "0001551152",
                "role": "symbol",
                "symbol": symbol,
                "sources": sources,
                "status": status,
                "valid_from": pd.Timestamp("2013-01-02"),
                "valid_to": None,
            }
            for symbol, sources, status in rows
        ]
    )

    scope = ftd_scope(lineage, {}, SecurityManual.empty())

    symbols = frozenset(scope.windows["key"])
    print(f"\nSANITY (FTD scope symbols): {sorted(symbols)} from lineage symbols {[symbol for symbol, _, _ in rows]}")
    assert symbols == frozenset({"ABBV"}), "a Notes debt symbol and a noise symbol are not FTD scope keys"


def test_a_new_period_run_rereads_every_cached_period_with_the_current_scope(sqlite_store, tmp_path, caplog, monkeypatch):
    """202601a/b are stored with ABC's lines only; 202602a is new. XYZ and LATE are lineage symbols with no stored line
    (whatever their stamp): the run that stores 202602a re-reads 202601a/b with the current scope and stores both; the
    same-day re-run reads no file."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(
        sqlite_store,
        [
            ("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")),
            ("ABC", "0000000778", "XYZ", pd.Timestamp("2026-02-03")),
            ("ABC", "0000000779", "LATE", pd.Timestamp("2026-01-25")),
        ],
    )
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER
            + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|555555XXX|XYZ|7|XYZ CORP|2.0\n20260105|777777LLL|LATE|5|LATE CO|1.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|555555XXX|XYZ|8|XYZ CORP|2.0\n",
            "202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n",
        },
    )
    stored_abc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-20"]),
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-16"]),
            "cusip": ["000111AAA", "000111AAA"],
            "source_symbol": ["ABC", "ABC"],
            "description": ["ABC CORP", "ABC CORP"],
            "price": [10.0, 10.0],
            "fails_quantity": [100.0, 50.0],
            "fails_value": [1000.0, 500.0],
            "period": ["202601a", "202601b"],
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, stored_abc)
    reads: list[str] = []
    reader = ftd.read_zip_text
    monkeypatch.setattr(ftd, "read_zip_text", lambda path: reads.append(path.stem.removeprefix("cnsfails")) or reader(path))
    with caplog.at_level(logging.INFO):
        ftd.fetch_fails_to_deliver(context)
    first, reads[:] = list(reads), []
    ftd.fetch_fails_to_deliver(context)

    stored = _stored(sqlite_store)
    print(f"\nSANITY (full re-read on a new-period run): run 1 read {first}, run 2 read {reads} (expected none);")
    print(f"  stored {stored[['period', 'cusip', 'fails_quantity']].values.tolist()}")
    assert first == ["202602a", "202601a", "202601b"], "the new period, then each other cached period once"
    assert reads == [], "a run storing no new period reads no file"
    assert stored[["period", "cusip", "fails_quantity"]].values.tolist() == [
        ["202601a", "000111AAA", 100.0],
        ["202601a", "555555XXX", 7.0],
        ["202601a", "777777LLL", 5.0],
        ["202601b", "000111AAA", 50.0],
        ["202601b", "555555XXX", 8.0],
        ["202602a", "000111AAA", 60.0],
    ], "XYZ's and LATE's history is upserted"
    assert any("full re-read of 2" in record.getMessage() for record in caplog.records), "one log line names the full re-read"


def test_a_new_key_no_file_lists_is_reread_once_not_on_every_run(sqlite_store, tmp_path, monkeypatch):
    """NEW joins ABC's lineage today and appears in no FTD file: the cold run reads both cached periods with the full scope
    and no separate growth pass; a second run the same day stores no new period, so it reads nothing."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")), ("ABC", "0000000778", "NEW", RUN_DATE)])
    write_ftd_zips(
        tmp_path / "ftd",
        {"202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n", "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n"},
    )
    reads: list[str] = []
    reader = ftd.read_zip_text
    monkeypatch.setattr(ftd, "read_zip_text", lambda path: reads.append(path.stem.removeprefix("cnsfails")) or reader(path))
    ftd.fetch_fails_to_deliver(context)
    first, reads[:] = list(reads), []

    ftd.fetch_fails_to_deliver(context)

    print(f"\nSANITY (rowless growth key): run 1 read {sorted(first)}, run 2 read {sorted(reads)} (expected none)")
    assert first == ["202601a", "202601b"], "the cold run reads each period once, with no separate growth pass"
    assert reads == [], "a never-matching growth key already re-read is not re-read on the next run"


def _recording_reader(monkeypatch, crash_on: str | None = None) -> list[str]:
    """Record each period whose cached ZIP is read; `crash_on` kills the process (a non-ValueError) at that period's read."""
    reads: list[str] = []

    def read(path):
        period = path.stem.removeprefix("cnsfails")
        if period == crash_on:
            raise RuntimeError(f"simulated kill while reading {period}")
        reads.append(period)
        return read_zip_text(path)

    monkeypatch.setattr(ftd, "read_zip_text", read)
    return reads


def test_an_interrupted_new_period_run_keeps_the_owed_reread_for_the_next_run(sqlite_store, tmp_path, monkeypatch):
    """S1: 202601a/b are stored, then XYZ (in both files) joins ABC's lineage. The run that stores 202602a is killed in its
    re-read of 202601a; the next same-day run must still store XYZ's two lines, and a later new-period run must not lose
    them either."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|555555XXX|XYZ|7|XYZ CORP|2.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|555555XXX|XYZ|8|XYZ CORP|2.0\n",
        },
    )
    ftd.fetch_fails_to_deliver(context)  # cold: XYZ is not yet in scope
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")), ("ABC", "0000000778", "XYZ", pd.Timestamp("2026-02-03"))])
    write_ftd_zips(tmp_path / "ftd", {"202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n"})
    _recording_reader(monkeypatch, crash_on="202601a")
    with pytest.raises(RuntimeError, match="simulated kill"):
        ftd.fetch_fails_to_deliver(context)
    after_kill = sorted(set(_stored(sqlite_store)["period"]))
    reads = _recording_reader(monkeypatch)
    ftd.fetch_fails_to_deliver(context)  # the next run, same day
    retry, reads[:] = list(reads), []
    write_ftd_zips(tmp_path / "ftd", {"202602b": _HEADER + "20260220|000111AAA|ABC|61|ABC CORP|10.0\n"})
    ftd.fetch_fails_to_deliver(context)  # a later new-period run

    xyz = _stored(sqlite_store).query("source_symbol == 'XYZ'")[["period", "cusip", "fails_quantity"]].values.tolist()
    print(f"\nSANITY (F-002 crash): periods stored after the kill {after_kill}; retry read {retry}; later run read {reads};")
    print(f"  XYZ lines {xyz} (expected 202601a 7 and 202601b 8)")
    assert after_kill == ["202601a", "202601b"], "a killed run commits nothing: its new period stays pending"
    assert retry == ["202602a", "202601a", "202601b"], "the retry repeats the new period and the full re-read"
    assert xyz == [["202601a", "555555XXX", 7.0], ["202601b", "555555XXX", 8.0]], "XYZ's history survives the killed run"
    assert reads == ["202602b", "202601a", "202601b", "202602a"], "the next new period re-reads every other cached period"


def test_a_key_with_stored_lines_under_one_cusip6_is_reread_for_its_other_families(sqlite_store, tmp_path, monkeypatch):
    """S2: WBD's Sharadar CUSIP 25470F104 is in scope, so the cold build stores DISCA's lines under it. DISCA then joins
    WBD's lineage: the next new-period run stores its lines under the other CUSIP 99999A101."""
    context = _context(sqlite_store, tmp_path, ("WBD",))
    monkeypatch.setattr(ftd, "sharadar_cusips", lambda universe, sharadar: {"WBD": frozenset({"25470F104"})})
    _lineage(sqlite_store, [("WBD", "0001437107", "WBD", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER + "20260105|25470F104|DISCA|100|DISCOVERY A|10.0\n20260105|99999A101|DISCA|7|DISCOVERY A OLD CUSIP|2.0\n",
            "202601b": _HEADER + "20260120|25470F104|DISCA|90|DISCOVERY A|10.0\n20260120|99999A101|DISCA|8|DISCOVERY A OLD CUSIP|2.0\n",
        },
    )
    ftd.fetch_fails_to_deliver(context)  # cold: the 25470F prefix is in scope, the DISCA symbol is not
    _lineage(sqlite_store, [("WBD", "0001437107", "WBD", pd.Timestamp("2020-01-01")), ("WBD", "0001437107", "DISCA", pd.Timestamp("2026-02-03"))])
    write_ftd_zips(tmp_path / "ftd", {"202602a": _HEADER + "20260205|934423104|WBD|60|WBD|10.0\n"})

    ftd.fetch_fails_to_deliver(context)

    other = _stored(sqlite_store).query("cusip == '99999A101'")[["period", "source_symbol", "fails_quantity"]].values.tolist()
    print(f"\nSANITY (F-003 S2 partial key): DISCA lines under 99999A101 {other} (expected 2: 202601a 7, 202601b 8)")
    assert other == [["202601a", "DISCA", 7.0], ["202601b", "DISCA", 8.0]], "a key with lines under one CUSIP-6 is re-read for its others"


def test_a_cusip6_first_stored_by_a_new_period_rereads_the_stored_periods_for_it(sqlite_store, tmp_path, monkeypatch):
    """S3: Sharadar adds ABC's class B CUSIP 000222BBB with no lineage change. The run that stores the new period also
    stores its 202601a/b lines (3 of 3)."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    monkeypatch.setattr(ftd, "sharadar_cusips", lambda universe, sharadar: {"ABC": frozenset({"000111AAA"})})
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|000222BBB|ABCB|7|ABC CL B|2.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|000222BBB|ABCB|8|ABC CL B|2.0\n",
        },
    )
    ftd.fetch_fails_to_deliver(context)
    monkeypatch.setattr(ftd, "sharadar_cusips", lambda universe, sharadar: {"ABC": frozenset({"000111AAA", "000222BBB"})})
    write_ftd_zips(tmp_path / "ftd", {"202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n20260205|000222BBB|ABCB|9|ABC CL B|2.0\n"})

    ftd.fetch_fails_to_deliver(context)

    class_b = _stored(sqlite_store).query("cusip == '000222BBB'")[["period", "fails_quantity"]].values.tolist()
    print(f"\nSANITY (F-003 S3 prefix growth): class B lines {class_b} (expected 3: 202601a 7, 202601b 8, 202602a 9)")
    assert class_b == [["202601a", 7.0], ["202601b", 8.0], ["202602a", 9.0]], "a CUSIP-6 new to the table is re-read once"


def test_a_same_day_rerun_reads_no_file_and_each_new_period_rereads_the_cache_once(sqlite_store, tmp_path, monkeypatch):
    """Guard: GHOST joins ABC's lineage and no file lists it. The run storing 202602a re-reads 202601a/b once; the same-day
    re-run reads nothing; the run storing 202602b re-reads the other cached periods once."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {"202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n", "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n"},
    )
    ftd.fetch_fails_to_deliver(context)
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")), ("ABC", "0000000778", "GHOST", pd.Timestamp("2026-02-03"))])
    write_ftd_zips(tmp_path / "ftd", {"202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n"})
    reads = _recording_reader(monkeypatch)
    runs = []
    for new in (None, None, "202602b"):
        if new:
            write_ftd_zips(tmp_path / "ftd", {new: _HEADER + "20260220|000111AAA|ABC|61|ABC CORP|10.0\n"})
        ftd.fetch_fails_to_deliver(context)
        runs.append(list(reads))
        reads[:] = []

    print(f"\nSANITY (bounded re-read): new-period run read {runs[0]}, same-day re-run {runs[1]}, next new period {runs[2]}")
    assert runs == [["202602a", "202601a", "202601b"], [], ["202602b", "202601a", "202601b", "202602a"]], "a same-day re-run reads 0 periods"


def test_a_curated_cusip6_no_later_file_lists_is_stored_after_the_next_new_period_run(sqlite_store, tmp_path, monkeypatch):
    """S4: the curation adds ABC's CUSIP 000333CCC after the cold build; only 202601a/b list it, the new 202602a does not.
    The run that stores 202602a re-reads 202601a/b with the curated CUSIP-6 and stores both lines."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|000333CCC|ABCX|7|ABC TRACKING|2.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|000333CCC|ABCX|8|ABC TRACKING|2.0\n",
        },
    )
    ftd.fetch_fails_to_deliver(context)
    cold = _stored(sqlite_store).query("cusip == '000333CCC'")["period"].tolist()
    ratio = {"ticker": "ABC", "cusip": "000333CCC", "ratio": 1.0, "valid_from": "2000-01-01", "source": "https://www.sec.gov/example"}
    monkeypatch.setattr(ftd, "load_security_manual", lambda config_dir: parse_security_manual({"conversion_ratios": [ratio]}))
    write_ftd_zips(tmp_path / "ftd", {"202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n"})

    ftd.fetch_fails_to_deliver(context)

    curated = _stored(sqlite_store).query("cusip == '000333CCC'")[["period", "fails_quantity"]].values.tolist()
    print(f"\nSANITY (S4 curated CUSIP-6, no later file): cold stored {cold}; after the new-period run {curated} (expected 202601a 7, 202601b 8)")
    assert cold == [], "before the curation the tracking CUSIP is out of scope"
    assert curated == [["202601a", 7.0], ["202601b", 8.0]], "a curated CUSIP-6 no later file lists gets its history"


def _meta_lineage(store) -> None:
    """META's dated lineage: FB from the 2012 IPO to the 2022-06-09 rename, META from then (open)."""
    windows = [("FB", "2012-05-17", "2022-06-09"), ("META", "2022-06-09", None)]
    rows = [("cik_window", "", "2012-05-17", None), *(("symbol", symbol, start, end) for symbol, start, end in windows)]
    frame = pd.DataFrame(
        [
            {
                "company_id": "E0001326801",
                "canonical_ticker": "META",
                "cik": "0001326801",
                "role": role,
                "symbol": symbol,
                "valid_from": pd.Timestamp(start),
                "valid_to": pd.Timestamp(end) if end else None,
                "status": "corroborated",
                "sources": "form345,roster",
                "scope_changed_at": pd.Timestamp("2026-01-01"),
            }
            for role, symbol, start, end in rows
        ]
    )
    store.save(Tables.universe_cik_symbol, frame)


def test_a_lineage_symbol_brings_a_line_and_its_cusip6_only_inside_its_window(sqlite_store, tmp_path):
    """FB is META's symbol until 2022-06-09. A 2021 FB line on Meta's CUSIP is stored and puts 30303M in scope (its class
    line 30303M999 too); a 2023 META line is stored; the 2025 FB line is a ProShares ETF (74349Y101): neither it nor the
    other ProShares ETF SH (74349Y102) is stored, as the out-of-window match adds no CUSIP-6."""
    context = _context(sqlite_store, tmp_path, ("META",))
    _meta_lineage(sqlite_store)
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202106a": _HEADER + "20210604|30303M102|FB|100|FACEBOOK INC CL A|330.0\n20210604|30303M999|FBB|5|FACEBOOK INC CL B|1.0\n",
            "202306a": _HEADER + "20230605|30303M102|META|50|META PLATFORMS INC|270.0\n",
            "202506a": _HEADER + "20250605|74349Y101|FB|10|PROSHARES S&P 500 DYNAMIC|20.0\n20250605|74349Y102|SH|20|PROSHARES SHORT S&P500|40.0\n",
        },
    )

    ftd.fetch_fails_to_deliver(context)

    stored = _stored(sqlite_store)[["period", "cusip", "source_symbol"]].values.tolist()
    print(f"\nSANITY (dated symbol scope): stored {stored}; the 2025 ProShares FB/SH lines (74349Y) are expected out")
    assert stored == [
        ["202106a", "30303M102", "FB"],
        ["202106a", "30303M999", "FBB"],
        ["202306a", "30303M102", "META"],
    ], "only in-window symbol lines (and the CUSIP-6 they bring) are stored"


def test_a_stored_out_of_window_symbol_line_puts_no_cusip6_in_scope(sqlite_store, tmp_path):
    """The table already holds a 2025 ProShares `FB` line (74349Y101, out of META's FB window): the next period's other
    ProShares ETF SH (74349Y102) is not stored, while META's own line is."""
    context = _context(sqlite_store, tmp_path, ("META",))
    _meta_lineage(sqlite_store)
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202506a": _HEADER + "20250605|74349Y101|FB|10|PROSHARES S&P 500 DYNAMIC|20.0\n",
            "202507a": _HEADER + "20250707|74349Y102|SH|20|PROSHARES SHORT S&P500|40.0\n20250707|30303M102|META|50|META PLATFORMS INC|700.0\n",
        },
    )
    bloat = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-06-05")],
            "trade_date": [pd.Timestamp("2025-06-03")],
            "cusip": ["74349Y101"],
            "source_symbol": ["FB"],
            "description": ["PROSHARES S&P 500 DYNAMIC"],
            "price": [20.0],
            "fails_quantity": [10.0],
            "fails_value": [200.0],
            "period": ["202506a"],
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, bloat)

    ftd.fetch_fails_to_deliver(context)

    stored = _stored(sqlite_store)[["period", "cusip", "source_symbol"]].values.tolist()
    print(f"\nSANITY (dated stored prefixes): stored {stored}; SH 74349Y102 expected out, META 30303M102 in")
    assert ["202507a", "74349Y102", "SH"] not in stored, "an out-of-window stored FB line adds no CUSIP-6 to the scope"
    assert ["202507a", "30303M102", "META"] in stored
