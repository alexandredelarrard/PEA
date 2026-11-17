"""RegSHO short-volume fetch: day-file plan, incremental write, `full` rebuild and its abort gate.

One RegSHO file covers the whole market, so the plan is table-wide: the business days from the earlier of the
`sec_short_volume_security` latest day and today less its overlap, to today. An empty table, `full` or the first run of a
new half-month starts at the source start; `full` aborts before any write on a fetch error or when nothing is served. The in-scope lines are stored as
filed (a keyed upsert).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.institutionals import fetch_short_interest as si
from src.data_store.schema import Tables
from src.utils.window import WindowRun
from tests.canonical_schema import create_canonical
from tests.data_extract.fake_context import extract_config
from tests.data_extract.institutionals.test_short_volume_security import _file

UNIVERSE = ["LEN"]
_AS_OF = pd.Timestamp("2024-06-28")


@pytest.fixture
def sqlite_store(sqlite_store, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(si.time, "sleep", lambda seconds: None)
    create_canonical(sqlite_store.engine, [Tables.universe_cik_symbol, Tables.sec_fails_to_deliver_security, Tables.sec_short_volume_security])
    lineage = pd.DataFrame(
        [
            ("0000920760", "cik_window", "", "register"),
            ("0000920760", "symbol", "LEN", "form345,roster"),
            ("0000920760", "symbol", "LEN-B", "form345"),
        ],
        columns=["cik", "role", "symbol", "sources"],
    )
    lineage = lineage.assign(company_id="E0000920760", canonical_ticker="LEN", valid_from=pd.Timestamp("1900-01-01"), status="corroborated")
    sqlite_store.save(Tables.universe_cik_symbol, lineage)
    return sqlite_store


def _context(store, tmp_path: Path | None = None, *, as_of: pd.Timestamp = _AS_OF, full: bool = False) -> Any:
    return SimpleNamespace(
        store=store,
        run=WindowRun("full" if full else "incremental", "FINRA fixture", as_of),
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger("test.regsho"),
        config=extract_config(),
    )


def _raw(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Stored lines (symbol, day, short, total) as filed."""
    frame = pd.DataFrame(rows, columns=["source_symbol", "date", "short_volume", "total_volume"])
    return frame.assign(date=pd.to_datetime(frame["date"]), market="Q", short_exempt_volume=0.0)[list(si.RAW_COLUMNS)]


def _serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, list[tuple[str, float, float]]]) -> list[pd.Timestamp]:
    """Serve `files` ({yyyymmdd: [(symbol, short, total)]}); every other day is not served. Returns the days asked."""
    asked: list[pd.Timestamp] = []

    def _fetch(day: pd.Timestamp, session: object = None) -> str | None:
        del session
        asked.append(day)
        rows = files.get(day.strftime("%Y%m%d"))
        return None if rows is None else _file(day.strftime("%Y%m%d"), [(s, sv, 0, tv, "Q,N") for s, sv, tv in rows])

    monkeypatch.setattr(si, "_fetch_day", _fetch)
    return asked


def _stored(store) -> dict[tuple[str, str], float]:
    frame = store.load(Tables.sec_short_volume_security)
    return {
        (str(s), str(pd.Timestamp(d).date())): float(v) for s, d, v in zip(frame["source_symbol"], frame["date"], frame["short_volume"], strict=True)
    }


def test_an_up_to_date_table_rereads_the_configured_thirty_day_overlap(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("AAA", "2024-06-27", 1.0, 2.0)]))
    plan = si._plan_days(_context(sqlite_store))
    assert plan.equals(pd.bdate_range("2024-05-28", _AS_OF))
    print("SANITY: the FINRA source frontier minus30CD determines the day-file window.")


def test_a_stale_table_catches_up_from_its_frontier_minus_overlap(sqlite_store) -> None:
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("AAA", "2024-06-03", 1.0, 2.0), ("BBB", "2024-06-17", 1.0, 2.0)]))
    plan = si._plan_days(_context(sqlite_store))
    assert plan.equals(pd.bdate_range("2024-05-20", _AS_OF))
    print("SANITY: FINRA catches up from the market-wide frontier with the same configured overlap.")


@pytest.mark.parametrize(
    ("latest", "day", "first"),
    [
        ("2026-06-15", "2026-06-17", "2026-05-18"),
        ("2026-06-30", "2026-07-02", "2026-06-01"),
        ("2026-12-31", "2027-01-04", "2026-12-01"),
        ("2026-05-15", "2026-05-19", "2026-04-15"),
    ],
)
def test_calendar_boundaries_cannot_trigger_an_implicit_historical_rebuild(sqlite_store, latest: str, day: str, first: str) -> None:
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("AAA", latest, 1.0, 2.0)]))
    plan = si._plan_days(_context(sqlite_store, as_of=pd.Timestamp(day)))
    assert plan.equals(pd.bdate_range(first, day))
    print(f"SANITY: a run on{day} starts at{first}; half-month/month/year boundaries never override config.")


def test_cold_and_explicit_full_keep_fixed_source_bounds(sqlite_store) -> None:
    context = _context(sqlite_store)
    cold = si._plan_days(context)
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("AAA", "2024-06-27", 1.0, 2.0)]))
    context.run = WindowRun("full", "explicit FINRA replay", context.run.instant)
    full = si._plan_days(context)
    assert cold.equals(pd.bdate_range("2018-08-01", _AS_OF)) and full.equals(cold)
    print("SANITY: cold/full runs use the configured2018-08-01 source floor, independently of relative years.")


def test_incremental_run_upserts_the_fetched_days_as_filed(sqlite_store, tmp_path, monkeypatch):
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2020-01-02", 1.0, 2.0), ("LEN", "2020-01-03", 3.0, 4.0)]))
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.bdate_range("2020-01-03", "2020-01-06"))
    asked = _serve(monkeypatch, {"20200103": [("LEN", 30.0, 40.0), ("LEN/B", 5.0, 6.0), ("ZZZZ", 1.0, 1.0)], "20200106": [("LEN", 50.0, 60.0)]})

    si.fetch_short_interest(_context(sqlite_store, tmp_path), UNIVERSE)

    assert [d.date().isoformat() for d in asked] == ["2020-01-03", "2020-01-06"]
    assert _stored(sqlite_store) == {
        ("LEN", "2020-01-02"): 1.0,
        ("LEN", "2020-01-03"): 30.0,
        ("LEN/B", "2020-01-03"): 5.0,
        ("LEN", "2020-01-06"): 50.0,
    }, "LEN/B as filed (no class sum), ZZZZ out of scope"
    assert sqlite_store.columns(Tables.sec_short_volume_security) == list(si.RAW_COLUMNS)
    print("\n=== SANITY CHECK: RegSHO incremental ===")
    print("  fetched days upserted as filed (LEN 30 and LEN/B 5 on 2020-01-03, not summed); the unfetched 2020-01-02 line kept; ZZZZ not stored")


def test_full_refetches_every_served_day_from_the_source_start(sqlite_store, monkeypatch):
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2018-08-02", 99.0, 99.0)]))
    asked = _serve(monkeypatch, {"20180801": [("LEN", 3.0, 4.0)], "20180802": [("LEN", 5.0, 6.0)], "20180803": [("LEN", 7.0, 8.0)]})

    si.fetch_short_interest(_context(sqlite_store, as_of=pd.Timestamp("2018-08-03"), full=True), UNIVERSE)

    assert {d.date().isoformat() for d in asked} == {"2018-08-01", "2018-08-02", "2018-08-03"}
    assert _stored(sqlite_store) == {("LEN", "2018-08-01"): 3.0, ("LEN", "2018-08-02"): 5.0, ("LEN", "2018-08-03"): 7.0}
    print("\n=== SANITY CHECK: RegSHO full ===")
    print("  full re-fetches every day from the registered 2018-08-01 source start and the stale 2018-08-02 value 99 is replaced")


def test_full_aborts_before_any_write_on_a_fetch_error_or_an_empty_source(sqlite_store, monkeypatch):
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2018-08-01", 1.0, 2.0), ("LEN", "2018-08-02", 3.0, 4.0)]))
    _serve(monkeypatch, {})

    with pytest.raises(RuntimeError, match="aborting"):
        si.fetch_short_interest(_context(sqlite_store, as_of=pd.Timestamp("2018-08-02"), full=True), UNIVERSE)
    assert _stored(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}

    def _boom(day: pd.Timestamp, session: object = None) -> str | None:
        raise ConnectionError("reset")

    monkeypatch.setattr(si, "_fetch_day", _boom)
    with pytest.raises(RuntimeError, match="aborting"):
        si.fetch_short_interest(_context(sqlite_store, as_of=pd.Timestamp("2018-08-02"), full=True), UNIVERSE)
    assert _stored(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}
    print("\n=== SANITY CHECK: RegSHO full abort gate ===")
    print("  nothing served, or a network error, aborts full before the write; the stored lines are kept")


def test_a_run_upserts_its_lines_and_leaves_every_other_symbol(sqlite_store, monkeypatch):
    """A `-t` run re-reads the planned days for its scope only: another symbol's stored line survives."""
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2018-08-01", 9.0, 9.0), ("OTHR", "2018-08-01", 7.0, 7.0)]))
    _serve(monkeypatch, {"20180801": [("LEN", 3.0, 4.0)], "20180802": [("LEN", 5.0, 6.0)]})

    si.fetch_short_interest(_context(sqlite_store, as_of=pd.Timestamp("2018-08-02"), full=True), ["LEN"])

    assert _stored(sqlite_store) == {("LEN", "2018-08-01"): 3.0, ("LEN", "2018-08-02"): 5.0, ("OTHR", "2018-08-01"): 7.0}
    print("\n=== SANITY CHECK: RegSHO scoped run ===")
    print("  -t LEN -F: LEN's served days re-read and upserted; OTHR's stored line untouched. Validated.")


def test_empty_download_leaves_the_table_untouched(sqlite_store, monkeypatch):
    sqlite_store.drop(Tables.sec_short_volume_security)
    monkeypatch.setattr(si, "_fetch_day", lambda day, session=None: None)
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.bdate_range("2024-06-24", "2024-06-28"))

    assert si.fetch_short_interest(_context(sqlite_store), tickers=["LEN"]) is None
    assert not sqlite_store.exists(Tables.sec_short_volume_security)
    print("\n=== SANITY CHECK: RegSHO empty window ===")
    print("  every day-file missing -> no crash, no table created; the frontier is unchanged, so they are planned again. Validated.")


def test_fetch_day_reuses_the_supplied_http_session():
    calls: list[tuple[str, dict]] = []

    class Session:
        def get(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=200, text="payload")

    assert si._fetch_day(pd.Timestamp("2026-09-22"), cast(Any, Session())) == "payload"
    assert len(calls) == 1 and calls[0][0].endswith("CNMSshvol20260922.txt")
    print("\n=== SANITY CHECK: RegSHO connection reuse ===")
    print("  one run-scoped HTTP session serves the daily FINRA request")


def test_a_lineage_symbol_brings_a_finra_line_only_inside_its_window(sqlite_store, monkeypatch):
    """FB is META's symbol until 2022-06-09 and META from then; an FTD line of Meta's CUSIP under FB is stored. A FINRA FB
    line is stored in 2021 but not in 2025 (a ProShares ETF), a META line in 2025 but not in 2021 (another ETF trust)."""
    lineage = pd.DataFrame(
        [
            ("cik_window", "", pd.Timestamp("2012-05-17"), pd.NaT),
            ("symbol", "FB", pd.Timestamp("2012-05-17"), pd.Timestamp("2022-06-09")),
            ("symbol", "META", pd.Timestamp("2022-06-09"), pd.NaT),
        ],
        columns=["role", "symbol", "valid_from", "valid_to"],
    )
    lineage = lineage.assign(cik="0001326801", company_id="E0001326801", canonical_ticker="META", sources="form345,roster", status="corroborated")
    sqlite_store.save(Tables.universe_cik_symbol, lineage)
    fb_line = {"date": pd.Timestamp("2021-06-04"), "trade_date": pd.Timestamp("2021-06-02"), "cusip": "30303M102", "source_symbol": "FB"}
    sqlite_store.save(Tables.sec_fails_to_deliver_security, pd.DataFrame([{**fb_line, "fails_quantity": 100.0, "period": "202106a"}]))
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.DatetimeIndex(["2021-06-01", "2025-06-02"]))
    _serve(monkeypatch, {"20210601": [("FB", 10.0, 20.0), ("META", 1.0, 2.0)], "20250602": [("FB", 3.0, 4.0), ("META", 30.0, 40.0)]})

    si.fetch_short_interest(_context(sqlite_store), ["LEN", "META"])

    stored = _stored(sqlite_store)
    print(f"\n=== SANITY CHECK: dated RegSHO scope === stored {sorted(stored)}; FB 2025 and META 2021 expected out")
    assert stored == {("FB", "2021-06-01"): 10.0, ("META", "2025-06-02"): 30.0}, "a lineage symbol is in scope only inside its window"


def test_a_lineage_symbol_is_also_in_scope_inside_its_stored_ftd_span(sqlite_store, monkeypatch):
    """News Corp's lineage dates NWSA from 2019-11-08, but its stored FTD NWSA lines (on its own CUSIP-6, so in scope by
    construction) run from 2013: a 2019-01-02 FINRA NWSA line is stored; a 2012 one, before both, is not."""
    lineage = pd.DataFrame(
        [("cik_window", "", pd.Timestamp("2013-06-28"), pd.NaT), ("symbol", "NWSA", pd.Timestamp("2019-11-08"), pd.NaT)],
        columns=["role", "symbol", "valid_from", "valid_to"],
    )
    lineage = lineage.assign(cik="0001564708", company_id="E0001564708", canonical_ticker="NWSA", sources="form345,roster", status="corroborated")
    sqlite_store.save(Tables.universe_cik_symbol, lineage)
    lines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-07-03", "2026-01-06"]),
            "trade_date": pd.to_datetime(["2013-07-01", "2026-01-02"]),
            "cusip": "65249B109",
            "source_symbol": "NWSA",
            "fails_quantity": 10.0,
            "period": ["201307a", "202601a"],
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, lines)
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.DatetimeIndex(["2012-06-01", "2019-01-02"]))
    _serve(monkeypatch, {"20120601": [("NWSA", 1.0, 2.0)], "20190102": [("NWSA", 5.0, 6.0)]})

    si.fetch_short_interest(_context(sqlite_store), ["NWSA"])

    stored = _stored(sqlite_store)
    print(f"\n=== SANITY CHECK: FTD-span RegSHO scope === stored {sorted(stored)}; 2019-01-02 in (FTD span), 2012-06-01 out")
    assert stored == {("NWSA", "2019-01-02"): 5.0}, "the stored FTD span dates a lineage symbol too"
