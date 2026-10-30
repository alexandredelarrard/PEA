"""RegSHO short-volume fetch: day-file plan, incremental write, `full` rebuild and its abort gate.

One RegSHO file covers the whole market, so the plan is table-wide: the business days from the earlier
of the table's latest day and today less its overlap, to today. An empty table or `full` starts at the
source start; `full` aborts before any write on a fetch error or when nothing is served.
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
from src.data_store.store import _create_canonical
from tests.data_extract.fake_context import extract_config
from tests.data_extract.institutionals.test_short_volume_security import UNIVERSE, _file, _identity, _master


@pytest.fixture
def sqlite_store(sqlite_store):
    _create_canonical(
        sqlite_store.engine,
        [
            Tables.sp500_tickers,
            Tables.entity_lineage,
            Tables.symbol_tenure,
            Tables.security_master,
            Tables.sec_fails_to_deliver_security,
            Tables.sec_fails_to_deliver,
            Tables.sec_short_volume_security,
            Tables.short_interest,
        ],
    )
    return sqlite_store


def _context(store, tmp_path: Path | None = None) -> Any:
    return SimpleNamespace(
        store=store, paths={"DATA_STORE": tmp_path}, log=logging.getLogger("test.regsho"), config=extract_config(data_extract={"redundant_ticks": []})
    )


def _raw(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Stored raw rows (symbol, day, short, total) stamped as LEN's canonical line."""
    frame = pd.DataFrame(rows, columns=["source_symbol", "date", "short_volume", "total_volume"])
    return frame.assign(
        date=pd.to_datetime(frame["date"]),
        market="Q",
        short_exempt_volume=0.0,
        security_id="C526057104",
        ticker="LEN",
        lineage_role="canonical_current",
        security_class="class_A",
    )[list(si.SECURITY_COLUMNS)]


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


def _grain(store) -> dict[tuple[str, str], float]:
    frame = store.load(Tables.short_interest)
    return {(str(t), str(pd.Timestamp(d).date())): float(v) for t, d, v in zip(frame["ticker"], frame["date"], frame["short_volume"], strict=True)}


_AS_OF = pd.Timestamp("2024-06-28")  # a Friday


def _si(ticker: str, dates) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": pd.DatetimeIndex(dates), "short_volume": 1.0, "total_volume": 2.0})


def test_an_up_to_date_table_rereads_the_overlap_window(sqlite_store):
    ctx = _context(sqlite_store)
    sqlite_store.save(Tables.short_interest, _si("AAA", ["2024-06-27"]))

    plan = si._plan_days(ctx, False, _AS_OF)

    assert plan.equals(pd.bdate_range("2024-06-21", _AS_OF))  # today - 7 d is earlier than the latest day
    print("\n=== SANITY CHECK: RegSHO day plan, up to date ===")
    print(f"  latest stored day 06-27 -> {len(plan)} day files from today - 7 d (06-21) to today.")


def test_a_stale_table_resumes_from_its_latest_day(sqlite_store):
    ctx = _context(sqlite_store)
    sqlite_store.save(Tables.short_interest, _si("AAA", ["2024-05-01"]))
    sqlite_store.save(Tables.short_interest, _si("BBB", ["2024-05-20"]))

    plan = si._plan_days(ctx, False, _AS_OF)

    assert plan.equals(pd.bdate_range("2024-05-20", _AS_OF)), "the table-wide latest day, not each ticker's own"
    print("\n=== SANITY CHECK: RegSHO day plan, stale ===")
    print(f"  AAA stops 05-01, BBB 05-20 -> {len(plan)} day files from the table's latest day 05-20.")


def test_an_empty_table_and_full_start_at_the_source_start(sqlite_store):
    ctx = _context(sqlite_store)
    assert Tables.short_interest.resume is not None and Tables.short_interest.resume.source_start == "2018-08-01"
    cold = si._plan_days(ctx, False, _AS_OF)
    sqlite_store.save(Tables.short_interest, _si("AAA", ["2024-06-27"]))
    full = si._plan_days(ctx, True, _AS_OF)

    expected = pd.bdate_range("2018-08-01", _AS_OF)
    assert cold.equals(expected) and full.equals(expected)
    print("\n=== SANITY CHECK: RegSHO day plan, cold and full ===")
    print(f"  no table or --full -> all {len(full)} business days from the 2018-08-01 source start to today.")


def test_incremental_run_rewrites_only_the_fetched_days(sqlite_store, tmp_path, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2020-01-02", 1.0, 2.0), ("LEN", "2020-01-03", 3.0, 4.0)]))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "ticker": "LEN", "short_volume": [1.0, 3.0], "total_volume": [2.0, 4.0]}),
    )
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2020-01-06")))
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.bdate_range("2020-01-03", "2020-01-06"))
    asked = _serve(monkeypatch, {"20200103": [("LEN", 30.0, 40.0), ("LEN/B", 5.0, 6.0), ("ZZZZ", 1.0, 1.0)], "20200106": [("LEN", 50.0, 60.0)]})

    si.fetch_short_interest(_context(sqlite_store, tmp_path), UNIVERSE, pause=0.0, identity=_identity(store=sqlite_store))

    assert [d.date().isoformat() for d in asked] == ["2020-01-03", "2020-01-06"]
    assert {key: value for key, value in _grain(sqlite_store).items() if value > 0} == {
        ("LEN", "2020-01-02"): 1.0,
        ("LEN", "2020-01-03"): 35.0,
        ("LEN", "2020-01-06"): 50.0,
    }
    quiet = sqlite_store.load(Tables.short_interest, where={"short_volume": 0})
    assert quiet is not None and quiet["total_volume"].isna().all() and quiet["source_checked_at"].notna().all()
    raw = sqlite_store.load(Tables.sec_short_volume_security)
    assert sorted(raw["source_symbol"]) == ["LEN", "LEN", "LEN", "LEN/B"], "ZZZZ is no universe security"
    print("\n=== SANITY CHECK: RegSHO incremental ===")
    print("  fetched days rewritten (LEN 2020-01-03 = 30 + LEN-B 5); the unfetched 2020-01-02 row kept; ZZZZ not stored")


def test_full_refetches_every_served_day_from_the_source_start(sqlite_store, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    legacy = pd.DataFrame({"date": pd.to_datetime(["2018-08-02"]), "ticker": "LEN", "short_volume": [99.0], "total_volume": [99.0]})
    sqlite_store.save(Tables.short_interest, legacy)
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2018-08-03")))
    files = {"20180801": [("LEN", 3.0, 4.0)], "20180802": [("LEN", 5.0, 6.0)], "20180803": [("LEN", 7.0, 8.0)]}
    asked = _serve(monkeypatch, files)

    si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity(store=sqlite_store))

    assert {d.date().isoformat() for d in asked} == {"2018-08-01", "2018-08-02", "2018-08-03"}
    assert pd.Timestamp("2018-07-31") not in asked
    assert {key: value for key, value in _grain(sqlite_store).items() if value > 0} == {
        ("LEN", "2018-08-01"): 3.0,
        ("LEN", "2018-08-02"): 5.0,
        ("LEN", "2018-08-03"): 7.0,
    }
    quiet = sqlite_store.load(Tables.short_interest, where={"short_volume": 0})
    assert quiet is not None and quiet["total_volume"].isna().all() and quiet["source_checked_at"].notna().all()
    print("\n=== SANITY CHECK: RegSHO full ===")
    print("  full re-fetches every day from the registered 2018-08-01 source start and the legacy 2018-08-02 value 99 is replaced")


def test_full_aborts_before_any_write_on_a_fetch_error_or_an_empty_source(sqlite_store, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame({"date": pd.to_datetime(["2018-08-01", "2018-08-02"]), "ticker": "LEN", "short_volume": [1.0, 3.0], "total_volume": [2.0, 4.0]}),
    )
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2018-08-02")))
    _serve(monkeypatch, {})

    with pytest.raises(RuntimeError, match="aborting"):
        si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity(store=sqlite_store))
    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}

    def _boom(day: pd.Timestamp, session: object = None) -> str | None:
        raise ConnectionError("reset")

    monkeypatch.setattr(si, "_fetch_day", _boom)
    with pytest.raises(RuntimeError, match="aborting"):
        si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity(store=sqlite_store))
    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}
    print("\n=== SANITY CHECK: RegSHO full abort gate ===")
    print("  nothing served, or a network error, aborts full before the write; the ticker table keeps its rows")


def test_a_run_upserts_its_tickers_and_leaves_every_other_key(sqlite_store, monkeypatch):
    """A `-t` run re-reads the planned days for its tickers only: another key's rows survive."""
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["LEN", "OTHR"], "cik": ["0000920760", "0000000009"]}))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame(
            {"date": pd.to_datetime(["2018-08-01", "2018-08-01"]), "ticker": ["LEN", "OTHR"], "short_volume": [9.0, 7.0], "total_volume": [9.0, 7.0]}
        ),
    )
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2018-08-02")))
    _serve(monkeypatch, {"20180801": [("LEN", 3.0, 4.0)], "20180802": [("LEN", 5.0, 6.0)]})
    context = _context(sqlite_store)
    context.config = extract_config(data_extract={"redundant_ticks": []})

    si.fetch_short_interest(context, ["LEN"], pause=0.0, full=True, identity=_identity(store=sqlite_store))

    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 3.0, ("LEN", "2018-08-02"): 5.0, ("OTHR", "2018-08-01"): 7.0}
    print("\n=== SANITY CHECK: RegSHO scoped run ===")
    print("  -t LEN -F: LEN's served days re-read and upserted; OTHR's stored row untouched. Validated.")


def test_empty_download_leaves_the_tables_untouched(sqlite_store, monkeypatch):
    sqlite_store.drop(Tables.short_interest)
    sqlite_store.drop(Tables.sec_short_volume_security)
    monkeypatch.setattr(si, "_fetch_day", lambda day, session=None: None)
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.bdate_range("2024-06-24", "2024-06-28"))

    assert si.fetch_short_interest(_context(sqlite_store), tickers=["LEN"], pause=0.0, identity=_identity(store=sqlite_store)) is None
    assert not sqlite_store.exists(Tables.short_interest) and not sqlite_store.exists(Tables.sec_short_volume_security)
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
