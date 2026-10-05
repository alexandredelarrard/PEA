"""RegSHO short-volume fetch: resume window, incremental write, `full` rebuild and its abort gate.

Resume is on the GLOBAL max date, not per ticker: one RegSHO file covers the whole market, so once day D is
stored every ticker has D. `full` re-fetches every served date (2018-08-01 on, plus the 2017-12-29 file) and
keeps no legacy row; a stored date the source fails to serve aborts it before any write.
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
from tests.data_extract.fake_context import extract_config
from tests.data_extract.institutionals.test_short_volume_security import UNIVERSE, _file, _identity, _master


def _context(store, tmp_path: Path | None = None) -> Any:
    return SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path}, log=logging.getLogger("test.regsho"), config=extract_config(data_extract={}))


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
    monkeypatch.setattr(si, "record_run", lambda *a, **k: None)
    return asked


def _grain(store) -> dict[tuple[str, str], float]:
    frame = store.load(Tables.short_interest)
    return {(str(t), str(pd.Timestamp(d).date())): float(v) for t, d, v in zip(frame["ticker"], frame["date"], frame["short_volume"], strict=True)}


def test_resume_day_replays_a_bounded_tail_from_the_global_max(sqlite_store):
    ctx = _context(sqlite_store)
    cold = si._resume_day(ctx, years_history=10)
    assert cold == pd.Timestamp.today().normalize() - pd.DateOffset(years=10)

    sqlite_store.replace(Tables.sec_short_volume_security, _raw([("LEN", "2024-05-01", 1.0, 2.0), ("LEN/B", "2024-06-04", 1.0, 2.0)]))
    assert si._resume_day(ctx, years_history=10) == pd.Timestamp("2024-05-24")

    print("\n=== SANITY CHECK: RegSHO resume day ===")
    print(f"  cold table -> {cold.date()} (years_history); stored raw max 2024-06-04 -> 2024-05-24 (7-session repair tail). Validated.")


def test_incremental_run_rewrites_only_the_fetched_days(sqlite_store, tmp_path, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2020-01-02", 1.0, 2.0), ("LEN", "2020-01-03", 3.0, 4.0)]))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "ticker": "LEN", "short_volume": [1.0, 3.0], "total_volume": [2.0, 4.0]}),
    )
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2020-01-06")))
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: pd.Timestamp("2020-01-03"))
    asked = _serve(monkeypatch, {"20200103": [("LEN", 30.0, 40.0), ("LEN/B", 5.0, 6.0), ("ZZZZ", 1.0, 1.0)], "20200106": [("LEN", 50.0, 60.0)]})

    si.fetch_short_interest(_context(sqlite_store, tmp_path), UNIVERSE, pause=0.0, identity=_identity())

    assert [d.date().isoformat() for d in asked] == ["2020-01-03", "2020-01-06"]
    assert _grain(sqlite_store) == {("LEN", "2020-01-02"): 1.0, ("LEN", "2020-01-03"): 35.0, ("LEN", "2020-01-06"): 50.0}
    raw = sqlite_store.load(Tables.sec_short_volume_security)
    assert sorted(raw["source_symbol"]) == ["LEN", "LEN", "LEN", "LEN/B"], "ZZZZ is no universe security"
    print("\n=== SANITY CHECK: RegSHO incremental ===")
    print("  fetched days rewritten (LEN 2020-01-03 = 30 + LEN-B 5); the unfetched 2020-01-02 row kept; ZZZZ not stored")


def test_full_refetches_every_served_day_and_keeps_no_legacy_row(sqlite_store, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(
        Tables.short_interest, pd.DataFrame({"date": pd.to_datetime(["2015-01-02"]), "ticker": "LEN", "short_volume": [9.0], "total_volume": [9.0]})
    )
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2015-01-02", 9.0, 9.0)]))
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2018-08-03")))
    files = {"20171229": [("LEN", 1.0, 2.0)], "20180801": [("LEN", 3.0, 4.0)], "20180802": [("LEN", 5.0, 6.0)], "20180803": [("LEN", 7.0, 8.0)]}
    asked = _serve(monkeypatch, files)

    with pytest.raises(RuntimeError, match="2015-01-02"):
        si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity())
    assert _grain(sqlite_store) == {("LEN", "2015-01-02"): 9.0}, "a stored date the source no longer serves aborts before any write"

    sqlite_store.drop(Tables.sec_short_volume_security)
    asked.clear()
    si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity())

    assert {d.date().isoformat() for d in asked} >= {"2017-12-29", "2018-08-01", "2018-08-02", "2018-08-03"}
    assert min(asked) == pd.Timestamp("2017-12-29") and pd.Timestamp("2018-07-31") not in asked
    assert _grain(sqlite_store) == {("LEN", "2017-12-29"): 1.0, ("LEN", "2018-08-01"): 3.0, ("LEN", "2018-08-02"): 5.0, ("LEN", "2018-08-03"): 7.0}
    print("\n=== SANITY CHECK: RegSHO full ===")
    print("  a stored 2015 date the source no longer serves aborts the run with both tables intact;")
    print("  without it, full re-fetches 2017-12-29 plus 2018-08-01 on and the legacy 2015 ticker row is gone")


def test_full_aborts_before_replace_when_a_served_date_fails(sqlite_store, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    sqlite_store.save(Tables.sec_short_volume_security, _raw([("LEN", "2018-08-01", 1.0, 2.0), ("LEN", "2018-08-02", 3.0, 4.0)]))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame({"date": pd.to_datetime(["2018-08-01", "2018-08-02"]), "ticker": "LEN", "short_volume": [1.0, 3.0], "total_volume": [2.0, 4.0]}),
    )
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, tz=None: pd.Timestamp("2018-08-02")))
    _serve(monkeypatch, {"20171229": [("LEN", 1.0, 2.0)], "20180801": [("LEN", 30.0, 40.0)]})  # 2018-08-02 is stored but not served

    with pytest.raises(RuntimeError, match="2018-08-02"):
        si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity())
    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}
    assert len(sqlite_store.load(Tables.sec_short_volume_security)) == 2

    def _boom(day: pd.Timestamp, session: object = None) -> str | None:
        raise ConnectionError("reset")

    monkeypatch.setattr(si, "_fetch_day", _boom)
    with pytest.raises(RuntimeError, match="aborting"):
        si.fetch_short_interest(_context(sqlite_store), UNIVERSE, pause=0.0, full=True, identity=_identity())
    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 1.0, ("LEN", "2018-08-02"): 3.0}
    print("\n=== SANITY CHECK: RegSHO full abort gate ===")
    print("  a stored date not served, or a network error, aborts full before replace; both tables keep their rows")


def test_empty_download_leaves_the_tables_untouched(sqlite_store, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(si, "record_run", lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(si, "_fetch_day", lambda day, session=None: None)
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: pd.Timestamp.today().normalize())

    assert si.fetch_short_interest(_context(sqlite_store), tickers=["LEN"], pause=0.0, identity=_identity()) is None
    assert not sqlite_store.exists(Tables.short_interest) and not sqlite_store.exists(Tables.sec_short_volume_security)
    assert recorded, "an empty window must still record the run"
    print("\n=== SANITY CHECK: RegSHO empty window ===")
    print("  every day-file missing -> no crash, no table created, run still recorded. Validated.")


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
