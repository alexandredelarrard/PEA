"""RegSHO short-volume fetch: day-file plan, incremental write, `full` rebuild and its abort gate.

One RegSHO file covers the whole market, so the run reads the UNION of the sessions any universe key
needs (`resume.series_windows` on `sec_short_interest`): each key's own last date minus the overlap
and the full window for a new key. A hole is a DAY: a calendar session inside the stored span on which
no key has a row (the file was never stored); `repair=True` re-reads per-key gaps once. An unscoped
`full` re-fetches every served date (2018-08-01 on, plus the 2017-12-29 file) and keeps no legacy row;
a stored date the source fails to serve aborts it before any write. A scoped `full` upserts its tickers.
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
    return asked


def _grain(store) -> dict[tuple[str, str], float]:
    frame = store.load(Tables.short_interest)
    return {(str(t), str(pd.Timestamp(d).date())): float(v) for t, d, v in zip(frame["ticker"], frame["date"], frame["short_volume"], strict=True)}


_AS_OF = pd.Timestamp("2024-06-29")  # last completed session: Friday 2024-06-28
_SESSIONS = pd.bdate_range("2024-05-01", "2024-06-28")


def _si(ticker: str, dates) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": pd.DatetimeIndex(dates), "short_volume": 1.0, "total_volume": 2.0})


def _seed_universe(store, tickers: list[str], added_on: dict[str, str]) -> None:
    store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {
                "ticker": tickers,
                "cik": [str(i) for i in range(len(tickers))],
                "added_on": [pd.Timestamp(added_on.get(t, "2000-01-01")) for t in tickers],
            }
        ),
    )


def test_the_day_plan_is_the_union_of_every_key_s_windows(sqlite_store):
    ctx = _context(sqlite_store)
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "CAL", "date": _SESSIONS, "close_split": 1.0}))  # the calendar
    cold = si._plan_days(ctx, ["AAA"], 1, False, _AS_OF)

    sqlite_store.save(Tables.short_interest, _si("AAA", _SESSIONS))
    sqlite_store.save(Tables.short_interest, _si("LAG", _SESSIONS[_SESSIONS <= pd.Timestamp("2024-06-10")]))
    _seed_universe(sqlite_store, ["AAA", "LAG", "NEW"], {"NEW": "2024-06-27"})
    warm = si._plan_days(ctx, ["AAA", "LAG"], 1, False, _AS_OF)
    with_new = si._plan_days(ctx, ["AAA", "LAG", "NEW"], 1, False, _AS_OF)

    assert cold.equals(_SESSIONS)  # no table: every calendar session from the 1-year floor
    assert warm.equals(_SESSIONS[_SESSIONS >= pd.Timestamp("2024-06-03")])  # LAG's own window from 06-10 - 7 d
    assert with_new.equals(_SESSIONS)  # NEW (added inside the overlap) reads back to the floor
    print("\n=== SANITY CHECK: RegSHO day plan ===")
    print(f"  cold -> {len(cold)} sessions; LAG's own window from 06-03 -> {len(warm)} day files;")
    print(f"  a new key widens the union to the whole window ({len(with_new)} sessions). Each day file is read once.")


def test_a_day_with_rows_for_some_keys_is_not_re_read_a_day_with_none_is(sqlite_store):
    ctx = _context(sqlite_store)
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "CAL", "date": _SESSIONS, "close_split": 1.0}))
    partial = pd.Timestamp("2024-05-14")  # AAA missing, BBB stored: the file was read
    empty = pd.Timestamp("2024-05-21")  # nobody stored: the file was never read
    sqlite_store.save(Tables.short_interest, _si("AAA", _SESSIONS.difference(pd.DatetimeIndex([partial, empty]))))
    sqlite_store.save(Tables.short_interest, _si("BBB", _SESSIONS.difference(pd.DatetimeIndex([empty]))))

    nightly = si._plan_days(ctx, ["AAA", "BBB"], 1, False, _AS_OF)
    repair = si._plan_days(ctx, ["AAA", "BBB"], 1, False, _AS_OF, repair=True)

    forward = _SESSIONS[_SESSIONS >= pd.Timestamp("2024-06-21")]
    assert nightly.equals(forward.union(pd.DatetimeIndex([empty])))
    assert repair.equals(forward.union(pd.DatetimeIndex([partial, empty])))
    print("\n=== SANITY CHECK: RegSHO day-level holes ===")
    print(f"  {partial.date()} has BBB's row -> not re-read nightly; {empty.date()} has no row at all -> re-read;")
    print("  repair=True (the one-time Phase 11 pass) also lists the per-key gap day.")


def test_days_after_the_calendar_are_business_days(sqlite_store):
    ctx = _context(sqlite_store)
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "CAL", "date": _SESSIONS[_SESSIONS <= pd.Timestamp("2024-06-26")], "close_split": 1.0}))
    sqlite_store.save(Tables.short_interest, _si("AAA", _SESSIONS[_SESSIONS <= pd.Timestamp("2024-06-26")]))

    days = si._plan_days(ctx, ["AAA"], 1, False, _AS_OF)

    assert list(days.strftime("%m-%d")) == ["06-19", "06-20", "06-21", "06-24", "06-25", "06-26", "06-27", "06-28"]
    print("\n=== SANITY CHECK: RegSHO days past the calendar ===")
    print("  prices end 06-26 -> 06-27 and 06-28 are still requested as business days.")


def test_days_before_the_source_start_are_never_requested(sqlite_store):
    # FINRA serves nothing before 2018-08-01 except one stray 2017-12-29 file, so the
    # never-stored sessions between them must not be listed as holes, nightly or on repair.
    assert Tables.short_interest.resume is not None and Tables.short_interest.resume.source_start == "2018-08-01"
    ctx = _context(sqlite_store)
    sessions = pd.bdate_range("2017-12-26", "2018-08-17")
    served = sessions[sessions >= pd.Timestamp("2018-08-01")]
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "CAL", "date": sessions, "close_split": 1.0}))
    sqlite_store.save(Tables.short_interest, _si("AAA", served.union(pd.DatetimeIndex(["2017-12-29"]))))
    sqlite_store.save(Tables.short_interest, _si("BBB", served))
    as_of = pd.Timestamp("2018-08-18")  # last completed session: Friday 2018-08-17

    nightly = si._plan_days(ctx, ["AAA", "BBB"], 1, False, as_of)
    repair = si._plan_days(ctx, ["AAA", "BBB"], 1, False, as_of, repair=True)

    forward = served[served >= pd.Timestamp("2018-08-10")]
    assert nightly.equals(forward)
    assert repair.equals(forward)
    print("\n=== SANITY CHECK: RegSHO source start ===")
    print("  stored span starts 2017-12-29 but source_start is 2018-08-01: nightly and repair both read only the")
    print(f"  {len(forward)} forward sessions from 08-10; the 2018-01..07 block FINRA answers 403 is never requested.")


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


def test_scoped_full_upserts_its_tickers_and_leaves_every_other_key(sqlite_store, monkeypatch):
    """A `-t` full run re-reads every served day for its tickers only: no replace, another key's rows survive."""
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

    si.fetch_short_interest(context, ["LEN"], pause=0.0, full=True, identity=_identity())

    assert _grain(sqlite_store) == {("LEN", "2018-08-01"): 3.0, ("LEN", "2018-08-02"): 5.0, ("OTHR", "2018-08-01"): 7.0}
    print("\n=== SANITY CHECK: RegSHO scoped full ===")
    print("  -t LEN -F: LEN's served days re-read and upserted; OTHR's stored row untouched (no table replace). Validated.")


def test_empty_download_leaves_the_tables_untouched(sqlite_store, monkeypatch):
    monkeypatch.setattr(si, "_fetch_day", lambda day, session=None: None)
    monkeypatch.setattr(si, "_plan_days", lambda *a, **k: pd.bdate_range("2024-06-24", "2024-06-28"))

    assert si.fetch_short_interest(_context(sqlite_store), tickers=["LEN"], pause=0.0, identity=_identity()) is None
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
