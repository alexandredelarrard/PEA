"""RegSHO short-volume fetch: resume window, universe filter, upsert shape.

Resume is on the GLOBAL max date, not per ticker: one RegSHO file covers the whole
market, so once day D is stored every ticker has D. A per-ticker frontier would let
a single lagging symbol (index churn, a renamed ticker) drag the loop back over
thousands of already-held day-files on every run -- the reason `fails_to_deliver`
is a separate table in the first place (schema.py).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.institutionals import fetch_short_interest as si
from src.data_store.schema import Tables


def _identity() -> Identity:
    lineage = pd.DataFrame(
        [
            {"cik": "0000000001", "entity_id": "E_AAA", "source": "roster"},
            {"cik": "0000000002", "entity_id": "E_FISV", "source": "roster"},
            {"cik": "0000000003", "entity_id": "E_TT", "source": "roster"},
            {"cik": "0000000004", "entity_id": "E_IR", "source": "roster"},
            {"cik": "0000000005", "entity_id": "E_WTW", "source": "roster"},
        ]
    )
    tenure = pd.DataFrame(
        [
            ("AAA", "0000000001", "2009-01-01", None, 100),
            ("FISV", "0000000002", "2018-01-01", "2023-06-01", 100),
            ("FI", "0000000002", "2023-06-01", None, 100),
            ("IR", "0000000003", "2009-01-01", "2020-03-01", 100),
            ("IR", "0000000004", "2020-03-05", None, 100),
            ("TT", "0000000003", "2020-03-01", None, 100),
            ("WTW", "0000000099", "2009-01-01", "2019-04-18", 100),
            ("WTW", "0000000005", "2019-04-18", None, 100),
        ],
        columns=["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings"],
    )
    tenure["source"] = "form345"
    tenure["evidence"] = "test"
    roster = pd.DataFrame(
        [
            ("AAA", "0000000001"),
            ("FISV", "0000000002"),
            ("TT", "0000000003"),
            ("IR", "0000000004"),
            ("WTW", "0000000005"),
        ],
        columns=["ticker", "cik"],
    )
    return build_identity(lineage, tenure, roster)


def _context(store) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger("test.regsho"))


def _regsho(day: str, rows: list[tuple[str, int, int]]) -> str:
    head = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    return head + "".join(f"{day}|{t}|{s}|0|{v}|Q\n" for t, s, v in rows)


def test_resume_day_is_the_day_after_the_global_max(sqlite_store):
    ctx = _context(sqlite_store)
    # cold table -> full years_history window
    cold = si._resume_day(ctx, years_history=10)
    assert cold == pd.Timestamp.today().normalize() - pd.DateOffset(years=10)

    sqlite_store.replace(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "BBB"],
                "date": pd.to_datetime(["2024-05-01", "2024-06-03", "2024-06-04"]),
                "short_volume": [1.0, 2.0, 3.0],
                "total_volume": [10.0, 20.0, 30.0],
            }
        ),
    )
    # GLOBAL max is 2024-06-04 (BBB's) -- AAA lagging at 05-01 must NOT pull it back
    assert si._resume_day(ctx, years_history=10) == pd.Timestamp("2024-06-05")

    print("\n=== SANITY CHECK: RegSHO resume day ===")
    print(
        f"  cold table -> {cold.date()} (years_history); stored max 2024-06-04 -> "
        "2024-06-05. A ticker stale at 2024-05-01 does not widen the window. Validated."
    )


def test_stored_rows_returns_empty_for_an_empty_bounded_prefix(sqlite_store):
    sqlite_store.replace(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2020-01-02"]),
                "short_volume": [1.0],
                "total_volume": [2.0],
            }
        ),
    )

    prefix = si._stored_rows(_context(sqlite_store), until=pd.Timestamp("2020-01-01"))

    assert prefix.empty
    assert list(prefix.columns) == ["date", "ticker", "short_volume", "total_volume"]

    print("\n=== SANITY CHECK: RegSHO empty legacy prefix ===")
    print("  bounded read before the first stored date -> typed empty frame")
    print("  OK: a fully reproducible source window needs no legacy rows")


def test_fetch_filters_to_the_universe_and_upserts(sqlite_store, monkeypatch):
    sqlite_store.replace(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2024-06-03"]),
                "short_volume": [1.0],
                "total_volume": [10.0],
            }
        ),
    )
    monkeypatch.setattr(si, "record_run", lambda *a, **k: None)

    # ⚠ THE WINDOW MUST NOT BE `today .. today`. `fetch_short_interest` builds
    # `pd.bdate_range(_resume_day(...), today)`, and a bdate_range whose start AND end are the
    # same WEEKEND day is EMPTY -- so on a Saturday or Sunday no day-file was fetched, nothing
    # was stored, and `len(stored) == 2` failed. That is exactly what happened in the
    # 2026-09-06 (Sunday) full-suite run, where this test passed in isolation on the Monday
    # and looked like ordering pollution.
    #
    # A 5-business-day window is non-empty on every day of the week, and serving the day-file
    # only ONCE keeps the assertion on "one new row" exact regardless of how many days the
    # range holds.
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: (pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(5)))
    served: list[pd.Timestamp] = []

    def _one_day(day, session=None):
        del session
        if served:
            return None
        served.append(day)
        return _regsho(day.strftime("%Y%m%d"), [("AAA", 500, 1000), ("ZZZ", 900, 1800)])

    monkeypatch.setattr(si, "_fetch_day", _one_day)

    si.fetch_short_interest(_context(sqlite_store), tickers=["AAA"], pause=0.0, identity=_identity())

    # the fetcher returns None -- it resumes from the DB and writes to it, so the stored
    # table is the only contract worth asserting on
    stored = sqlite_store.load(Tables.short_interest)
    assert served, "the window must contain at least one business day on any weekday"
    assert set(stored["ticker"]) == {"AAA"}, "ZZZ leaked past the universe filter"
    assert len(stored) == 2  # prior row kept, one new day added

    print("\n=== SANITY CHECK: RegSHO universe filter + upsert ===")
    print(
        f"  day-file for {served[0].date()} had AAA+ZZZ -> stored "
        f"{sorted(set(stored['ticker']))} only; table {len(stored)} rows "
        f"(1 prior + 1 new). Validated."
    )


def test_empty_download_leaves_the_table_untouched(sqlite_store, monkeypatch):
    """A holiday / all-404 window must not crash: `store.save` warns on the empty frame, the
    run is still recorded, and nothing is written."""
    recorded: list = []
    monkeypatch.setattr(si, "record_run", lambda *a, **k: recorded.append(a))
    monkeypatch.setattr(si, "_fetch_day", lambda day: None)
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: pd.Timestamp.today().normalize())

    assert si.fetch_short_interest(_context(sqlite_store), tickers=["AAA"], pause=0.0, identity=_identity()) is None
    assert not sqlite_store.exists(Tables.short_interest)  # nothing written, nothing created
    assert recorded, "an empty window must still record the run"

    print("\n=== SANITY CHECK: RegSHO empty window ===")
    print("  every day-file missing -> no crash, no table created, run still recorded. Validated.")


def test_parse_keeps_source_symbol_until_identity_resolution():
    parsed = si._parse_regsho(_regsho("20200305", [("FI", 10, 20), ("FI", 5, 10)]))
    assert parsed.iloc[0]["source_symbol"] == "FI"
    assert parsed.iloc[0]["short_volume"] == 15
    assert "ticker" not in parsed.columns

    print("\n=== SANITY CHECK: RegSHO parse identity boundary ===")
    print("  two FI rows aggregate as source_symbol=FI; no destination ticker exists yet")
    print("  OK: canonical identity is assigned only after the dated source parse")


def test_fetch_day_reuses_the_supplied_http_session():
    calls: list[tuple[str, dict]] = []

    class Session:
        def get(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=200, text="payload")

    assert si._fetch_day(pd.Timestamp("2026-09-22"), Session()) == "payload"
    assert len(calls) == 1 and calls[0][0].endswith("CNMSshvol20260922.txt")

    print("\n=== SANITY CHECK: RegSHO connection reuse ===")
    print("  one run-scoped HTTP session serves the daily FINRA request")
    print("  OK: full replay avoids a new TLS handshake for every date")


def test_full_refresh_reconciles_legacy_preserves_failed_dates_and_is_idempotent(sqlite_store, monkeypatch):
    identity = _identity()
    context = _context(sqlite_store)
    stored = pd.DataFrame(
        {
            "ticker": ["IR", "WTW", "IR", "AAA"],
            "date": pd.to_datetime(["2015-01-02", "2015-01-02", "2020-03-03", "2020-03-06"]),
            "short_volume": [10.0, 20.0, 30.0, 40.0],
            "total_volume": [100.0, 200.0, 300.0, 400.0],
        }
    )
    sqlite_store.replace(Tables.short_interest, stored)
    monkeypatch.setattr(si, "_today", lambda: pd.Timestamp("2020-03-06"))
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: pd.Timestamp("2020-03-02"))
    monkeypatch.setattr(si, "record_run", lambda *a, **k: None)

    def _fetch(day: pd.Timestamp, session=None) -> str | None:
        del session
        if day == pd.Timestamp("2020-03-05"):
            return _regsho("20200305", [("FI", 50, 100), ("FISV", 25, 50)])
        return None

    monkeypatch.setattr(si, "_fetch_day", _fetch)
    universe = ["AAA", "FISV", "TT", "IR", "WTW"]
    si.fetch_short_interest(context, universe, pause=0.0, full=True, identity=identity)
    first = sqlite_store.load(Tables.short_interest).sort_values(["date", "ticker"]).reset_index(drop=True)
    si.fetch_short_interest(context, universe, pause=0.0, full=True, identity=identity)
    second = sqlite_store.load(Tables.short_interest).sort_values(["date", "ticker"]).reset_index(drop=True)

    assert set(first["ticker"]) == {"AAA", "FISV", "IR", "TT"}
    assert "WTW" not in set(first["ticker"])
    assert first[(first.ticker == "TT") & (pd.to_datetime(first.date) == pd.Timestamp("2015-01-02"))].short_volume.iloc[0] == 10.0
    assert first[(first.ticker == "IR") & (pd.to_datetime(first.date) == pd.Timestamp("2020-03-03"))].short_volume.iloc[0] == 30.0
    assert first[(first.ticker == "AAA") & (pd.to_datetime(first.date) == pd.Timestamp("2020-03-06"))].short_volume.iloc[0] == 40.0
    fisv = first[(first.ticker == "FISV") & (pd.to_datetime(first.date) == pd.Timestamp("2020-03-05"))]
    assert fisv.short_volume.iloc[0] == 25.0 and fisv.total_volume.iloc[0] == 50.0
    assert not first.duplicated(["ticker", "date"]).any()
    pd.testing.assert_frame_equal(first, second)

    print("\n=== SANITY CHECK: RegSHO retention-safe full refresh ===")
    print("  legacy IR -> TT; prior Weight Watchers removed; seam-gap IR preserved")
    print("  failed-date AAA preserved; only date-eligible FISV publishes; future FI is rejected; rerun identical")
    print("  OK: the rolling source rebuilds what it can without erasing what it cannot")


def test_full_refresh_all_source_failures_abort_without_erasing_history(sqlite_store, monkeypatch):
    identity = _identity()
    context = _context(sqlite_store)
    sqlite_store.replace(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2020-03-02"]),
                "short_volume": [1.0],
                "total_volume": [2.0],
            }
        ),
    )
    monkeypatch.setattr(si, "_today", lambda: pd.Timestamp("2020-03-06"))
    monkeypatch.setattr(si, "_resume_day", lambda *a, **k: pd.Timestamp("2020-03-02"))
    monkeypatch.setattr(si, "_fetch_day", lambda day, session=None: None)

    with pytest.raises(RuntimeError, match="preserving the table by aborting"):
        si.fetch_short_interest(context, ["AAA"], pause=0.0, full=True, identity=identity)
    saved = sqlite_store.load(Tables.short_interest)
    assert len(saved) == 1 and saved.iloc[0]["ticker"] == "AAA"

    print("\n=== SANITY CHECK: RegSHO all-403 safety gate ===")
    print("  zero reproducible days -> abort before replace; stored AAA row survives")
    print("  OK: a provider outage cannot be mistaken for an empty market")
