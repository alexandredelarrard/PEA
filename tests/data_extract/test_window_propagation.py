"""Source windows use configured dates, units and the single Context run anchor."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import resume
from src.data_extract.utils.common.bulk_cache import quarter_periods
from src.data_extract.utils.common.resume import SeriesRequest, planner_bounds, series_windows
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.institutionals import fetch_short_interest as short
from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_extract.utils.prices import fetch_prices as prices
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_extract.utils.zips import fetch_financial_notes as notes
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.data_policy import window_policy
from src.utils.window import Window, WindowRun
from tests.data_extract.edgar_fixtures import fake_context, master_text, seed_index
from tests.data_extract.prices.test_one_download import FakeYahoo, _recorded
from tests.support.store_identity import store_identity
from tests.support.universe import seed_universe
from tests.utils.test_data_policy import _config


def _context(tmp_path: Path, store: DataStore) -> Any:
    context = fake_context(tmp_path, store, ["AAA", "BBB"])
    context.config = deepcopy(_config())
    context.run = WindowRun("incremental", "scoped source check", pd.Timestamp("2026-11-16"))
    item = context.config.data.tables.prices
    item.start_date, item.end_date = "2026-10-01", "2026-11-10"
    return context


def test_per_key_frontier_and_both_configured_bounds(tmp_path, sqlite_store) -> None:
    context = _context(tmp_path, sqlite_store)
    dates = pd.bdate_range("2026-10-01", "2026-11-10")
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "AAA", "date": dates, "close_split": 100.0}))
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "BBB", "date": dates[dates <= "2026-11-03"], "close_split": 100.0}))
    request = SeriesRequest(["AAA", "BBB"], None)
    work = series_windows(context, Tables.prices, request)
    assert work.windows == {
        "AAA": [Window(pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-10"))],
        "BBB": [Window(pd.Timestamp("2026-10-27"), pd.Timestamp("2026-11-10"))],
    }
    context.config.data.tables.prices.reprocess.overlap = {"value": 1, "unit": "months"}
    changed = series_windows(context, Tables.prices, request)
    assert changed.windows["AAA"][0].start == pd.Timestamp("2026-10-10")
    assert changed.windows["BBB"][0].start == pd.Timestamp("2026-10-03")
    print("SANITY: each source key uses its own frontier; configured bounds and month units reach actual source windows.")


def test_initial_full_and_provider_ceiling_share_fixed_dates(tmp_path, sqlite_store) -> None:
    context = _context(tmp_path, sqlite_store)
    request = SeriesRequest(["AAA"], None, until=pd.Timestamp("2026-11-06"))
    first = series_windows(context, Tables.prices, request)
    assert first.windows["AAA"] == [Window(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-06"))]
    context.run = WindowRun("full", "explicit source replay", context.run.as_of)
    assert series_windows(context, Tables.prices, request).windows == first.windows
    print("SANITY: initial and explicit full requests retain the same fixed lower bound and provider ceiling.")


def test_out_of_history_source_performs_no_store_reads(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, sqlite_store)
    context.run = WindowRun("incremental", "before configured history", pd.Timestamp("2020-01-01"))
    monkeypatch.setattr(sqlite_store, "key_stats", lambda *args, **kwargs: pytest.fail("no-work source read"))
    work = series_windows(context, Tables.prices, SeriesRequest(["AAA"], None))
    assert work.windows == {"AAA": []}
    print("SANITY: an empty configured source intersection stops before frontier reads or provider work.")


def test_price_provider_rows_obey_finite_configured_scope(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, sqlite_store)
    context.run = WindowRun("full", "finite source replay", pd.Timestamp("2024-06-29"))
    sqlite_store.drop(Tables.universe_tickers)
    context.universe = seed_universe(sqlite_store, ["AAPL", "MO", "NVDA"])
    context._identity_resolver = store_identity(sqlite_store)
    item = context.config.data.tables.prices
    item.start_date, item.end_date = "2024-05-10", "2024-06-07"
    fake = FakeYahoo(_recorded())
    monkeypatch.setattr(prices.yf, "download", lambda tickers, **kwargs: fake(tickers, start="1900-01-01", end="2100-01-01", actions=True))
    monkeypatch.setattr(prices.time, "sleep", lambda seconds: None)
    protected = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-05-01", "2024-06-28"]),
            "close_split": [17.0, 19.0],
            "close_total": [17.0, 19.0],
            "open": [17.0, 19.0],
            "high": [17.0, 19.0],
            "low": [17.0, 19.0],
            "volume": [100, 100],
        }
    )
    sqlite_store.save(Tables.prices, protected)
    prices.fetch_prices(context, prices.YahooSession(pause=0.0), ["AAPL"])
    rows = sqlite_store.load(Tables.prices).sort_values("date")
    assert rows.loc[~rows["date"].between("2024-05-10", "2024-06-07"), "close_split"].tolist() == [17.0, 19.0]
    assert set(rows["ticker"]) == {"AAPL"}
    assert len(rows) > 2
    print("SANITY: real recorded provider rows outside both config endpoints are discarded and retained prefix/suffix values survive.")


def test_surprise_candidates_use_configured_dates_and_revisit_recent_publications(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, sqlite_store)
    item = context.config.data.tables.earnings_surprises
    item.start_date, item.end_date = "2020-01-01", "2026-11-10"
    initial = pd.DataFrame(
        {
            "ticker": "AAA",
            "earnings_date": pd.to_datetime(["2025-01-01", "2026-11-09", "2026-11-12"]),
            "eps_estimate": 1.0,
            "eps_actual": [1.0, 1.0, None],
            "surprise_pct": 0.0,
        }
    )
    sqlite_store.save(Tables.earnings_surprises, initial)
    calls = []

    def download(symbol: str, window: Window) -> pd.DataFrame:
        calls.append((symbol, window))
        return pd.DataFrame(
            {
                "ticker": symbol,
                "earnings_date": pd.to_datetime(["2025-01-01", "2026-11-09", "2026-11-11"]),
                "eps_estimate": 1.0,
                "eps_actual": 2.0,
                "surprise_pct": 100.0,
            }
        )

    monkeypatch.setattr(surprises, "_download_one", download)
    monkeypatch.setattr(surprises.time, "sleep", lambda seconds: None)
    surprises.fetch_earnings_surprises(context, ["AAA"])
    assert calls == [("AAA", Window(pd.Timestamp("2026-10-12"), pd.Timestamp("2026-11-10")))]
    rows = (
        sqlite_store.load(Tables.earnings_surprises)
        .assign(earnings_date=lambda frame: pd.to_datetime(frame["earnings_date"]))
        .set_index("earnings_date")
    )
    assert rows.loc["2025-01-01", "eps_actual"] == 1.0 and rows.loc["2026-11-09", "eps_actual"] == 2.0
    assert pd.isna(rows.loc["2026-11-12", "eps_actual"]) and pd.Timestamp("2026-11-11") not in rows.index
    print(
        "SANITY:35CD candidate checks revisit a recent publication despite a future stored estimate, while both configured write bounds stay protected."
    )


def test_finra_day_units_follow_the_configured_window_without_half_month_rebuild(tmp_path, sqlite_store) -> None:
    context = _context(tmp_path, sqlite_store)
    item = context.config.data.tables.sec_short_volume_security
    item.start_date, item.end_date = "2020-01-01", "2026-11-10"
    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-11-10")],
            "source_symbol": ["AAA"],
            "market": ["Q"],
            "short_volume": [1.0],
            "short_exempt_volume": [0.0],
            "total_volume": [2.0],
        }
    )
    sqlite_store.save(Tables.sec_short_volume_security, raw)
    days = short._plan_days(context)
    assert days.equals(pd.bdate_range("2026-10-12", "2026-11-10"))
    item.reprocess.overlap.value = 3
    assert short._plan_days(context).equals(pd.bdate_range("2026-11-09", "2026-11-10"))
    print("SANITY: FINRA day-file units inherit30CD/3CD from config and retain the finite ceiling across the half-month boundary.")


def test_cached_edgar_index_is_filtered_at_both_configured_ends(tmp_path: Path, sqlite_store: DataStore) -> None:
    context = _context(tmp_path, sqlite_store)
    item = context.config.data.tables.sec_8k
    item.start_date, item.end_date = "2026-01-02", "2026-07-05"
    seed_index(
        context,
        [
            (1, "AAA", "8-K", date, accession)
            for date, accession in [
                ("2025-12-31", "prefix"),
                ("2026-01-02", "start"),
                ("2026-07-02", "inside"),
                ("2026-07-06", "same-quarter-future"),
                ("2026-10-01", "later-quarter"),
            ]
        ],
    )
    policy = window_policy(context.config, Tables.sec_8k, context.run.as_of)
    assert policy is not None
    rows = edgar_index.entries(context, {"0000000001"}, ["8-K"], window=policy.requested)
    assert rows["accession"].tolist() == ["start", "inside"]
    print("SANITY: cached index partitions and rows obey both configured filing-date bounds, including future rows in the same cached quarter.")


def test_document_worklist_uses_config_bounds_and_the_context_mode(tmp_path: Path, sqlite_store: DataStore) -> None:
    context = _context(tmp_path, sqlite_store)
    item = context.config.data.tables.sec_8k
    item.start_date, item.end_date = "2026-01-01", "2026-06-30"
    seed_index(
        context,
        [
            (1, "AAA", "8-K", date, accession)
            for date, accession in [
                ("2025-12-31", "prefix"),
                ("2026-01-02", "completed"),
                ("2026-05-06", "missing"),
                ("2026-07-01", "suffix"),
            ]
        ],
    )
    sqlite_store.save(
        Tables.sec_8k,
        pd.DataFrame(
            {"ticker": ["AAA"], "accession_number": ["completed"], "item": ["8.01"], "filing_date": [pd.Timestamp("2026-01-02")], "form": ["8-K"]}
        ),
    )
    request = WorklistRequest(keys=pd.DataFrame({"ticker": ["AAA"], "cik": ["0000000001"]}), forms=["8-K"], identity=None)
    work = document_worklist(context, Tables.sec_8k, request)
    assert work.units["AAA"]["accession"].tolist() == ["missing"]
    context.run = WindowRun("full", "explicit filing replay", context.run.instant)
    replay = document_worklist(context, Tables.sec_8k, request)
    assert replay.units["AAA"]["accession"].tolist() == ["completed", "missing"]
    item.start_date = "2026-05-01"
    changed = document_worklist(context, Tables.sec_8k, request)
    assert changed.units["AAA"]["accession"].tolist() == ["missing"]
    print("SANITY: fixed table bounds and one Context mode govern missing/completed filing work, with no year/as-of/full override in the request.")


def test_edgar_index_refresh_enumerates_only_quarters_intersecting_configured_dates(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    item = context.config.data.tables.sec_8k
    item.start_date, item.end_date = "2026-05-10", "2026-08-08"
    policy = window_policy(context.config, Tables.sec_8k, context.run.as_of)
    assert policy is not None
    asked = []

    def download(context, url, path, **kwargs):
        year, quarter = int(url.split("/")[-3]), int(url.split("/")[-2][-1])
        first = pd.Timestamp(year=year, month=quarter * 3 - 2, day=1)
        asked.append((year, quarter))
        path.write_bytes(master_text([(1, "AAA", "8-K", first.strftime("%Y-%m-%d"), f"q{quarter}")]).encode("latin-1"))
        return 200

    monkeypatch.setattr(edgar_index.sec_io, "download", download)
    written = edgar_index.refresh(context, window=policy.requested)
    assert asked == [(2026, 2), (2026, 3)] and set(written) == {"2026Q2", "2026Q3"}
    assert edgar_index.refreshed_today(context, policy.requested)
    print("SANITY: the metadata refresh touches only complete quarter units intersecting fixed config dates, without a year-based scope.")


def test_market_filing_index_uses_the_same_bounded_reader(tmp_path: Path, sqlite_store: DataStore) -> None:
    context = _context(tmp_path, sqlite_store)
    seed_index(
        context,
        [
            (1, "Manager A", "13F-HR", "2026-05-01", "prefix"),
            (2, "Manager B", "13F-HR", "2026-05-15", "inside"),
            (3, "Manager C", "13F-HR/A", "2026-08-01", "suffix"),
        ],
    )
    window = Window(pd.Timestamp("2026-05-10"), pd.Timestamp("2026-06-30"))
    rows = edgar_index.entries(context, None, ["13F-HR", "13F-HR/A"], window)
    assert rows["accession"].tolist() == ["inside"]
    print("SANITY: global-manager 13F metadata shares the configured bounded index reader; security ticker scope never truncates whole books.")


def test_ended_grace_uses_configured_actual_sessions_and_invalidates_its_cache(tmp_path: Path, sqlite_store: DataStore) -> None:
    context = _context(tmp_path, sqlite_store)
    sqlite_store.drop(Tables.universe_tickers)
    context.universe = seed_universe(
        sqlite_store,
        ["AAA"],
        rows={
            "AAA": {
                "first_price_date": "2026-06-01",
                "ended_on": "2026-06-11",
            }
        },
    )
    dates = pd.bdate_range("2026-06-01", "2026-07-02").difference(pd.DatetimeIndex(["2026-06-19"]))
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "AAA", "date": dates, "close_split": 1.0}))
    assert planner_bounds(context)["AAA"][1] == pd.Timestamp("2026-06-26")
    context.config.data.planning.ended_grace.value = 3
    assert planner_bounds(context)["AAA"][1] == pd.Timestamp("2026-06-16")
    print("SANITY: configured end grace counts actual sessions across Juneteenth; changing YAML policy invalidates the cached bounds.")


def test_incomplete_grace_calendar_does_not_invent_weekday_sessions(tmp_path: Path, sqlite_store: DataStore) -> None:
    context = _context(tmp_path, sqlite_store)
    sqlite_store.drop(Tables.universe_tickers)
    context.universe = seed_universe(
        sqlite_store,
        ["AAA"],
        rows={
            "AAA": {
                "first_price_date": "2026-06-01",
                "ended_on": "2026-06-11",
            }
        },
    )
    dates = pd.bdate_range("2026-06-01", "2026-06-18")
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "AAA", "date": dates, "close_split": 1.0}))
    assert planner_bounds(context)["AAA"][1] is None
    print("SANITY: insufficient retained sessions keep the listing end open within the requested source Window; no invented weekday close.")


def test_archive_unit_enumeration_has_no_history_year_or_future_unit_override() -> None:
    assert quarter_periods(Window(pd.Timestamp("2026-03-10"), pd.Timestamp("2026-08-02"))) == ["2026q1", "2026q2"]
    assert ftd.ftd_periods(Window(pd.Timestamp("2026-10-10"), pd.Timestamp("2026-11-10"))) == ["202610a", "202610b"]
    print("SANITY: quarter and half-month source units are bounded by their period ends, including both finite endpoints.")


def test_notes_source_candidates_respect_both_tables_independent_dates(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    numeric, text = context.config.data.tables.notes_num, context.config.data.tables.notes_text
    numeric.start_date, numeric.end_date = "2026-03-01", "2026-06-30"
    text.start_date, text.end_date = "2026-09-01", "2026-09-30"
    published = ["2025q4", "2026q1", "2026_04", "2026q2", "2026q3", "2026_10"]
    monkeypatch.setattr(notes, "_scrape_available_periods", lambda ctx: published)
    assert notes.notes_periods(context) == ["2026_04", "2026q1", "2026q2", "2026q3"]
    numeric.end_date = "2026-05-31"
    assert notes.notes_periods(context) == ["2026_04", "2026q1", "2026q3"]
    print("SANITY: each notes table's configured bounds reach published-unit selection, without widening the other table or sliding by years.")


def test_archive_completion_considers_only_the_tables_requiring_each_unit(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    numeric, text = context.config.data.tables.notes_num, context.config.data.tables.notes_text
    numeric.start_date, numeric.end_date = "2026-01-01", "2026-06-30"
    text.start_date, text.end_date = "2026-07-01", "2026-09-30"
    held = {Tables.notes_num: frozenset({"2026q1"}), Tables.notes_text: frozenset({"2026q3"})}
    monkeypatch.setattr(resume, "stored_values", lambda ctx, table, column: held[table])
    listed, completed = resume.archive_periods(context, (Tables.notes_num, Tables.notes_text), ["2025q4", "2026q1", "2026q2", "2026q3", "2026q4"])
    assert listed == ["2026q1", "2026q2", "2026q3"] and completed == frozenset({"2026q1", "2026q3"})
    print(
        "SANITY: a unit is complete in every table whose config requires it; disjoint sibling bounds do not cause repeated work or unauthorized scope."
    )
