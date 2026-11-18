"""Source windows use configured dates, units and the single Context run anchor."""

from __future__ import annotations

import os
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import resume
from src.data_extract.utils.common.bulk_cache import quarter_periods
from src.data_extract.utils.common.resume import SeriesRequest, planner_bounds, series_windows
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.fundamentals_sharadar import sharadar_history as vendor_history
from src.data_extract.utils.identity import evidence
from src.data_extract.utils.institutionals import fetch_short_interest as short
from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_extract.utils.prices import fetch_prices as prices
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_extract.utils.zips import fetch_financial_notes as notes
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.data_policy import window_policy
from src.utils.window import Window, WindowRun
from tests.canonical_schema import create_canonical
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
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


def test_vendor_history_recovery_protects_outside_dates_and_keys(tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, sqlite_store)
    dates = pd.to_datetime(["2026-02-01", "2026-05-01", "2026-08-01", "2026-11-01"])
    stored = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4 + ["BBB"],
            "as_of": [*dates, dates[1]],
            "fiscal_end": [*dates, dates[1]],
            "totalAssets": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    sqlite_store.save(Tables.sharadar_fundamentals_history, stored)
    stored_snapshot = sqlite_store.load(Tables.sharadar_fundamentals_history).sort_values(["ticker", "as_of"]).reset_index(drop=True)
    vendor = pd.DataFrame(
        {"ticker": ["AAA"] * 4, "date": dates, "reportperiod": dates, "dimension": ["ARQ"] * 4, "assets": [100.0, 200.0, 300.0, 400.0]}
    )
    sqlite_store.save(Tables.sharadar_fundamentals_facts, vendor)
    policy = context.config.data.tables.sharadar_fundamentals_history
    policy.start_date, policy.end_date = "2026-05-01", "2026-08-01"
    context.run = WindowRun("full", "finite vendor history recovery", context.run.instant)
    monkeypatch.setattr(vendor_history, "load_field_map", lambda directory: None)
    monkeypatch.setattr(vendor_history, "with_predecessor_series", lambda context, vendor, names, splits, **kwargs: (vendor, splits))
    monkeypatch.setattr(vendor_history, "vendor_share_basis", lambda *args: SimpleNamespace(vendor=pd.DataFrame()))

    def transformed(vendor: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        result = vendor.rename(columns={"date": "as_of", "reportperiod": "fiscal_end", "assets": "totalAssets"})
        return result.assign(as_of=pd.to_datetime(result.as_of), fiscal_end=pd.to_datetime(result.fiscal_end))[
            ["ticker", "as_of", "fiscal_end", "totalAssets"]
        ]

    monkeypatch.setattr(vendor_history, "build_frame", transformed)
    vendor_history.build_sharadar_history(context, ["AAA"])
    after = sqlite_store.load(Tables.sharadar_fundamentals_history).sort_values(["ticker", "as_of"]).reset_index(drop=True)
    before = stored_snapshot
    protected = before.ticker.eq("BBB") | ~pd.to_datetime(before.as_of).between("2026-05-01", "2026-08-01")
    pd.testing.assert_frame_equal(after.loc[protected].reset_index(drop=True), before.loc[protected].reset_index(drop=True))
    assert after.loc[~protected, "totalAssets"].tolist() == [200.0, 300.0]
    print("SANITY: vendor-history recovery rewrites only its configured dates/key and protects the prefix, suffix and other issuer.")


def test_notes_identity_capture_uses_the_same_configured_archive_units(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    numeric, text = context.config.data.tables.notes_num, context.config.data.tables.notes_text
    numeric.start_date, numeric.end_date = "2026-04-01", "2026-06-30"
    text.start_date, text.end_date = "2026-09-01", "2026-09-30"
    cached = ["2025q4", "2026q1", "2026_04", "2026q2", "2026q3", "2026q4"]
    monkeypatch.setattr(evidence, "cached_notes_periods", lambda cache: cached)
    monkeypatch.setattr(evidence, "captured_dei_periods", lambda context: {"2026_04"})
    assert evidence.pending_dei_periods(context, tmp_path) == ["2026q2", "2026q3"]
    context.run = WindowRun("full", "explicit bounded evidence recovery", context.run.instant)
    assert evidence.pending_dei_periods(context, tmp_path) == ["2026_04", "2026q2", "2026q3"]
    print("SANITY: identity capture and notes ingestion share the same source-unit bounds; only explicit recovery replays completed units.")


def test_direct_identity_capture_refuses_an_outside_unit_before_opening_it(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    for table in (Tables.notes_num, Tables.notes_text):
        context.config.data.tables[table.name].start_date = "2026-01-01"
    monkeypatch.setattr(evidence, "read_zip_tables", lambda *args, **kwargs: pytest.fail("out-of-window archive opened"))
    with pytest.raises(ValueError, match="outside"):
        evidence.capture_dei_period(context, tmp_path, "2025q4")
    print("SANITY: standalone identity evidence capture enforces the source Window before opening an excluded archive.")


@pytest.mark.parametrize("operation", ["download", "ingest"])
def test_direct_notes_subruns_filter_provided_units_before_io(
    operation: str, tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    for table in (Tables.notes_num, Tables.notes_text):
        item = context.config.data.tables[table.name]
        item.start_date, item.end_date = "2026-01-01", "2026-06-30"
    periods = ["2025q4", "2026q1", "2026q2", "2026q3"]
    visited: list[str] = []
    if operation == "download":
        monkeypatch.setattr(notes, "ensure_zip", lambda context, path, url, **kwargs: visited.append(path.stem.removesuffix("_notes")) or path)
        assert notes.download_notes_periods(context, tmp_path, periods) == 2
    else:
        monkeypatch.setattr(notes, "_repair_stored_clocks", lambda *args, **kwargs: {})
        monkeypatch.setattr(notes, "load_identity", lambda context: SimpleNamespace(source_cohort={}))
        monkeypatch.setattr(notes, "stored_period_clock", lambda context, tables, period: visited.append(period))
        monkeypatch.setattr(notes, "ensure_archive", lambda *args, **kwargs: None)
        assert notes.ingest_financial_notes(context, tmp_path, ["AAA", "BBB"], periods) == 0
    assert visited == ["2026q1", "2026q2"]
    print(f"SANITY: direct notes {operation} enforces configured unit bounds before cache/provider/clock I/O.")


def _cached_notes_unit(cache: Path, period: str) -> None:
    end = pd.Period(period.upper(), freq="Q").end_time.normalize().strftime("%Y%m%d")
    accession = f"new-{period}"
    members = {
        "sub.tsv": f"adsh\tcik\tform\tfy\tfp\tfiled\n{accession}\t7\t10-Q\t2026\tQ1\t{end}\n",
        "num.tsv": (
            "adsh\ttag\tddate\tqtrs\tuom\tdimn\tcoreg\tfootnote\tvalue\n"
            f"{accession}\tDefinedBenefitPlanBenefitObligation\t{end}\t0\tUSD\t0\t\t\t100\n"
        ),
        "txt.tsv": (
            "adsh\ttag\tddate\tqtrs\tdimn\tcoreg\tescaped\ttxtlen\tfootnote\tvalue\n"
            f"{accession}\tDefinedBenefitPlanDisclosureTextBlock\t{end}\t0\t0\t\t0\t12\t\tPension note\n"
        ),
    }
    path = cache / f"{period}_notes.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    observed = pd.Timestamp("2026-10-21", tz="UTC").timestamp()
    os.utime(path, (observed, observed))


def test_completed_notes_units_stay_untouched_until_explicit_bounded_recovery(
    tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, sqlite_store)
    create_canonical(sqlite_store.engine, [Tables.notes_num, Tables.notes_text])
    sqlite_store.delete(Tables.universe_tickers, {"ticker": ["AAA", "BBB"]})
    context.universe = seed_universe(sqlite_store, ["AAPL"])
    identity = dated_identity(
        [("AAPL", "0000000007", "cik_window", SENTINEL, None)],
        {"AAPL": "0000000007"},
        changed_at={"AAPL": pd.Timestamp("2026-07-05")},
    )
    monkeypatch.setattr(notes, "load_identity", lambda ctx: identity)
    periods = ["2026q1", "2026q2", "2026q3"]
    for period in periods:
        _cached_notes_unit(tmp_path, period)
    for table in (Tables.notes_num, Tables.notes_text):
        item = context.config.data.tables[table.name]
        item.start_date, item.end_date = "2026-01-01", "2026-09-30"
        sqlite_store.save(
            table,
            pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "cik": "0000000001",
                        "adsh": f"old-{p}",
                        "tag": "existing",
                        "ddate": pd.Timestamp("2026-03-31"),
                        "qtrs": 0,
                        "period": p,
                        "available_at": pd.Timestamp("2026-07-21").date(),
                        "value": 1.0 if table == Tables.notes_num else "kept",
                    }
                    for p in periods[:2]
                ]
            ),
        )
    parsed: list[str] = []
    read = notes._read_notes

    def observe(path: Path, identity: Any, universe: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        unit = path.stem.removesuffix("_notes")
        parsed.append(unit)
        assert context.run.mode == "full" or unit == "2026q3", "normal ingest reopened a completed archive"
        return read(path, identity, universe)

    monkeypatch.setattr(notes, "_read_notes", observe)
    assert notes.ingest_financial_notes(context, tmp_path, ["AAPL"], periods) == 2
    assert parsed == ["2026q3"], "new lineage and a new archive must not replay completed units"
    parsed.clear()
    assert notes.ingest_financial_notes(context, tmp_path, ["AAPL"], periods) == 0 and parsed == []
    before = {table: sqlite_store.load(table) for table in (Tables.notes_num, Tables.notes_text)}
    for table in before:
        item = context.config.data.tables[table.name]
        item.start_date, item.end_date = "2026-04-01", "2026-06-30"
    context.run = WindowRun("full", "explicit bounded notes recovery", context.run.instant)
    assert notes.ingest_financial_notes(context, tmp_path, ["AAPL"], periods) == 2 and parsed == ["2026q2"]
    for table, original in before.items():
        after = sqlite_store.load(table)
        protected = original.loc[original.period.ne("2026q2")].sort_values("adsh").reset_index(drop=True)
        actual = after.loc[after.period.ne("2026q2")].sort_values("adsh").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, protected)
    print(
        "SANITY: real cached notes parse only the missing quarter; repeat is a no-op, while explicit recovery replays its configured unit and protects other units."
    )
