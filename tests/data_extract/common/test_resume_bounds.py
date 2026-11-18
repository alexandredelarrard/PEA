"""AC-008 / REQ-011: every per-ticker planner clamps its work to the ticker's identity bounds.

`Universe.bounds` = `(first_price_date, ended_on)`. On the phase-0 fixture store (read-only) with the identity columns
the phase-5 publish wrote: WBD ended on 2026-10-05 (prices stored to 2026-10-06, its completion 8-K filed 2026-10-06),
AAPL open from 1995-10-09. Each planner yields nothing after WBD's end plus the U-65 grace of 10 trading sessions
(2026-10-19 on the fixture's price calendar) and the same plan for AAPL as with no bounds:
the series planner (`prices`, `sharadar_fundamentals_facts`), the EDGAR document planner, the DEF 14A (LLM) worklist, the
earnings-call planner and the earnings-surprise plan.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.behavioral import fetch_earnings_call_transcripts as ect
from src.data_extract.utils.common.resume import SeriesRequest, planner_bounds, series_windows
from src.data_extract.utils.edgar_extract import document_resume
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_extract.utils.structure.def14a import fetch as def14a_fetch
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.data_extract.behavioral.test_earnings_call_transcripts import _AAA, _CCC, UNIVERSE, _ctx, _source, _write
from tests.data_extract.fake_context import extract_config
from tests.support.universe import seed_universe

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"
AS_OF = pd.Timestamp("2026-11-03")
WBD_END = pd.Timestamp("2026-10-05")
#: U-65: the 10th session after WBD's end on the fixture's price calendar (Columbus Day 2026-10-12 is a session).
WBD_GRACE_END = pd.Timestamp("2026-10-19")
AAPL_FIRST = pd.Timestamp("1995-10-09")
SAMPLE = ["WBD", "AAPL"]
#: CHTR lists 8-Ks of its pre-2009 registrant before its current listing's first price (2010-01-05): kept.
PREDECESSOR = "CHTR"


def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        if (cache / "fixture_store" / "identity.sqlite").exists() and (cache / "parity" / "published_5_runA_universe_tickers.parquet").exists():
            return cache
    pytest.skip("the run's fixture store and phase-5 publish are not reachable from this checkout")


class _UniverseView:
    """The `Universe` reads the planners make, over the published identity columns of the sample (or no bounds)."""

    def __init__(self, rows: pd.DataFrame, bounded: bool) -> None:
        first, ended = pd.to_datetime(rows["first_price_date"]), pd.to_datetime(rows["ended_on"])
        pairs = [(None if pd.isna(f) else f, None if pd.isna(e) else e) for f, e in zip(first, ended, strict=True)]
        self.bounds = dict(zip(rows["ticker"], pairs, strict=True)) if bounded else {}
        self.trading_symbols = dict(zip(rows["ticker"], rows["trading_symbol"], strict=True))
        self.tickers = list(rows["ticker"])

    def added_on_dates(self, keys: list[str], as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
        return {}


@pytest.fixture(scope="module")
def fixture() -> tuple[DataStore, pd.DataFrame]:
    """The fixture store (read-only) and the sample's published `universe_tickers` rows."""
    cache = _run_cache()
    path = cache / "fixture_store" / "identity.sqlite"
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    rows = pd.read_parquet(cache / "parity" / "published_5_runA_universe_tickers.parquet")
    return store, rows[rows["ticker"].isin(SAMPLE)].reset_index(drop=True)


def _contexts(fixture: tuple[DataStore, pd.DataFrame]) -> tuple[Any, Any]:
    store, rows = fixture
    return (
        types.SimpleNamespace(
            store=store,
            universe=_UniverseView(rows, bounded=True),
            config=extract_config(),
            run=WindowRun("incremental", "bounded real identity fixture", AS_OF),
        ),
        types.SimpleNamespace(
            store=store,
            universe=_UniverseView(rows, bounded=False),
            config=extract_config(),
            run=WindowRun("incremental", "bounded real identity fixture", AS_OF),
        ),
    )


@pytest.mark.parametrize("table", [Tables.prices, Tables.sharadar_fundamentals_facts], ids=["prices", "sharadar_fundamentals_facts"])
def test_series_planner_stops_at_the_end_and_starts_at_the_first_price(fixture: tuple[DataStore, pd.DataFrame], table: Any) -> None:
    bounded, control = _contexts(fixture)
    until = pd.Timestamp("2026-11-02")
    work = series_windows(bounded, table, SeriesRequest(SAMPLE, None, until=until))
    open_work = series_windows(control, table, SeriesRequest(SAMPLE, None, until=until))
    bounded.run = WindowRun("full", "explicit bounded series fixture", AS_OF)
    full = series_windows(bounded, table, SeriesRequest(SAMPLE, None, until=until))

    wbd_last = work.last["WBD"]
    assert all(end <= WBD_GRACE_END for _, end in work.windows["WBD"]), work.windows["WBD"]
    if pd.Timestamp(wbd_last) >= WBD_END:
        assert work.windows["WBD"] == []
    else:
        assert work.windows["WBD"][-1].end == WBD_GRACE_END and open_work.windows["WBD"][-1].end == until
    assert work.windows["AAPL"] == open_work.windows["AAPL"]
    assert (
        full.windows["AAPL"][0].start == max(AAPL_FIRST, pd.Timestamp(bounded.config.data.tables[table.name].start_date))
        and open_work.windows["AAPL"] != []
    )

    print(f"\n=== SANITY CHECK: series planner bounds on {table.name} ===")
    print(
        f"  WBD last stored {pd.Timestamp(wbd_last).date()}, ended 2026-10-05: windows {work.windows['WBD']} (no bounds: {open_work.windows['WBD']})"
    )
    print(
        f"  AAPL windows unchanged {work.windows['AAPL']}; a full pull starts at its first price {full.windows['AAPL'][0].start.date()} (floor 1986)"
    )
    print("  Validated: nothing after WBD's end + grace, none once the run is past grace end + overlap, nothing before AAPL's first price.")


def test_edgar_document_planner_drops_filings_after_the_end(fixture: tuple[DataStore, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> None:
    store, rows = fixture
    bounded, control = _contexts(fixture)
    index = store.load(Tables.sec_8k, columns=["cik", "accession_number", "form", "filing_date"], where={"ticker": [*SAMPLE, PREDECESSOR]})
    index = index.drop_duplicates("accession_number").rename(columns={"accession_number": "accession", "filing_date": "filed"})
    index = index.assign(company="", form="8-K", filed=pd.to_datetime(index["filed"]))
    monkeypatch.setattr(
        document_resume.edgar_index, "entries", lambda context, ciks, forms, window: index.loc[window.mask(index["filed"])].reset_index(drop=True)
    )
    chtr = pd.read_parquet(_run_cache() / "parity" / "published_5_runA_universe_tickers.parquet").query("ticker == @PREDECESSOR")
    bounded.universe.bounds[PREDECESSOR] = (pd.Timestamp(chtr["first_price_date"].iloc[0]), None)
    keys = pd.concat([rows[["ticker", "cik"]], chtr[["ticker", "cik"]]], ignore_index=True)
    bounded.run = control.run = WindowRun("full", "bounded cached filing sample", AS_OF)
    request = WorklistRequest(keys=keys, forms=["8-K"], identity=None)

    work = document_worklist(bounded, Tables.sec_8k, request)
    open_work = document_worklist(control, Tables.sec_8k, request)

    assert WBD_END < work.units["WBD"]["filed"].max() <= WBD_GRACE_END, work.units["WBD"]["filed"].max()
    assert work.units["WBD"]["filed"].eq(pd.Timestamp("2026-10-06")).any(), "U-65: the completion 8-K (filed the day after the end) is listed"
    assert work.units["AAPL"].equals(open_work.units["AAPL"]) and work.units[PREDECESSOR].equals(open_work.units[PREDECESSOR])
    first_chtr = bounded.universe.bounds[PREDECESSOR][0]
    early = int((work.units[PREDECESSOR]["filed"] < first_chtr).sum())
    assert early > 0
    dropped = sorted(set(open_work.units["WBD"]["accession"]) - set(work.units["WBD"]["accession"]))
    print("\n=== SANITY CHECK: EDGAR document planner bounds (sec_8k index rows of the fixture) ===")
    print(f"  WBD: {len(open_work.units['WBD'])} listed without bounds, {len(work.units['WBD'])} with; dropped {dropped} (filed after 2026-10-19)")
    print("  WBD's completion 8-K filed 2026-10-06 (the day after its end) is listed (U-65 grace)")
    print(f"  AAPL: {len(work.units['AAPL'])} units, identical with and without bounds")
    print(f"  CHTR: {early} filings before its listing's first price {first_chtr.date()} kept (predecessor registrant)")
    print("  Validated: no WBD filing after its end + grace; AAPL and CHTR's predecessor history unchanged.")


def _def14a_worklist(context: Any, walked: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """The filings `fetch_def14a_llm` hands its ticker loop when `sec_def14a_text` holds `walked`, under the bounds of
    `context` (the store read and the universe names are stubbed; nothing is extracted)."""
    seen: list[pd.DataFrame] = []
    monkeypatch.setattr(def14a_fetch, "planner_bounds", lambda ctx: planner_bounds(context))
    monkeypatch.setattr(def14a_fetch, "universe_rows", lambda ctx, tickers: pd.DataFrame({"ticker": SAMPLE, "name": SAMPLE}))
    monkeypatch.setattr(def14a_fetch, "ticker_works", lambda ctx, ticker, df_filings, floor: seen.append(df_filings) or [])
    config = context.config
    store = types.SimpleNamespace(load=lambda table, **kwargs: walked, create_declared=lambda table: None)
    run = types.SimpleNamespace(store=store, log=logging.getLogger(__name__), config=config, run=context.run, universe=context.universe)
    def14a_fetch.fetch_def14a_llm(run, config, SAMPLE, {})  # type: ignore[arg-type]
    return pd.concat(seen, ignore_index=True) if seen else pd.DataFrame(columns=["ticker"])


def test_def14a_worklist_keeps_only_proxies_inside_the_bounds(fixture: tuple[DataStore, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = fixture
    bounded, control = _contexts(fixture)
    stored = store.load(Tables.def14a_llm, columns=["ticker", "accession_number", "as_of"], where={"ticker": SAMPLE}).rename(
        columns={"as_of": "filing_date"}
    )
    stored["cik"] = stored["ticker"].map({"WBD": "0001437107", "AAPL": "0000320193"})
    # known truth: one walked proxy of each ticker dated after WBD's end
    late = pd.DataFrame(
        {
            "ticker": SAMPLE,
            "cik": ["0001437107", "0000320193"],
            "accession_number": ["late-wbd", "late-aapl"],
            "filing_date": ["2026-10-20", "2026-10-20"],
        }
    )
    walked = pd.concat([stored, late], ignore_index=True).assign(filing_date=lambda f: pd.to_datetime(f["filing_date"]), period="")
    worklist = _def14a_worklist(bounded, walked, monkeypatch)
    kept = {ticker: worklist[worklist["ticker"] == ticker].reset_index(drop=True) for ticker in SAMPLE}
    open_list = _def14a_worklist(control, walked, monkeypatch)
    open_aapl = open_list[open_list["ticker"] == "AAPL"].reset_index(drop=True)

    assert "late-wbd" not in set(kept["WBD"]["accession_number"]) and pd.to_datetime(kept["WBD"]["filing_date"]).max() <= WBD_GRACE_END
    assert kept["AAPL"].equals(open_aapl) and "late-aapl" in set(kept["AAPL"]["accession_number"])
    print("\n=== SANITY CHECK: DEF 14A (LLM) worklist bounds ===")
    print(
        f"  WBD: {len(kept['WBD'])} of {int((walked['ticker'] == 'WBD').sum())} walked proxies kept; the 2026-10-20 proxy is past its end + grace (2026-10-19)"
    )
    print(f"  AAPL: {len(kept['AAPL'])} kept, identical with and without bounds")
    print("  Validated.")


def test_earnings_surprises_skip_an_ended_ticker_and_drop_rows_after_its_end(
    fixture: tuple[DataStore, pd.DataFrame], sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, rows = fixture
    stored = store.load(Tables.earnings_surprises, where={"ticker": SAMPLE})
    pulls: list[str] = []

    def download(symbol: str, window: Window) -> pd.DataFrame:
        pulls.append(symbol)
        return stored[stored["ticker"] == symbol].reset_index(drop=True)

    monkeypatch.setattr(surprises, "_download_one", download)
    context = types.SimpleNamespace(
        store=sqlite_store,
        universe=_UniverseView(rows, bounded=True),
        log=logging.getLogger(__name__),
        config=extract_config(),
        run=WindowRun("incremental", "surprise fixture", AS_OF),
    )

    # first run: WBD unseen -> a full pull, its post-end forward row dropped
    surprises.fetch_earnings_surprises(context, SAMPLE)
    saved = sqlite_store.load(Tables.earnings_surprises)
    wbd = pd.to_datetime(saved.loc[saved["ticker"] == "WBD", "earnings_date"])
    wbd_source = pd.to_datetime(stored.loc[stored["ticker"] == "WBD", "earnings_date"])
    assert wbd.max() <= WBD_GRACE_END < wbd_source.max()

    # second run, past WBD's grace end plus the table's overlap (R-05): WBD has stored rows and has ended -> not pulled again
    after = _run_dates(Tables.earnings_surprises)["after"]
    context.run = WindowRun("incremental", "after-ended fixture", after)
    pulls.clear()
    surprises.fetch_earnings_surprises(context, SAMPLE)
    assert "WBD" not in pulls
    print("\n=== SANITY CHECK: earnings-surprise plan bounds ===")
    print(f"  WBD first run: {len(wbd)} of {len(wbd_source)} source rows saved, last {wbd.max().date()} (source to {wbd_source.max().date()})")
    print(f"  WBD second run on {after.date()}: not pulled (ended, rows stored, past grace end + overlap); pulls {pulls}")
    print("  Validated.")


#: R-05: a synthetic ended key with no stored row in any table (ended with WBD).
ROWLESS = "ZZZ"


def _run_dates(table: Any) -> dict[str, pd.Timestamp]:
    """R-05 run dates around WBD's end: inside the grace, the last day of the table's overlap after the grace end, the day after."""
    work = extract_config().data.tables[table.name].reprocess
    offset = work.overlap if work.mode == "tail" else work.lookback
    assert offset.unit == "calendar_days"
    last_day = WBD_GRACE_END + pd.Timedelta(days=offset.value)
    return {"inside the grace": pd.Timestamp("2026-10-12"), "last overlap day": last_day, "after": last_day + pd.Timedelta(days=1)}


class _NewRowlessView(_UniverseView):
    """The sample plus `ROWLESS`, ended with WBD and listed as a new key of the run."""

    def __init__(self, rows: pd.DataFrame) -> None:
        super().__init__(rows, bounded=True)
        self.bounds[ROWLESS] = (None, WBD_END)

    def added_on_dates(self, keys: list[str], as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
        return {ROWLESS: as_of}


@pytest.mark.parametrize("table", [Tables.prices, Tables.sharadar_fundamentals_facts], ids=["prices", "sharadar_fundamentals_facts"])
def test_an_ended_series_key_with_rows_is_done_once_the_run_is_past_its_grace_end_plus_the_overlap(
    fixture: tuple[DataStore, pd.DataFrame], table: Any
) -> None:
    """R-05: WBD (ended 2026-10-05, grace end 2026-10-19; prices to 2026-10-06, SF1 `date` to 2026-08-06) keeps a window
    clamped to its grace end until the run date passes the grace end plus the table's overlap, then gets none; an
    ended key with no row still gets its full pull from the floor."""
    store, rows = fixture
    keys = [*SAMPLE, ROWLESS]
    windows: dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]] = {}
    for label, day in _run_dates(table).items():
        context = types.SimpleNamespace(
            store=store, universe=_NewRowlessView(rows), config=extract_config(), run=WindowRun("incremental", "ended series fixture", day)
        )
        windows[label] = series_windows(context, table, SeriesRequest(keys, None, until=day)).windows

    assert windows["inside the grace"]["WBD"] and windows["last overlap day"]["WBD"], windows
    assert windows["after"]["WBD"] == [], windows["after"]["WBD"]
    assert all(end <= WBD_GRACE_END for spans in windows.values() for _, end in spans["WBD"])
    assert all(
        spans[ROWLESS]
        and spans[ROWLESS][0][1] <= WBD_GRACE_END
        and spans[ROWLESS][0][0] == pd.Timestamp(extract_config().data.tables[table.name].start_date)
        for spans in windows.values()
    )
    assert windows["after"]["AAPL"], "an open key keeps its window"
    print(f"\n=== SANITY CHECK: ended series key done after grace + overlap (R-05) on {table.name} ===")
    for label, day in _run_dates(table).items():
        print(f"  run {day.date()} ({label}): WBD {windows[label]['WBD']}; {ROWLESS} (no rows) {windows[label][ROWLESS]}")
    print(
        f"  Validated: WBD planned up to its grace end until {_run_dates(table)['last overlap day'].date()}, then done; a rowless ended key keeps its full pull."
    )


def test_earnings_surprises_refetch_an_ended_ticker_until_its_grace_end_plus_the_overlap(
    fixture: tuple[DataStore, pd.DataFrame], sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R-05: WBD's stored reported surprises end 2026-08-06 (ended 2026-10-05): it is asked of Yahoo inside the grace and the
    overlap after it, skipped once the run is past grace end + 35 days; an ended ticker with no row gets its full pull."""
    store, rows = fixture
    source = store.load(Tables.earnings_surprises, where={"ticker": SAMPLE})
    reported = source[source["eps_actual"].notna()].reset_index(drop=True)
    sqlite_store.save(Tables.earnings_surprises, reported)
    pulls: dict[str, list[tuple[str, Window]]] = {}

    def download(symbol: str, window: Window) -> pd.DataFrame | None:
        pulls[label].append((symbol, window))
        found = source[source["ticker"] == symbol].reset_index(drop=True)
        return found if not found.empty else None

    monkeypatch.setattr(surprises, "_download_one", download)
    context = types.SimpleNamespace(
        store=sqlite_store,
        universe=_NewRowlessView(rows),
        log=logging.getLogger(__name__),
        config=extract_config(),
        run=WindowRun("incremental", "ended surprise fixture", AS_OF),
    )
    for label, day in _run_dates(Tables.earnings_surprises).items():
        pulls[label] = []
        context.run = WindowRun("incremental", "ended surprise fixture", day)
        with caplog.at_level(logging.INFO, logger=__name__):
            surprises.fetch_earnings_surprises(context, [*SAMPLE, ROWLESS])

    assert "WBD" in dict(pulls["inside the grace"]) and "WBD" in dict(pulls["last overlap day"]), pulls
    assert "WBD" not in dict(pulls["after"]), pulls["after"]
    assert all(dict(plan)[ROWLESS] == Window(pd.Timestamp("2002-10-01"), WBD_GRACE_END) for plan in pulls.values()), pulls
    assert pd.to_datetime(sqlite_store.load(Tables.earnings_surprises, where={"ticker": ["WBD"]})["earnings_date"]).max() <= WBD_GRACE_END
    print("\n=== SANITY CHECK: earnings-surprise ended-ticker cut-off (R-05) ===")
    print(
        f"  WBD last reported {pd.to_datetime(reported.loc[reported['ticker'] == 'WBD', 'earnings_date']).max().date()}, ended 2026-10-05, grace end 2026-10-19"
    )
    for label, day in _run_dates(Tables.earnings_surprises).items():
        print(f"  run {day.date()} ({label}): pulls {pulls[label]}")
    print(f"  skip log: {[r.getMessage() for r in caplog.records if 'ended ticker' in r.getMessage()]}")
    print("  Validated: WBD re-asked inside grace + overlap, skipped after; a rowless ended ticker keeps its full pull.")


def test_earnings_call_planner_drops_calls_after_the_end(tmp_path: Path, sqlite_store: DataStore) -> None:
    # known truth: AAA ended 2026-01-01, its 2026-04-21 call is past the end; CCC is open
    seed_universe(sqlite_store, UNIVERSE, rows={"AAA": {"ended_on": "2026-01-01"}})
    # Actual January2026 sessions are required; an absent calendar cannot supply a fabricated grace.
    sessions = pd.to_datetime(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15"]
    )
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "AAA", "date": sessions, "close": 1.0}))
    source, _ = _source(_write(tmp_path / "t.parquet", [_AAA, _CCC]), "bounds")
    summary = ect.extract_earnings_calls(_ctx(tmp_path, sqlite_store), tickers=["AAA", "CCC"], source=source)
    calls = sqlite_store.load(Tables.earnings_call_sections, columns=["ticker", "quarter"]).drop_duplicates()
    assert sorted(map(tuple, calls.to_numpy())) == [("AAA", "2010Q1"), ("CCC", "2012Q1"), ("CCC", "2012Q2")]
    print("\n=== SANITY CHECK: earnings-call planner bounds ===")
    print(
        f"  {summary.calls_new} new call(s) written: {sorted(map(tuple, calls.to_numpy()))}; AAA's 2026Q1 call (2026-04-21) is after its end 2026-01-01"
    )
    print("  Validated.")


def test_the_planner_end_is_the_tenth_session_after_the_end(fixture: tuple[DataStore, pd.DataFrame]) -> None:
    """U-65: `planner_bounds` moves each `ended_on` 10 sessions later on the `prices` calendar (fixture: 2026-10-05 ->
    2026-10-19); an open key and the first price date are unchanged; with no stored calendar, no invented session grace."""
    bounded, _ = _contexts(fixture)
    empty = types.SimpleNamespace(
        store=types.SimpleNamespace(distinct=lambda *args, **kwargs: []), universe=bounded.universe, config=bounded.config, run=bounded.run
    )

    bounds, fallback = planner_bounds(bounded), planner_bounds(empty)

    assert bounded.config.data.planning.ended_grace.value == 10
    assert bounds["WBD"][1] == WBD_GRACE_END and bounds["WBD"][0] == bounded.universe.bounds["WBD"][0], bounds["WBD"]
    assert bounds["AAPL"] == (AAPL_FIRST, None) and fallback["WBD"][1] is None, fallback["WBD"]
    print("\n=== SANITY CHECK: planner end with grace (U-65) ===")
    print(
        f"  WBD ended {WBD_END.date()} -> planners stop at {bounds['WBD'][1].date()} (10th session); AAPL open; no calendar -> no invented session grace"
    )
