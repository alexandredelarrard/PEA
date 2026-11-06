"""AC-008 / REQ-011: every per-ticker planner clamps its work to the ticker's identity bounds.

`Universe.bounds` = `(first_price_date, ended_on)`. On the phase-0 fixture store (read-only) with the identity columns
the phase-5 publish wrote: WBD ended on 2026-10-05 (prices stored to 2026-10-06, its completion 8-K filed 2026-10-06),
AAPL open from 1995-10-09. Each planner yields nothing after WBD's end plus the U-65 grace of 10 trading sessions
(2026-10-19 on the fixture's price calendar) and the same plan for AAPL as with no bounds:
the series planner (`prices`, `fundamentals_sharadar`), the EDGAR document planner, the DEF 14A (LLM) listing, the
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
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.common.resume import END_GRACE_SESSIONS, planner_bounds, series_windows
from src.data_extract.utils.edgar_extract import document_resume
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_extract.utils.structure.def14a import fetch as def14a_fetch
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.behavioral.test_earnings_call_transcripts import _AAA, _CCC, _CFG, UNIVERSE, _ctx, _source, _write
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

    def new_since(self, overlap_days: int, as_of: pd.Timestamp) -> set[str]:
        return set()


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
        types.SimpleNamespace(store=store, universe=_UniverseView(rows, bounded=True)),
        types.SimpleNamespace(store=store, universe=_UniverseView(rows, bounded=False)),
    )


@pytest.mark.parametrize("table", [Tables.prices, Tables.sharadar_fundamentals], ids=["prices", "sharadar_fundamentals"])
def test_series_planner_stops_at_the_end_and_starts_at_the_first_price(fixture: tuple[DataStore, pd.DataFrame], table: Any) -> None:
    bounded, control = _contexts(fixture)
    until = pd.Timestamp("2026-11-02")
    work = series_windows(bounded, table, SAMPLE, AS_OF, until=until, years_history=40, calendar=None)
    open_work = series_windows(control, table, SAMPLE, AS_OF, until=until, years_history=40, calendar=None)
    full = series_windows(bounded, table, SAMPLE, AS_OF, until=until, years_history=40, full=True, calendar=None)

    wbd_last = work.last["WBD"]
    assert all(end <= WBD_GRACE_END for _, end in work.windows["WBD"]), work.windows["WBD"]
    if pd.Timestamp(wbd_last) >= WBD_END:
        assert work.windows["WBD"] == []
    else:
        assert work.windows["WBD"][-1][1] == WBD_GRACE_END and open_work.windows["WBD"][-1][1] == until
    assert work.windows["AAPL"] == open_work.windows["AAPL"]
    assert full.windows["AAPL"][0][0] == AAPL_FIRST and open_work.windows["AAPL"] != []

    print(f"\n=== SANITY CHECK: series planner bounds on {table.name} ===")
    print(
        f"  WBD last stored {pd.Timestamp(wbd_last).date()}, ended 2026-10-05: windows {work.windows['WBD']} (no bounds: {open_work.windows['WBD']})"
    )
    print(f"  AAPL windows unchanged {work.windows['AAPL']}; a full pull starts at its first price {full.windows['AAPL'][0][0].date()} (floor 1986)")
    print("  Validated: nothing after WBD's end + grace, none once its history reaches the end, nothing before AAPL's first price.")


def test_edgar_document_planner_drops_filings_after_the_end(fixture: tuple[DataStore, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> None:
    store, rows = fixture
    bounded, control = _contexts(fixture)
    index = store.load(Tables.sec_8k, columns=["cik", "accession_number", "form", "filing_date"], where={"ticker": [*SAMPLE, PREDECESSOR]})
    index = index.drop_duplicates("accession_number").rename(columns={"accession_number": "accession", "filing_date": "filed"})
    index = index.assign(company="", form="8-K", filed=pd.to_datetime(index["filed"]))
    monkeypatch.setattr(
        document_resume.edgar_index, "entries", lambda context, ciks, forms, since=None: index[index["filed"] >= since].reset_index(drop=True)
    )
    chtr = pd.read_parquet(_run_cache() / "parity" / "published_5_runA_universe_tickers.parquet").query("ticker == @PREDECESSOR")
    bounded.universe.bounds[PREDECESSOR] = (pd.Timestamp(chtr["first_price_date"].iloc[0]), None)
    keys = pd.concat([rows[["ticker", "cik"]], chtr[["ticker", "cik"]]], ignore_index=True)
    request = WorklistRequest(keys=keys, as_of=AS_OF, forms=["8-K"], identity=None, years_history=40, full=True)

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


def test_def14a_listing_keeps_only_proxies_inside_the_bounds(fixture: tuple[DataStore, pd.DataFrame], monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = fixture
    bounded, control = _contexts(fixture)
    stored = store.load(Tables.def14a_llm, columns=["ticker", "accession_number", "as_of"], where={"ticker": SAMPLE}).rename(
        columns={"as_of": "filing_date"}
    )
    stored["cik"] = stored["ticker"].map({"WBD": "0001437107", "AAPL": "0000320193"})
    # known truth: one listed proxy of each ticker dated after WBD's end
    late = pd.DataFrame(
        {
            "ticker": SAMPLE,
            "cik": ["0001437107", "0000320193"],
            "accession_number": ["late-wbd", "late-aapl"],
            "filing_date": ["2026-10-20", "2026-10-20"],
        }
    )
    listing = pd.concat([stored, late], ignore_index=True).assign(filing_date=lambda f: pd.to_datetime(f["filing_date"]))
    monkeypatch.setattr(def14a_fetch, "_list_scope_windows", lambda context, scope, company, years, since: listing[listing["ticker"] == scope.ticker])
    kept = {
        ticker: def14a_fetch._listed_proxies(bounded, ticker, FilingScope.roster_only(ticker, ""), "", 40, pd.Timestamp("1990-01-01"))
        for ticker in SAMPLE
    }
    open_aapl = def14a_fetch._listed_proxies(control, "AAPL", FilingScope.roster_only("AAPL", ""), "", 40, pd.Timestamp("1990-01-01"))

    assert "late-wbd" not in set(kept["WBD"]["accession_number"]) and pd.to_datetime(kept["WBD"]["filing_date"]).max() <= WBD_GRACE_END
    assert kept["AAPL"].equals(open_aapl) and "late-aapl" in set(kept["AAPL"]["accession_number"])
    print("\n=== SANITY CHECK: DEF 14A (LLM) listing bounds ===")
    print(
        f"  WBD: {len(kept['WBD'])} of {int((listing['ticker'] == 'WBD').sum())} listed proxies kept; the 2026-10-20 proxy is past its end + grace (2026-10-19)"
    )
    print(f"  AAPL: {len(kept['AAPL'])} kept, identical with and without bounds")
    print("  Validated.")


def test_earnings_surprises_skip_an_ended_ticker_and_drop_rows_after_its_end(
    fixture: tuple[DataStore, pd.DataFrame], sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, rows = fixture
    stored = store.load(Tables.earnings_surprises, where={"ticker": SAMPLE})
    pulls: list[str] = []

    def download(symbol: str, limit: int) -> pd.DataFrame:
        pulls.append(symbol)
        return stored[stored["ticker"] == symbol].reset_index(drop=True)

    monkeypatch.setattr(surprises, "_download_one", download)
    context = types.SimpleNamespace(store=sqlite_store, universe=_UniverseView(rows, bounded=True), log=logging.getLogger(__name__))

    # first run: WBD unseen -> a full pull, its post-end forward row dropped
    surprises.fetch_earnings_surprises(context, SAMPLE, years_history=31, pause=0.0)
    saved = sqlite_store.load(Tables.earnings_surprises)
    wbd = pd.to_datetime(saved.loc[saved["ticker"] == "WBD", "earnings_date"])
    wbd_source = pd.to_datetime(stored.loc[stored["ticker"] == "WBD", "earnings_date"])
    assert wbd.max() <= WBD_GRACE_END < wbd_source.max()

    # second run: an ended WBD is skipped only once its last reported date reaches the grace end (R-05, REQ-011);
    # either way no row after the grace end is stored
    reported = pd.to_datetime(saved.loc[(saved["ticker"] == "WBD") & saved["eps_actual"].notna(), "earnings_date"]).max()
    pulls.clear()
    surprises.fetch_earnings_surprises(context, SAMPLE, years_history=31, pause=0.0)
    again = pd.to_datetime(sqlite_store.load(Tables.earnings_surprises, where={"ticker": "WBD"})["earnings_date"])
    assert ("WBD" in pulls) == (reported < WBD_GRACE_END), (reported, pulls)
    assert again.max() <= WBD_GRACE_END, again.max()
    print("\n=== SANITY CHECK: earnings-surprise plan bounds ===")
    print(f"  WBD first run: {len(wbd)} of {len(wbd_source)} source rows saved, last {wbd.max().date()} (source to {wbd_source.max().date()})")
    print(f"  WBD second run: last reported {reported.date()} vs grace end {WBD_GRACE_END.date()} -> pulls {pulls}; stored to {again.max().date()}")
    print("  Validated.")


def test_earnings_call_planner_drops_calls_after_the_end(tmp_path: Path, sqlite_store: DataStore) -> None:
    # known truth: AAA ended 2026-01-01, its 2026-04-21 call is past the end; CCC is open
    seed_universe(sqlite_store, UNIVERSE, rows={"AAA": {"ended_on": "2026-01-01"}})
    source, _ = _source(_write(tmp_path / "t.parquet", [_AAA, _CCC]), "bounds")
    summary = ect.extract_earnings_calls(_ctx(tmp_path, sqlite_store), _CFG, tickers=["AAA", "CCC"], source=source)
    calls = sqlite_store.load(Tables.earnings_call_sections, columns=["ticker", "quarter"]).drop_duplicates()
    assert sorted(map(tuple, calls.to_numpy())) == [("AAA", "2010Q1"), ("CCC", "2012Q1"), ("CCC", "2012Q2")]
    print("\n=== SANITY CHECK: earnings-call planner bounds ===")
    print(
        f"  {summary.calls_new} new call(s) written: {sorted(map(tuple, calls.to_numpy()))}; AAA's 2026Q1 call (2026-04-21) is after its end 2026-01-01"
    )
    print("  Validated.")


def test_the_planner_end_is_the_tenth_session_after_the_end(fixture: tuple[DataStore, pd.DataFrame]) -> None:
    """U-65: `planner_bounds` moves each `ended_on` 10 sessions later on the `prices` calendar (fixture: 2026-10-05 ->
    2026-10-19); an open key and the first price date are unchanged; with no stored calendar, 10 weekdays."""
    bounded, _ = _contexts(fixture)
    empty = types.SimpleNamespace(store=types.SimpleNamespace(distinct=lambda *args, **kwargs: []), universe=bounded.universe)

    bounds, fallback = planner_bounds(bounded), planner_bounds(empty)

    assert END_GRACE_SESSIONS == 10
    assert bounds["WBD"][1] == WBD_GRACE_END and bounds["WBD"][0] == bounded.universe.bounds["WBD"][0], bounds["WBD"]
    assert bounds["AAPL"] == (AAPL_FIRST, None) and fallback["WBD"][1] == pd.Timestamp("2026-10-19"), fallback["WBD"]
    print("\n=== SANITY CHECK: planner end with grace (U-65) ===")
    print(f"  WBD ended {WBD_END.date()} -> planners stop at {bounds['WBD'][1].date()} (10th session); AAPL open; no calendar -> 10 weekdays")
