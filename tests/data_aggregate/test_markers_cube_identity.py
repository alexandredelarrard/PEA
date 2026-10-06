"""AC-016 check 3: empty-filing markers never change a cube cell.

Real rows of five tickers are copied from the live DB (read only) into an in-memory SQLite
store. The governance, insider and beneficial-ownership panels and the fundamentals history
are built from that store, 20 marker rows are added, and everything is built again. The
all-filer 13F panel gets the same check against the new-ticker gap-walk markers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
from src.data_aggregate.utils.governance.directors import finalize_board_source
from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_aggregate.utils.institutionals.inputs import load_source
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.institutional_features import build_institutional_feature_panel
from src.data_aggregate.utils.institutionals.ownership_features import build_ownership_feature_panel
from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_extract.utils.fundamentals.build_history import FACT_COLUMNS, build_ticker_history
from src.data_extract.utils.institutionals.fetch_13f_backfill import mark_gap_walked
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from tests.conftest import _store, make_frames

TICKERS = ["AAPL", "DIS", "HPQ", "JPM", "PG"]
GRID_START = pd.Timestamp("2024-01-02")
#: The fundamentals history replay is slow, so it runs on two tickers' facts filed since FACTS_SINCE.
HISTORY_TICKERS = ["HPQ", "PG"]
FACTS_SINCE = pd.Timestamp("2023-01-01")
#: Synthetic accessions per table: 20 in total.
MARKERS_PER_TABLE = {
    Tables.sec_13d: 3,
    Tables.sec_13g: 4,
    Tables.insider_transactions: 4,
    Tables.def14a_llm: 3,
    Tables.sec_8k_votes: 3,
    Tables.fundamentals_facts: 3,
}
#: The 13F gap-marker check: two tickers' `sec13f_hr` rows from F13_SINCE and one marker each.
F13_TICKERS = ["HPQ", "PG"]
F13_SINCE = pd.Timestamp("2023-01-01")
F13_GAP_START = pd.Timestamp("2024-09-01")


def _copy_live_rows(live: DataStore, sqlite: DataStore) -> pd.DataFrame:
    """Copy the five tickers' source rows into `sqlite`; return their wide close prices."""
    for table in (Tables.sec_13d, Tables.sec_13g, Tables.insider_transactions, Tables.def14a_llm, Tables.sec_8k_votes):
        sqlite.save(table, cast(pd.DataFrame, live.load(table, where={"ticker": TICKERS})))
    facts = cast(pd.DataFrame, live.load(Tables.fundamentals_facts, columns=list(FACT_COLUMNS), where={"ticker": HISTORY_TICKERS}))
    facts = facts[pd.to_datetime(facts["filing_date"]) >= FACTS_SINCE]
    sqlite.save(Tables.fundamentals_facts, facts)
    prices = cast(pd.DataFrame, live.load(Tables.prices, columns=["date", "ticker", "close_split", "close_total"], where={"ticker": TICKERS}))
    prices = prices[pd.to_datetime(prices["date"]) >= GRID_START]
    return prices.assign(date=pd.to_datetime(prices["date"]))


def _marker_dates(store: DataStore, table: Table, n: int) -> list[tuple[str, pd.Timestamp]]:
    """`n` (ticker, date) pairs spread over the stored frontier dates, never after the latest one."""
    assert table.resume is not None and table.resume.frontier_col is not None
    stored = cast(pd.DataFrame, store.load(table, columns=["ticker", table.resume.frontier_col]))
    stored = stored.assign(day=pd.to_datetime(stored[table.resume.frontier_col])).sort_values("day")
    picks = stored.iloc[[int(i * (len(stored) - 1) / max(n - 1, 1)) for i in range(n)]]
    return list(zip(picks["ticker"].astype(str), picks["day"], strict=True))


def _add_markers(store: DataStore) -> int:
    """Save one marker per synthetic accession, built the way each fetcher builds it."""
    forms = {Tables.sec_13d: "SC 13D", Tables.sec_13g: "SC 13G", Tables.insider_transactions: "4", Tables.fundamentals_facts: "10-Q"}
    n_saved = 0
    for table, n in MARKERS_PER_TABLE.items():
        for i, (ticker, day) in enumerate(_marker_dates(store, table, n)):
            accession = f"9999999999-26-{n_saved:06d}"
            if table is Tables.def14a_llm:
                frame = marker_frame(table, {"ticker": ticker, "accession_number": accession, "as_of": day})
            elif table is Tables.sec_8k_votes:
                frame = marker_frame(table, {"ticker": ticker, "accession_number": accession, "filing_date": day, "cik": "0000000001", "form": "8-K"})
            else:
                frame = marker_row(table, ticker, FilingStamp(accession, forms[table], "0000000001", day + pd.Timedelta(hours=i), False, None, None))
            store.save(table, frame)
            n_saved += 1
    return n_saved


def _builders(prices: pd.DataFrame) -> dict[str, Callable[[DataStore], pd.DataFrame]]:
    """Each consumer of a marker table, built from a store the way its cube step reads it."""
    close = prices.pivot(index="date", columns="ticker", values="close_split").sort_index()
    close_total = prices.pivot(index="date", columns="ticker", values="close_total").sort_index()
    peers = {t: {p: 1.0 for p in TICKERS if p != t} for t in TICKERS}
    frames = make_frames(close.index, peers, close_split=close, close_total=close_total)

    def ownership(store: DataStore) -> pd.DataFrame:
        return build_ownership_feature_panel(
            frames,
            store.load(Tables.sec_13d, project=True),
            store.load(Tables.sec_13g, project=True),
            complete_through_13d=schedule_complete_through(store, Tables.sec_13d, close.index.max()),
            complete_through_13g=schedule_complete_through(store, Tables.sec_13g, close.index.max()),
        )

    def insider(store: DataStore) -> pd.DataFrame:
        complete_through = schedule_complete_through(store, Tables.insider_transactions, close.index.max())
        return build_insider_feature_panel(frames, store.load(Tables.insider_transactions, project=True), complete_through=complete_through)

    def governance(store: DataStore) -> pd.DataFrame:
        proxies, _ = impute_def14a(cast(pd.DataFrame, store.load(Tables.def14a_llm)))
        proxies, _ = finalize_board_source(proxies)
        panel, _ = build_governance_feature_panel(
            proxies, peers, frames.trading_index, votes=store.load(Tables.sec_8k_votes), close_total=close_total, availability=frames.availability
        )
        return panel

    def history(store: DataStore) -> pd.DataFrame:
        parts = [
            build_ticker_history(t, store.load(Tables.fundamentals_facts, columns=list(FACT_COLUMNS), where={"ticker": t})) for t in HISTORY_TICKERS
        ]
        return pd.concat(parts, ignore_index=True)

    return {"ownership": ownership, "insider": insider, "governance": governance, "fundamentals history": history}


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("ticker", "date", "as_of") if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True)[sorted(df.columns)]


def test_markers_never_change_a_cube_cell(sqlite_store: DataStore) -> None:
    """Every marker-table consumer is cell-identical with and without 20 marker rows."""
    live = _store()
    try:
        prices = _copy_live_rows(live, sqlite_store)
    except Exception as exc:  # noqa: BLE001 -- a missing live table means no real data to test on
        pytest.skip(f"live source rows unavailable: {type(exc).__name__}: {exc}")
    builders = _builders(prices)
    frontiers = {
        t: schedule_complete_through(sqlite_store, t, prices["date"].max()) for t in (Tables.sec_13d, Tables.sec_13g, Tables.insider_transactions)
    }
    before = {name: _canonical(build(sqlite_store)) for name, build in builders.items()}

    n_markers = _add_markers(sqlite_store)
    stored = {
        t: len(cast(pd.DataFrame, sqlite_store.load(t, markers=True))) - len(cast(pd.DataFrame, sqlite_store.load(t))) for t in MARKERS_PER_TABLE
    }
    after = {name: _canonical(build(sqlite_store)) for name, build in builders.items()}

    assert n_markers == 20 and stored == {t: n for t, n in MARKERS_PER_TABLE.items()}
    assert {t: schedule_complete_through(sqlite_store, t, prices["date"].max()) for t in frontiers} == frontiers
    for name in builders:
        assert not before[name].empty, f"{name}: built nothing from real rows"
        pd.testing.assert_frame_equal(before[name], after[name], check_exact=True)

    print("\n=== SANITY CHECK: markers never change a cube cell (AC-016 check 3) ===")
    print(f"  tickers {TICKERS}, grid {prices['date'].min().date()}..{prices['date'].max().date()}")
    print(f"  markers stored: {', '.join(f'{t.name}={n}' for t, n in stored.items())} (total {n_markers})")
    for name, frame in before.items():
        print(f"  {name:<20} {frame.shape[0]:>7,} rows x {frame.shape[1]:>3} cols: identical")
    print(f"  13D/13G/insider frontiers unchanged: {', '.join(f'{t.name}={d.date() if d is not None else None}' for t, d in frontiers.items())}")
    print("  CONCLUSION: the store's marker filter keeps every consumer cell-identical. Validated.")


def test_13f_gap_markers_never_change_an_institutional_cell(sqlite_store: DataStore) -> None:
    """The all-filer 13F panel, read the way the cube step reads `sec13f_hr`, is cell-identical with and
    without one gap-walk marker per ticker."""
    live = _store()
    try:
        holdings = live.load(Tables.sec13f_hr, columns=[*Tables.sec13f_hr.read_columns, "cusip"], where={"ticker": F13_TICKERS}, since=F13_SINCE)
        prices = cast(
            pd.DataFrame, live.load(Tables.prices, columns=["date", "ticker", "close_split"], where={"ticker": F13_TICKERS}, since=GRID_START)
        )
    except Exception as exc:  # noqa: BLE001 -- a missing live table means no real data to test on
        pytest.skip(f"live source rows unavailable: {type(exc).__name__}: {exc}")
    sqlite_store.save(Tables.sec13f_hr, cast(pd.DataFrame, holdings))
    close = prices.assign(date=pd.to_datetime(prices["date"])).pivot(index="date", columns="ticker", values="close_split").sort_index()
    frames = make_frames(close.index, {t: {p: 1.0 for p in F13_TICKERS if p != t} for t in F13_TICKERS}, close_split=close)
    log = logging.getLogger(__name__)

    def build() -> pd.DataFrame:
        return build_institutional_feature_panel(frames, load_source(sqlite_store, log, Tables.sec13f_hr, F13_TICKERS))

    before = _canonical(build())
    saved = mark_gap_walked(cast(Any, SimpleNamespace(store=sqlite_store)), F13_TICKERS, F13_GAP_START)
    hidden = len(cast(pd.DataFrame, sqlite_store.load(Tables.sec13f_hr, markers=True))) - len(cast(pd.DataFrame, sqlite_store.load(Tables.sec13f_hr)))
    after = _canonical(build())

    assert saved == hidden == len(F13_TICKERS)
    assert not before.empty
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    print("\n=== SANITY CHECK: 13F gap markers never change an institutional cell ===")
    print(
        f"  {len(cast(pd.DataFrame, holdings)):,} live sec13f_hr rows of {F13_TICKERS} since {F13_SINCE.date()}, {saved} markers dated {F13_GAP_START.date()}"
    )
    print(f"  institutional panel {before.shape[0]:,} rows x {before.shape[1]} cols: identical. Validated.")
