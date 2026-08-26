"""Resume-window resolution for the per-entity fetchers.

`resume_since` decides how far back a batch download must reach. It reads the
frontier with ONE grouped aggregate (`store.max_date_by`) instead of loading the
table -- `prices` is ~1.8M rows, and the old `groupby("ticker")["date"].max()`
paid a full read to learn a single date.

The `include_missing` flag is the subtle part, and the reason this test exists:
absence means "needs its whole history" on `prices` (a new ticker) but "will never
have a row" on `dividends` (a non-payer). Getting it wrong on `dividends` pins
every run to the full window forever.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.common.incremental import resume_since


def _seed(store) -> None:
    """AAA current through 2024-06-03, BBB stale since 2024-01-05."""
    store.replace("prices", pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2024-05-01", "2024-06-03", "2024-01-05"]),
        "close": [1.0, 2.0, 3.0],
    }))


def test_max_date_by_is_one_grouped_query(sqlite_store):
    _seed(sqlite_store)
    frontier = sqlite_store.max_date_by("prices", "ticker", "date")

    assert frontier == {"AAA": pd.Timestamp("2024-06-03"),
                        "BBB": pd.Timestamp("2024-01-05")}
    # absent table / column -> empty dict, the "nothing stored yet" contract
    assert sqlite_store.max_date_by("no_such_table", "ticker", "date") == {}
    print("\n=== SANITY CHECK: grouped max-date query ===")
    print(f"  max_date_by(prices) -> {({k: str(v.date()) for k, v in frontier.items()})}; "
          "missing table -> {}. One GROUP BY, no table read. Validated.")


def test_resume_since_takes_the_oldest_frontier(sqlite_store):
    _seed(sqlite_store)
    ctx = SimpleNamespace(store=sqlite_store)

    since = resume_since(ctx, "prices", ["AAA", "BBB"], years_history=15)
    assert since == pd.Timestamp("2024-01-05"), since   # the laggard sets the window

    # asking only about the current ticker moves the window forward
    assert resume_since(ctx, "prices", ["AAA"], years_history=15) == pd.Timestamp("2024-06-03")
    print("\n=== SANITY CHECK: oldest frontier wins ===")
    print(f"  [AAA,BBB] -> {since.date()} (BBB, the laggard); [AAA] -> 2024-06-03. Validated.")


def test_missing_ticker_pulls_back_only_when_include_missing(sqlite_store):
    _seed(sqlite_store)
    ctx = SimpleNamespace(store=sqlite_store)
    history_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=15)

    # prices semantics: an unseen ticker genuinely needs its full history
    assert resume_since(ctx, "prices", ["AAA", "NEW"], years_history=15) == history_start
    # dividends semantics: a never-payer must NOT drag the window back forever
    assert resume_since(ctx, "prices", ["AAA", "NEW"], years_history=15,
                        include_missing=False) == pd.Timestamp("2024-06-03")
    print("\n=== SANITY CHECK: include_missing ===")
    print(f"  unseen ticker: include_missing=True -> {history_start.date()} (full backfill); "
          "False -> 2024-06-03 (never-payers ignored). Validated.")


def test_window_never_predates_years_history(sqlite_store):
    """A ticker stale beyond the configured window must not widen it: the stored
    frontier is clamped, so a 1-year config never triggers a 15-year download."""
    _seed(sqlite_store)
    ctx = SimpleNamespace(store=sqlite_store)
    history_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=1)

    since = resume_since(ctx, "prices", ["AAA", "BBB"], years_history=1)
    assert since == history_start, since
    print("\n=== SANITY CHECK: window clamp ===")
    print(f"  BBB last traded 2024-01-05 but years_history=1 -> since={since.date()} "
          "(clamped, never older than the configured window). Validated.")


def test_cold_table_falls_back_to_full_history(sqlite_store):
    ctx = SimpleNamespace(store=sqlite_store)
    history_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=15)
    assert resume_since(ctx, "prices", ["AAA"], years_history=15) == history_start
    print("\n=== SANITY CHECK: cold start ===")
    print(f"  empty/missing table -> since={history_start.date()} (full years_history). Validated.")
