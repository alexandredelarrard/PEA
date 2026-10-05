"""`prices` holds the universe and its companies' current secondary share classes (Q2d, D-Q2-6).

The fetch list is read from `security_master` through the store: rows with role `secondary_class` and an
open end, under their Yahoo symbol. Delisted classes are not fetched; a new class takes the full window.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_prices as fp
from src.data_store.schema import Tables

MASTER_COLUMNS = ("security_id", "canonical_company", "market_symbol", "lineage_role", "valid_from", "valid_to")


def _master() -> pd.DataFrame:
    rows = [
        ("C084670702", "BRK-B", "BRK-B", "canonical_current", "2010-01-21", None),
        ("C084670108", "BRK-B", "BRK/A", "secondary_class", "2009-06-26", None),
        ("C02079K305", "GOOGL", "GOOGL", "canonical_current", "2015-10-02", None),
        ("C02079K107", "GOOGL", "GOOG", "secondary_class", "2015-10-02", None),
        ("C38259P706", "GOOGL", "GOOG", "secondary_class", "2014-04-01", "2015-10-02"),
        ("C934423104", "WBD", "WBD", "canonical_current", "2022-04-11", None),
        ("C25470F203", "WBD", "DISCB", "secondary_class", "2009-06-26", "2022-04-11"),
        ("C25470F302", "WBD", "DISCK", "secondary_class", "2009-06-26", "2022-04-11"),
        ("C060505682", "BAC", "BACPRL", "excluded", "2010-01-01", None),
        ("C526057302", "LEN", "LEN.B", "secondary_class", "2009-06-26", None),
    ]
    frame = pd.DataFrame(rows, columns=list(MASTER_COLUMNS))
    frame["source"] = "ftd"
    frame["source_symbol"] = frame["market_symbol"].str.replace(r"[/.-]", "", regex=True)
    for column in ("valid_from", "valid_to"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _bars(symbols: list[str], day: str = "2026-09-30") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(day),
            "ticker": symbols,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1_000.0,
        }
    )


@pytest.fixture
def fetched(sqlite_store, monkeypatch) -> tuple[Any, list[tuple[list[str], pd.Timestamp]]]:
    calls: list[tuple[list[str], pd.Timestamp]] = []

    def download(batch, since, until, *args, **kwargs):
        calls.append((list(batch), since))
        return _bars(list(batch))

    monkeypatch.setattr(fp, "download_ohlcv", download)
    monkeypatch.setattr(fp, "record_run", lambda *a, **k: None)
    monkeypatch.setattr(fp, "last_completed_session", lambda: pd.Timestamp("2026-10-02"))
    sqlite_store.save(Tables.security_master, _master())
    return sqlite_store, calls


@pytest.mark.parametrize(
    ("symbol", "yahoo"), [("BRK/A", "BRK-A"), ("BRK.A", "BRK-A"), ("brk-a", "BRK-A"), ("LEN/B", "LEN-B"), ("GOOG", "GOOG"), (" DISCB ", "DISCB")]
)
def test_yahoo_symbol_spelling(symbol, yahoo):
    assert fp.yahoo_symbol(symbol) == yahoo


def test_fetch_list_is_the_universe_plus_its_current_secondary_classes(fetched):
    store, calls = fetched
    ctx = SimpleNamespace(store=store)
    assert fp.secondary_class_symbols(cast(Any, ctx), ["BRK-B", "GOOGL", "WBD"]) == {"BRK-A": "BRK-B", "GOOG": "GOOGL"}

    fp.fetch_price_history(cast(Any, ctx), tickers=["BRK-B", "GOOGL", "WBD"], years_history=15)
    stored = set(store.load(Tables.prices, columns=["ticker"])["ticker"])
    assert stored == {"BRK-B", "GOOGL", "WBD", "BRK-A", "GOOG"}, stored
    assert not {"DISCB", "DISCK", "BACPRL", "LEN-B", "LEN.B"} & stored
    window_start = pd.Timestamp("2026-10-02") - pd.DateOffset(years=15)
    new_classes = [batch for batch, since in calls if since == window_start]
    assert new_classes == [["BRK-A", "GOOG"]], calls
    print("\n=== SANITY CHECK: AC-111 fetch list ===")
    print(f"  universe BRK-B, GOOGL, WBD + current classes BRK-A, GOOG -> prices {sorted(stored)}")
    print("  DISCB/DISCK (delisted), a preferred and another company's class are not fetched; new classes take the full window")


def test_a_company_split_repulls_its_secondary_class(fetched):
    store, calls = fetched
    store.save(Tables.prices, pd.concat([_bars(["GOOGL", "GOOG"], "2022-07-01"), _bars(["BRK-B", "BRK-A"], "2026-10-01")]))
    store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["GOOGL"], "date": pd.to_datetime(["2022-07-18"]), "split_ratio": [20.0]}))
    ctx = SimpleNamespace(store=store)
    owners = fp.secondary_class_symbols(cast(Any, ctx), ["BRK-B", "GOOGL"])
    assert fp.tickers_needing_repull(cast(Any, ctx), ["BRK-B", "GOOGL", *owners], owners) == ["GOOG", "GOOGL"]
    fp.fetch_price_history(cast(Any, ctx), tickers=["BRK-B", "GOOGL"], years_history=15)
    assert calls[0][0] == ["GOOG", "GOOGL"]
    print("\n=== SANITY CHECK: split vintage ===\n  GOOGL's 2022-07-18 split re-pulls GOOG's whole history with GOOGL's (both bars predate it)")


def test_a_ticker_subset_fetches_only_its_own_classes(fetched):
    store, calls = fetched
    fp.fetch_price_history(cast(Any, SimpleNamespace(store=store)), tickers=["WBD"], years_history=15)
    assert set(store.load(Tables.prices, columns=["ticker"])["ticker"]) == {"WBD"}
    print("\n=== SANITY CHECK: subset ===\n  -t WBD fetches WBD alone: its classes DISCB and DISCK are delisted")
