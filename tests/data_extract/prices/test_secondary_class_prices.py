"""`prices` holds the universe and its companies' current secondary share classes (Q2d, D-Q2-6).

The fetch list is read from `security_master` through the store: rows with role `secondary_class` and an
open end, under their Yahoo symbol. Delisted classes are not fetched; a class with no stored bar takes the
whole window. Dividends and splits are stored for the universe only.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_extract.utils.prices import fetch_prices as fp
from src.data_store.schema import Tables

MASTER_COLUMNS = ("security_id", "canonical_company", "market_symbol", "lineage_role", "valid_from", "valid_to")
_AS_OF = pd.Timestamp("2026-10-03")
_UNTIL = pd.Timestamp("2026-10-02")
_FLOOR = _AS_OF - pd.DateOffset(years=15)


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


def _bars(symbols: list[str], day: str = "2026-09-30", split: float = 0.0, dividend: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(day),
            "ticker": symbols,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1_000.0,
            "dividends": dividend,
            "stock splits": split,
        }
    )


@pytest.fixture
def fetched(sqlite_store, monkeypatch) -> tuple[Any, list[tuple[list[str], pd.Timestamp]], dict[str, pd.DataFrame]]:
    """A fake one-download yfinance: `responses[symbol]` is that symbol's bars (default one plain bar)."""
    calls: list[tuple[list[str], pd.Timestamp]] = []
    responses: dict[str, pd.DataFrame] = {}

    def download(batch, since, until, *args, **kwargs):
        calls.append((sorted(batch), pd.Timestamp(since)))
        return pd.concat([responses.get(symbol, _bars([symbol])) for symbol in batch], ignore_index=True)

    monkeypatch.setattr(fp, "download_ohlcv", download)
    monkeypatch.setattr(fp, "last_completed_session", lambda now=None: _UNTIL)
    sqlite_store.save(Tables.security_master, _master())
    return sqlite_store, calls, responses


def _seed_stored(store: Any, universe: list[str], classes: list[str]) -> None:
    """Stored bars on 2026-09-25 for `universe` and `classes`, and a stored dividend row for the universe only."""
    store.save(Tables.prices, _bars([*universe, *classes], "2026-09-25").drop(columns=["dividends", "stock splits"]))
    store.save(Tables.dividends, pd.DataFrame({"ticker": universe, "date": pd.Timestamp("2026-09-25"), "dividends": 0.0}))


def _context(store: Any) -> Any:
    return SimpleNamespace(
        store=store, log=logging.getLogger("test.secondary_prices"), config=OmegaConf.create({"data_extract": {"years_history": 15}})
    )


@pytest.mark.parametrize(
    ("symbol", "yahoo"), [("BRK/A", "BRK-A"), ("BRK.A", "BRK-A"), ("brk-a", "BRK-A"), ("LEN/B", "LEN-B"), ("GOOG", "GOOG"), (" DISCB ", "DISCB")]
)
def test_yahoo_symbol_spelling(symbol, yahoo):
    assert fp.yahoo_symbol(symbol) == yahoo


def test_fetch_list_is_the_universe_plus_its_current_secondary_classes(fetched):
    store, calls, _ = fetched
    ctx = _context(store)
    assert fp.secondary_class_symbols(cast(Any, ctx), ["BRK-B", "GOOGL", "WBD"]) == {"BRK-A": "BRK-B", "GOOG": "GOOGL"}
    _seed_stored(store, ["BRK-B", "GOOGL", "WBD"], [])

    fp.fetch_prices_and_actions(cast(Any, ctx), tickers=["BRK-B", "GOOGL", "WBD"], years_history=15, as_of=_AS_OF)

    stored = set(store.load(Tables.prices, columns=["ticker"])["ticker"])
    assert stored == {"BRK-B", "GOOGL", "WBD", "BRK-A", "GOOG"}, stored
    assert not {"DISCB", "DISCK", "BACPRL", "LEN-B", "LEN.B"} & stored
    assert (["BRK-A", "GOOG"], _FLOOR) in calls, calls
    assert set(store.load(Tables.dividends, columns=["ticker"])["ticker"]) <= {"BRK-B", "GOOGL", "WBD"}
    print("\n=== SANITY CHECK: AC-111 fetch list ===")
    print(f"  universe BRK-B, GOOGL, WBD + current classes BRK-A, GOOG -> prices {sorted(stored)}; download groups {calls}")
    print("  DISCB/DISCK (delisted), a preferred and another company's class are not fetched; the new classes take the whole window")


def test_a_split_in_a_class_response_repulls_that_class_and_stores_only_the_universe_split(fetched):
    store, calls, responses = fetched
    _seed_stored(store, ["GOOGL", "BRK-B"], ["GOOG", "BRK-A"])
    responses["GOOGL"] = _bars(["GOOGL"], "2026-09-29", split=20.0)
    responses["GOOG"] = _bars(["GOOG"], "2026-09-29", split=20.0)

    fp.fetch_prices_and_actions(cast(Any, _context(store)), tickers=["BRK-B", "GOOGL"], years_history=15, as_of=_AS_OF)

    assert calls[-1] == (["GOOG", "GOOGL"], _FLOOR), calls
    assert store.load(Tables.prices_splits)["ticker"].tolist() == ["GOOGL"]
    print("\n=== SANITY CHECK: split vintage ===")
    print("  the 2026-09-29 split in GOOGL's and GOOG's own responses re-pulls both over the whole window; only GOOGL's split is stored")


def test_a_ticker_subset_fetches_only_its_own_classes(fetched):
    store, _, _ = fetched
    fp.fetch_prices_and_actions(cast(Any, _context(store)), tickers=["WBD"], years_history=15, as_of=_AS_OF)
    assert set(store.load(Tables.prices, columns=["ticker"])["ticker"]) == {"WBD"}
    print("\n=== SANITY CHECK: subset ===\n  -t WBD fetches WBD alone: its classes DISCB and DISCK are delisted")
