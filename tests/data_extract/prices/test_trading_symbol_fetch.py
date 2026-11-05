"""REQ-013: ticker-keyed sources are queried under the dated trading symbol and stored under the frozen universe key.

PSKY's company trades as SKYD from 2026-10-06 (phase-5 publish of the fixture store: `trading_symbol = SKYD`,
`status = renamed`). Yahoo prices, Yahoo earnings surprises and Sharadar SF1 are asked for SKYD (Sharadar in its
vendor spelling) and every row is stored under PSKY; AAPL is asked for under its own key. Network calls are stubbed
(known truth: the stub records the symbols requested).
"""

from __future__ import annotations

import logging
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar as fetch_mod
from src.data_extract.utils.prices import fetch_earnings_surprises as surprises
from src.data_extract.utils.prices import fetch_prices as fp
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.edgar_fixtures import fake_context
from tests.support.universe import seed_universe

ROOT = Path(__file__).resolve().parents[3]
PUBLISHED = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache" / "parity" / "published_5_runA_universe_tickers.parquet"
TICKERS = ["AAPL", "PSKY"]


def _published_symbols() -> dict[str, str]:
    """The fixture's published trading symbols of the sample (PSKY -> SKYD); skips when the run dir is absent."""
    for base in (ROOT, *ROOT.parents):
        if (base / PUBLISHED).exists():
            rows = pd.read_parquet(base / PUBLISHED)
            return dict(zip(rows["ticker"], rows["trading_symbol"], strict=True))
    pytest.skip("the phase-5 publish of the fixture store is not reachable from this checkout")


def _universe(store: DataStore) -> Any:
    symbols = _published_symbols()
    return seed_universe(store, TICKERS, rows={t: {"trading_symbol": symbols[t]} for t in TICKERS})


def test_universe_and_identity_expose_the_trading_symbol(sqlite_store: DataStore) -> None:
    universe = _universe(sqlite_store)
    identity = build_identity(
        lineage=pd.DataFrame(
            {"cik": ["0000000001"], "company_id": ["E0000000001"], "role": ["cik_window"], "symbol": [""], "valid_from": ["2000-01-01"]}
        ),
        tenure=pd.DataFrame({"symbol": ["AAPL"], "issuer_cik": ["0000000001"], "valid_from": ["2000-01-01"], "valid_to": [None], "n_filings": [1]}),
        roster_cik=universe.cik_by_ticker,
    )
    identity = replace(identity, trading_symbols=universe.trading_symbols, bounds=universe.bounds)
    assert universe.trading_symbols == {"AAPL": "AAPL", "PSKY": "SKYD"}
    assert identity.trading_symbol("psky") == "SKYD" and identity.trading_symbol("AAPL") == "AAPL" and identity.ended_on("PSKY") is None
    print("\n=== SANITY CHECK: trading-symbol API ===")
    print(f"  Universe.trading_symbols {universe.trading_symbols}; Identity.trading_symbol('PSKY') = {identity.trading_symbol('PSKY')}")
    print("  Validated.")


def test_yahoo_prices_query_the_trading_symbol_and_store_the_key(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    universe = _universe(sqlite_store)
    requested: list[list[str]] = []

    def download(tickers: list[str], since: pd.Timestamp, until: pd.Timestamp, *args: Any, **kwargs: Any) -> pd.DataFrame:
        requested.append(list(tickers))
        days = pd.bdate_range(since, until)
        return pd.concat(
            [
                pd.DataFrame(
                    {"date": days, "ticker": t, "open": 1.0, "high": 1.0, "low": 1.0, "close_split": 1.0, "close_total": 1.0, "volume": 10.0}
                )
                for t in tickers
            ],
            ignore_index=True,
        )

    monkeypatch.setattr(fp, "download_ohlcv", download)
    session = fp.YahooSession(pause=0.0, symbols=universe.trading_symbols)
    rows = session.pull([(pd.Timestamp("2026-10-26"), pd.Timestamp("2026-10-30"), TICKERS)], "test")
    bars = session.bars(["PSKY"], pd.Timestamp("2026-10-26"), pd.Timestamp("2026-10-30"))

    assert requested[0] == ["AAPL", "SKYD"] and requested[1] == ["SKYD"]
    assert set(rows["ticker"]) == {"AAPL", "PSKY"} and set(bars["ticker"]) == {"PSKY"}
    print("\n=== SANITY CHECK: Yahoo prices under the trading symbol ===")
    print(f"  requested {requested}; stored keys {sorted(set(rows['ticker']))} ({len(rows)} bars)")
    print("  Validated: PSKY is asked for as SKYD and stored as PSKY.")


def test_earnings_surprises_query_the_trading_symbol(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    universe = _universe(sqlite_store)
    requested: list[str] = []

    def download(symbol: str, limit: int) -> pd.DataFrame:
        requested.append(symbol)
        return pd.DataFrame(
            {"ticker": [symbol], "earnings_date": [pd.Timestamp("2026-08-04")], "eps_estimate": [0.1], "eps_actual": [0.1], "surprise_pct": [0.0]}
        )

    monkeypatch.setattr(surprises, "_download_one", download)
    context = types.SimpleNamespace(store=sqlite_store, universe=universe, log=logging.getLogger(__name__))
    surprises.fetch_earnings_surprises(context, TICKERS, years_history=31, pause=0.0)
    saved = sqlite_store.load(Tables.earnings_surprises)

    assert sorted(requested) == ["AAPL", "SKYD"] and sorted(saved["ticker"]) == ["AAPL", "PSKY"]
    print("\n=== SANITY CHECK: earnings surprises under the trading symbol ===")
    print(f"  requested {sorted(requested)}; stored {sorted(saved['ticker'])}. Validated.")


def test_sharadar_sf1_queries_the_trading_symbol(tmp_path: Path, sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context = fake_context(tmp_path, sqlite_store, TICKERS, sharadar_request_pace=0.0)
    context.universe = _universe(sqlite_store)
    context.log.debug = lambda msg, *a: None
    sqlite_store.save(
        Tables.sharadar_tickers,
        pd.DataFrame({"table": "SF1", "permaticker": [1, 2], "ticker": ["AAPL", "SKYD"], "currency": "USD", "isdelisted": "N"}),
    )
    requested: list[str] = []

    def get(ctx: Any, table: str, /, **filters: Any) -> pd.DataFrame:
        requested.append(filters["ticker"])
        return pd.DataFrame()

    monkeypatch.setattr(fetch_mod, "sharadar_get", get)
    monkeypatch.setattr(fetch_mod, "sleep_pace", lambda *a, **k: None)
    fetch_mod.fetch_sharadar_fundamentals(context, TICKERS, years_history=10, as_of=pd.Timestamp("2026-11-03"))

    assert set(requested) == {"AAPL", "SKYD"}
    print("\n=== SANITY CHECK: Sharadar SF1 under the trading symbol ===")
    print(f"  requested {sorted(set(requested))} (PSKY's vendor row is SKYD). Validated.")
