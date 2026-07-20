"""
universe.py  (src/utils/universe.py)
------------------------------------
THE single entry point for the analysis universe. Every step (extract, peers,
cube, modelling, backtest) resolves which tickers to analyse from ONE place: the
`sp500_tickers` DB table. Populate that table with whatever set you want — the
S&P 500 (the default seeder `fetch_prices.get_sp500_tickers` scrapes it), the
Russell 1000, or a hand-picked list — and the whole flow follows automatically:

    extraction  -> seeds the table if empty, then FETCHES only these names (+ the
                   benchmark/macro `other_tickers`),
    peers       -> builds baskets ONLY among these names,
    cube        -> builds features ONLY for these names,
    modelling/backtest -> inherit the universe through the cube.

To switch universe you change ONLY what fills `sp500_tickers` (e.g. `store.replace(
"sp500_tickers", russell_1000_df)`); no step code changes. The table name is kept
as `sp500_tickers` for backward-compat regardless of which index actually fills it.
"""
from __future__ import annotations

from src.context import Context

UNIVERSE_TABLE = "sp500_tickers"


def load_universe_tickers(context: Context) -> list[str]:
    """The analysis universe: sorted, de-duplicated, upper-cased tickers from the
    `sp500_tickers` table. Returns [] when the table is absent or empty (not yet
    seeded) so callers can fall back / warn rather than crash."""
    df = context.store.load(UNIVERSE_TABLE, columns=["ticker"])
    if df is None or df.empty or "ticker" not in df.columns:
        return []
    return sorted({str(t).strip().upper() for t in df["ticker"].dropna() if str(t).strip()})
