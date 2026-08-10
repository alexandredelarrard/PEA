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

from src.data_store.schema import Tables
from src.context import Context
from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS


def load_universe_tickers(context: Context) -> list[str]:
    """The analysis universe: sorted, de-duplicated, upper-cased tickers from the
    `sp500_tickers` table, EXCLUDING names with insufficient history
    (`INSUFFICIENT_HISTORY_TICKERS` -- recent IPOs / spin-offs with < 4 years of data
    that can't support the multi-year look-backs / walk-forward backtest). Returns []
    when the table is absent or empty (not yet seeded) so callers can fall back / warn."""
    # `optional=True` is what makes the documented "[] when not yet seeded" true: a plain `load`
    # raises on an absent/empty table, which on a cold DB would abort the seeding run itself.
    df = context.store.load(Tables.sp500_tickers, columns=["ticker"], optional=True)
    if df is None or "ticker" not in df.columns:
        return []
    return sorted({t for raw in df["ticker"].dropna()
                   if (t := str(raw).strip().upper()) and t not in INSUFFICIENT_HISTORY_TICKERS})
