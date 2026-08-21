"""
universe.py  (src/utils/universe.py)
------------------------------------
THE single entry point for the analysis universe. Every step (extract, peers,
cube, modelling, backtest) resolves which tickers to analyse from ONE place: the
`sp500_tickers` DB table.
"""

from __future__ import annotations

from src.data_store.schema import Tables
from src.context import Context
from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS

def load_universe_tickers(context: Context) -> list[str]:
    """The analysis universe: sorted, de-duplicated, upper-cased tickers from the
    `sp500_tickers` table, EXCLUDING names with insufficient history and the
    redundant share classes listed in `data_extract.redundant_ticks`."""
    # Guard BEFORE touching the frame: `optional=True` returns None on a cold/empty table,
    # which is the seeding run's own first call -- it must get [] back, not an AttributeError.
    df = context.store.load(Tables.sp500_tickers, columns=["ticker"], optional=True)
    if df is None or "ticker" not in df.columns:
        return []
    # Exclude after normalising, so a lower-cased "goog" is dropped like "GOOG".
    excluded = INSUFFICIENT_HISTORY_TICKERS | {
        str(t).strip().upper() for t in context.config.data_extract.redundant_ticks}
    return sorted({t for raw in df["ticker"].dropna()
                   if (t := str(raw).strip().upper()) and t not in excluded})
