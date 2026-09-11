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


#: The filing tables a company keyed by CIK must appear in if its CIK is the right one. Kept
#: short and cheap on purpose: this runs on every seed, and one hit is enough to prove the ID
#: resolves at EDGAR. `prices` is NOT in the list -- it keys on the TICKER via yfinance and so
#: is populated for a company whose CIK is wrong, which is exactly how XOM's error survived.
CIK_EVIDENCE_TABLES: tuple[str, ...] = ("def14a_llm", "sec_def14a", "sec_8k", "sec_13d")


def unverified_ciks(context: Context) -> list[dict]:
    """Universe rows whose CIK appears in NO filing table, with the evidence counted.

    ⚠ THIS CHECK EXISTS BECAUSE EXXON HAD THE WRONG COMPANY ID FOR AN UNKNOWN LENGTH OF TIME
    and nothing noticed. `sp500_tickers` carried CIK 0002115436 for XOM where Exxon Mobil's
    real CIK is 34088; that ID does resolve at EDGAR and does file proxies, but only from
    2023, so XOM contributed 4,092 rows to the governance cube with ZERO non-null values in
    any proxy-derived feature -- the largest energy name in the index with no board, pay or
    provision history at all. It took a feature-level audit thirty phases downstream to find
    it, because every stage in between was structurally healthy: `prices` keys on the TICKER,
    so XOM had 7,803 price rows and looked entirely present.

    A ticker with no rows in ANY filing table is not necessarily a bad CIK -- a genuine recent
    IPO or spin-off has none either, and measured on 2026-09-09 nine of the twelve such
    tickers (Kenvue, Solventum, Veralto, GE Vernova, GE HealthCare, Sandisk, FedEx Freight,
    Qnity, Honeywell Aerospace) have no `prices` rows either and are upstream of everything.
    So this REPORTS rather than raises, and the discriminator it reports alongside is the
    price-row count.

    ⚠ MEASURED LIMIT, STATED HERE SO NOBODY READS A CLEAN RUN AS A CLEAN UNIVERSE: **this
    check does NOT catch the Exxon defect that motivated it**, and the reason is worth
    knowing. CIK 0002115436 is not a dead ID -- it resolves at EDGAR and it files, carrying 4
    proxies and 179 8-Ks in this database. What is wrong with it is its DEPTH, not its
    existence, and depth cannot be judged from inside the database either: a filing history
    far shorter than the price history is the NORMAL shape for a holding-company
    reorganisation, and 15+ index members have it legitimately (DIS -> TWDC 2019, LIN -> Linde
    plc 2018, DD -> DowDuPont 2017, APA Corp 2021, Viatris 2020, Evergy 2018, BLK 2024).
    Nor does SEC's own ticker file settle it: `sec_utils.load_cik_mapping` records that
    `company_tickers.json` was dropped as a source precisely because it mismaps XOM to "a
    non-filing ExxonMobil Holdings Corp shell", and the CIK in this table comes from
    WIKIPEDIA's S&P 500 list, which carries the same value.

    So this function catches the ID-resolves-to-nothing case, which is cheap and worth
    catching, and the deep-history case is caught downstream by
    `reports/validate/governance/_scripts/04_coverage_by_ticker.py` -- a ticker with price
    history and zero non-null proxy features. That is the check that actually found XOM.
    """
    df = context.store.load(Tables.sp500_tickers, columns=["ticker", "cik"], optional=True)
    if df is None or df.empty or "cik" not in df.columns:
        return []

    # ONE `distinct` per table, not one count per (ticker, table): the universe is ~500 names
    # and five tables, and a per-cell count is 2,500 round trips on a check that runs at seed.
    def _tickers(table: str) -> set[str]:
        try:
            return {str(t).strip().upper() for t in context.store.distinct(table, "ticker") if t}
        except Exception:                           # noqa: BLE001 -- a table may not exist yet
            return set()

    seen = {table: _tickers(table) for table in CIK_EVIDENCE_TABLES}
    filed = set().union(*seen.values()) if seen else set()
    priced = _tickers("prices")

    out: list[dict] = []
    for raw_t, raw_c in zip(df["ticker"], df["cik"]):
        ticker = str(raw_t).strip().upper()
        if not ticker or ticker in filed:
            continue
        out.append({
            "ticker": ticker, "cik": str(raw_c),
            "in_prices": ticker in priced,
            "filing_tables_checked": list(CIK_EVIDENCE_TABLES),
            # a company that has traded for years with no filing row ANYWHERE is the XOM
            # shape; one absent from `prices` too is simply too new to have been fetched
            "shape": "SUSPECT CIK" if ticker in priced else "too new / not yet fetched",
        })
    return out
