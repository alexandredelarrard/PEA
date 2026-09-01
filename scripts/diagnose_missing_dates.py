"""
diagnose_missing_dates.py
-------------------------
Localize WHERE a date window disappears from the cube pipeline: raw equity prices -> market
trading calendar -> per-stock coverage -> macro/factor series -> final cube. Runs against the
real data store, no rebuild needed.

Reads through `context.store` like everything else. It used to read `paths["PRICES_PATH"]`,
`paths["MACRO_PATH"]` and `paths["CUBE_PATH"]` parquet files -- keys that no longer exist in
`context.paths`, so it raised KeyError before doing any work. Sections 4a (the `other_tickers`
columns inside `prices`) and 4b (the macro parquet) were separate then; they are ONE section
now, because both sets of series live in `prices_macro`.

Usage (from the repo root):
    python scripts/diagnose_missing_dates.py 2025-03-01 2025-04-30
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.constants.constants import MACRO_ALL_SERIES, MACRO_MARKET_SERIES
from src.context import get_config_context
from src.data_store.schema import Tables
from src.utils.macro import load_macro_wide


def _window(a, b):
    return pd.Timestamp(a), pd.Timestamp(b)


def main(lo: str, hi: str):
    lo, hi = _window(lo, hi)
    config, context = get_config_context("./configs", use_cache=False, save=False)
    store = context.store

    print(f"\n==== Diagnosing missing window {lo.date()} .. {hi.date()} "
          f"(market series={MACRO_MARKET_SERIES}) ====\n")

    # ---- 1. RAW EQUITY PRICES ------------------------------------------------
    print(f"1. RAW PRICES  {Tables.prices}")
    dmin, dmax = store.bounds(Tables.prices, "date")
    if dmin is None:
        print(f"   [FATAL] {Tables.prices} is empty -> run `data_extract price-history`.")
        return
    dmin, dmax = pd.Timestamp(dmin), pd.Timestamp(dmax)
    print(f"   overall date range: {dmin.date()} .. {dmax.date()}")
    if hi > dmax:
        print(f"   >>> window END is AFTER the last price date ({dmax.date()}). "
              f"This is a TAIL case: forward-return targets need ~max(horizon) trading days "
              f"of FUTURE prices, so the last months are trimmed. FIX = extend the price "
              f"history (re-run extraction).")

    # projected + bounded: `prices` is ~1.8M rows and only the window is needed
    # `close_split` is never null when `close_total` is, so it defines the widest grid --
    # which is what a coverage diagnostic wants.
    win = store.load(Tables.prices, columns=["date", "ticker", "close_split"],
                     since=lo, until=hi, optional=True)
    if win is None:
        print("   rows in window: 0  -> the whole window is absent from `prices`.")
        close = pd.DataFrame()
    else:
        win["date"] = pd.to_datetime(win["date"]).dt.normalize()
        print(f"   rows in window: {len(win):,}  distinct dates: {win['date'].nunique()}  "
              f"distinct tickers: {win['ticker'].nunique()}")
        close = win.pivot_table(index="date", columns="ticker", values="close_split",
                                aggfunc="last")

    # ---- 2. MARKET TRADING CALENDAR (the killer filter) ---------------------
    print(f"\n2. {MACRO_MARKET_SERIES} TRADING CALENDAR (StepCubePrices drops every date "
          f"where the market close is NaN)")
    macro = load_macro_wide(store)
    if macro is None:
        print(f"   >>> {Tables.prices_macro} is EMPTY -> the whole cube calendar is "
              f"undefined. FIX = run `data_extract macro`.")
        macro = pd.DataFrame(columns=["date"])
        mkt = None
    else:
        macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
        macro = macro.sort_values("date").set_index("date")
        mkt = macro[MACRO_MARKET_SERIES] if MACRO_MARKET_SERIES in macro.columns else None

    if mkt is None:
        print(f"   >>> {MACRO_MARKET_SERIES} ABSENT from {Tables.prices_macro} entirely -> "
              f"the whole cube calendar is undefined. FIX = run `data_extract macro`.")
    else:
        mw = mkt.loc[(mkt.index >= lo) & (mkt.index <= hi)]
        n_dates, n_valid = len(mw), int(mw.notna().sum())
        print(f"   {MACRO_MARKET_SERIES} rows present in window: {n_dates}  non-NaN: {n_valid}")
        if n_dates == 0:
            print(f"   >>> NO ROWS in this window -> ALL stocks dropped here. ROOT CAUSE = "
                  f"interior gap in the market series. This is the 'no stock' symptom.")
        elif n_valid < n_dates:
            print(f"   >>> present but {n_dates - n_valid} NaN closes -> those dates dropped "
                  f"for all stocks.")
        else:
            print(f"   calendar is intact here (not the cause).")

    # ---- 3. PER-STOCK COVERAGE in window ------------------------------------
    print(f"\n3. PER-STOCK CLOSE COVERAGE in window")
    if len(close):
        per_date_names = close.notna().sum(axis=1)
        print(f"   dates in window: {len(close)}; median stocks/day with a close: "
              f"{int(per_date_names.median())}")
    else:
        print(f"   no close rows in window at all.")

    # ---- 4. MACRO / FACTOR SERIES -------------------------------------------
    print(f"\n4. MACRO / FACTOR SERIES from {Tables.prices_macro} (a gap here NaNs the "
          f"target for EVERY stock over the affected forward window)")
    for name in MACRO_ALL_SERIES:
        if name not in macro.columns:
            print(f"   {name:<20} ABSENT from {Tables.prices_macro}")
            continue
        s = macro[name]
        sw = s.loc[(s.index >= lo) & (s.index <= hi)]
        print(f"   {name:<20} rows {len(sw):>3}  non-NaN {int(sw.notna().sum()):>3}  "
              f"(history {s.dropna().index.min().date() if s.notna().any() else '-'} ..)")

    # ---- 5. FINAL CUBE ------------------------------------------------------
    print(f"\n5. FINAL CUBE {Tables.cube}")
    cmin, cmax = store.bounds(Tables.cube, "date")
    if cmin is None:
        print("   cube not built yet.")
    else:
        cw = store.load(Tables.cube, columns=["date", "ticker", "target_horizon"],
                        since=lo, until=hi, optional=True)
        if cw is None:
            print("   cube rows in window: 0")
        else:
            cw["date"] = pd.to_datetime(cw["date"]).dt.normalize()
            print(f"   cube rows in window: {len(cw):,}  dates: {cw['date'].nunique()}  "
                  f"tickers: {cw['ticker'].nunique()}")
            by_h = cw.groupby("target_horizon")["date"].nunique()
            print(f"   distinct dates per horizon in window:\n{by_h.to_string()}")
        print(f"   cube overall range: {pd.Timestamp(cmin).date()} .. "
              f"{pd.Timestamp(cmax).date()}")

    print("\n==== VERDICT GUIDE ====")
    print("  * Step 1 says window is after last price date -> TAIL trim: extend prices.")
    print(f"  * Step 2 shows {MACRO_MARKET_SERIES} missing/NaN in window -> EXTRACT bug "
          f"(interior gap in the market series); re-run `data_extract macro`.")
    print("  * Step 2 intact but Step 4 macro/factor gap -> EXTRACT bug in a shared factor; "
          "target NaN'd for all stocks over its forward window.")
    print("  * Steps 1-4 all intact but cube empty -> CUBE PREP bug (targets/merge).")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "2025-03-01"
    b = sys.argv[2] if len(sys.argv) > 2 else "2025-04-30"
    main(a, b)
