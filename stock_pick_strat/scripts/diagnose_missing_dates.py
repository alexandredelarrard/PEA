"""
diagnose_missing_dates.py
-------------------------
Localize WHERE a date window disappears from the cube pipeline: raw prices ->
SPY trading calendar -> per-stock coverage -> factor/macro panel -> targets ->
final cube. Run against the real data store (no rebuild needed).

Usage (from stock_pick_strat/):
    python scripts/diagnose_missing_dates.py 2025-03-01 2025-04-30
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.context import get_config_context
from src.data_aggregate.utils.common import data_utils as du


def _window(a, b):
    return pd.Timestamp(a), pd.Timestamp(b)


def main(lo: str, hi: str):
    lo, hi = _window(lo, hi)
    config, context = get_config_context("./configs", use_cache=False, save=False)
    mkt = config.build_cube.market_ticker
    paths = context.paths

    print(f"\n==== Diagnosing missing window {lo.date()} .. {hi.date()} "
          f"(market_ticker={mkt}) ====\n")

    # ---- 1. RAW PRICES ---------------------------------------------------
    prices_path = paths["PRICES_PATH"]
    if not prices_path.exists():
        print(f"[FATAL] prices file not found: {prices_path}")
        return
    pl = pd.read_parquet(prices_path)
    pl["date"] = pd.to_datetime(pl["date"]).dt.normalize()
    dmin, dmax = pl["date"].min(), pl["date"].max()
    print(f"1. RAW PRICES  {prices_path}")
    print(f"   overall date range: {dmin.date()} .. {dmax.date()}  "
          f"({pl['ticker'].nunique()} tickers, {len(pl):,} rows)")
    if hi > dmax:
        print(f"   >>> window END is AFTER the last price date ({dmax.date()}). "
              f"This is a TAIL case: forward-return targets need ~max(horizon) "
              f"trading days of FUTURE prices, so the last months are trimmed. "
              f"FIX = extend the price history (re-run extraction).")
    win = pl[(pl["date"] >= lo) & (pl["date"] <= hi)]
    print(f"   rows in window: {len(win):,}  distinct dates: {win['date'].nunique()}  "
          f"distinct tickers: {win['ticker'].nunique()}")

    # ---- 2. SPY TRADING CALENDAR (the killer filter) --------------------
    raw = du.prices_long_to_multiindex(pl)
    close = du.extract_field(raw, "Close")
    spy = close[mkt] if mkt in close.columns else None
    print(f"\n2. {mkt} TRADING CALENDAR (normalize_prices drops every date where "
          f"{mkt} close is NaN)")
    if spy is None:
        print(f"   >>> {mkt} column ABSENT from prices entirely -> the WHOLE cube "
              f"calendar is undefined. FIX = ensure {mkt} is downloaded.")
    else:
        spy_win = spy.loc[(spy.index >= lo) & (spy.index <= hi)]
        n_dates = len(spy_win)
        n_valid = int(spy_win.notna().sum())
        print(f"   {mkt} rows present in window: {n_dates}  non-NaN: {n_valid}")
        if n_dates == 0:
            print(f"   >>> {mkt} has NO ROWS in this window -> ALL stocks dropped "
                  f"here. ROOT CAUSE = interior gap in {mkt} price history "
                  f"(extraction). This is the 'no stock' symptom.")
        elif n_valid < n_dates:
            print(f"   >>> {mkt} present but {n_dates - n_valid} NaN closes -> those "
                  f"dates dropped for all stocks.")
        else:
            print(f"   {mkt} calendar is intact here (not the cause).")

    # ---- 3. PER-STOCK COVERAGE in window --------------------------------
    print(f"\n3. PER-STOCK CLOSE COVERAGE in window")
    in_win = close.loc[(close.index >= lo) & (close.index <= hi)]
    if len(in_win):
        per_date_names = in_win.notna().sum(axis=1)
        print(f"   dates in window: {len(in_win)}; median stocks/day with a close: "
              f"{int(per_date_names.median())}")
    else:
        print(f"   no close rows in window at all (calendar dropped them).")

    # ---- 4. MACRO / FACTOR TICKERS --------------------------------------
    others = list(config.data_extract.get("other_tickers", []))
    print(f"\n4. SHARED FACTOR TICKERS {others} (a gap here NaNs the target for "
          f"EVERY stock over the affected forward window)")
    for tk in others:
        if tk not in close.columns:
            print(f"   {tk:<10} ABSENT from prices")
            continue
        s = close[tk]
        sw = s.loc[(s.index >= lo) & (s.index <= hi)]
        print(f"   {tk:<10} rows {len(sw):>3}  non-NaN {int(sw.notna().sum()):>3}")

    macro_path = paths["MACRO_PATH"]
    if macro_path.exists():
        m = pd.read_parquet(macro_path)
        dcol = "date" if "date" in m.columns else m.columns[0]
        m[dcol] = pd.to_datetime(m[dcol]).dt.normalize()
        mw = m[(m[dcol] >= lo) & (m[dcol] <= hi)]
        print(f"   MACRO file rows in window: {len(mw)} "
              f"(range {m[dcol].min().date()}..{m[dcol].max().date()})")

    # ---- 5. FINAL CUBE ---------------------------------------------------
    cube_path = paths["CUBE_PATH"]
    print(f"\n5. FINAL CUBE {cube_path}")
    if not cube_path.exists():
        print("   cube not built yet.")
    else:
        cube = pd.read_parquet(cube_path, columns=["date", "ticker", "target_horizon"])
        cube["date"] = pd.to_datetime(cube["date"]).dt.normalize()
        cw = cube[(cube["date"] >= lo) & (cube["date"] <= hi)]
        print(f"   cube rows in window: {len(cw):,}  dates: {cw['date'].nunique()}  "
              f"tickers: {cw['ticker'].nunique()}")
        by_h = cw.groupby("target_horizon")["date"].nunique()
        print(f"   distinct dates per horizon in window:\n{by_h.to_string()}")
        print(f"   cube overall range: {cube['date'].min().date()} .. "
              f"{cube['date'].max().date()}")

    print("\n==== VERDICT GUIDE ====")
    print("  * Step 1 says window is after last price date -> TAIL trim: extend prices.")
    print(f"  * Step 2 shows {mkt} missing/NaN in window -> EXTRACT bug (interior gap "
          f"in benchmark); heals with the interior-gap backfill fix.")
    print("  * Step 2 intact but Step 4 factor/macro gap -> EXTRACT bug in a shared "
          "factor; target NaN'd for all stocks over its forward window.")
    print("  * Steps 1-4 all intact but cube empty -> CUBE PREP bug (targets/merge).")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "2025-03-01"
    b = sys.argv[2] if len(sys.argv) > 2 else "2025-04-30"
    main(a, b)
