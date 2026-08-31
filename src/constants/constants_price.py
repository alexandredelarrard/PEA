
# --------------------------------------------------------------------------- #
# MACRO / MARKET series registry -- everything in `prices_macro`               #
# --------------------------------------------------------------------------- #
# ONE registry for the single long table (date, ticker, close). It replaced two wide
# tables (`macro`, FRED features on a 16y window; `macro_asset_prices`, allocation legs
# on 31y) that double-stored yield_10y and vix from two source paths at two depths, plus
# the non-equity rows that used to sit in `prices`. The invariant is now: every series
# exists exactly ONCE, from exactly one source. Breaking it is what made "which gold is
# this?" a real question.
#
# yfinance symbol -> series name. CLOSE only: `fetch_macro` calls `download_ohlcv` and
# drops OHLV+volume, so nothing here ever reaches the `prices` table (which is the equity
# universe and nothing else). Auto-adjusted, so each price leg is a total-return proxy.
#   SPY   = S&P 500 total return (since 1993)     ^VIX = CBOE VIX (since 1990)
#   CL=F  = WTI front future                      GC=F = COMEX gold front future (2000)
#   XLE   = Energy Select SPDR (1998), the "commodity via ENERGY EQUITIES" leg (no futures):
#           the rate/inflation-shock diversifier that was +~60% in the 2022 selloff
MACRO_PRICE_SERIES = {
    "SPY": "equity_tr",
    "^VIX": "vix",
    "CL=F": "oil",
    "GC=F": "gold",
    "XLE": "energy",
}

# FRED series id -> series name. LEVELS ONLY; the spreads below are derived from these.
# No DGS3MO: it is the coupon-equivalent quote of the same 3-month bill as DTB3, so the
# pipeline was fetching one instrument twice. cash_rate (DTB3) is the survivor -- it has a
# real consumer (allocation.py's cash leg) and drives the freshness gate.
# FRED no longer serves a broad daily S&P (SP500 is license-truncated to ~10y) or ANY gold
# series (the London fixes were removed ~2025), which is why those legs are yfinance above.
MACRO_FRED_SERIES = {
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",           # -> bond_10y_tr
    "DGS30": "yield_30y",
    "DTB3": "cash_rate",            # 3-month T-bill secondary market rate (cash leg)
    "BAA10Y": "baa_credit_spread",  # Moody's Baa over 10Y: one consistently-defined series
    "T10YIE": "breakeven_10y",      # 10Y breakeven inflation (since 2003)
    # FX from FRED, not Yahoo's USDEUR=X: DEXUSEU starts 1999-01 (the euro's own first
    # quote) where Yahoo starts 2003-12, and it is already quoted USD per EUR -- the
    # convention every consumer uses -- so no reciprocal to invert on ingest. Yahoo also
    # carried stale 2008 bars (2008-12-08 read 1.49 against a real 1.29).
    # COST: DEXUSEU rides the WEEKLY H.10 release, so FX trails the calendar by up to a
    # week where Yahoo was same-day. Hence its absence from MACRO_CORE_LEVEL_SERIES -- it
    # must not hold the freshness gate open -- and the newest ~3 trading days carry no FX.
    # Consumers see NaN there (never a stale ffill), which is the safe direction.
    "DEXUSEU": "fx_usdeur",
}
# Derived spread -> (minuend, subtrahend). FRED's own T10Y2Y IS DGS10-DGS2, so deriving it
# is numerically identical; deriving BOTH is what makes every FRED leg a same-cadence level
# and retires the freshness bug where a same-day-publishing computed spread marked the table
# current while the 1-BDay-lagged level block was still stale.
# CAVEAT: yield_curve_10y3m now differs from FRED's T10Y3M by the ~5bp discount-vs-coupon
# basis between DTB3 and DGS3MO. Free here -- the series swings hundreds of bp and has no
# consumer; if it gains one it will be as a CHANGE, where a constant offset differences out.
MACRO_SPREAD_SERIES = {
    "yield_curve_10y2y": ("yield_10y", "yield_2y"),
    "yield_curve_10y3m": ("yield_10y", "cash_rate"),
}
# Reconstructed 10Y total-return index + its maturity assumption (build_bond_total_return).
MACRO_BOND_TR_SERIES = "bond_10y_tr"
MACRO_BOND_MATURITY_YEARS = 10
# CORE daily level series the freshness gate keys on (all lag ~1 business day). Judging
# freshness on the overall max let a fast series mask a stale level block.
MACRO_CORE_LEVEL_SERIES = ("equity_tr", "yield_10y", "cash_rate", "vix")

# The market series: the cube's beta/epsilon reference and every sleeve's benchmark. Named,
# not configured -- it identifies a row in `prices_macro`, not a tunable.
MACRO_MARKET_SERIES = "equity_tr"
# Cube factor-panel column -> prices_macro series. The panel KEYS are preserved from when
# these came out of `prices` via cube_part_market, so no beta/feature name changes
# downstream; only USD/EUR actually remaps. `energy` is deliberately absent: wiring it in
# would add a factor and a beta column, which is a modelling decision, not a refactor.
MACRO_CUBE_FACTORS = {"oil": "oil", "gold": "gold", "USD/EUR": "fx_usdeur"}

# Macro level -> daily-change factor name. ONLY daily-moving series belong here.
# NOTE: cpi_yoy_pct (monthly) and fed_balance_sheet (weekly) are deliberately
# EXCLUDED -- their daily change is ~always zero. Inflation risk is captured by
# the daily breakeven instead.
DAILY_MACRO_LEVELS = {
    "yield_10y": "d_yield_10y",
    "yield_curve_10y2y": "d_yield_curve",
    "vix": "d_vix",
    "breakeven_10y": "d_breakeven_10y",
    "baa_credit_spread": "d_baa_credit_spread",
}

# Every series name written to `prices_macro`, derived from the registries above so there is
# no second list to drift. Used by the freshness gate, the tests and the sanity prints.
MACRO_ALL_SERIES = (tuple(MACRO_PRICE_SERIES.values()) + tuple(MACRO_FRED_SERIES.values())
                    + tuple(MACRO_SPREAD_SERIES) + (MACRO_BOND_TR_SERIES,))