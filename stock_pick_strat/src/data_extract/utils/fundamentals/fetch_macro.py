"""
Fetch macro series from FRED (free, complete history, needs a free API key) and save
them to the `macro` DB table (PK: date). Series: Treasury yields (3M/2Y/10Y/30Y), the
10Y-2Y / 10Y-3M curve spreads, VIX, IG/HY credit spreads, 10Y breakeven inflation —
see SERIES below.

Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html
Put it in .env as FRED_API_KEY=...
"""
import pandas as pd
import os 
from fredapi import Fred

from src.context import Context

SERIES = {
    # Risk-free rate
    "DGS3MO": "yield_3m",
    # Yield curve
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",
    "DGS30": "yield_30y",
    "T10Y2Y": "yield_curve_10y2y",   # FRED computes this spread directly
    "T10Y3M": "yield_curve_10y3m",
    # Volatility
    "VIXCLS": "vix",
    # Credit spreads (bonus: useful risk-sentiment overlay)
    "BAMLC0A0CM": "ig_credit_spread",
    "BAMLH0A0HYM2": "hy_credit_spread",
    "T10YIE": "breakeven_10y",
}


def _macro_is_up_to_date(context: Context) -> bool:
    """True when the `macro` DB table already covers the last business day (the daily
    yield / VIX series set the max date). Skips redundant re-pulls on weekends /
    same-day re-runs."""
    existing = context.store.load("macro", columns=["date"])
    if existing.empty:
        return False
    max_date = pd.to_datetime(existing["date"]).max()
    last_expected = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    return max_date >= last_expected


def fetch_macro(context: Context):
    if not os.getenv("FRED_API_KEY"):
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it "
            "to your .env file."
        )

    if _macro_is_up_to_date(context):
        print("Macro data already up to date - skipping (DB table 'macro')")
        return

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    start = pd.Timestamp.today() - pd.DateOffset(years=context.config.data_extract.years_history + 1)

    frames = {}
    for series_id, name in SERIES.items():
        s = fred.get_series(series_id, observation_start=start)
        frames[name] = s

    macro = pd.DataFrame(frames)
    macro.index.name = "date"
    macro = macro.reset_index()

    context.store.save("macro", macro)
    print(f"Saved {len(macro)} rows of macro data to DB table 'macro'")
