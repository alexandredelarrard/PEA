"""
Fetch macro series from FRED (free, complete history, needs a free API key):
  - CPIAUCSL: CPI, all urban consumers (inflation)
  - WALCL: Fed total assets (balance sheet size)

Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html
Put it in .env as FRED_API_KEY=...

Run:
    python -m data.fetch_macro
"""
import pandas as pd
import os 
from fredapi import Fred

from src.context import Context

SERIES = {
    # Risk-free rate
    "DGS3MO": "yield_3m",
    "TB3MS": "tbill_3m_secondary",  # alt monthly 3m series, useful if DGS3MO has gaps
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
    """True when the cached macro file already covers the last business day
    (the daily yield/VIX series set the max date). Skips redundant re-pulls
    on weekends / same-day re-runs."""
    path = context.paths["MACRO_PATH"]
    if not path.exists():
        return False
    existing = pd.read_parquet(path, columns=["date"])
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
        print(f"Macro data already up to date — skipping {context.paths['MACRO_PATH']}")
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

    macro["cpi_yoy_pct"] = macro["cpi"].pct_change(periods=12) * 100

    macro.to_parquet(context.paths["MACRO_PATH"], index=False)
    print(f"Saved {len(macro)} rows of macro data to {context.paths["MACRO_PATH"]}")
