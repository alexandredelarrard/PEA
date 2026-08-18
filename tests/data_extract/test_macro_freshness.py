"""
macro up-to-date check (src/data_extract/utils/prices/fetch_macro.py).

FRED publishes the raw LEVEL series (Treasury yields, VIX) with a ~1 business-day
lag, but the COMPUTED spreads/breakeven (T10Y2Y/T10Y3M/T10YIE) publish SAME-DAY.
Keying "up to date" on the overall max date let a same-day spread mask a still-stale
level block, so the refresh that would fill the yields/VIX was skipped (the observed
"1-day gap": levels stuck a day behind what FRED already had). The check must key on
the LEVEL block reaching the previous business day.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from conftest import FakeStore     # the ONE shared store double
from src.data_extract.utils.fundamentals import fetch_macro as fm


def _ctx(df):
    return SimpleNamespace(store=FakeStore({"macro": df}))


def _macro_df(level_last: pd.Timestamp, spread_last: pd.Timestamp) -> pd.DataFrame:
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(prev - pd.Timedelta(days=16), prev)
    df = pd.DataFrame({"date": dates})
    for c in fm._CORE_LEVEL_SERIES:                                    # yields + vix
        df[c] = [1.0 if d <= level_last else float("nan") for d in dates]
    for c in ("yield_curve_10y2y", "yield_curve_10y3m", "breakeven_10y"):
        df[c] = [1.0 if d <= spread_last else float("nan") for d in dates]
    return df


def test_up_to_date_keys_on_level_block_not_fast_spreads():
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    prev2 = prev - pd.tseries.offsets.BDay(1)

    # spreads current (prev business day) but LEVEL series a day behind -> NOT up to
    # date, so the refresh that fills the yields/VIX runs (the bug was: it skipped)
    assert fm._macro_is_up_to_date(_ctx(_macro_df(level_last=prev2, spread_last=prev))) is False
    # level block reaches the previous business day -> up to date (skip; no perpetual pull)
    assert fm._macro_is_up_to_date(_ctx(_macro_df(level_last=prev, spread_last=prev))) is True
    # empty / unseeded -> not up to date
    assert fm._macro_is_up_to_date(_ctx(pd.DataFrame(columns=["date"]))) is False

    print("\n=== SANITY CHECK: macro freshness keys on the level block ===")
    print("  stale yields/VIX behind current spreads -> refresh (not skipped); level block "
          "at prev business day -> up to date. Fixes the level-series '1-day gap'. Validated.")


if __name__ == "__main__":
    test_up_to_date_keys_on_level_block_not_fast_spreads()
