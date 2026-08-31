"""
`prices_macro` up-to-date check (src/data_extract/utils/prices/fetch_macro.py).

The sources publish the raw LEVEL series (market close, Treasury yields, cash, VIX) with a
~1 business-day lag. Keying "up to date" on the table's OVERALL max date let ANY one fast
series mask a still-stale level block, so the refresh that would fill the levels was skipped
(the observed "1-day gap": levels stuck a day behind what the source already had). Both old
wide fetchers hit this and patched it separately; the gate must key on the CORE LEVEL block
reaching the previous business day.

Long-format twist that did not exist before: a core series can now be entirely ABSENT (zero
rows) rather than present-but-NaN, because the melt drops NaN. That must read as "not fresh"
too, otherwise a series that never downloaded would never be retried.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from conftest import FakeStore     # the ONE shared store double
from src.constants.constants_price import MACRO_CORE_LEVEL_SERIES
from src.data_extract.utils.prices import fetch_macro as fm
from src.data_store.schema import Tables


def _ctx(df):
    return SimpleNamespace(store=FakeStore({Tables.prices_macro: df}))


def _long(core_last: pd.Timestamp, other_last: pd.Timestamp,
          drop_series: str | None = None) -> pd.DataFrame:
    """Long `prices_macro` rows where the core level series stop at `core_last` and the rest
    run to `other_last`. `drop_series` omits one core series entirely (zero rows)."""
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(prev - pd.Timedelta(days=16), prev)
    rows = []
    for name in MACRO_CORE_LEVEL_SERIES:
        if name == drop_series:
            continue
        rows += [{"date": d, "ticker": name, "close": 1.0} for d in dates if d <= core_last]
    for name in ("yield_curve_10y2y", "breakeven_10y", "bond_10y_tr"):
        rows += [{"date": d, "ticker": name, "close": 1.0} for d in dates if d <= other_last]
    return pd.DataFrame(rows, columns=["date", "ticker", "close"])


def test_up_to_date_keys_on_core_level_block():
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    prev2 = prev - pd.tseries.offsets.BDay(1)

    # derived/fast series current but the CORE LEVEL block a day behind -> NOT up to date, so
    # the refresh that fills the levels runs (the bug was: it skipped)
    assert fm._is_up_to_date(_ctx(_long(core_last=prev2, other_last=prev))) is False
    # core block reaches the previous business day -> up to date (skip; no perpetual re-pull)
    assert fm._is_up_to_date(_ctx(_long(core_last=prev, other_last=prev))) is True
    # a core series MISSING ENTIRELY must not read as fresh just because the others are
    assert fm._is_up_to_date(
        _ctx(_long(core_last=prev, other_last=prev, drop_series="vix"))) is False
    # empty / unseeded -> not up to date
    assert fm._is_up_to_date(
        _ctx(pd.DataFrame(columns=["date", "ticker", "close"]))) is False

    print("\n=== SANITY CHECK: prices_macro freshness keys on the core level block ===")
    print(f"  core series {list(MACRO_CORE_LEVEL_SERIES)}")
    print("  stale levels behind current derived series -> refresh (not skipped);")
    print("  core block at prev business day -> up to date;")
    print("  one core series with ZERO rows -> not fresh (long-format case). Validated.")


if __name__ == "__main__":
    test_up_to_date_keys_on_core_level_block()
