"""Pure transform math of the unified macro fetcher (src/.../prices/fetch_macro.py):
short-gap mean-fill, the derived spreads, the untransformed price leg, and the wide->long melt.

Synthetic known-truth fixtures throughout -- this is parsing/algebra, not an economic claim,
so real data would only make the assertions weaker.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import (MACRO_ALL_SERIES, MACRO_FRED_SERIES,
                                     MACRO_PRICE_SERIES, MACRO_SPREAD_SERIES)
from src.data_extract.utils.prices import fetch_macro as fm
from src.data_extract.utils.prices.fetch_macro import derive_series, fill_short_gaps, to_long


def test_fill_short_gaps_mean_and_week_guard():
    """A short interior gap (< 1 week) is filled with the MEAN of the two bracketing days;
    a >= 1 week gap and leading/trailing NaNs are left untouched."""
    idx = pd.date_range("2024-01-01", "2024-01-20", freq="D")
    s = pd.Series(np.nan, index=idx)
    s["2024-01-01"] = 10.0            # valid
    # 01-02, 01-03 missing -> gap span 01-01..01-04 = 3 days (< 7) -> fill mean(10,16)=13
    s["2024-01-04"] = 16.0            # valid
    # 01-05 .. 01-14 missing -> gap span 01-04..01-15 = 11 days (>= 7) -> NOT filled
    s["2024-01-15"] = 20.0            # valid
    s["2024-01-16":] = np.nan         # trailing NaNs -> untouched
    df = pd.DataFrame({"x": s})

    out = fill_short_gaps(df, ["x"], max_gap_days=7)
    assert out.loc["2024-01-02", "x"] == pytest.approx(13.0)   # mean(10, 16)
    assert out.loc["2024-01-03", "x"] == pytest.approx(13.0)
    assert pd.isna(out.loc["2024-01-08", "x"])                 # long gap left NaN
    assert pd.isna(out.loc["2024-01-18", "x"])                 # trailing NaN untouched
    assert out.loc["2024-01-01", "x"] == 10.0 and out.loc["2024-01-15", "x"] == 20.0
    print("\n=== SANITY CHECK: short-gap mean fill ===")
    print("  2-day gap filled with mean(10,16)=13; 11-day gap + trailing NaNs untouched. Validated.")


def test_derived_spreads_are_exact_differences():
    """Both curve spreads are exact differences of the stored levels, and `yield_curve_10y3m`
    is built off `cash_rate` -- `yield_3m` (DGS3MO) was dropped as the second quote of the
    same 3-month bill, so nothing may still reference it."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    wide = pd.DataFrame({"yield_10y": [4.0, 4.1, 4.2, 4.3, 4.4],
                         "yield_2y": [3.5, 3.5, 3.6, 3.8, 4.5],
                         "cash_rate": [5.0, 5.0, 5.0, 4.9, 4.8]}, index=idx)

    out = derive_series(wide)
    pd.testing.assert_series_equal(out["yield_curve_10y2y"],
                                  wide["yield_10y"] - wide["yield_2y"],
                                  check_names=False)
    pd.testing.assert_series_equal(out["yield_curve_10y3m"],
                                   wide["yield_10y"] - wide["cash_rate"],
                                   check_names=False)
    # an inverted curve must come out NEGATIVE, not absolute -- the sign IS the signal
    assert out["yield_curve_10y3m"].iloc[0] == pytest.approx(-1.0)
    assert out["yield_curve_10y2y"].iloc[-1] == pytest.approx(-0.1)
    # bond TR index reconstructed off yield_10y, normalised to ~100 at the first valid obs
    assert out["bond_10y_tr"].dropna().iloc[0] == pytest.approx(100.0, rel=0.02)

    assert "yield_3m" not in MACRO_FRED_SERIES.values()
    assert "yield_3m" not in MACRO_ALL_SERIES
    assert MACRO_SPREAD_SERIES["yield_curve_10y3m"] == ("yield_10y", "cash_rate")
    print("\n=== SANITY CHECK: derived spreads ===")
    print(f"  10y2y == yield_10y - yield_2y; 10y3m == yield_10y - cash_rate (exact).")
    print(f"  inverted curve stays signed: 10y3m[0] = {out['yield_curve_10y3m'].iloc[0]:+.2f}")
    print("  yield_3m/DGS3MO absent from the registry (collapsed into cash_rate). Validated.")


def test_derive_skips_a_series_it_cannot_build():
    """A spread whose input is missing is SKIPPED, not emitted as an all-NaN series -- the
    long table must never carry a series with no data."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    out = derive_series(pd.DataFrame({"yield_10y": [4.0, 4.1, 4.2]}, index=idx))
    assert "yield_curve_10y2y" not in out.columns      # needs yield_2y
    assert "yield_curve_10y3m" not in out.columns      # needs cash_rate
    assert "bond_10y_tr" in out.columns                # yield_10y is present
    print("\n=== SANITY CHECK: derive skips unbuildable series ===")
    print("  missing yield_2y/cash_rate -> spreads omitted entirely, not all-NaN. Validated.")


def test_fx_comes_from_fred_already_usd_per_eur():
    """FX is a FRED LEVEL (`DEXUSEU`), not a Yahoo price. Two reasons, both load-bearing:
    DEXUSEU starts 1999-01 where Yahoo's `USDEUR=X` only starts 2003-12, and it is already
    quoted USD per EUR -- the convention every consumer uses -- so there is no reciprocal to
    invert. Yahoo's `XXXYYY=X` means "YYY per one XXX", i.e. EUR per USD, and silently
    storing that would flip a real long position in the long-book sleeve."""
    assert MACRO_FRED_SERIES["DEXUSEU"] == "fx_usdeur"
    assert "fx_usdeur" not in MACRO_PRICE_SERIES.values()
    assert not any(s.endswith("=X") for s in MACRO_PRICE_SERIES), "no FX pair on the price leg"
    print("\n=== SANITY CHECK: FX source + convention ===")
    print("  fx_usdeur <- FRED DEXUSEU (USD per EUR, from 1999-01), not Yahoo USDEUR=X")
    print("  (EUR per USD, from 2003-12). No inversion step to get wrong. Validated.")


def test_price_leg_stores_closes_untransformed(monkeypatch):
    """The price leg is a pure symbol->name relabel of `close`: no inversion, no rescaling.
    There WAS a reciprocal here while FX came from Yahoo; this pins that nothing re-grows one,
    which would be invisible in every downstream test (the series is still plausible)."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    raw = pd.concat([pd.DataFrame({"date": idx, "ticker": sym, "close": [c, c + 1.0, c + 2.0],
                                   "volume": 1.0})
                     for sym, c in zip(MACRO_PRICE_SERIES, [100.0, 200.0, 300.0, 400.0, 500.0])],
                    ignore_index=True)
    monkeypatch.setattr(fm, "download_ohlcv", lambda *a, **k: raw)

    ctx = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None,
                                              warning=lambda *a, **k: None))
    wide = fm._fetch_price_leg(ctx, idx[0], idx[-1])

    assert set(wide.columns) == set(MACRO_PRICE_SERIES.values())
    for sym, name in MACRO_PRICE_SERIES.items():
        expected = raw.loc[raw["ticker"] == sym, "close"].to_numpy()
        np.testing.assert_allclose(wide[name].to_numpy(), expected)
    assert "volume" not in wide.columns          # the "trim the volume" step
    print("\n=== SANITY CHECK: price leg is untransformed ===")
    print(f"  {len(MACRO_PRICE_SERIES)} symbols -> {sorted(wide.columns)}, closes identical to "
          f"the yfinance response, volume dropped. Validated.")


def test_to_long_drops_nan_and_is_one_row_per_series_date():
    """The melt is where the wide layout's NaN padding disappears: a series that starts late
    contributes NO rows before it starts, instead of a NaN block per date."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    wide = pd.DataFrame({"equity_tr": [100.0, 101.0, 102.0, 103.0],
                         "breakeven_10y": [np.nan, np.nan, 2.3, 2.4]}, index=idx)

    long = to_long(wide)
    assert list(long.columns) == ["date", "ticker", "close"]
    assert len(long) == 4 + 2                                    # 4 equity + 2 breakeven
    assert not long["close"].isna().any()
    assert not long.duplicated(subset=["date", "ticker"]).any()   # the (ticker, date) pk
    bk = long[long["ticker"] == "breakeven_10y"]
    assert bk["date"].min() == idx[2]                             # ragged start, no padding
    assert to_long(pd.DataFrame()).empty                          # empty-safe

    print("\n=== SANITY CHECK: wide -> long melt ===")
    print(f"  4 dates x 2 series with a late start -> {len(long)} rows, not 8; "
          f"0 NaN; breakeven starts {bk['date'].min().date()}. Validated.")


if __name__ == "__main__":
    test_fill_short_gaps_mean_and_week_guard()
    test_derived_spreads_are_exact_differences()
    test_derive_skips_a_series_it_cannot_build()
    test_fx_comes_from_fred_already_usd_per_eur()
    test_to_long_drops_nan_and_is_one_row_per_series_date()
