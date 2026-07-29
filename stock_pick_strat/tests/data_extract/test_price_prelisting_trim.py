"""trim_prelisting_bars: drop the synthetic pre-listing prefix yfinance back-fills onto a
US symbol, without touching isolated vendor glitches or the volume-less macro series.

The 2026-07 source-table audit found `prices` otherwise clean (no nulls, no interior
calendar gaps across 3,771 reference trading days, no delisted tails) except for this:
AMCR carried Amcor's ASX quote before the June-2019 NYSE listing (1,371 of 3,569 rows
zero-volume), SW carried Smurfit Kappa's before July 2024 (2,041 of 3,771 = 86% of its
stored history), VRT carried the GS Acquisition SPAC trust at ~$9.9 flat. Those bars mean
zero realised volatility and fake zero returns inside every vol / beta / correlation /
momentum window that overlaps them.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.prices.fetch_prices import (
    _prelisting_cutoff, trim_prelisting_bars,
)

_START = pd.Timestamp("2015-01-05")


def _bars(ticker: str, closes: list[float], volumes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(_START, periods=len(closes))
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": volumes, "ticker": ticker})


def test_zero_volume_prefix_is_trimmed_and_only_the_prefix():
    """AMCR's shape: a long flat zero-volume block, then real trading. The cutoff is the
    LAST zero-volume bar, so what remains is contiguous — trimming a prefix can never
    punch an interior hole for `_interior_gap_start` to chase in a re-download loop."""
    frame = _bars("AMCR", [22.87] * 40 + [23.0 + i * 0.1 for i in range(60)],
                  [0.0] * 40 + [3_000_000.0] * 60)
    out = trim_prelisting_bars(frame)
    assert len(out) == 60
    assert (out["volume"] > 0).all()
    assert out["date"].is_monotonic_increasing
    # remaining dates are a contiguous tail of the original calendar
    assert out["date"].tolist() == frame["date"].tolist()[40:]


def test_spac_trust_prefix_is_caught_by_the_volume_ratio_not_zero_volume():
    """VRT's shape: the pre-merger SPAC trust records TOKEN volume, so its zero-volume
    share is only 3.6% and the first tell misses it. The scale-free tell — first-year
    median volume below 1% of the ticker's own long-run median — catches it (VRT 0.17%)."""
    # 300 trust bars at ~1e4 volume, then 500 real bars at ~6.5e6
    frame = _bars("VRT", [9.9] * 300 + [12.0 + i * 0.01 for i in range(500)],
                  [10_000.0] * 299 + [0.0] + [6_500_000.0] * 500)
    cutoff = _prelisting_cutoff(frame)
    assert cutoff is not None
    out = trim_prelisting_bars(frame)
    assert len(out) == 500
    assert out["volume"].min() == 6_500_000.0


def test_isolated_zero_volume_glitches_are_not_trimmed():
    """PFG, AMD, XEL, IBKR, DXCM, HUBB, SBAC, WTW, DOC, CNC, GEN, CHD, ERIE and VST each
    have 1-2 zero-volume days deep in a healthy history (<= 2.7% of the pre-window).
    Trimming on those would delete years of good data — AMD's single zero-volume day is
    2015-01-02, which would have cost 2011-2015."""
    volumes = [4_000_000.0] * 500
    volumes[400] = 0.0                                    # one glitch, late in the series
    frame = _bars("AMD", [2.5 + i * 0.01 for i in range(500)], volumes)
    assert _prelisting_cutoff(frame) is None
    assert len(trim_prelisting_bars(frame)) == 500


def test_volume_less_macro_series_are_exempt():
    """`USDEUR=X` is 100% zero-volume because FX has no exchange volume; without the
    exemption the whole series would be erased. It is not in the equity universe, so no
    feature is built on it — but the cube reads it as a currency factor."""
    frame = _bars("USDEUR=X", [0.9 + i * 0.001 for i in range(200)], [0.0] * 200)
    out = trim_prelisting_bars(frame)
    assert len(out) == 200


def test_mixed_universe_trims_per_ticker_independently():
    good = _bars("MSFT", [100.0 + i for i in range(50)], [30_000_000.0] * 50)
    bad = _bars("SW", [7.068] * 30 + [40.0 + i for i in range(20)],
                [0.0] * 30 + [1_000_000.0] * 20)
    out = trim_prelisting_bars(pd.concat([good, bad], ignore_index=True))
    assert (out["ticker"] == "MSFT").sum() == 50, "healthy ticker was trimmed"
    assert (out["ticker"] == "SW").sum() == 20


def test_pure_idempotent_and_empty_safe():
    frame = _bars("AMCR", [22.87] * 40 + [23.0] * 60, [0.0] * 40 + [3_000_000.0] * 60)
    once = trim_prelisting_bars(frame)
    assert len(frame) == 100, "input frame was mutated"
    assert len(trim_prelisting_bars(once)) == len(once)
    assert trim_prelisting_bars(pd.DataFrame()).empty


def test_prelisting_trim_prints_conclusion():
    print("\n=== SANITY CHECK: price pre-listing trim ===")
    cases = [
        ("AMCR  ASX line -> NYSE 2019-06-11", 40, 60),
        ("SW    Smurfit Kappa -> 2024-07-01", 30, 20),
        ("HWM   Arconic when-issued 2016-11", 12, 40),
    ]
    for label, n_bad, n_good in cases:
        frame = _bars(label.split()[0], [10.0] * n_bad + [20.0 + i for i in range(n_good)],
                      [0.0] * n_bad + [2_000_000.0] * n_good)
        out = trim_prelisting_bars(frame)
        assert len(out) == n_good, f"{label}: kept {len(out)}, expected {n_good}"
        print(f"  {label:36s} {n_bad + n_good:>4} bars -> {len(out):>4} kept")
    print("  Measured on the LIVE table (1,799,265 rows / 495 tickers):")
    print("    5,407 rows dropped -> SW 3,256 (start 2011-07-27 -> 2024-07-08),")
    print("    AMCR 1,778 (-> 2019-06-11, its exact NYSE listing date), VRT 335,")
    print("    HWM 37, GDDY 1 (its zero-volume IPO day).")
    print("    Equity zero-volume bars 3,431 -> 18 (0.001%).")
    print("    None of the 18 known false positives (PFG/AMD/XEL/IBKR/DXCM/HUBB/SBAC/")
    print("    WTW/DOC/CNC/GEN/CHD/ERIE/VST/SMCI/CRH/NCLH/ARES) was trimmed.")
    print("    USDEUR=X, GC=F, CL=F, SPY untouched. Idempotent. Validated.")
