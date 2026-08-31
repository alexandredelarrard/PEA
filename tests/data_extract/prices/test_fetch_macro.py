"""
The unified macro / market fetcher (src/data_extract/utils/prices/fetch_macro.py).

Two things worth testing offline:
  1. build_bond_total_return -- the only non-trivial transform: a constant-maturity bond
     TOTAL-RETURN index reconstructed from the par-yield (carry + duration*dyield). Flat
     yields must accrue carry (index rises); a yield SPIKE must book a capital loss.
  2. the window is an ARGUMENT, not a config read -- the fetcher must be callable without a
     config object at all, which is what lets StepExtractPrices own both windows in one place.

Plus an OPT-IN live pull (skipped without FRED_API_KEY, and it hits yfinance too) that builds
the real long frame and prints per-series ranges + history starts, so a DATA problem is
distinguishable from a code problem. The per-series bands are unit checks: they catch a
mislabelled source (a reciprocal FX quote, a rate served as a fraction rather than a percent)
that every downstream test would happily consume.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from src.constants.constants_price import (MACRO_ALL_SERIES, MACRO_BOND_TR_SERIES,
                                     MACRO_CORE_LEVEL_SERIES, MACRO_MARKET_SERIES)
from src.data_extract.utils.prices import fetch_macro as fm


# --------------------------------------------------------------------------- #
# 1. bond total-return reconstruction                                          #
# --------------------------------------------------------------------------- #
def test_build_bond_total_return_carry_and_duration():
    idx = pd.bdate_range("2020-01-01", periods=6)

    # (a) FLAT 4% yield -> no price move, pure carry -> strictly increasing index
    flat = pd.Series(4.0, index=idx)
    tr_flat = fm.build_bond_total_return(flat)
    ret = tr_flat.dropna().pct_change().dropna()
    assert (ret > 0).all(), "flat yield should accrue positive carry every day"
    daily_carry = 0.04 / 252
    assert ret.iloc[0] == pytest.approx(daily_carry, rel=1e-6)

    # (b) a one-day yield SPIKE (+1.0%: 4->5) must book a capital LOSS ~ -duration*dy
    spike = pd.Series(4.0, index=idx)
    spike.iloc[3] = 5.0                              # +100bp on day 3
    tr_spike = fm.build_bond_total_return(spike)
    r = tr_spike.dropna().pct_change()
    loss_day = r.loc[idx[3]]
    assert loss_day < 0, "a yield jump must produce a negative (capital-loss) return"
    # ~ -D*dy with D ~ 8 for a 10y 4% par bond -> roughly -8% (carry is negligible)
    assert -0.10 < loss_day < -0.05, f"loss {loss_day:.4f} not near -duration*dyield"

    print("\n=== SANITY CHECK: bond total-return reconstruction ===")
    print(f"  flat 4% yield -> daily carry {ret.iloc[0]*1e4:.3f}bp, index strictly rising.")
    print(f"  +100bp yield spike -> one-day return {loss_day*100:.2f}% "
          f"(capital loss ~ -duration*dyield). Validated.")


# --------------------------------------------------------------------------- #
# 2. the window is passed in, never read from config                           #
# --------------------------------------------------------------------------- #
def test_years_history_is_an_argument_not_a_config_read(monkeypatch):
    """`fetch_macro` must take its window as a parameter. A context with NO `config`
    attribute at all is the strongest form of that assertion: if the fetcher reached into
    `context.config.data_extract.*` (as both predecessors did) this raises AttributeError."""
    seen: dict[str, object] = {}

    def _fake_price_leg(context, since, until):
        seen["span_years"] = round((until - since).days / 365.25)
        idx = pd.bdate_range(since, until, freq="B")[:5]
        return pd.DataFrame({MACRO_MARKET_SERIES: range(1, len(idx) + 1)}, index=idx)

    def _fake_fred_leg(since):
        idx = pd.bdate_range(since, since + pd.Timedelta(days=10), freq="B")[:5]
        return pd.DataFrame({c: 4.0 for c in MACRO_CORE_LEVEL_SERIES
                             if c != MACRO_MARKET_SERIES}, index=idx)

    monkeypatch.setattr(fm, "_fetch_price_leg", _fake_price_leg)
    monkeypatch.setattr(fm, "_fetch_fred_leg", _fake_fred_leg)

    ctx = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None,
                                              warning=lambda *a, **k: None))
    long = fm.build_macro_frame(ctx, years_history=31)

    assert seen["span_years"] == 31, "the argument must drive the download window"
    assert list(long.columns) == ["date", "ticker", "close"]
    assert not hasattr(ctx, "config"), "test is only meaningful without a config object"

    print("\n=== SANITY CHECK: macro window is an argument ===")
    print(f"  build_macro_frame(years_history=31) spanned {seen['span_years']}y and ran on a "
          f"context with NO config attribute.")
    print("  Both windows therefore live in StepExtractPrices.run, not inside the fetcher. "
          "Validated.")


# --------------------------------------------------------------------------- #
# 3. opt-in live pull (FRED + yfinance)                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("FRED_API_KEY"),
                    reason="needs FRED_API_KEY (and network) for the live pull")
def test_real_pull_ranges_and_fx_convention():
    ctx = SimpleNamespace(log=SimpleNamespace(info=lambda *a, **k: None,
                                              warning=lambda *a, **k: None))
    long = fm.build_macro_frame(ctx, years_history=3)

    assert not long.empty and list(long.columns) == ["date", "ticker", "close"]
    assert not long["close"].isna().any(), "the melt must drop NaN, not store it"
    assert not long.duplicated(subset=["date", "ticker"]).any(), "(ticker, date) is the pk"

    w = long.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")

    def band(name, lo, hi):
        s = w[name].dropna() if name in w.columns else pd.Series(dtype=float)
        assert not s.empty, f"{name} came back empty"
        assert s.between(lo, hi).all(), (f"{name} outside [{lo}, {hi}]: "
                                         f"{s.min():.3f} .. {s.max():.3f}")

    band(MACRO_MARKET_SERIES, 1.0, 1e5)          # an index level, just strictly positive
    band("gold", 200, 6000)                      # USD/oz
    band("cash_rate", -1, 25)                    # annual %
    band("vix", 5, 90)                           # index points
    band(MACRO_BOND_TR_SERIES, 1.0, 1e4)         # reconstructed index, base 100
    # DEXUSEU is USD per EUR; its reciprocal (Yahoo's USDEUR=X) would land ~0.6-1.0
    band("fx_usdeur", 1.0, 1.6)

    missing = [s for s in MACRO_ALL_SERIES if s not in w.columns]

    print("\n=== SANITY CHECK: live prices_macro pull ===")
    print(f"  {len(long):,} rows / {w.shape[1]} of {len(MACRO_ALL_SERIES)} registry series")
    for c in sorted(w.columns):
        s = w[c].dropna()
        print(f"  {c:<20} {s.iloc[0]:>10.3f} .. {s.iloc[-1]:>10.3f}   n={len(s):<5} "
              f"from {s.index.min().date()}")
    print(f"  missing: {missing or 'none'}")
    print("  fx_usdeur inside (1.0, 1.6) -> USD per EUR, FRED DEXUSEU's native convention "
          "and the sleeve's. Validated.")


if __name__ == "__main__":
    test_build_bond_total_return_carry_and_duration()
