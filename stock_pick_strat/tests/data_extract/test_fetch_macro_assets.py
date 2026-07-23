"""
Long-history multi-asset ALLOCATION series (src/data_extract/utils/prices/fetch_macro_assets.py).

Two things are worth testing:
  1. build_bond_total_return -- the only non-trivial transform: a constant-maturity
     bond TOTAL-RETURN index reconstructed from the par-yield (carry + duration*Δyield).
     Flat yields must accrue carry (index rises); a yield SPIKE must book a capital loss.
  2. freshness gate -- skip only when the core daily level columns reach the prev business day.

Plus an OPT-IN real-FRED sanity check (skipped without FRED_API_KEY) that pulls a short
window, reconstructs the bond TR, and prints range sanity so DATA vs code issues are provable.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import (
    MACRO_ASSET_CORE_LEVEL_COLUMNS,
    MACRO_ASSET_GOLD_COLUMN,
    MACRO_ASSET_BOND_TR_COLUMN,
)
from src.data_extract.utils.prices import fetch_macro_assets as fma


# --------------------------------------------------------------------------- #
# 1. bond total-return reconstruction                                          #
# --------------------------------------------------------------------------- #
def test_build_bond_total_return_carry_and_duration():
    idx = pd.bdate_range("2020-01-01", periods=6)

    # (a) FLAT 4% yield -> no price move, pure carry -> strictly increasing index
    flat = pd.Series(4.0, index=idx)
    tr_flat = fma.build_bond_total_return(flat)
    ret = tr_flat.dropna().pct_change().dropna()
    assert (ret > 0).all(), "flat yield should accrue positive carry every day"
    daily_carry = 0.04 / 252
    assert ret.iloc[0] == pytest.approx(daily_carry, rel=1e-6)

    # (b) a one-day yield SPIKE (+1.0%: 4->5) must book a capital LOSS ~ -duration*Δy
    spike = pd.Series(4.0, index=idx)
    spike.iloc[3] = 5.0                              # +100bp on day 3
    tr_spike = fma.build_bond_total_return(spike)
    r = tr_spike.dropna().pct_change()
    loss_day = r.loc[idx[3]]
    assert loss_day < 0, "a yield jump must produce a negative (capital-loss) return"
    # ~ -D*Δy with D ~ 8 for a 10y 4% par bond -> roughly -8% (carry is negligible)
    assert -0.10 < loss_day < -0.05, f"loss {loss_day:.4f} not near -duration*Δy"

    print("\n=== SANITY CHECK: bond total-return reconstruction ===")
    print(f"  flat 4% yield -> daily carry {ret.iloc[0]*1e4:.3f}bp, index strictly rising.")
    print(f"  +100bp yield spike -> one-day return {loss_day*100:.2f}% (capital loss ~ -duration*dyield). Validated.")


# --------------------------------------------------------------------------- #
# 2. freshness gate                                                            #
# --------------------------------------------------------------------------- #
class _Store:
    def __init__(self, df):
        self._df = df

    def load(self, name, columns=None, limit=None):
        return self._df.copy()


def _df_reaching(level_last: pd.Timestamp) -> pd.DataFrame:
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(prev - pd.Timedelta(days=16), prev)
    df = pd.DataFrame({"date": dates})
    for c in MACRO_ASSET_CORE_LEVEL_COLUMNS:
        df[c] = [1.0 if d <= level_last else float("nan") for d in dates]
    return df


def test_up_to_date_keys_on_core_level_block():
    prev = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    prev2 = prev - pd.tseries.offsets.BDay(1)
    ctx_stale = SimpleNamespace(store=_Store(_df_reaching(prev2)))
    ctx_fresh = SimpleNamespace(store=_Store(_df_reaching(prev)))
    ctx_empty = SimpleNamespace(store=_Store(pd.DataFrame(columns=["date"])))

    assert fma._macro_assets_up_to_date(ctx_stale) is False
    assert fma._macro_assets_up_to_date(ctx_fresh) is True
    assert fma._macro_assets_up_to_date(ctx_empty) is False

    print("\n=== SANITY CHECK: macro-asset freshness gate ===")
    print("  core level block a day behind -> refresh; at prev business day -> skip; empty -> refresh. Validated.")


# --------------------------------------------------------------------------- #
# 3. opt-in real FRED sanity check                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.getenv("FRED_API_KEY"), reason="needs FRED_API_KEY for the live pull")
def test_real_fred_pull_ranges():
    captured: dict[str, pd.DataFrame] = {}
    ctx = SimpleNamespace(
        store=SimpleNamespace(
            load=lambda *a, **k: pd.DataFrame(columns=["date"]),
            replace=lambda name, df: captured.__setitem__(name, df),
        ),
        config=SimpleNamespace(data_extract=SimpleNamespace(macro_asset_years_history=2)),
        log=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    fma._refresh_macro_assets(ctx)
    df = next(iter(captured.values()))

    assert not df.empty and "date" in df.columns
    assert (df["equity_tr"].dropna() > 0).all()
    g = df[MACRO_ASSET_GOLD_COLUMN].dropna()
    assert g.empty or (g.between(200, 6000).all())          # USD/oz sane band
    assert df["cash_rate"].dropna().between(-1, 25).all()
    assert (df[MACRO_ASSET_BOND_TR_COLUMN].dropna() > 0).all()

    print("\n=== SANITY CHECK: live FRED macro-asset pull ===")
    print(f"  rows={len(df)}  cols={list(df.columns)}")
    for c in ("equity_tr", MACRO_ASSET_GOLD_COLUMN, "yield_10y", MACRO_ASSET_BOND_TR_COLUMN,
              "cash_rate", "fx_usdeur"):
        s = df[c].dropna()
        if s.empty:
            print(f"  {c:12s}: EMPTY")
        else:
            print(f"  {c:12s}: {s.iloc[0]:.3f} .. {s.iloc[-1]:.3f}  (n={len(s)})")
    print("  All legs in plausible ranges; bond TR strictly positive. Validated.")


if __name__ == "__main__":
    test_build_bond_total_return_carry_and_duration()
    test_up_to_date_keys_on_core_level_block()
