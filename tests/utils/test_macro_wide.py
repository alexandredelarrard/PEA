"""
The long->wide macro adapter (src/utils/macro.py).

`prices_macro` is stored long, but every consumer -- the cube's beta/target step, the
long-book and trend-CTA sleeves, the portfolio benchmark, the L/S diagnostics -- was written
against a WIDE frame with `date` as a column. This adapter is the single place that conversion
happens, so its contract is what makes the refactor a one-line change at each of those six
call sites instead of six rewrites:

  * `date` is a COLUMN, not the index (drop-in for the `store.load` it replaced),
  * columns are the SERIES names, so the old wide-column vocabulary survives verbatim,
  * None (not an empty frame) when the table is missing/empty -- the `if df is None` guards.

Runs on `sqlite_store`, a real DataStore, so the projection and `where=` go through real SQL.
"""
from __future__ import annotations

import pandas as pd

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_store.schema import Tables
from src.utils.macro import load_macro_series, load_macro_wide

# the wide columns each real consumer reads, so a rename in `prices_macro` breaks HERE
_TREND_COLS = ["equity_tr", "gold", "energy", "bond_10y_tr", "fx_usdeur"]   # trend/signal.py
_ALLOC_COLS = _TREND_COLS + ["cash_rate"]                                   # long_book/allocation.py
_LONGBOOK_COLS = _ALLOC_COLS + ["vix"]                                      # step_long_book.py
_CUBE_COLS = ["equity_tr", "oil", "gold", "fx_usdeur", "yield_10y",
              "yield_curve_10y2y", "vix", "breakeven_10y", "baa_credit_spread"]


def _seed(store, series: list[str], n: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = [{"date": d, "ticker": s, "close": 100.0 + i + j}
            for j, s in enumerate(series) for i, d in enumerate(dates)]
    long = pd.DataFrame(rows)
    store.replace(Tables.prices_macro, long)
    return long


def test_load_macro_wide_shape_and_vocabulary(sqlite_store):
    long = _seed(sqlite_store, sorted(set(_LONGBOOK_COLS + _CUBE_COLS)))

    wide = load_macro_wide(sqlite_store)
    assert wide is not None
    # `date` is a COLUMN, not the index -- the drop-in contract
    assert "date" in wide.columns and wide.index.name is None
    assert wide["date"].is_monotonic_increasing
    # one row per date, one column per series
    assert len(wide) == long["date"].nunique()
    for consumer, cols in (("trend", _TREND_COLS), ("allocation", _ALLOC_COLS),
                           ("long_book", _LONGBOOK_COLS), ("cube", _CUBE_COLS)):
        missing = [c for c in cols if c not in wide.columns]
        assert not missing, f"{consumer} would lose {missing}"

    print("\n=== SANITY CHECK: long -> wide adapter ===")
    print(f"  {len(long)} long rows -> {wide.shape[0]} dates x {wide.shape[1] - 1} series")
    print(f"  'date' is a column (index.name={wide.index.name}); every consumer's wide "
          f"vocabulary present. Validated.")


def test_series_narrows_the_read(sqlite_store):
    _seed(sqlite_store, sorted(set(_LONGBOOK_COLS + _CUBE_COLS)))

    wide = load_macro_wide(sqlite_store, series=["equity_tr", "vix"])
    assert sorted(wide.columns) == ["date", "equity_tr", "vix"]

    s = load_macro_series(sqlite_store, MACRO_MARKET_SERIES)
    assert isinstance(s, pd.Series) and s.name == MACRO_MARKET_SERIES
    assert s.index.name == "date" and s.notna().all()          # date-INDEXED here, by design

    print("\n=== SANITY CHECK: projection ===")
    print(f"  series=['equity_tr','vix'] -> columns {sorted(wide.columns)}")
    print(f"  load_macro_series -> date-indexed Series of {len(s)} values. Validated.")


def test_none_when_empty_or_absent(sqlite_store):
    # absent table -> None, so the `if df is None` guards fire instead of KeyError
    assert load_macro_wide(sqlite_store) is None
    assert load_macro_series(sqlite_store, MACRO_MARKET_SERIES) is None

    _seed(sqlite_store, ["equity_tr"])
    # present table, but the requested series has no rows -> also None
    assert load_macro_series(sqlite_store, "gold") is None
    assert load_macro_wide(sqlite_store, series=["gold"]) is None

    print("\n=== SANITY CHECK: empty/absent contract ===")
    print("  missing table -> None; present table but requested series has no rows -> None.")
    print("  Consumers branch on `is None`, never on a fabricated empty frame. Validated.")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
