"""`labels_to_wide`: the WIDE target part contract.

  * column naming          -> target_<label>_h<horizon>, via `target_column` / TARGET_COL_RE
  * grain                  -> one row per (date, ticker), never duplicated
  * immature labels        -> a date whose h90 is still NaN KEEPS its row (the defect the
                              pivot fixes: the old `stack()` dropped it and the newest
                              ~max_horizon trading days vanished from the part)
  * all-NaN rows           -> and only those -- are dropped
  * legacy single-DataFrame labels -> TypeError with an actionable message

Synthetic fixtures on purpose: this is stacking/alignment math with a known truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.assemble.cube import (
    TARGET_COL_RE, horizons_in, labels_to_wide, target_column,
)


def _grid(dates: pd.DatetimeIndex, tkrs: list[str], values) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates, columns=tkrs)


# --------------------------------------------------------------------------- #
# 1. naming contract                                                          #
# --------------------------------------------------------------------------- #
def test_target_column_and_regex_round_trip():
    for label, h in [("rank", 30), ("zscore", 60), ("epsilon", 90), ("ret_fwd", 5)]:
        col = target_column(label, h)
        m = TARGET_COL_RE.match(col)
        assert m, f"{col} does not match TARGET_COL_RE"
        assert m["label"] == label and int(m["horizon"]) == h

    cols = ["date", "ticker", "f_mom_12m", "target_rank_h30", "target_rank_h90",
            "target_zscore_h60"]
    assert horizons_in(cols, "rank") == [30, 90]
    assert horizons_in(cols, "zscore") == [60]
    assert horizons_in(cols, "epsilon") == []

    print("\n=== SANITY CHECK: wide target naming contract ===")
    print(f"  target_column('rank', 30) -> {target_column('rank', 30)}; the regex round-trips "
          f"an underscored label ('ret_fwd'); horizons_in(cols,'rank') -> "
          f"{horizons_in(cols, 'rank')} from the SCHEMA alone, no data scan. Validated.")


# --------------------------------------------------------------------------- #
# 2. shape + grain                                                            #
# --------------------------------------------------------------------------- #
def test_labels_to_wide_columns_and_unique_keys():
    dates = pd.bdate_range("2020-01-01", periods=4)
    tkrs = ["AAA", "BBB", "CCC"]
    mk = lambda v: _grid(dates, tkrs, [v] * len(dates))
    labels = {30: {"rank": mk([0.2, 0.5, 0.8]), "zscore": mk([-1.0, 0.0, 1.0])},
              90: {"rank": mk([0.1, 0.4, 0.9]), "zscore": mk([-2.0, 0.1, 1.5])}}

    wide = labels_to_wide(labels)

    assert set(wide.columns) == {"date", "ticker",
                                 "target_rank_h30", "target_zscore_h30",
                                 "target_rank_h90", "target_zscore_h90"}
    assert not wide.duplicated(["date", "ticker"]).any()
    assert len(wide) == len(dates) * len(tkrs)
    # the values land on the right (date, ticker), not merely in the right column
    row = wide[(wide["ticker"] == "CCC")].iloc[0]
    assert row["target_rank_h30"] == 0.8 and row["target_zscore_h90"] == 1.5

    print("\n=== SANITY CHECK: labels_to_wide shape ===")
    print(f"  2 horizons x 2 labels -> {sorted(c for c in wide.columns if c.startswith('target'))}"
          f"\n  {len(wide)} rows = {len(dates)} dates x {len(tkrs)} tickers, "
          f"0 duplicate (date,ticker). Validated.")


def test_labels_to_wide_column_count_matches_grid():
    """The configured grid (horizons [30,60,90] x labels [rank,zscore,epsilon]) must emit
    exactly 2 key columns + 9 target columns."""
    dates = pd.bdate_range("2020-01-01", periods=2)
    tkrs = ["AAA", "BBB"]
    mk = lambda: _grid(dates, tkrs, np.arange(4, dtype=float).reshape(2, 2))
    labels = {h: {lab: mk() for lab in ("rank", "zscore", "epsilon")}
              for h in (30, 60, 90)}

    wide = labels_to_wide(labels)

    assert wide.shape[1] == 2 + 9, f"expected 11 columns, got {list(wide.columns)}"
    print("\n=== SANITY CHECK: 3 horizons x 3 labels ===")
    print(f"  {wide.shape[1]} columns = 2 keys + 9 targets. Validated.")


# --------------------------------------------------------------------------- #
# 3. THE regression: immature labels keep their row                           #
# --------------------------------------------------------------------------- #
def test_immature_horizon_keeps_its_dates():
    """h30 matured through the last date, h90 has not. Those trailing dates must SURVIVE with
    `target_*_h90` NaN -- under the old long writer's `stack()` (dropna=True) a (date, ticker)
    row existed only where a label was non-null, so the newest ~max_horizon days were missing
    from the part entirely and `predict_latest` could not see them."""
    dates = pd.bdate_range("2020-01-01", periods=8)
    tkrs = ["AAA", "BBB"]
    h30 = _grid(dates, tkrs, np.arange(16, dtype=float).reshape(8, 2))
    h90 = h30.copy()
    h90.iloc[-3:, :] = np.nan                      # last 3 dates not yet mature at h=90

    wide = labels_to_wide({30: {"rank": h30}, 90: {"rank": h90}})

    tail = dates[-3:]
    kept = wide[wide["date"].isin(tail)]
    assert len(kept) == 3 * len(tkrs), "immature dates were dropped"
    assert kept["target_rank_h90"].isna().all(), "h90 should be NaN on the immature tail"
    assert kept["target_rank_h30"].notna().all(), "h30 matured there and must carry a value"
    assert len(wide) == len(dates) * len(tkrs), "no row lost anywhere on the grid"

    print("\n=== SANITY CHECK: immature h90 tail survives ===")
    print(f"  last {len(tail)} dates x {len(tkrs)} tickers = {len(kept)} rows kept, "
          f"target_rank_h90 all NaN, target_rank_h30 all present. "
          f"Total {len(wide)} rows = the full {len(dates)}x{len(tkrs)} grid. Validated.")


def test_only_all_nan_rows_are_dropped():
    """A (date, ticker) with nothing known at ANY (label, horizon) stores nothing; a row with
    even one non-null target is kept."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    tkrs = ["AAA", "BBB"]
    h30 = _grid(dates, tkrs, [[1.0, 2.0], [np.nan, 4.0], [np.nan, 6.0]])
    h90 = _grid(dates, tkrs, [[1.0, 2.0], [np.nan, 4.0], [7.0, np.nan]])
    # (date[1], AAA) is NaN in both -> dropped. (date[2], AAA) has h90 only -> kept.
    # (date[2], BBB) has h30 only -> kept.

    wide = labels_to_wide({30: {"rank": h30}, 90: {"rank": h90}})
    keys = set(zip(wide["date"], wide["ticker"]))

    assert (dates[1], "AAA") not in keys, "all-NaN row should be dropped"
    assert (dates[2], "AAA") in keys and (dates[2], "BBB") in keys
    assert len(wide) == 5, f"expected 6 - 1 all-NaN row = 5, got {len(wide)}"

    print("\n=== SANITY CHECK: all-NaN drop is the ONLY drop ===")
    print(f"  {len(dates) * len(tkrs)} grid cells -> {len(wide)} rows: only the one "
          f"(date,ticker) NaN across every target column was dropped; partial rows kept. "
          f"Validated.")


# --------------------------------------------------------------------------- #
# 4. legacy mapping is refused, loudly                                        #
# --------------------------------------------------------------------------- #
def test_legacy_single_dataframe_labels_raise():
    dates = pd.bdate_range("2020-01-01", periods=2)
    legacy = {30: _grid(dates, ["AAA"], [[0.1], [0.2]])}     # {horizon: DataFrame}

    with pytest.raises(TypeError, match="build_targets_multi"):
        labels_to_wide(legacy)

    print("\n=== SANITY CHECK: legacy {horizon: DataFrame} refused ===")
    print("  the single-target fallback is gone; the TypeError names build_targets_multi as "
          "the fix rather than producing a silently column-less part. Validated.")
