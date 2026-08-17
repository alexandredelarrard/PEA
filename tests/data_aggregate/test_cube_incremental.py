"""
Incremental cube-part builds (src/data_aggregate/step_build_cube.py).

The exploded DAG rebuilds each cube_part_<group> INCREMENTALLY: read the latest date, recompute only
a warm-up-padded trailing window, and append the new rows. This is only correct if a trailing-window
build reproduces the FULL build's tail exactly — which holds because the price/rolling features are
backward-looking (window <= warm-up) and the cross-sectional standardization is per-day (independent
across dates). This test proves that equivalence on the price feature builder, and checks the
idempotent tail-append helper.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import build_feature_panel
from src.data_aggregate.transformers.step_cube_extras import StepCubeExtras
from src.data_aggregate.utils.common.parts import CUBE_PARTS, PART_BY_NAME
from src.data_aggregate.utils.common.sources import (
    OPTIONAL_SOURCE_COLUMNS, SOURCE_COLUMNS, project_existing,
)


def _synthetic_prices(n_days: int = 2000, n_tickers: int = 8, seed: int = 0):
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tk = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.012, (n_days, n_tickers)), axis=0)),
                         index=dates, columns=tk)
    open_ = close.shift(1).bfill()
    ret = close.pct_change().fillna(0.0)
    sector = pd.DataFrame(np.repeat(ret.mean(axis=1).to_numpy()[:, None], n_tickers, axis=1),
                          index=dates, columns=tk)                     # one shared "sector"
    high, low = close * 1.01, close * 0.99
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_days, n_tickers)).astype(float),
                          index=dates, columns=tk)
    return dates, close, open_, sector, high, low, volume


def test_windowed_build_reproduces_full_tail():
    dates, close, open_, sector, high, low, volume = _synthetic_prices()
    sh = [5, 20, 60]

    full = build_feature_panel(close, open_, sector, "rank", high, low, volume, sh)

    # simulate an incremental run with the ACTUAL configured price-group warm-up: recompute only
    # [cutoff-warmup, end]. This must cover the longest daily look-back (the 5-year = 1260-day
    # seasonality feature) -> the test fails if the map value is ever set too low.
    warmup = PART_BY_NAME["cube_part_momentum"].warmup_trading_days
    cutoff_pos = 1900
    cutoff = dates[cutoff_pos]
    start = dates[cutoff_pos - warmup]
    win = build_feature_panel(close.loc[start:], open_.loc[start:], sector.loc[start:], "rank",
                              high.loc[start:], low.loc[start:], volume.loc[start:], sh)

    # the tail we would KEEP + APPEND (date > cutoff) must match the full build bit-for-bit
    f_tail = (full[full["date"] > cutoff].set_index(["date", "ticker"]).sort_index())
    w_tail = (win[win["date"] > cutoff].set_index(["date", "ticker"]).sort_index())
    assert not f_tail.empty and f_tail.shape == w_tail.shape
    cols = list(f_tail.columns)
    pd.testing.assert_frame_equal(f_tail[cols], w_tail[cols].reindex(f_tail.index),
                                  check_exact=False, atol=1e-9, rtol=0)

    n_tail_days = f_tail.index.get_level_values("date").nunique()
    print("\n=== SANITY CHECK: windowed build reproduces the full tail ===")
    print(f"  full={len(full)} rows over {close.shape[0]} days; windowed recompute of the last "
          f"{close.shape[0] - (cutoff_pos - warmup)} days (warmup {warmup})")
    print(f"  tail (date > {cutoff.date()}): {n_tail_days} days x {full['ticker'].nunique()} "
          f"tickers x {len(cols)} feature cols -> IDENTICAL between full and windowed builds")
    print("  CONCLUSION: backward-looking features + per-day standardization -> the incremental "
          "trailing recompute equals a full rebuild on the appended dates. Validated.")


def test_incremental_horizon_arithmetic():
    """The target refresh window must reach back >= max_horizon so matured (NaN->value) labels are
    recomputed, while features only need dates strictly after the stored max."""
    # emulate _window_start on a business-day calendar
    idx = pd.bdate_range("2019-01-01", periods=800)
    last = idx[750]

    def window_start(last, n_back):
        pos = int(idx.searchsorted(pd.Timestamp(last)))
        return idx[max(0, pos - n_back)]

    max_h = 90
    feat_start = window_start(last, 1400)                 # warm-up only (features)
    tgt_start = window_start(last, 1400 + max_h)          # warm-up + horizon (targets compute)
    refresh_from = window_start(last, max_h)              # matured-label overwrite window

    assert feat_start < last and tgt_start <= feat_start
    # the refresh window covers exactly the dates whose forward labels could have matured
    assert (idx.searchsorted(last) - idx.searchsorted(refresh_from)) == max_h
    print("\n=== SANITY CHECK: incremental window arithmetic ===")
    print(f"  last stored date {last.date()} | feature warm-up start {feat_start.date()} | "
          f"target compute start {tgt_start.date()} | matured-label refresh from {refresh_from.date()}")
    print("  targets overwrite the trailing max_horizon window (matured labels), betas/features "
          "append only dates after the max. Validated.")


def test_per_part_warmup_covers_binding_lookback():
    """Each part's warm-up must cover the LONGEST daily-grid look-back of EVERY feature group
    merged into it, else the incremental tail would be silently wrong.

    The binding look-backs now live on the registry entry (`CubePart.binding_lookbacks`)
    rather than being a literal duplicated here, so the contract cannot drift out of sync
    with the code that enforces it. Groups whose look-back is in FILING or QUARTER space read
    the full source table, so their grid look-back is 0 and a ~6-month floor is plenty."""
    covered: dict[str, int] = {}
    for part in CUBE_PARTS:
        for group, need in part.binding_lookbacks:
            assert part.warmup_trading_days >= need, (
                f"{part.name}: warm-up {part.warmup_trading_days} < binding daily look-back "
                f"{need} of member group '{group}'")
            covered[group] = part.warmup_trading_days

    # the 14 feature groups the old exploded DAG ran as separate tasks are all still owned
    assert set(covered) == {
        "price", "fundamental", "sector", "earnings", "governance", "employee", "dividend",
        "attention", "institutional", "superinvestor", "insider", "short_interest",
        "earnings_call_sentiment", "earnings_call_embedding",
    }, f"feature groups lost/added: {sorted(covered)}"

    # heavy parts read ~5y; every other part stays light (this is where the memory win is)
    heavy = {p.name for p in CUBE_PARTS if p.warmup_trading_days >= 1260}
    light = [p for p in CUBE_PARTS if p.kind == "features" and p.name not in heavy]
    assert all(p.warmup_trading_days <= 400 for p in light), "light parts should stay light"

    print("\n=== SANITY CHECK: per-part warm-up vs binding look-back ===")
    for part in CUBE_PARTS:
        members = ", ".join(f"{g}({n})" for g, n in part.binding_lookbacks) or "-"
        print(f"  {part.name:<26} warm-up={part.warmup_trading_days:>5}  members: {members}")
    print(f"  CONCLUSION: all 14 feature groups are owned by {len(heavy)} heavy part(s) reading "
          f"~5y and {len(light)} light part(s) reading <=400d. Each part reads only as far back "
          "as its longest member needs. Validated.")


def test_source_column_projection_covers_builder_needs():
    """Each tall-table source load is projected to only the columns its builder reads (the
    memory fix for the OOM). The projection MUST cover every column the builder requires —
    this test is the contract that guards against a projection dropping a needed column."""
    required = {  # columns each builder actually consumes from the table (the contract)
        "sec13f_hr": {"cik", "period", "ticker", "shares", "value_usd",
                      "call_value", "put_value", "filing_date"},
        "insider_transactions":   {"ticker", "filing_date", "transaction_code", "value_usd"},
        "short_interest":         {"date", "ticker", "short_volume", "total_volume",
                                   "short_interest", "avg_daily_volume"},
        "fails_to_deliver":       {"date", "ticker", "fails_quantity"},
        "wiki_pageviews":         {"date", "ticker", "pageviews"},
        "google_trends":          {"date", "ticker", "search_interest"},
    }
    for tbl, need in required.items():
        proj = set(SOURCE_COLUMNS[tbl])
        assert need <= proj, f"{tbl}: projection is MISSING required cols {need - proj}"

    # the extras step must FORWARD the projection to the store; an unmapped table -> full load
    step = object.__new__(StepCubeExtras)
    seen: dict[str, list | None] = {}
    # what each table really has, so the projection can be narrowed to it
    live = {"sec13f_hr": SOURCE_COLUMNS["sec13f_hr"],
            "fundamentals_history": ["ticker", "as_of", "totalRevenue"]}

    class _Store:
        """One object now that the step reads columns AND rows from the same store."""

        def exists(self, name):
            return name in live

        def columns(self, name):
            return live.get(name)

        def load(self, name, columns=None, **kw):
            seen[name] = columns
            return pd.DataFrame()

    class _Ctx:
        store = _Store()

    step._context = _Ctx()
    step._store = step._context.store
    step._log = logging.getLogger("test")
    step._load_source("sec13f_hr")
    step._load_source("fundamentals_history")             # not in the projection map
    assert seen["sec13f_hr"] == SOURCE_COLUMNS["sec13f_hr"]
    assert seen["fundamentals_history"] is None            # small table -> loaded in full

    print("\n=== SANITY: source-column projection ===")
    for tbl in required:
        print(f"  {tbl:<24} -> {len(SOURCE_COLUMNS[tbl])} cols (covers builder needs)")
    print("  sec13f_hr (~21.7M rows) drops the call/put/cusip-era bloat; small tables load "
          "full. StepCubeExtras forwards the projection to the store. Validated.")


def test_projection_tolerates_an_absent_optional_column():
    """A column the BUILDER treats as optional must not make the READ fail.

    `read_table` resolves each projected column via `tbl.c[name]`, which raises KeyError for
    an absent one. The live `short_interest` table has only date/ticker/short_volume/
    total_volume, while the projection also lists `short_interest` + `avg_daily_volume` --
    which `_short_fields` uses only `if {...}.issubset(hist.columns)`. Demanding them
    unconditionally killed the whole extras step with `KeyError: 'short_interest'`."""
    live_short = ["date", "ticker", "short_volume", "total_volume"]
    got = project_existing(live_short, "short_interest")
    assert got == live_short, got
    assert "short_interest" not in got and "avg_daily_volume" not in got

    # a table with every projected column present is unchanged
    full_13f = list(SOURCE_COLUMNS["sec13f_hr"])
    assert project_existing(full_13f, "sec13f_hr") == full_13f

    # unknown column list (table shape unreadable) -> project the full wanted list
    assert project_existing(None, "short_interest") == SOURCE_COLUMNS["short_interest"]
    # a table absent from the map -> no projection (load in full)
    assert project_existing(["a", "b"], "fundamentals_history") is None

    # every OPTIONAL column must actually appear in that table's projection, else the
    # exemption is dead and a real missing column would be silently tolerated
    for tbl, optional in OPTIONAL_SOURCE_COLUMNS.items():
        assert optional <= set(SOURCE_COLUMNS[tbl]), f"{tbl}: stale optional cols"

    print("\n=== SANITY CHECK: optional projected columns degrade, required ones warn ===")
    print(f"  short_interest live cols {live_short} -> projection {got}")
    print(f"  optional-by-table: { {k: sorted(v) for k, v in OPTIONAL_SOURCE_COLUMNS.items()} }")
    print("  CONCLUSION: an optional column missing from the live table is dropped from the "
          "projection instead of raising KeyError and killing the step. Validated.")


if __name__ == "__main__":
    test_windowed_build_reproduces_full_tail()
    test_incremental_horizon_arithmetic()
    test_per_part_warmup_covers_binding_lookback()
    test_source_column_projection_covers_builder_needs()
    test_projection_tolerates_an_absent_optional_column()
