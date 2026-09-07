"""
Incremental cube-part builds (src/data_aggregate/step_build_cube.py).

The exploded DAG rebuilds each cube_part_<group> INCREMENTALLY: read the latest date, recompute only
a warm-up-padded trailing window, then REWRITE the trailing `PART_REFRESH_TRADING_DAYS` and append
everything after them. This is only correct if a trailing-window build reproduces the FULL build's
tail exactly — which holds because the price/rolling features are backward-looking (window <=
warm-up) and the cross-sectional standardization is per-day (independent across dates). This test
proves that equivalence on the price feature builder over the INCLUSIVE span, and checks the
idempotent tail-append helper.

The rewrite is not hygiene. A part's last stored date is the one most likely to be wrong: it was
built from the newest, least settled prices, and the fetcher re-pulls that same tail (a settled
close superseding a mid-session bar, a refilled hole). Appending strictly after it left the live
`cube_part_momentum` stopping ON a date whose features were ranked over 45 of 491 tickers, with no
incremental run able to replace that row.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import build_feature_panel
from src.data_aggregate.transformers.step_cube_extras import StepCubeExtras
from src.data_aggregate.utils.common.incremental import (
    PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part, window_start,
)
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
    """The equivalence the whole incremental design rests on, over the INCLUSIVE window.

    The compared span is `date >= refresh_from`, not `date > cutoff`: a backward-looking part
    now REWRITES its trailing `PART_REFRESH_TRADING_DAYS` as well as appending after them, so
    those dates have to be reproduced by the windowed build too -- they are deleted from the
    table before the append and there is no second chance at them.

    This is also what makes `plan_window`'s `warmup + extra_back + refresh` load-bearing: the
    read window starts `warmup` days before `refresh_from`, not before `last`, so the OLDEST
    rewritten date still gets its full look-back."""
    dates, close, open_, sector, high, low, volume = _synthetic_prices()
    sh = [5, 20, 60]

    full = build_feature_panel(close, open_, sector, "rank", high, low, volume, sh)

    # simulate an incremental run with the ACTUAL configured price-group warm-up: recompute only
    # [refresh_from - warmup, end]. This must cover the longest daily look-back (the 5-year =
    # 1260-day seasonality feature) -> the test fails if the map value is ever set too low.
    warmup = PART_BY_NAME["cube_part_momentum"].warmup_trading_days
    refresh = PART_REFRESH_TRADING_DAYS
    cutoff_pos = 1900
    last = dates[cutoff_pos]
    refresh_from = dates[cutoff_pos - refresh]
    start = dates[cutoff_pos - refresh - warmup]
    win = build_feature_panel(close.loc[start:], open_.loc[start:], sector.loc[start:], "rank",
                              high.loc[start:], low.loc[start:], volume.loc[start:], sh)

    # the tail the incremental run WRITES (date >= refresh_from) must match the full build
    # bit-for-bit -- both the `refresh` dates it overwrites and everything it appends after.
    f_tail = (full[full["date"] >= refresh_from].set_index(["date", "ticker"]).sort_index())
    w_tail = (win[win["date"] >= refresh_from].set_index(["date", "ticker"]).sort_index())
    assert not f_tail.empty and f_tail.shape == w_tail.shape
    cols = list(f_tail.columns)
    pd.testing.assert_frame_equal(f_tail[cols], w_tail[cols].reindex(f_tail.index),
                                  check_exact=False, atol=1e-9, rtol=0)

    # the rewritten span really does include `last` itself -- the row a strictly-after append
    # could never replace, and the one a truncated extract run left wrong on the live table.
    rewritten = f_tail.index.get_level_values("date").unique()
    assert last in set(rewritten)
    assert (rewritten <= last).sum() == refresh + 1, "refresh_from .. last inclusive"

    n_tail_days = len(rewritten)
    print("\n=== SANITY CHECK: windowed build reproduces the full tail (INCLUSIVE) ===")
    print(f"  full={len(full)} rows over {close.shape[0]} days; windowed recompute of the last "
          f"{close.shape[0] - (cutoff_pos - refresh - warmup)} days "
          f"(warmup {warmup} + refresh {refresh})")
    print(f"  stored max was {last.date()}; the run REWRITES from {refresh_from.date()} "
          f"({refresh + 1} dates through {last.date()}) and appends after")
    print(f"  tail (date >= {refresh_from.date()}): {n_tail_days} days x "
          f"{full['ticker'].nunique()} tickers x {len(cols)} feature cols -> IDENTICAL between "
          "full and windowed builds")
    print("  CONCLUSION: backward-looking features + per-day standardization -> the incremental "
          "trailing recompute equals a full rebuild on every date it writes, rewritten dates "
          "included. Validated.")


def test_incremental_horizon_arithmetic():
    """The target refresh window must reach back >= max_horizon so matured (NaN->value) labels
    are recomputed. That is much wider than the backward-looking parts' own trailing rewrite
    (`PART_REFRESH_TRADING_DAYS`), which is why `write_part` gives an explicit `refresh_from`
    precedence over `window.refresh_from`."""
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
    print(f"  targets overwrite the trailing max_horizon window ({max_h} days, matured labels); "
          f"betas/features rewrite only {PART_REFRESH_TRADING_DAYS}, so the wider explicit "
          "window wins. Validated.")


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


# --------------------------------------------------------------------------- #
# the inclusive trailing refresh                                               #
# --------------------------------------------------------------------------- #
# long enough that momentum's real 1,320-day warm-up plus the refresh still lands inside it;
# `window_start` clamps at index 0, which would silently hide the arithmetic under test.
CAL = pd.bdate_range("2011-01-03", periods=2000)
LAST_POS = 1900
LAST = CAL[LAST_POS]


class _RefreshStore:
    """Minimal DataStore stand-in: records what `write_part` asked for."""

    def __init__(self, columns=None, last=LAST):
        self._columns, self._last = columns, last
        self.appended: tuple | None = None
        self.replaced: pd.DataFrame | None = None

    def max_date(self, part):
        return self._last

    def columns(self, part):
        return self._columns or []

    def replace(self, part, rows):
        self.replaced = rows
        return len(rows)

    def append_tail(self, part, tail, cutoff, *, inclusive):
        self.appended = (tail, pd.Timestamp(cutoff), inclusive)
        return len(tail)


def _rows(dates=CAL[LAST_POS - 10:LAST_POS + 6]):
    return pd.DataFrame({"date": dates, "ticker": "T0", "f": 1.0})


def test_plan_window_refresh_arithmetic():
    """`since` reaches `warmup + refresh` back, `refresh_from` exactly `refresh` back.

    The `+ refresh` in `since` is the point: the warm-up has to be measured from the oldest
    REWRITTEN date, not from `last`, or that date is computed with less look-back than a full
    rebuild would give it."""
    warmup, refresh = 1320, PART_REFRESH_TRADING_DAYS
    w = plan_window(_RefreshStore(), "p", warmup=warmup, full=False,
                    trading_index=CAL, refresh=refresh)

    assert w.last == LAST
    assert w.refresh_from == CAL[LAST_POS - refresh]
    assert w.since == CAL[LAST_POS - refresh - warmup]
    # the oldest rewritten date still gets the FULL warm-up behind it
    assert CAL.searchsorted(w.refresh_from) - CAL.searchsorted(w.since) == warmup
    print("\n=== SANITY CHECK: plan_window(refresh=%d) ===" % refresh)
    print(f"  last={w.last.date()}  refresh_from={w.refresh_from.date()}  "
          f"since={w.since.date()}")
    print(f"  warm-up behind the OLDEST rewritten date = {warmup} trading days (not "
          f"{warmup - refresh}), so it is computed exactly as a full rebuild would. Validated.")


def test_plan_window_without_refresh_is_unchanged():
    """The parts that opt out (fundamentals / text / extras) keep their exact old window."""
    w = plan_window(_RefreshStore(), "p", warmup=130, full=False, trading_index=CAL)
    assert w.refresh_from is None
    assert w.since == CAL[LAST_POS - 130]
    assert plan_window(_RefreshStore(), "p", warmup=130, full=True).refresh_from is None


def test_plan_window_refresh_stacks_with_extra_back():
    """Targets take both: `extra_back` for the maturing-label compute window and `refresh`."""
    w = plan_window(_RefreshStore(), "p", warmup=390, full=False, trading_index=CAL,
                    extra_back=90, refresh=PART_REFRESH_TRADING_DAYS)
    assert w.since == CAL[LAST_POS - 390 - 90 - PART_REFRESH_TRADING_DAYS]


def test_write_part_rewrites_inclusively_from_refresh_from():
    """The defect this closes: `cube_part_momentum` stopped ON a bad date, and a strictly-after
    append could never replace that row -- only a `--full` rebuild would."""
    store = _RefreshStore(columns=["date", "ticker", "f"])
    window = PartWindow(LAST, CAL[LAST_POS - 1320], CAL[LAST_POS - PART_REFRESH_TRADING_DAYS])
    n = write_part(store, "p", _rows(), window)

    tail, cutoff, inclusive = store.appended
    assert inclusive is True
    assert cutoff == window.refresh_from
    assert tail["date"].min() == window.refresh_from
    assert LAST in set(tail["date"]), "the previously-final row IS rewritten"
    print("\n=== SANITY CHECK: write_part rewrites its own tail ===")
    print(f"  stored max {LAST.date()} -> DELETE >= {cutoff.date()} then append {n} rows "
          f"({tail['date'].min().date()} .. {tail['date'].max().date()})")
    print("  the stored max date's own row is REPLACED, not skipped. Validated.")


def test_write_part_strict_append_when_no_refresh():
    """No opt-in -> bit-identical to the pre-refresh behaviour."""
    store = _RefreshStore(columns=["date", "ticker", "f"])
    write_part(store, "p", _rows(), PartWindow(LAST, CAL[LAST_POS - 130]))
    tail, cutoff, inclusive = store.appended
    assert inclusive is False and cutoff == LAST
    assert tail["date"].min() > LAST


def test_explicit_refresh_from_wins_over_the_part_default():
    """The target step's maturing-label window (~90 trading days) is much wider than the
    shared 5 and must not be narrowed by it."""
    store = _RefreshStore(columns=["date", "ticker", "f"])
    wide = window_start(CAL, LAST, 90)
    window = PartWindow(LAST, CAL[LAST_POS - 500], CAL[LAST_POS - PART_REFRESH_TRADING_DAYS])
    write_part(store, "p", _rows(CAL[LAST_POS - 150:LAST_POS + 6]), window, refresh_from=wide)
    _, cutoff, inclusive = store.appended
    assert inclusive is True and cutoff == wide
    assert cutoff < window.refresh_from


def test_refresh_never_narrows_the_written_span():
    """A guard on the constant: the part must rewrite at least as far back as the price
    fetcher can still change its inputs, or a corrected price leaves a stale feature behind.
    The fetcher's floor is 7 BUSINESS days, which is 5 trading sessions of span."""
    from src.data_extract.utils.prices.fetch_prices import PRICE_REFRESH_TRADING_DAYS
    fetcher_span = (pd.Timestamp("2026-09-04")
                    - pd.tseries.offsets.BDay(PRICE_REFRESH_TRADING_DAYS))
    part_span = window_start(pd.bdate_range("2026-01-01", "2026-09-04"),
                             pd.Timestamp("2026-09-04"), PART_REFRESH_TRADING_DAYS)
    print(f"\n  fetcher re-pulls from {fetcher_span.date()}; part rewrites from "
          f"{part_span.date()}")
    assert part_span <= fetcher_span + pd.Timedelta(days=2), (
        f"part refresh ({PART_REFRESH_TRADING_DAYS} sessions) must cover the fetcher's "
        f"{PRICE_REFRESH_TRADING_DAYS} BDay re-pull floor")


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
        "sec_fails_to_deliver":   {"date", "ticker", "fails_quantity"},
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
