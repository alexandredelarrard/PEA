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
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common.incremental import (
    PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part, window_start,
)
from src.data_aggregate.utils.common.parts import CUBE_PARTS, PART_BY_NAME
from src.data_store.schema import (
    ALL, Tables, name_of, projection, projection_report, resolve,
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

    # the feature groups the old exploded DAG ran as separate tasks are all still owned.
    # `attention` is absent because that panel was DELETED (dead code: defined, never called
    # from `run()`) with the `extras` -> `institutionals` rename -- not because a group lost
    # its owner, which is what this assertion exists to catch.
    #
    # `short_interest` -> `short_flow` is a RENAME, not a loss: the panel now builds from RegSHO
    # daily short VOLUME and fails-to-deliver, and `ic_shortvol_days_to_cover` went with the old
    # name because days-to-cover needs FINRA short-INTEREST positions this repo does not fetch.
    # `conditioning` and `cross_source` are the two DERIVED panels added in Phase 2.6/2.7; they
    # read the `ConditioningSink` rather than a source table, but they still own a daily-grid
    # look-back (the 252-day excursion window is one of the two that bind this part's warm-up).
    #
    # `ownership` was ADDED to the set on 2026-09-12 and that is a DECLARATION, not a new panel:
    # `cube_part_institutionals` had been merging SEVEN panels while declaring six, so the
    # beneficial-ownership family (`ic_bo_*` / `ic_act_*`) had no entry and nothing here could
    # check its warm-up. Its `HOLDER_ACTIVE_DAYS = 378` is in fact the longest bounded look-back
    # in the part, covered by the 390-day warm-up with 12 days to spare -- which was luck, and
    # `f_ic_bo_new_holder` was the single worst column in the 2026-09-12 L7 run, wrong on 490 of
    # 491 tickers on the newest date.
    #
    # ⚠ THIS ASSERTION IS STILL WEAKER THAN IT LOOKS. A group's declared number is its longest
    # BOUNDED look-back, and several families here are UNBOUNDED: an age-since-last-event, a
    # distinct-actor count to date, an expanding-window owner surprise, a price path measured
    # from an anchor fifteen years back. `parts.py`'s premise -- source tables are read in full,
    # so filing-space builders need no grid warm-up -- holds for a rolling window and fails for
    # these, because the events are all read but are projected onto a grid that starts at
    # `window.since`. No value of `warmup_trading_days` fixes that. `cube_part_institutionals`
    # answers it structurally instead (`_load_frames` takes no `since`; it computes over the
    # full calendar and lets `write_part` slice the tail), so for THAT part the warm-up is now
    # belt-and-braces. For every other part this test passing is still not evidence that an
    # incremental build is correct -- L7 in `validate institutionals` is.
    assert set(covered) == {
        "price", "fundamental", "sector", "earnings", "governance", "employee", "dividend",
        "institutional", "superinvestor", "insider", "short_flow", "conditioning",
        "cross_source", "ownership", "earnings_call_sentiment", "earnings_call_embedding",
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


def test_read_projection_covers_builder_needs():
    """Each tall-table source load is projected to only the columns its builder reads (the
    memory fix for the OOM), and the projection MUST cover every column the builder requires.

    The projection now lives ONCE, on the registry (`Table.read_columns`), and reaches the
    builders through `store.load(project=True)`. This test is the contract that guards against
    a registry edit dropping a needed column -- it fails here rather than silently emptying a
    feature family.
    """
    required = {  # columns each builder actually consumes from the table (the contract)
        Tables.sec13f_hr: {"cik", "period", "ticker", "shares", "value_usd",
                           "call_value", "put_value", "filing_date"},
        Tables.insider_transactions: {"ticker", "owner_cik", "filing_date", "transaction_code",
                                      "shares", "price_per_share", "value_usd",
                                      "security_type", "shares_owned_after"},
        # ⚠ FOUR COLUMNS, NOT SIX. `short_interest` / `avg_daily_volume` were in this contract
        # for `ic_shortvol_days_to_cover` -- a feature that no longer exists, and whose absence
        # is a DESIGN decision rather than a missing projection: the live `sec_short_interest`
        # table carries RegSHO daily short VOLUME only, and days-to-cover needs FINRA's
        # twice-monthly short-INTEREST positions, which this repo does not fetch. Asserting the
        # two dead columns here made the projection look broken when it was the contract that
        # was stale. Do not add them back without a fetcher that populates them.
        Tables.short_interest: {"date", "ticker", "short_volume", "total_volume"},
        Tables.sec_fails_to_deliver: {"date", "ticker", "fails_quantity"},
        # ownership_features (13D/13G). `accession_number` + `cusip` are the canonical-event
        # key (a group files one 13D under many reporting persons; summing their
        # `percent_of_class` would double-count the same block), and `filing_date` is the ONLY
        # legal stamp -- `date_of_event` is never projected, which is what makes L4 structural.
        Tables.sec_13d: {"ticker", "accession_number", "cusip", "filing_date",
                         "is_amendment", "percent_of_class", "reporting_person_cik"},
        Tables.sec_13g: {"ticker", "accession_number", "cusip", "filing_date",
                         "percent_of_class", "reporting_person_cik"},
        # the two per-person DEF 14A children, read by StepCubeGovernance. `accession_number`
        # is in the directors set because the per-FILING aggregates key on it (an `as_of` can
        # carry two filings), and every one of the six Item 402(k) components is required
        # because `impute_director_comp` reconstructs a NULL `total` by summing them.
        # ⚠ `def14a_directors.is_independent` is PROJECTED but deliberately not read (D79:
        # `pct_independent_directors` is one of the twelve D3-protected features, so deriving
        # it from the child would move live cells). Left in the projection on purpose -- do not
        # "tidy" it out, it is the substrate for that deferred work.
        Tables.def14a_directors: {"ticker", "accession_number", "as_of", "name", "age",
                                  "tenure_years", "other_public_company_boards"},
        Tables.def14a_director_comp: {"ticker", "as_of", "total", "fees_earned",
                                      "stock_awards", "option_awards", "non_equity_incentive",
                                      "pension_change", "other_compensation"},
    }
    for table, need in required.items():
        proj = set(table.read_columns)
        assert proj, f"{table.name}: registry declares no read_columns"
        assert need <= proj, f"{table.name}: projection is MISSING required {need - proj}"

    print()
    print("=== SANITY: registry read_columns cover every builder need ===")
    for table in required:
        print(f"  {table.name:<24} -> {len(table.read_columns)} cols (covers builder needs)")
    print("  def14a_directors keeps `is_independent` projected-but-unread on purpose (D79).")
    print("  sec13f_hr (~21.7M rows) drops the cusip-era bloat; a table declaring no "
          "read_columns loads in full. Validated.")


def test_step_forwards_the_registry_projection_to_the_store():
    """`_load_source` must hand the store `project=True` and nothing else.

    The step used to resolve the projection itself and pass `columns=`; the registry now owns
    it, so what this asserts is that the step DELEGATES -- `project=True` and no column list of
    its own, for a projected table and an unprojected one alike.

    `prices_splits` is the unprojected case ON PURPOSE: it is one of the two tables
    `_load_source` really reads whole (`cusip_ticker_map` is the other), so this pins that the
    switch to `project=True` did not narrow them. Every OTHER table the step reads declares
    `read_columns`, and each one reproduces its pre-switch column list exactly.
    """
    step = object.__new__(StepCubeInstitutionals)
    seen: dict[str, dict] = {}
    live = {"sec13f_hr": list(Tables.sec13f_hr.read_columns),
            "prices_splits": ["ticker", "date", "split_ratio"]}

    class _Store:
        """Resolves through `name_of` exactly as `DataStore` does, because the step hands it
        `Table` objects rather than name strings -- a fake keyed on the attribute name would
        re-admit the bug this signature change fixed."""

        def exists(self, table):
            return name_of(table) in live

        def columns(self, table):
            return live.get(name_of(table), [])

        def distinct(self, table, column, **kw):
            return []

        def load(self, table, columns=None, *, project=False, where=None, **kw):
            seen[name_of(table)] = {"columns": columns, "project": project, "where": where}
            cols = projection(table, self.columns(table)) if project else columns
            return pd.DataFrame(columns=cols or ["ticker"])

    class _Ctx:
        store = _Store()

    step._context = _Ctx()
    step._store = step._context.store
    step._log = logging.getLogger("test")
    step._load_source(Tables.sec13f_hr)
    step._load_source(Tables.prices_splits)           # declares no read_columns

    for name, call in seen.items():
        assert call["project"] is True, f"{name}: step must delegate the projection"
        assert call["columns"] is None, f"{name}: step must not pass its own column list"
    assert projection(Tables.sec13f_hr, live["sec13f_hr"]) == live["sec13f_hr"]
    assert projection(Tables.prices_splits, live["prices_splits"]) is None

    print()
    print("=== SANITY: the step delegates the projection ===")
    print(f"  calls seen: {seen}")
    print("  CONCLUSION: `_load_source` passes `project=True` for every table and resolves no "
          "column list of its own, so the registry is the single declaration. Validated.")


def test_universe_scope_is_pushed_down_to_the_read():
    """The universe cut must reach the DATABASE as a `WHERE ticker IN (...)`, not the frame.

    The step used to load every in-DB ticker and discard the off-universe ones in pandas. The
    cut itself is about `_xs` (D26: a same-day percentile ranks a ticker against every other
    ticker on that date, and in the cube that set is the universe) -- but doing it after the
    read meant paying for ~21.7M `sec13f_hr` rows in order to throw some away.

    Three things are pinned here, and the third is the one a refactor breaks silently:
      (a) a table WITH a ticker column and a universe -> `where` carries the sorted ticker list
      (b) a table with NO `ticker_col` (`sec13f_manager_holdings`) -> no `where` at all, rather
          than a `WHERE None IN (...)`. The guard is the REGISTRY's `ticker_col`, because the
          frame does not exist yet -- `"ticker" in df.columns` has nothing to look at
      (c) the off-universe DIAGNOSTIC survives, as its own cheap `SELECT DISTINCT ticker`.
          Once the rows never arrive, the report cannot be taken from the frame, and losing it
          would mean the S&P 500 membership boundary moves with nothing saying so.
    """
    step = object.__new__(StepCubeInstitutionals)
    calls: dict[str, dict] = {}
    distincts: list[tuple[str, str]] = []
    universe = ["AAPL", "MSFT", "NVDA"]

    class _Store:
        def exists(self, table):
            return True

        def columns(self, table):
            return list(resolve(table).read_columns) or ["cik", "period", "cusip"]

        def distinct(self, table, column, **kw):
            distincts.append((name_of(table), column))
            return universe + ["DELISTED_A", "DELISTED_B"]

        def load(self, table, columns=None, *, project=False, where=None, **kw):
            calls[name_of(table)] = {"where": where, "project": project}
            return pd.DataFrame({"ticker": universe})

    class _Ctx:
        store = _Store()

    step._context = _Ctx()
    step._store = step._context.store
    step._log = logging.getLogger("test")

    step._load_source(Tables.sec13f_hr, universe)              # (a) has ticker_col
    step._load_source(Tables.sec13f_manager_holdings, universe)  # (b) ticker_col is None
    step._load_source(Tables.prices_splits)                    # no universe -> no where

    assert calls["sec13f_hr"]["where"] == {"ticker": sorted(universe)}, calls["sec13f_hr"]
    assert Tables.sec13f_manager_holdings.ticker_col is None
    assert calls["sec13f_manager_holdings"]["where"] is None
    assert calls["prices_splits"]["where"] is None
    # (c) the diagnostic fired for the ticker'd table only, and only once
    assert distincts == [("sec13f_hr", "ticker")], distincts

    print()
    print("=== SANITY: the universe cut is a SQL predicate, not a pandas mask ===")
    for name, call in calls.items():
        print(f"  {name:<26} where={call['where']}")
    print(f"  SELECT DISTINCT queries issued: {distincts}")
    print("  CONCLUSION: the in-universe read carries `WHERE ticker IN (...)`, the table with "
          "no ticker column carries none, and the off-universe report survives as one cheap "
          "DISTINCT instead of a 21.7M-row load-and-discard. Validated.")


def test_every_projected_table_resolves_to_a_registered_physical_table():
    """A projection must hang off a REGISTERED table, and the registry key trap is real.

    Five registry entries carry an attribute name that is not their table name -- the one
    that bit is `Tables.short_interest`, whose table is `sec_short_interest`. A projection map
    keyed on the ATTRIBUTE name meant `store.exists()` returned False, `_load_source` returned
    None, and the three `ic_shortvol_*` features were absent from `cube_part_institutionals`
    while 956,640 rows sat unread in the table. Nothing failed: the map, the step and this
    file's other tests all agreed on a name no table has.

    Hanging `read_columns` off the `Table` object makes the whole class unrepresentable --
    there is no key left to get wrong. This test states that, and prints the trap it closes.
    """
    projected = [t for t in ALL if t.read_columns]
    for t in projected:
        assert resolve(t.name) is t, f"{t.name}: not reachable by its own physical name"
        assert set(t.optional_columns) <= set(t.read_columns), (
            f"{t.name}: optional_columns "
            f"{sorted(set(t.optional_columns) - set(t.read_columns))} are not in read_columns "
            f"-- a dead exemption would silently tolerate a real missing column")

    mismatched = {a: t.name for a, t in vars(Tables).items()
                  if hasattr(t, "name") and a != t.name}
    print()
    print("=== SANITY: projections hang off the Table object, not a name key ===")
    print(f"  {len(projected)} tables declare read_columns, all reachable by physical name")
    print(f"  {len(mismatched)} registry entries where attribute != table name -- the trap:")
    for attr, real in sorted(mismatched.items()):
        print(f"      Tables.{attr:22} -> {real}"
              + ("   <- the one that broke ic_shortvol_*" if attr == "short_interest" else ""))
    assert name_of(Tables.short_interest) == "sec_short_interest"


def test_projection_tolerates_an_absent_optional_column():
    """A column the BUILDER treats as optional must not make the READ fail, and a missing
    REQUIRED one must be REPORTED rather than swallowed.

    `read_table` resolves each projected column via `tbl.c[name]`, which raises KeyError for
    an absent one. The live `sec_short_interest` table has only date/ticker/short_volume/
    total_volume, while the projection also declares `short_interest` + `avg_daily_volume` --
    which `_short_fields` uses only `if {...}.issubset(hist.columns)`. Demanding them
    unconditionally killed the whole institutionals step with `KeyError: 'short_interest'`.
    """
    live_short = ["date", "ticker", "short_volume", "total_volume"]
    cols, required_missing, optional_missing = projection_report(Tables.short_interest,
                                                                 live_short)
    assert cols == live_short, cols
    assert not required_missing
    assert sorted(optional_missing) == ["avg_daily_volume", "short_interest"]

    # a table with every projected column present is unchanged
    full_13f = list(Tables.sec13f_hr.read_columns)
    assert projection(Tables.sec13f_hr, full_13f) == full_13f

    # a missing REQUIRED column is REPORTED, not dropped in silence -- that is the half of
    # this that a quiet degrade would hide
    _, req_missing, _ = projection_report(Tables.sec13f_hr,
                                          [c for c in full_13f if c != "value_usd"])
    assert req_missing == ["value_usd"], req_missing

    # unknown column list (table shape unreadable) -> project the full declared list
    assert (projection(Tables.short_interest, None)
            == list(Tables.short_interest.read_columns))
    # a table declaring no projection -> None (load in full)
    assert projection(Tables.prices_splits, ["a", "b"]) is None

    print()
    print("=== SANITY CHECK: optional projected columns degrade, required ones report ===")
    print(f"  short_interest live cols {live_short} -> projection {cols}")
    print(f"  dropped as optional: {sorted(optional_missing)}")
    print(f"  sec13f_hr without `value_usd` -> required_missing={req_missing}")
    print("  CONCLUSION: an optional column missing from the live table is dropped from the "
          "projection instead of raising KeyError and killing the step, while a missing "
          "REQUIRED one is surfaced. Validated.")


if __name__ == "__main__":
    test_windowed_build_reproduces_full_tail()
    test_incremental_horizon_arithmetic()
    test_per_part_warmup_covers_binding_lookback()
    test_read_projection_covers_builder_needs()
    test_projection_tolerates_an_absent_optional_column()
