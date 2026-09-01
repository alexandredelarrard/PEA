"""Roster-wide regression test for the Sharadar TTM window
(src/data_extract/utils/fundamentals_sharadar/build_ttm.py).

Separate from `test_sharadar_field_map.py` because it asks a different question. That file
proves the window ARITHMETIC on synthetic known truth -- four quarters sum to a year, a gap
nulls, one ticker never borrows another's. This file proves the window fires on the RIGHT ROWS
across the whole live roster, which no fixture can answer: the two defects it guards against
were both invisible to every named-ticker spot check, because a spot check that passes on
eleven tickers says nothing about the other 305.

The instrument is an INDEPENDENT recomputation of wholeness by SET MEMBERSHIP, asserted to
agree with `build_ttm`'s own shift-and-roll answer for every ticker. Set membership is blind to
exactly the two things that were wrong -- a repeated quarter does not change a set, and an
absolute date drift is not a set operation -- which is what makes it a cross-check rather than
a restatement of the code under test.

Reads `sharadar_fundamentals` once, module-scoped, and shares one `build_ttm` result across
all three tests: the ARQ table is ~51.8k rows x ~112 columns and rebuilding per ticker would
cost minutes for no extra coverage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals_sharadar.build_ttm import (
    ARQ, _one_row_per_quarter, build_ttm)
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    TranslationReport, load_field_map, translate)
from src.data_store.schema import Tables
from src.utils.quarters import quarter_ordinal

CONFIG_DIR = Path("./configs")

#: The duration column the wholeness comparison is made on. `totalRevenue` maps from
#: `revenue`, which carries NO zero rule -- so a NULL here is the window refusing, never the
#: zero rule nulling a cell underneath it. Every other duration candidate (`interestExpense`,
#: `inventory`) is zero-ruled `null` and would conflate the two causes.
FIELD = "totalRevenue"

#: The four tickers where TWO REAL quarters normalise onto ONE `calendardate`. The observed
#: SET collapses them into a single ordinal and so claims a whole window where the shift
#: arithmetic correctly refuses -- the predicate's one blind spot, carved out by name rather
#: than absorbed by a tolerance. Asserted to be EXACTLY these four, so a fifth such ticker
#: appearing in a later vendor drop fails this test instead of vanishing into the carve-out.
CLASS_A_TICKERS: frozenset[str] = frozenset({"BBY", "GPN", "OKE", "KR"})

#: A floor, not an equality -- the roster drifts as S&P 500 membership changes. Measured
#: 2026-08-31: 49,280 non-null `FIELD` rows over 489 tickers.
#:
#: ⚠ That is deliberately SMALLER than the 49,500 windows the contiguity check accepts. The two
#: count different things: 49,500 windows are four contiguous quarters, and 220 of them hold a
#: quarter whose `revenue` is NULL (399 ARQ rows carry none), which `rolling(4)` nulls by
#: contract. This floor tracks published VALUES, so it is the 49,280.
MIN_WHOLE_TTM_ROWS = 49_200

#: `(ticker, calendardate)` pairs that legitimately survive de-duplication -- one per class-A
#: collision group. There are 7 groups (BBY 2012-03-31, GPN 2016-12-31, KR 1995/96/97/98-12-31,
#: OKE 1999-12-31) and each holds exactly 2 distinct `reportperiod`s, so 7 rows survive. KR's
#: 1998 group is 3 ROWS over those 2 periods -- an amendment on top of a collision -- which is
#: why the raw extra-row count is 8 and the post-dedup survivor count is 7.
#:
#: Dedup keys on `reportperiod`, so these are kept on purpose: keying on `calendardate` would
#: DELETE a real quarter from each of these 7.
EXPECTED_SURVIVING_CALENDAR_DUPES = 7


# --------------------------------------------------------------------------- #
# fixtures -- one read, one build, shared                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def field_map():
    return load_field_map(str(CONFIG_DIR))


@pytest.fixture(scope="module")
def context():
    from src.context import get_config_context
    try:
        _, ctx = get_config_context(str(CONFIG_DIR), use_cache=False, save=False)
        with ctx.store.engine.connect():
            pass
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"context/database unavailable ({type(exc).__name__}: {exc})")
    if ctx.store.row_count(Tables.sharadar_fundamentals) == 0:
        pytest.skip(f"{Tables.sharadar_fundamentals} is empty -- run fundamentals-sharadar")
    return ctx


@pytest.fixture(scope="module")
def roster(context, field_map):
    """`(vendor_arq, discrete, built)` for the WHOLE live roster, computed once.

    `discrete` is `build_ttm`'s own INPUT -- the translated per-quarter frame, where `FIELD`
    still holds a single quarter rather than a trailing twelve. The expected set is measured
    there, not on `built`, and that distinction is the whole point: a row the 45-day gate
    deleted is absent from `built` entirely, so a predicate computed on the output would agree
    with the defect by construction and AVGO's total erasure would read as a pass.

    `actions` is passed so the build is the production one; the share block is irrelevant to
    `totalRevenue`, but running a different code path than production would weaken the guard.
    """
    vendor = context.store.load(Tables.sharadar_fundamentals, project=True)
    vendor = vendor[vendor["dimension"] == ARQ].copy()
    actions = context.store.load(Tables.sharadar_actions, project=True)
    report = TranslationReport()
    discrete = translate(vendor, field_map, report=report)
    built = build_ttm(discrete, field_map, actions=actions, report=report)
    built["_ordinal"] = quarter_ordinal(built["calendardate"])
    return vendor, discrete, built


def _expected_whole(group: pd.DataFrame) -> set[int]:
    """Wholeness by SET MEMBERSHIP -- the independent predicate, over DISCRETE quarters.

    A row at quarter ordinal `q` ends a trailing twelve iff all four of `{q-3, q-2, q-1, q}`
    are USABLE for that ticker: present in the observed set AND carrying a value, since
    `rolling(4)` nulls a window holding a NaN and that is the contract.

    No shift, no sort, no rolling window -- so it shares no arithmetic with `_window_is_whole`
    and cannot fail in the same direction. What is cross-checked here is the WINDOW, and only
    the window.

    `_one_row_per_quarter` is applied first as a stated PRECONDITION rather than reimplemented:
    which of two filings of one quarter supplies the value is a separate contract with its own
    three unit tests in `test_sharadar_field_map.py`, and folding it in here would make this
    test fail for a reason that is not a window defect. TMUS is the case that forces the point
    -- its 2006-Q4 is filed twice, on 2007-03-30 and 2007-04-16, and the EARLIEST of the two
    carries no `revenue` at all, so the quarter is correctly unusable even though a value for
    it exists in the table.
    """
    deduped = _one_row_per_quarter(group)
    ordinals = quarter_ordinal(deduped["calendardate"])
    usable = {int(q) for q, ok in zip(ordinals, deduped[FIELD].notna())
              if pd.notna(q) and ok}
    return {q for q in usable if {q - 3, q - 2, q - 1, q} <= usable}


# --------------------------------------------------------------------------- #
# the roster-wide assertion                                                    #
# --------------------------------------------------------------------------- #
def test_no_window_is_nulled_except_by_a_genuine_missing_quarter(roster):
    """`build_ttm` nulls a trailing twelve ONLY where a quarter is genuinely absent.

    WATCHED TO FAIL on the pre-fix code, which is the only reason this is known to test
    anything. It disagreed on **274 tickers**, and the whole-window total was 48,015 against
    49,280 now. Two causes, both fixed in `build_ttm`:

      * Sharadar's ARQ grain is one row per FILING, so an amendment repeated a quarter and
        `ordinal.shift(3)` read the repeat as a gap -- 543 duplicate groups over 316 tickers,
        each nulling three trailing twelves. Fewer than 316 tickers show up here because a
        duplicate inside a ticker's first three quarters, or beside an already-missing one,
        nulls nothing that was not already null.
      * a 45-day cap on `|calendardate - reportperiod|` deleted 239 correct rows from four
        off-calendar filers outright -- every one of AVGO's, leaving it absent from
        `fundamentals_history` entirely, with KR and AZO at 100% NULL revenue and COST 97.6%.
    """
    _, discrete, built = roster
    actual_by_ticker = {
        ticker: {int(q) for q in group.loc[group[FIELD].notna(), "_ordinal"].dropna()}
        for ticker, group in built.groupby("ticker", sort=False)}
    disagree: dict[str, tuple[int, int]] = {}
    for ticker, group in discrete.groupby("ticker", sort=True):
        if ticker in CLASS_A_TICKERS:
            continue
        actual = actual_by_ticker.get(ticker, set())
        expected = _expected_whole(group)
        if actual != expected:
            disagree[ticker] = (len(expected - actual), len(actual - expected))

    print(f"\n=== SANITY CHECK: roster-wide TTM wholeness ({FIELD}) ===")
    print(f"  tickers compared      : {discrete['ticker'].nunique() - len(CLASS_A_TICKERS)}")
    print(f"  whole windows built   : {int(built[FIELD].notna().sum()):,}")
    print(f"  disagreeing tickers   : {len(disagree)}")
    if disagree:
        for ticker, (missing, extra) in sorted(disagree.items())[:25]:
            print(f"    {ticker:<6} {missing:>4} window(s) wrongly NULLED, "
                  f"{extra:>4} wrongly published")
        if len(disagree) > 25:
            print(f"    ... and {len(disagree) - 25} more")
    else:
        print("  OK: every ticker's built wholeness equals independent set membership")
    print("  -> the window fires on exactly the rows that have four usable quarters.")
    assert not disagree, (f"{len(disagree)} ticker(s) disagree with set membership: "
                          f"{sorted(disagree)}")


def test_class_a_carve_out_is_exactly_the_four_known_tickers(roster):
    """Two REAL quarters on one `calendardate` -- the carve-out is pinned, not open-ended.

    These are the only tickers where the set predicate is WRONG and the shift is right, so the
    carve-out above hides a real disagreement. It is bounded here: a fifth such ticker in a
    later vendor drop fails this test rather than being silently absorbed.
    """
    vendor, _, _ = roster
    counts = vendor.groupby(["ticker", "calendardate"])["reportperiod"].nunique()
    found = set(counts[counts > 1].index.get_level_values("ticker").unique())
    print("\n=== SANITY CHECK: class-A calendardate collisions ===")
    print(f"  tickers with 2 real quarters on 1 calendardate : {sorted(found)}")
    print(f"  pinned carve-out                               : {sorted(CLASS_A_TICKERS)}")
    print("  -> the predicate's blind spot is bounded and named.")
    assert found == set(CLASS_A_TICKERS)


def test_ttm_coverage_does_not_regress(roster):
    """Aggregate floors, so a change that quietly drops rows is caught even if the set
    predicate still agrees with it.

    Four independent floors, each pinning a distinct failure:
      * total whole windows -- the headline coverage
      * no ticker with ZERO whole windows -- the assertion AVGO failed before the gate went
      * no duplicate `(ticker, reportperiod)` reaching the window maths -- the dedup itself
      * exactly 8 surviving `(ticker, calendardate)` duplicates -- proof the dedup keyed on
        `reportperiod` and so did NOT delete a real class-A quarter
    """
    _, discrete, built = roster
    # Reindexed over the INPUT's tickers, so a ticker the build dropped entirely counts as 0
    # rather than vanishing from the index -- that erasure is exactly what AVGO suffered, and
    # a `groupby` on the output alone cannot see it.
    per_ticker = (built.groupby("ticker")[FIELD].apply(lambda s: int(s.notna().sum()))
                  .reindex(sorted(discrete["ticker"].unique()), fill_value=0))
    empty = sorted(per_ticker[per_ticker == 0].index)
    reportperiod_dupes = int(built.duplicated(["ticker", "reportperiod"]).sum())
    calendar_dupes = int(built.duplicated(["ticker", "calendardate"]).sum())
    named = ["AVGO", "KR", "AZO", "COST", "GPN", "GOOGL", "IBM", "KO", "AAPL", "BBY", "OKE"]

    print("\n=== SANITY CHECK: TTM coverage floors ===")
    print(f"  total whole TTM rows          : {int(per_ticker.sum()):,} "
          f"(floor {MIN_WHOLE_TTM_ROWS:,})")
    print(f"  tickers with 0 whole windows  : {len(empty)} {empty or ''}")
    print(f"  duplicate (ticker, reportperiod) reaching the window : {reportperiod_dupes}")
    print(f"  surviving (ticker, calendardate) duplicates          : {calendar_dupes} "
          f"(expected {EXPECTED_SURVIVING_CALENDAR_DUPES}, the class-A rows)")
    print("  per-ticker whole windows, the named 11:")
    for ticker in named:
        print(f"    {ticker:<6} {per_ticker.get(ticker, 0):>4}")
    print("  -> coverage is at or above every measured floor.")

    assert int(per_ticker.sum()) >= MIN_WHOLE_TTM_ROWS
    assert not empty, f"ticker(s) with no trailing twelve at all: {empty}"
    assert reportperiod_dupes == 0
    assert calendar_dupes == EXPECTED_SURVIVING_CALENDAR_DUPES
