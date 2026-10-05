"""Tests for the Sharadar acceptance gates
(src/data_extract/utils/fundamentals_sharadar/diagnostics.py).

REAL data, from POSTGRES -- the whole point of the phase is that the gates are measured
against what was stored, not against what the API said (D29). Every test prints its
conclusion, including one that exists specifically to make an ABSENCE visible: D19 is
unverified until the stored roster covers a CIK-cutover ticker. A gap nobody printed is a gap
nobody knows about.

The gates are PURE functions of frames, so the fixture performs the one projected read the
production path performs and every test shares it. Nothing here calls `run_diagnostics`, so a
test run can never overwrite the report.

"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.constants.constants import (
    SHARADAR_CONFIG_SUBDIR,
    SHARADAR_ZERO_FILLED_FIELDS,
    SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_extract.utils.common.registrant import Registrant, load_registrants
from src.data_extract.utils.fundamentals_sharadar.diagnostics import (
    confirm_sign_conventions,
    cross_check_shares,
    gate_completeness,
    gate_zero_fill,
    load_sec,
    load_sharadar,
)
from src.data_store.schema import Tables
from src.utils import cutover_continuity as cc

CONFIG_DIR = Path("./configs")

#: Ceiling on the share of rows carrying a POSITIVE `capex`. Measured at 0.97% (13 of 1,346,
#: 11 of them GS) on 2026-08-26. Bounded rather than zero because the plan's universal claim
#: was taken from AAPL alone and is false; see `test_sign_conventions_hold`.
MAX_POSITIVE_CAPEX_RATE = 0.05

#: Ratio span (max/min of `sharesbas / sharesOutstanding` across a ticker's dates) above which
#: the history has been retroactively re-based. A real share-class or reporting difference is
#: a LEVEL shift and holds flat over time; only a split moves the ratio within one ticker.
SPLIT_RATIO_SPAN = 1.5

#: The only tickers allowed a `sharefactor != 1.0`. Sharadar documents `sharefactor` as a
#: multiplicant in its `marketcap` calculation that adjusts for DUAL SHARE CLASSES, and these
#: are the two dual-class names in the universe. Measured 2026-09-07 over all 117,391 rows:
#: 447 carry `sharefactor != 1` and every one belongs to BRK-B (276) or V (171); every
#: split-adjusted name (NVDA, WMT, AMZN) is still exactly 1.0.
#:
#: Lives here, not in `constants.py`: its only consumer is this test
#: (the constants-placement rule is 2+ non-test `src/` consumers).
DUAL_CLASS_SHAREFACTOR_TICKERS = frozenset({"BRK-B", "V"})


@pytest.fixture(scope="module")
def context():
    """A real Context (DB + .env), skipping rather than erroring when either is missing."""
    from src.context import get_config_context

    try:
        _, ctx = get_config_context(str(CONFIG_DIR), use_cache=False, save=False)
        with ctx.store.engine.connect():
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"context/database unavailable ({type(exc).__name__}: {exc})")
    if ctx.store.row_count(Tables.sharadar_fundamentals) == 0:
        pytest.skip(f"{Tables.sharadar_fundamentals} is empty -- run fundamentals-sharadar")
    return ctx


@pytest.fixture(scope="module")
def frames(context):
    """The ONE projected read the gates run off, sliced per dimension.

    Mirrors `run_diagnostics`: the table is 112 columns x 3 dimensions, so reading it once per
    test would re-read the widest extract table in the schema five times.
    """
    frame = load_sharadar(context, None)
    by_dimension = {dim: group for dim, group in frame.groupby("dimension", sort=False)}
    arq = by_dimension.get("ARQ", frame.iloc[:0])
    if arq.empty:
        pytest.skip(f"{Tables.sharadar_fundamentals} has no ARQ rows")
    return SimpleNamespace(
        all=frame, arq=arq, art=by_dimension.get("ART", frame.iloc[:0]), sec=load_sec(context, sorted(arq["ticker"].astype(str).unique()))
    )


# --------------------------------------------------------------------------- #
# Gate 1 -- completeness                                                       #
# --------------------------------------------------------------------------- #
def test_completeness_gate_runs(frames):
    """Every stored ticker is measured against ITS OWN observed window, so the only thing this
    can report is a hole -- a late start (an IPO, or a shallower entitlement) is not a gap."""
    frame = gate_completeness(frames.arq)
    with_gaps = frame[frame["n_missing"] > 0]

    print("\n=== SANITY CHECK: gate 1, completeness ===")
    print(f"  tickers measured    : {len(frame)}")
    print(f"  quarters stored     : {int(frame['n_quarters'].sum())} across {frame['first_quarter'].min()}..{frame['last_quarter'].max()}")
    print(f"  tickers with a gap  : {len(with_gaps)}")
    print(f"  missing quarters    : {int(frame['n_missing'].sum())}")
    print(f"  duplicate quarters  : {int(frame['n_duplicate_quarters'].sum())}")
    for row in with_gaps.itertuples(index=False):
        print(f"    {row.ticker:6s} {row.n_missing} missing -> {row.missing_quarters}")

    assert not frame.empty, "the completeness gate measured no ticker at all"
    assert (frame["n_quarters"] > 0).all(), "a ticker was measured with zero quarters"
    print(f"  OK: {len(frame)} tickers measured, {int(frame['n_missing'].sum())} missing quarter(s) found.")


# --------------------------------------------------------------------------- #
# Sign conventions -- the stop condition for the field map                     #
# --------------------------------------------------------------------------- #
def test_sign_conventions_hold(frames):
    """`fcf == ncfo + capex` to the cent, and `capex <= 0` on all but a bounded few.

    !! The plan asserted `capex <= 0` UNIVERSALLY, off a measurement taken on AAPL alone. It is
    not universal: measured over all three stored dimensions, a small number of rows carry a
    POSITIVE capex, concentrated in GS. So this test pins the two properties differently, and
    deliberately:

      * `fcf == ncfo + capex` is asserted STRICTLY, because it did hold on every row and
        `freeCashflow <- fcf` depends on it entirely;
      * `capex <= 0` is asserted as a BOUNDED exception rate, with every offending row printed.
        Asserting the universal would be asserting something false; asserting nothing would let
        the rate grow silently on a wider roster. The bound is what makes the guarded sign flip
        in `field_map._negate_if_non_positive` safe.
    """
    result = confirm_sign_conventions(frames.all)
    rate = result["capex_positive_total"] / max(result["capex_rows_total"], 1)

    print("\n=== SANITY CHECK: sign conventions, from stored data ===")
    for dimension, block in result["dimensions"].items():
        print(
            f"  {dimension}: capex rows={block['capex_rows']}, "
            f"positive={block['capex_positive']} (max {block['capex_max']:,.0f}) | "
            f"fcf rows={block['fcf_rows']}, "
            f"max |fcf-(ncfo+capex)|={block['fcf_max_abs_residual']:,.4f} "
            f"at {block['fcf_worst_row']}, violations={block['fcf_violations']}"
        )
    print(f"  fcf == ncfo + capex   : {result['fcf_identity_holds']}  <- asserted strictly")
    print(
        f"  capex <= 0 throughout : {result['capex_sign_holds']}  "
        f"({result['capex_positive_total']} of {result['capex_rows_total']} rows positive "
        f"= {rate:.2%}, on {result['capex_positive_tickers']})"
    )
    for row in result["capex_positive_rows"].head(20).itertuples(index=False):
        print(f"    +capex  {row.ticker:5s} {row.dimension} {row.fiscalperiod} {pd.Timestamp(row.date).date()}  {row.capex:>16,.0f}")

    assert result["fcf_identity_holds"], "fcf is not ncfo + capex -- `freeCashflow <- fcf` needs a reconstruction after all"
    assert rate < MAX_POSITIVE_CAPEX_RATE, (
        f"positive-capex rows are {rate:.2%} of the table, over the {MAX_POSITIVE_CAPEX_RATE:.0%} "
        f"bound. A guarded sign flip is no longer good enough -- the map needs a real "
        f"capex mapping, not an exception list"
    )
    print(
        f"  OK: fcf identity exact; capex sign violated on {rate:.2%} of rows, which the map "
        f"handles by NULLing the exceptions rather than flipping them."
    )


# --------------------------------------------------------------------------- #
# The zero rule covers every flagged field                                     #
# --------------------------------------------------------------------------- #
def test_zero_rules_cover_every_flagged_field(frames):
    """`field_map` reads `sharadar_zero_rules.json` and fails loudly on a field with no entry,
    so the file must cover all 41 documented zero-filled fields -- no defaults, no omissions."""
    path = CONFIG_DIR / SHARADAR_CONFIG_SUBDIR / SHARADAR_ZERO_RULES_FILENAME
    if not path.exists():
        pytest.skip(f"{path} is missing -- the transform cannot run without it")
    blob = json.loads(path.read_text(encoding="utf-8"))
    rules = {k: v for k, v in blob.items() if not k.startswith("_")}
    missing = sorted(SHARADAR_ZERO_FILLED_FIELDS - set(rules))
    extra = sorted(set(rules) - SHARADAR_ZERO_FILLED_FIELDS)
    bad_rule = {k: v.get("rule") for k, v in rules.items() if v.get("rule") not in ("null", "keep")}
    no_reason = sorted(k for k, v in rules.items() if not str(v.get("reason", "")).strip())
    measured = gate_zero_fill(frames.arq, frames.art, frames.sec)
    nulled = sorted(k for k, v in rules.items() if v["rule"] == "null")

    print("\n=== SANITY CHECK: the zero rule covers every flagged field ===")
    print(f"  fields in SHARADAR_ZERO_FILLED_FIELDS : {len(SHARADAR_ZERO_FILLED_FIELDS)}")
    print(f"  entries in {path.name:26s}: {len(rules)}")
    print(f"  missing entries    : {missing or 'none'}")
    print(f"  unknown entries    : {extra or 'none'}")
    print(f"  entries with no reason : {no_reason or 'none'}")
    print(f"  rule=null          : {len(nulled)} -> {nulled or 'none'}")
    removed = int(measured[measured["field"].isin(nulled)]["n_zero"].sum())
    cells = int(measured[measured["n_rows"] > 0]["n_rows"].sum())
    print(f"  cells 0 -> NULL    : {removed:,} of {cells:,} measured ({removed / cells:.2%})")

    approved = blob.get("_APPROVED")
    print(f"  _APPROVED block    : {'yes, on ' + approved['on'] if approved else 'NO'}")
    if approved:
        print(f"    approved scope   : {approved.get('scope')}")

    assert approved and approved.get("on"), (
        "the rule file has no `_APPROVED` block. A regenerated PROPOSAL is byte-identical to a "
        "reviewed decision, so without this marker `human-approved` is only a claim in a "
        "docstring -- and the one thing this file exists to guarantee is that somebody looked "
        "at the `null` rules before they nulled real cells"
    )
    assert not missing, f"the transform would fail loudly on {len(missing)} field(s): {missing}"
    assert not extra, f"the rule file names fields Sharadar does not zero-fill: {extra}"
    assert not bad_rule, f"only 'null' and 'keep' are valid rules, found: {bad_rule}"
    assert not no_reason, f"every rule needs a stated reason, missing on: {no_reason}"
    print(f"  OK: all {len(rules)} fields ruled, {len(nulled)} nulled.")


# --------------------------------------------------------------------------- #
# `sharesbas` is NOT point-in-time -- the finding the field map must not skip   #
# --------------------------------------------------------------------------- #
def test_sharesbas_is_split_adjusted_not_point_in_time(frames):
    """Not in the plan's test list, and added because the cross-check answered a DIFFERENT
    question than the one D-decision `sharesOutstanding <- sharesbas` asked.

    The decision asked whether `sharesbas` sums share classes. It does not -- 12 of 14
    overlapping tickers sit at a ratio of exactly 1.0 against the SEC cover-page count. What
    the measurement found instead is that Sharadar restates the WHOLE HISTORY onto the current
    split basis: NVDA's 2021 rows carry ~25bn shares against the ~2.5bn actually outstanding
    before its June 2024 10-for-1. `sharefactor` is 1.0 on every one of those rows.

    That makes `sharesbas` unusable as a point-in-time count without de-adjustment, and this
    test exists so it cannot be mapped as one by accident.
    """
    frame = cross_check_shares(frames.arq, frames.sec)
    if frame.empty:
        pytest.skip("no overlapping ticker has both a sharesbas and a SEC sharesOutstanding")
    split = frame[frame["ratio_span"] >= SPLIT_RATIO_SPAN]
    agree = frame[(frame["median_ratio"] - 1).abs() <= 0.05]

    print("\n=== SANITY CHECK: sharesbas vs the SEC cover-page count ===")
    print(f"  tickers compared            : {len(frame)}")
    print(f"  median ratio == 1.0         : {len(agree)}  <- so NOT a share-class problem")
    print(f"  ratio_span >= {SPLIT_RATIO_SPAN}          : {len(split)}  <- SPLIT-ADJUSTED history")
    for row in frame.head(30).itertuples(index=False):
        print(
            f"    {row.ticker:5s} n={row.n_dates:3d} median={row.median_ratio:8.4f} "
            f"span={row.ratio_span:7.4f} sharefactor={row.median_sharefactor:.1f}  "
            f"{row.verdict}"
        )
    if len(split):
        print(f"  => `sharesbas` is NOT point-in-time for {', '.join(split['ticker'].head(10))}. Multiplying it by an as-filed price")
        print("     yields a market cap wrong by the split factor for every pre-split date.")
        print("     `build_ttm` de-adjusts using sharadar_actions, which carries the splits.")

    assert len(agree) >= len(frame) - len(split), (
        "a ticker disagrees with the SEC cover-page count for a reason that is NOT a split -- "
        "that would be the share-class summing question D-decision actually asked about"
    )
    # `sharefactor` must stay 1.0 on every SINGLE-CLASS name, because that is the assumption
    # the split de-adjustment rests on: if it started carrying the split factor, de-adjusting
    # with `sharadar_actions` on top of it would double-count.
    #
    # ⚠ It is NOT uniformly 1.0 across the table, and asserting that it was is what made this
    # test fail on the paid full-universe pull. Re-measured 2026-09-07: 447 of 117,391 rows
    # carry `sharefactor != 1` and they belong to exactly two DUAL-CLASS tickers --
    # BRK-B (1493.472 in 1996 -> 1.52 in 2026) and V (0.800 -> 1.099). It DECLINES as
    # `sharesbas` grows, which is Sharadar's documented dual-class multiplicant for
    # `marketcap`, not a split factor; every split name (NVDA, WMT, AMZN) is still exactly 1.0.
    #
    # So the exemption is by NAME and stays narrow on purpose: a third ticker appearing here,
    # or a split-shaped factor on a single-class name, must still fail.
    off = frame[frame["median_sharefactor"] != 1.0]
    unexpected = sorted(set(off["ticker"]) - set(DUAL_CLASS_SHAREFACTOR_TICKERS))
    if len(off):
        print(
            f"  sharefactor != 1.0 on {len(off)} ticker(s): " + ", ".join(f"{r.ticker}={r.median_sharefactor:g}" for r in off.itertuples(index=False))
        )
    assert not unexpected, (
        f"`sharefactor` is no longer 1.0 for {unexpected}, which are not known dual-class "
        f"names -- it may now encode the split adjustment, which would change how the "
        f"de-adjustment has to work (de-adjusting on top of it would double-count)"
    )
    print(f"  OK: {len(agree)}/{len(frame)} agree on level; {len(split)} carry a split-adjusted history that `build_ttm` de-adjusts.")


# --------------------------------------------------------------------------- #
# D19 -- verified the moment a cutover ticker is stored                        #
# --------------------------------------------------------------------------- #
def _cutover_registrants() -> dict[str, Registrant]:
    """`{ticker: registrant chain}` for every register ticker with at least one boundary.

    Read through `load_registrants` rather than off the JSON: one parser over one file is
    what stops a schema change in `configs/sec/` from breaking a test in this package.
    """
    return {t: r for t, r in load_registrants(str(CONFIG_DIR)).items() if r.boundaries}


def _windows(registrant: Registrant) -> tuple[cc.CikWindow, ...]:
    return tuple(cc.CikWindow(s.cik, s.valid_from, s.valid_to) for s in registrant.segments)


def test_cik_cutover_continuity(context, frames):
    """D19 joins Sharadar to the SEC layer on `ticker`, and a CIK cutover is where that join
    can lose half a history. Every vendor quarter missing within `CUTOVER_WINDOW_YEARS` of any
    boundary of the series as merged (predecessor vendor series applied) is collected and
    classified before anything is asserted; the test fails once, on the full table, unless each
    one is a recorded `vendor_coverage_gap` of `configs/sec/vendor_coverage_exceptions.json`.
    """
    registrants = _cutover_registrants()
    stored = set(frames.arq["ticker"].astype(str).unique())
    testable = sorted(set(registrants) & stored)

    print("\n=== SANITY CHECK: D19, CIK-cutover continuity ===")
    print(f"  cutover tickers in the register : {sorted(registrants)}")
    print(f"  tickers stored in {Tables.sharadar_fundamentals} : {len(stored)}")
    print(f"  testable (register and stored)    : {testable or 'NONE'}")
    if not testable:
        print("  => D19 IS UNVERIFIED. None of the register's cutover tickers has been")
        print("     extracted yet. This test runs as soon as one of them is stored.")
        pytest.skip(f"no cutover ticker stored: register={sorted(registrants)} vs {len(stored)} stored tickers. D19 UNVERIFIED.")

    windows = {ticker: _windows(registrants[ticker]) for ticker in testable}
    for ticker in testable:
        print(f"    {ticker}: boundaries {[str(b.date()) for b in registrants[ticker].boundaries]}")
    vendor_tickers = context.store.load(Tables.sharadar_tickers, columns=["ticker", "secfilings", "lastquarter"], optional=True)
    series = cc.predecessor_series(vendor_tickers if vendor_tickers is not None else pd.DataFrame(), windows, registrants)
    merged, events = cc.apply_predecessor_series(frames.arq, frames.arq, series)
    print(
        f"  predecessor vendor series: {[f'{s.ticker}<-{s.vendor_ticker}' for s in series] or 'none'}; stored: {sorted(set(events['vendor_ticker'])) or 'none'}"
    )
    print(f"  predecessor quarters: {events['event'].value_counts().to_dict() if not events.empty else 'none'}")

    columns = list(cc.FILING_COLUMNS)
    facts = context.store.load(Tables.fundamentals_facts, columns=columns, where={"ticker": testable, "form": list(cc.PERIODIC_FORMS)}, optional=True)
    exceptions = cc.load_vendor_exceptions(CONFIG_DIR)
    report = cc.assess_continuity(merged, windows, cc.prepare_filings(facts), exceptions)
    table = [(r.ticker, r.quarter, r.boundary, r["class"], bool(r.explained), r.evidence) for _, r in report.table.iterrows()]
    unobserved = list(report.unobserved)
    healed = [f"{r.ticker} {r.quarter}" for r in report.healed]

    print(f"  discontinuities: {len(table)} ({sum(row[4] for row in table)} explained)")
    print(f"    {'ticker':6s} {'quarter':7s} {'boundary':10s} {'class':20s} {'ok':3s} evidence")
    for ticker, quarter, seam, cls, explained, evidence in table:
        print(f"    {ticker:6s} {quarter:7s} {seam:10s} {cls:20s} {'yes' if explained else 'NO':3s} {evidence}")
    unexplained = [f"{ticker} {quarter} {cls}" for ticker, quarter, _, cls, explained, _ in table if not explained]
    print(f"  unexplained: {unexplained or 'none'} | healed records: {healed or 'none'} | unobserved boundaries: {unobserved or 'none'}")

    assert not unobserved, f"no vendor quarter within {cc.CUTOVER_WINDOW_YEARS}y of these boundaries, so D19 is untested there: {unobserved}"
    assert not report.duplicated, f"{cc.EXCEPTIONS_FILE} records one (ticker, quarter) twice: {report.duplicated}"
    assert not unexplained, (
        f"{len(unexplained)} cutover discontinuities are not recorded vendor gaps: {unexplained}. "
        f"A {cc.MISSING_SEC_FILING} or {cc.INCORRECT_CIK_WINDOW} is a register or SEC-side defect; a "
        f"{cc.VENDOR_COVERAGE_GAP} needs its EDGAR accession in {cc.EXCEPTIONS_FILE}."
    )
    assert not healed, (
        f"{healed} are in {cc.EXCEPTIONS_FILE} but the quarter is present now. Delete the rows: a stale exception hides the next real hole."
    )
    print(f"  OK: {len(testable)} cutover tickers; all {len(table)} discontinuities are recorded vendor gaps with their SEC filing.")
