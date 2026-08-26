"""Tests for the Sharadar phase-2 acceptance gates
(src/data_extract/utils/fundamentals_sharadar/diagnostics.py).

REAL data, from POSTGRES -- the whole point of the phase is that the gates are measured
against what was stored, not against what the API said (D29). Every test prints its
conclusion, and two of them exist specifically to make an ABSENCE visible: one records that
the spec's acceptance check #3 carries no information, the other that D19 is unverified until
the roster widens. A gap nobody printed is a gap nobody knows about.

!! Nothing here touches `src/validate/` or the `fundamentals_check*` tables (D25). The
diagnostic writes no production data, and this module writes no files at all -- it calls the
gate functions directly rather than `run_diagnostics`, so a test run can never overwrite the
human-approved rule file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.constants.constants import (
    FUNDAMENTALS_CATALOGUE_SUBDIR, FUNDAMENTALS_CIK_CUTOVER_FILENAME,
    SHARADAR_CONFIG_SUBDIR, SHARADAR_ZERO_FILLED_FIELDS, SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_store.schema import Tables
from src.data_extract.utils.fundamentals_sharadar.diagnostics import (
    Q4_TAUTOLOGY_MAX_PCT, confirm_q4_tautology, confirm_sign_conventions, cross_check_shares,
    gate_completeness, gate_zero_fill, q4_tautology_summary,
)

CONFIG_DIR = Path("./configs")

#: Ceiling on the share of rows carrying a POSITIVE `capex`. Measured at 0.97% (13 of 1,346,
#: 11 of them GS) on 2026-08-26. Bounded rather than zero because the plan's universal claim
#: was taken from AAPL alone and is false; see `test_sign_conventions_hold`.
MAX_POSITIVE_CAPEX_RATE = 0.05

#: Floor on the share of sum ARQ-vs-ARY triples that are EXACTLY zero. Measured at 96.1%. This is
#: the number that keeps the spec's acceptance check #3 dead.
MIN_EXACT_TAUTOLOGY_SHARE = 0.90

#: Ratio span (max/min of `sharesbas / sharesOutstanding` across a ticker's dates) above which
#: the history has been retroactively re-based. A real share-class or reporting difference is
#: a LEVEL shift and holds flat over time; only a split moves the ratio within one ticker.
SPLIT_RATIO_SPAN = 1.5


@pytest.fixture(scope="module")
def context():
    """A real Context (DB + .env), skipping rather than erroring when either is missing."""
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


# --------------------------------------------------------------------------- #
# Gate 1 -- completeness                                                       #
# --------------------------------------------------------------------------- #
def test_completeness_gate_runs(context):
    """Every stored ticker is measured against ITS OWN observed window, so the only thing this
    can report is a hole -- a late start is not a gap on a 5-year entitlement."""
    frame = gate_completeness(context)
    with_gaps = frame[frame["n_missing"] > 0]

    print("\n=== SANITY CHECK: gate 1, completeness ===")
    print(f"  tickers measured    : {len(frame)}")
    print(f"  quarters stored     : {int(frame['n_quarters'].sum())} across "
          f"{frame['first_quarter'].min()}..{frame['last_quarter'].max()}")
    print(f"  tickers with a gap  : {len(with_gaps)}")
    print(f"  missing quarters    : {int(frame['n_missing'].sum())}")
    print(f"  duplicate quarters  : {int(frame['n_duplicate_quarters'].sum())}")
    for row in with_gaps.itertuples(index=False):
        print(f"    {row.ticker:6s} {row.n_missing} missing -> {row.missing_quarters}")

    assert not frame.empty, "the completeness gate measured no ticker at all"
    assert (frame["n_quarters"] > 0).all(), "a ticker was measured with zero quarters"
    print(f"  OK: {len(frame)} tickers measured, "
          f"{int(frame['n_missing'].sum())} missing quarter(s) found.")


# --------------------------------------------------------------------------- #
# Sign conventions -- the stop condition for phase 3                           #
# --------------------------------------------------------------------------- #
def test_sign_conventions_hold(context):
    """`fcf == ncfo + capex` to the cent, and `capex <= 0` on all but a bounded few.

    !! The plan asserted `capex <= 0` UNIVERSALLY, off a measurement taken on AAPL alone. It is
    not universal: measured over all three stored dimensions, a small number of rows carry a
    POSITIVE capex, concentrated in GS. So this test pins the two properties differently, and
    deliberately:

      * `fcf == ncfo + capex` is asserted STRICTLY, because it did hold on every row and
        `freeCashflow <- fcf` depends on it entirely;
      * `capex <= 0` is asserted as a BOUNDED exception rate, with every offending row printed.
        Asserting the universal would be asserting something false; asserting nothing would let
        the rate grow silently on a wider roster. The bound is what makes phase 3's guarded
        sign flip safe.
    """
    result = confirm_sign_conventions(context)
    rate = result["capex_positive_total"] / max(result["capex_rows_total"], 1)

    print("\n=== SANITY CHECK: sign conventions, from stored data ===")
    for dimension, block in result["dimensions"].items():
        print(f"  {dimension}: capex rows={block['capex_rows']}, "
              f"positive={block['capex_positive']} (max {block['capex_max']:,.0f}) | "
              f"fcf rows={block['fcf_rows']}, "
              f"max |fcf-(ncfo+capex)|={block['fcf_max_abs_residual']:,.4f} "
              f"at {block['fcf_worst_row']}, violations={block['fcf_violations']}")
    print(f"  fcf == ncfo + capex   : {result['fcf_identity_holds']}  <- asserted strictly")
    print(f"  capex <= 0 throughout : {result['capex_sign_holds']}  "
          f"({result['capex_positive_total']} of {result['capex_rows_total']} rows positive "
          f"= {rate:.2%}, on {result['capex_positive_tickers']})")
    for row in result["capex_positive_rows"].itertuples(index=False):
        print(f"    +capex  {row.ticker:5s} {row.dimension} {row.fiscalperiod} "
              f"{pd.Timestamp(row.date).date()}  {row.capex:>16,.0f}")

    assert result["fcf_identity_holds"], (
        "fcf is not ncfo + capex -- `freeCashflow <- fcf` needs a reconstruction after all")
    assert rate < MAX_POSITIVE_CAPEX_RATE, (
        f"positive-capex rows are {rate:.2%} of the table, over the {MAX_POSITIVE_CAPEX_RATE:.0%} "
        f"bound. A guarded sign flip is no longer good enough -- phase 3 needs a real "
        f"capex mapping, not an exception list")
    print(f"  OK: fcf identity exact; capex sign violated on {rate:.2%} of rows, which phase 3 "
          f"handles by NULLing the exceptions rather than flipping them.")


# --------------------------------------------------------------------------- #
# The Q4 identity -- a test that documents a DEAD check                        #
# --------------------------------------------------------------------------- #
def test_q4_identity_is_tautological(context):
    """sum ARQ == ARY holds by arithmetic, because Sharadar CONSTRUCTS Q4 as `ARY - sum (Q1..Q3)`.

    This test exists to keep the spec's acceptance check #3 dead. A check that cannot fail
    cannot inform, and the only way that stays known is to measure it once and write the number
    down where the next reader will find it.
    """
    frame = confirm_q4_tautology(context)
    if frame.empty:
        pytest.skip("no fiscal year has all four ARQ quarters AND an ARY row in this window")
    summary = q4_tautology_summary(frame)
    worst = frame.iloc[0]

    print("\n=== SANITY CHECK: the Q4 identity is a TAUTOLOGY, not a check ===")
    print(f"  (ticker, fiscal year, field) triples : {summary['n']:,}")
    print(f"  EXACTLY zero      : {summary['n_exact']:,} ({summary['share_exact']:.2%})")
    print(f"  float noise only  : {summary['n_float_noise']:,} "
          f"(0 < dev <= {Q4_TAUTOLOGY_MAX_PCT:.2%})")
    print(f"  materially off    : {summary['n_over_bar']:,}, max {summary['max_dev']:.2%}")
    print(f"  worst triple      : {worst['ticker']} FY{worst['fiscal_year']} {worst['field']}"
          f"  sum ARQ={worst['sum_arq']:,.0f} vs ARY={worst['ary']:,.0f}")
    print("  where the deviations sit:")
    for row in summary["concentration"].head(6).itertuples(index=False):
        print(f"    {row.ticker:6s} FY{row.fiscal_year}  {row.n_fields} field(s)")
    print("  => the spec's acceptance check #3 (Q4 = FY - 9M) CARRIES NO INFORMATION on this")
    print("     vendor: Sharadar builds Q4 by SUBTRACTION, so wherever the identity holds it")
    print("     holds EXACTLY -- it can never detect a bad quarter, only a restatement.")
    print("     gate_implausible_quarters replaces it (D28).")

    assert summary["share_exact"] >= MIN_EXACT_TAUTOLOGY_SHARE, (
        f"only {summary['share_exact']:.1%} of triples are exactly zero, under the "
        f"{MIN_EXACT_TAUTOLOGY_SHARE:.0%} bound. If Q4 is no longer built by subtraction, "
        f"check #3 would start carrying information and this whole finding needs redoing")
    # the residual is restatements, which cluster; drift would be spread evenly instead
    assert summary["n_over_bar"] == 0 or len(summary["concentration"]) < summary["n_over_bar"], (
        "the non-zero deviations are one-per-(ticker, year), which looks like drift rather "
        "than the restatement clustering this finding claims")
    print(f"  OK: exact on {summary['share_exact']:.2%}; the {summary['n_over_bar']} exceptions "
          f"cluster into {len(summary['concentration'])} (ticker, year) restatements.")


# --------------------------------------------------------------------------- #
# The zero rule covers every flagged field                                     #
# --------------------------------------------------------------------------- #
def test_zero_rules_cover_every_flagged_field(context):
    """Phase 3 reads `sharadar_zero_rules.json` and fails loudly on a field with no entry, so
    the file must cover all 41 documented zero-filled fields -- no defaults, no omissions."""
    path = CONFIG_DIR / SHARADAR_CONFIG_SUBDIR / SHARADAR_ZERO_RULES_FILENAME
    if not path.exists():
        pytest.skip(f"{path} not written yet -- run `data_extract sharadar-diagnostics`")
    blob = json.loads(path.read_text(encoding="utf-8"))
    rules = {k: v for k, v in blob.items() if not k.startswith("_")}
    missing = sorted(SHARADAR_ZERO_FILLED_FIELDS - set(rules))
    extra = sorted(set(rules) - SHARADAR_ZERO_FILLED_FIELDS)
    bad_rule = {k: v.get("rule") for k, v in rules.items()
                if v.get("rule") not in ("null", "keep")}
    no_reason = sorted(k for k, v in rules.items() if not str(v.get("reason", "")).strip())
    measured = gate_zero_fill(context)
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
        "at the `null` rules before they nulled real cells")
    assert not missing, f"phase 3 would fail loudly on {len(missing)} field(s): {missing}"
    assert not extra, f"the rule file names fields Sharadar does not zero-fill: {extra}"
    assert not bad_rule, f"only 'null' and 'keep' are valid rules, found: {bad_rule}"
    assert not no_reason, f"every rule needs a stated reason, missing on: {no_reason}"
    print(f"  OK: all {len(rules)} fields ruled, {len(nulled)} nulled.")


# --------------------------------------------------------------------------- #
# `sharesbas` is NOT point-in-time -- the finding phase 3 must not walk past    #
# --------------------------------------------------------------------------- #
def test_sharesbas_is_split_adjusted_not_point_in_time(context):
    """Not in the plan's test list, and added because the cross-check answered a DIFFERENT
    question than the one D-decision `sharesOutstanding <- sharesbas` asked.

    The decision asked whether `sharesbas` sums share classes. It does not -- 12 of 14
    overlapping tickers sit at a ratio of exactly 1.0 against the SEC cover-page count. What
    the measurement found instead is that Sharadar restates the WHOLE HISTORY onto the current
    split basis: NVDA's 2021 rows carry ~25bn shares against the ~2.5bn actually outstanding
    before its June 2024 10-for-1. `sharefactor` is 1.0 on every one of those rows.

    That makes `sharesbas` unusable as a point-in-time count without de-adjustment, and this
    test exists so phase 3 cannot map it as one by accident.
    """
    frame = cross_check_shares(context)
    if frame.empty:
        pytest.skip("no overlapping ticker has both a sharesbas and a SEC sharesOutstanding")
    split = frame[frame["ratio_span"] >= SPLIT_RATIO_SPAN]
    agree = frame[(frame["median_ratio"] - 1).abs() <= 0.05]

    print("\n=== SANITY CHECK: sharesbas vs the SEC cover-page count ===")
    print(f"  tickers compared            : {len(frame)}")
    print(f"  median ratio == 1.0         : {len(agree)}  <- so NOT a share-class problem")
    print(f"  ratio_span >= {SPLIT_RATIO_SPAN}          : {len(split)}  <- SPLIT-ADJUSTED history")
    for row in frame.itertuples(index=False):
        print(f"    {row.ticker:5s} n={row.n_dates:3d} median={row.median_ratio:8.4f} "
              f"span={row.ratio_span:7.4f} sharefactor={row.median_sharefactor:.1f}  "
              f"{row.verdict}")
    if len(split):
        print("  => `sharesbas` is NOT point-in-time for "
              f"{', '.join(split['ticker'])}. Multiplying it by an as-filed price yields a")
        print("     market cap wrong by the split factor for every date before the split.")
        print("     Phase 3 must take sharesOutstanding from the SEC layer on the overlap, or")
        print("     de-adjust using sharadar_actions (already ingested, carries the splits).")

    assert len(agree) >= len(frame) - len(split), (
        "a ticker disagrees with the SEC cover-page count for a reason that is NOT a split -- "
        "that would be the share-class summing question D-decision actually asked about")
    assert (frame["median_sharefactor"] == 1.0).all(), (
        "`sharefactor` is no longer uniformly 1.0 -- it may now encode the split adjustment, "
        "which would change how phase 3 should de-adjust")
    print(f"  OK: {len(agree)}/{len(frame)} agree on level; {len(split)} carry a split-adjusted "
          f"history that phase 3 must de-adjust.")


# --------------------------------------------------------------------------- #
# D19 -- written now, SKIPPED with a printed reason                            #
# --------------------------------------------------------------------------- #
def _cutover_tickers() -> dict[str, str]:
    """`{ticker: cutover_date}` from the registrant-boundary register."""
    path = CONFIG_DIR / FUNDAMENTALS_CATALOGUE_SUBDIR / FUNDAMENTALS_CIK_CUTOVER_FILENAME
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["cutover_date"] for k, v in blob.items() if not k.startswith("_")}


def test_cik_cutover_continuity(context):
    """D19 joins Sharadar to the SEC layer on `ticker`. A CIK cutover is exactly where that
    join could silently lose half a history: the SEC side resolves per REGISTRANT, so a
    predecessor's filings hang off a different CIK, while Sharadar's series is ticker-keyed and
    continuous. This asserts both sides span the cutover date without a hole.

    It is expected to SKIP today -- none of APA / GOOGL / ETN is in the DJIA, and the free tier
    entitles the DJIA only. The skip prints its reason rather than vanishing, because D19 being
    UNVERIFIED is a fact the phase has to report, not a test that quietly did not run.
    """
    cutovers = _cutover_tickers()
    stored = set(context.store.distinct(Tables.sharadar_fundamentals, "ticker"))
    testable = sorted(set(cutovers) & stored)

    print("\n=== SANITY CHECK: D19, CIK-cutover continuity ===")
    print(f"  cutover tickers in the register : {sorted(cutovers)}")
    print(f"  tickers stored in {Tables.sharadar_fundamentals} : {len(stored)}")
    print(f"  testable (register and stored)    : {testable or 'NONE'}")
    if not testable:
        print("  => D19 IS UNVERIFIED. The free tier entitles the DJIA only, and none of")
        print(f"     {sorted(cutovers)} is a DJIA member. This test runs the day the roster")
        print("      widens (the Full tier, or an S&P 500 entitlement) and not before.")
        pytest.skip(f"no cutover ticker is entitled: register={sorted(cutovers)} "
                    f"and stored={len(stored)} tickers = empty. D19 UNVERIFIED.")

    completeness = gate_completeness(context, testable).set_index("ticker")
    sec = context.store.load(Tables.fundamentals_history_sec, columns=["ticker", "as_of"],
                             where={"ticker": testable}, optional=True)
    for ticker in testable:
        row = completeness.loc[ticker]
        sec_rows = 0 if sec is None else int((sec["ticker"] == ticker).sum())
        print(f"    {ticker}: cutover {cutovers[ticker]}, sharadar "
              f"{row['first_quarter']}..{row['last_quarter']} with {row['n_missing']} gap(s), "
              f"{sec_rows} SEC row(s)")
        assert row["n_missing"] == 0, (
            f"{ticker} has {row['n_missing']} missing quarter(s) around its CIK cutover "
            f"({cutovers[ticker]}): {row['missing_quarters']}")
    print(f"  OK: {len(testable)} cutover ticker(s) span their boundary with no gap.")
