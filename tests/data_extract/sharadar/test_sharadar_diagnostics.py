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
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.constants.constants import (
    SHARADAR_CONFIG_SUBDIR,
    SHARADAR_ZERO_FILLED_FIELDS,
    SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_extract.utils.common.identity import SEAM_MARGIN_DAYS
from src.data_extract.utils.common.registrant import Registrant, Segment, load_registrants
from src.data_extract.utils.fundamentals_sharadar.diagnostics import (
    confirm_sign_conventions,
    cross_check_shares,
    gate_completeness,
    gate_zero_fill,
    load_sec,
    load_sharadar,
)
from src.data_store.schema import Tables
from src.utils.quarters import quarter_label, quarter_ordinal

CONFIG_DIR = Path("./configs")

#: Ceiling on the share of rows carrying a POSITIVE `capex`. Measured at 0.97% (13 of 1,346,
#: 11 of them GS) on 2026-08-26. Bounded rather than zero because the plan's universal claim
#: was taken from AAPL alone and is false; see `test_sign_conventions_hold`.
MAX_POSITIVE_CAPEX_RATE = 0.05

#: Ratio span (max/min of `sharesbas / sharesOutstanding` across a ticker's dates) above which
#: the history has been retroactively re-based. A real share-class or reporting difference is
#: a LEVEL shift and holds flat over time; only a split moves the ratio within one ticker.
SPLIT_RATIO_SPAN = 1.5

#: Half-width of the window D19's continuity is measured in, around each ticker's own cutover
#: date. A registrant change can only lose filings near its own boundary, so two years each
#: side is eight quarters of margin on both -- wide enough that a predecessor's last filings
#: and a successor's first ones are both inside it, narrow enough that a vendor's sparse early
#: history (BG carries one ARQ row per year before 2004, 21 years before its 2023 cutover)
#: cannot be mistaken for a hole the cutover caused.
CUTOVER_WINDOW_YEARS = 2

#: The three classes of a vendor quarter missing near a cutover. Only the last can be explained:
#: the other two are defects of the SEC side or of the register, never of the vendor.
MISSING_SEC_FILING = "missing_sec_filing"
INCORRECT_CIK_WINDOW = "incorrect_cik_window"
VENDOR_COVERAGE_GAP = "vendor_coverage_gap"

#: The periodic forms whose period of report is a fiscal quarter or year end.
PERIODIC_FORMS = ("10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT", "10-KT/A")

VENDOR_GAP_LABEL = "Sharadar/source coverage gap; SEC filings present"


@dataclass(frozen=True)
class CutoverException:
    """One vendor quarter missing at a cutover, explained by the SEC filing that reports it."""

    ticker: str
    quarter: str
    period_end: str
    cik: str
    accession: str
    filed: str
    label: str


#: Every recorded cutover vendor gap, one row per missing quarter, each read on EDGAR. A missing
#: quarter with no row here fails, and a row whose quarter is present again fails too.
CUTOVER_VENDOR_EXCEPTIONS: tuple[CutoverException, ...] = (
    CutoverException("BKR", "2017Q1", "2017-03-31", "0000808362", "0000808362-17-000025", "2017-04-28", VENDOR_GAP_LABEL),
    CutoverException("BKR", "2017Q2", "2017-06-30", "0000808362", "0000808362-17-000034", "2017-07-28", VENDOR_GAP_LABEL),
    CutoverException("DOW", "2018Q1", "2018-03-31", "0000029915", "0000029915-18-000013", "2018-05-04", VENDOR_GAP_LABEL),
    CutoverException("DOW", "2018Q2", "2018-06-30", "0000029915", "0000029915-18-000020", "2018-08-03", VENDOR_GAP_LABEL),
    CutoverException("STE", "2014Q2", "2014-06-30", "0000815065", "0000815065-14-000008", "2014-08-08", VENDOR_GAP_LABEL),
    CutoverException("STE", "2014Q3", "2014-09-30", "0000815065", "0000815065-14-000012", "2014-11-04", VENDOR_GAP_LABEL),
    CutoverException("STE", "2014Q4", "2014-12-31", "0000815065", "0001628280-15-000532", "2015-02-09", VENDOR_GAP_LABEL),
    CutoverException("STE", "2015Q1", "2015-03-31", "0000815065", "0000815065-15-000004", "2015-05-27", VENDOR_GAP_LABEL),
    CutoverException("STE", "2015Q2", "2015-06-30", "0000815065", "0000815065-15-000008", "2015-08-07", VENDOR_GAP_LABEL),
    CutoverException("STE", "2015Q3", "2015-09-30", "0000815065", "0000815065-15-000009", "2015-10-30", VENDOR_GAP_LABEL),
    CutoverException("VMC", "2006Q1", "2006-03-31", "0000103973", "0000103973-06-000112", "2006-04-28", VENDOR_GAP_LABEL),
    CutoverException("VMC", "2006Q2", "2006-06-30", "0000103973", "0000103973-06-000197", "2006-08-01", VENDOR_GAP_LABEL),
    CutoverException("VMC", "2006Q3", "2006-09-30", "0000103973", "0000103973-06-000268", "2006-10-31", VENDOR_GAP_LABEL),
)

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


def _admitted(registrant: Registrant, cik: str, filed: pd.Timestamp) -> bool:
    """Whether the register's seam-widened window for `cik` admits a filing made on `filed`."""
    margin = pd.Timedelta(days=SEAM_MARGIN_DAYS)
    return any(
        segment.cik == cik
        and (segment.valid_from is None or filed >= segment.valid_from - margin)
        and (segment.valid_to is None or filed < segment.valid_to + margin)
        for segment in registrant.segments
    )


def _discontinuities(arq: pd.DataFrame, registrant: Registrant) -> tuple[dict[str, pd.Timestamp], list[pd.Timestamp]]:
    """Missing vendor quarters within `CUTOVER_WINDOW_YEARS` of every boundary, each with its
    boundary, plus the boundaries with no vendor quarter in their window at all."""
    dates = pd.to_datetime(arq["calendardate"], errors="coerce")
    offset = pd.DateOffset(years=CUTOVER_WINDOW_YEARS)
    missing: dict[str, pd.Timestamp] = {}
    unobserved: list[pd.Timestamp] = []
    for seam in registrant.boundaries:
        near = arq[dates.between(seam - offset, seam + offset)]
        if near.empty:
            unobserved.append(seam)
            continue
        labels = str(gate_completeness(near)["missing_quarters"].iloc[0]).split(",")
        for label in (part.strip() for part in labels):
            if label and label != "-":
                missing.setdefault(label, seam)
    return dict(sorted(missing.items())), unobserved


def _quarter_of(day: str) -> str:
    """The `2025Q1` label of one date; empty when the date is missing or unparseable."""
    ordinal = quarter_ordinal(pd.Series([day])).iloc[0]
    return "" if pd.isna(ordinal) else quarter_label(int(ordinal))


def _classify(registrant: Registrant, quarter: str, filings: pd.DataFrame, record: CutoverException | None) -> tuple[str, bool, str]:
    """`(class, explained, evidence)` for one missing vendor quarter.

    The SEC filings stored for the quarter decide the class when there are any; otherwise the
    exception record does. Only a recorded `vendor_coverage_gap` is explained.
    """
    if not filings.empty:
        filed = pd.to_datetime(filings["filing_date"])
        admitted = [_admitted(registrant, cik, day) for cik, day in zip(filings["cik"], filed, strict=True)]
        cls = VENDOR_COVERAGE_GAP if any(admitted) else INCORRECT_CIK_WINDOW
        evidence = "stored " + "; ".join(
            f"{cik} {accession} {form} filed {day.date()}" + ("" if ok else " OUTSIDE window")
            for cik, accession, form, day, ok in zip(filings["cik"], filings["accession_number"], filings["form"], filed, admitted, strict=True)
        )
    elif record is not None:
        cls = VENDOR_COVERAGE_GAP if _admitted(registrant, record.cik, pd.Timestamp(record.filed)) else INCORRECT_CIK_WINDOW
        evidence = "nothing stored for the period"
    else:
        return MISSING_SEC_FILING, False, "nothing stored for the period and no record"
    if record is None:
        return cls, False, evidence + " | no exception row"
    record_ok = _quarter_of(record.period_end) == quarter and _admitted(registrant, record.cik, pd.Timestamp(record.filed))
    evidence += f" | record {record.cik} {record.accession} filed {record.filed}" + ("" if record_ok else " INCONSISTENT")
    return cls, cls == VENDOR_COVERAGE_GAP and record_ok, evidence


def test_cutover_classification_rules():
    """Known-truth chain (old CIK to 2020-01-01, new CIK after): each class and the record
    checks come out as defined, without the database."""
    seam = pd.Timestamp("2020-01-01")
    chain = Registrant("TST", "reorganisation", (Segment("0000000001", None, seam, "fixture"), Segment("0000000002", seam, None, "fixture")))
    columns = ["cik", "accession_number", "form", "filing_date"]

    def stored(cik: str, filed: str) -> pd.DataFrame:
        return pd.DataFrame([[cik, f"{cik}-19-000001", "10-Q", pd.Timestamp(filed)]], columns=columns)

    record = CutoverException("TST", "2019Q2", "2019-06-30", "0000000001", "0000000001-19-000001", "2019-08-01", VENDOR_GAP_LABEL)
    cases = {
        "old CIK files inside its window": (stored("0000000001", "2019-08-01"), record, VENDOR_COVERAGE_GAP, True),
        "new CIK inside the 31-day seam margin": (stored("0000000002", "2019-12-15"), record, VENDOR_COVERAGE_GAP, True),
        "new CIK long before its window": (stored("0000000002", "2019-08-01"), record, INCORRECT_CIK_WINDOW, False),
        "a CIK outside the chain": (stored("0000000009", "2019-08-01"), record, INCORRECT_CIK_WINDOW, False),
        "vendor gap with no exception row": (stored("0000000001", "2019-08-01"), None, VENDOR_COVERAGE_GAP, False),
        "nothing stored, no record": (pd.DataFrame(columns=columns), None, MISSING_SEC_FILING, False),
        "nothing stored, record admitted": (pd.DataFrame(columns=columns), record, VENDOR_COVERAGE_GAP, True),
        "record for another quarter": (pd.DataFrame(columns=columns), replace(record, period_end="2019-09-30"), VENDOR_COVERAGE_GAP, False),
        "record CIK outside its window": (pd.DataFrame(columns=columns), replace(record, filed="2020-06-01"), INCORRECT_CIK_WINDOW, False),
    }
    print("\n=== SANITY CHECK: cutover discontinuity classes on a known-truth chain ===")
    for name, (filings, rec, want_cls, want_ok) in cases.items():
        cls, ok, evidence = _classify(chain, "2019Q2", filings, rec)
        print(f"  {name:40s} -> {cls:20s} explained={ok}")
        assert (cls, ok) == (want_cls, want_ok), f"{name}: got {cls}/{ok}, expected {want_cls}/{want_ok} ({evidence})"
    print(f"  OK: all {len(cases)} cases classify as defined; only a recorded, admitted vendor gap is explained.")


def test_cik_cutover_continuity(context, frames):
    """D19 joins Sharadar to the SEC layer on `ticker`, and a CIK cutover is where that join
    can lose half a history. Every vendor quarter missing within `CUTOVER_WINDOW_YEARS` of any
    boundary is collected and classified before anything is asserted; the test fails once,
    on the full table, unless each one is a recorded `vendor_coverage_gap`.
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

    found: dict[tuple[str, str], pd.Timestamp] = {}
    unobserved: list[str] = []
    for ticker in testable:
        missing, blind = _discontinuities(frames.arq[frames.arq["ticker"] == ticker], registrants[ticker])
        found.update({(ticker, quarter): seam for quarter, seam in missing.items()})
        unobserved.extend(f"{ticker} {seam.date()}" for seam in blind)
        boundaries = [str(b.date()) for b in registrants[ticker].boundaries]
        print(f"    {ticker}: boundaries {boundaries}, missing within +/-{CUTOVER_WINDOW_YEARS}y: {list(missing) or '-'}")

    columns = ["ticker", "cik", "accession_number", "form", "filing_date", "period_of_report"]
    gap_tickers = sorted({ticker for ticker, _ in found})
    facts = (
        context.store.load(Tables.fundamentals_facts, columns=columns, where={"ticker": gap_tickers, "form": list(PERIODIC_FORMS)}, optional=True)
        if gap_tickers
        else None
    )
    filings = (facts if facts is not None else pd.DataFrame(columns=columns)).drop_duplicates(["ticker", "accession_number"])
    filings = filings.assign(
        cik=filings["cik"].fillna("").astype(str).str.zfill(10), quarter=[_quarter_of(day) for day in filings["period_of_report"]]
    )

    records = {(r.ticker, r.quarter): r for r in CUTOVER_VENDOR_EXCEPTIONS}
    table: list[tuple[str, str, str, str, bool, str]] = []
    for (ticker, quarter), seam in sorted(found.items()):
        period = filings[(filings["ticker"] == ticker) & (filings["quarter"] == quarter)].sort_values("filing_date")
        cls, explained, evidence = _classify(registrants[ticker], quarter, period, records.get((ticker, quarter)))
        table.append((ticker, quarter, str(seam.date()), cls, explained, evidence))

    print(f"  discontinuities: {len(table)} ({sum(row[4] for row in table)} explained)")
    print(f"    {'ticker':6s} {'quarter':7s} {'boundary':10s} {'class':20s} {'ok':3s} evidence")
    for ticker, quarter, seam, cls, explained, evidence in table:
        print(f"    {ticker:6s} {quarter:7s} {seam:10s} {cls:20s} {'yes' if explained else 'NO':3s} {evidence}")
    unexplained = [f"{ticker} {quarter} {cls}" for ticker, quarter, _, cls, explained, _ in table if not explained]
    # A recorded gap that has been FILLED must be deleted, or the record goes stale.
    healed = sorted(f"{ticker} {quarter}" for ticker, quarter in records if (ticker, quarter) not in found)
    print(f"  unexplained: {unexplained or 'none'} | healed records: {healed or 'none'} | unobserved boundaries: {unobserved or 'none'}")

    assert not unobserved, f"no vendor quarter within {CUTOVER_WINDOW_YEARS}y of these boundaries, so D19 is untested there: {unobserved}"
    assert len(records) == len(CUTOVER_VENDOR_EXCEPTIONS), "CUTOVER_VENDOR_EXCEPTIONS records one (ticker, quarter) twice"
    assert not unexplained, (
        f"{len(unexplained)} cutover discontinuities are not recorded vendor gaps: {unexplained}. "
        f"A {MISSING_SEC_FILING} or {INCORRECT_CIK_WINDOW} is a register or SEC-side defect; a "
        f"{VENDOR_COVERAGE_GAP} needs its EDGAR accession in CUTOVER_VENDOR_EXCEPTIONS."
    )
    assert not healed, (
        f"{healed} are in CUTOVER_VENDOR_EXCEPTIONS but the quarter is present now. Delete the rows: a stale exception hides the next real hole."
    )
    print(f"  OK: {len(testable)} cutover tickers; all {len(table)} discontinuities are recorded vendor gaps with their SEC filing.")
