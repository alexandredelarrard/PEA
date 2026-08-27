"""Tests for the Sharadar phase-4 merge
(src/data_extract/utils/fundamentals_sharadar/merge_history.py, gap_check.py).

Split the way the repo's testing rule requires. **The no-leakage property gets a synthetic
known-truth fixture** -- you cannot prove a join never reaches forward by observing a dataset
where it happened not to, because the dataset would be encoding the answer -- and **every
basis, coverage and grain claim gets real data from Postgres**, since whether Sharadar's
`date` IS the SEC filing date is a measurement, not a fixture.

Each test prints its conclusion. Two exist specifically to pin a DECISION a later reader will
otherwise take for an oversight: that the four amendment columns are absent (D15), and that an
unapproved override entry changes nothing (D22).

!! Nothing here touches `src/validate/` or the `fundamentals_check*` tables (D25).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_PROVENANCE
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ
from src.data_extract.utils.fundamentals_sharadar.field_map import load_field_map
from src.data_extract.utils.fundamentals_sharadar.gap_check import (
    candidates, measure_gaps)
from src.data_extract.utils.fundamentals_sharadar.merge_history import (
    EMPLOYEES_COLUMN, NON_VALUE_COLUMNS, SEC_AS_OF, Overrides, build_frame,
    collapse_same_date, join_sec_block, load_overrides, merged_columns, sec_column,
    write_overrides)
from src.data_store.schema import Tables, name_of

CONFIG_DIR = Path("./configs")

#: The three CIK-cutover tickers D19 names. None of them is on the DJIA roster the free tier
#: covers, which is why the continuity test SKIPS rather than passing vacuously.
CIK_CUTOVER = ("APA", "GOOGL", "ETN")

#: The measured floor for `as_of` agreement. The plan measured 279/280 = 99.64%; 99% leaves
#: room for one more amendment-shaped miss without leaving room for a grain bug.
AS_OF_MATCH_FLOOR = 0.99


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def field_map():
    return load_field_map(str(CONFIG_DIR))


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


@pytest.fixture(scope="module")
def sources(context, field_map):
    """The four real inputs, loaded once: Sharadar ARQ, the SEC block, employees, actions."""
    vendor = context.store.load(Tables.sharadar_fundamentals, project=True,
                                where={"dimension": ARQ})
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    sec = context.store.load(Tables.fundamentals_history_sec,
                             columns=["ticker", "as_of", *sec_owned])
    employees = context.store.load(Tables.fundamentals_employees, optional=True)
    actions = context.store.load(Tables.sharadar_actions, project=True, optional=True)
    return vendor, sec, employees, actions


@pytest.fixture(scope="module")
def merged(sources, field_map):
    vendor, sec, employees, actions = sources
    return build_frame(vendor, sec, employees, actions, field_map,
                       Overrides(approved={}, pending={}))


@pytest.fixture(scope="module")
def overlap(sources):
    vendor, sec, *_ = sources
    return sorted(set(vendor["ticker"]) & set(sec["ticker"]))


# --------------------------------------------------------------------------- #
# the grain: is Sharadar's `date` the SEC filing date?                         #
# --------------------------------------------------------------------------- #
def test_as_of_matches_sec(sources, overlap):
    """`ARQ.date` vs `fundamentals_history_sec.as_of`, on the overlapping tickers.

    THE premise of the whole phase. If these two are not the same event, the merged table's
    `as_of` means one thing for 76 columns and another for 15, and no amount of downstream
    care fixes that.

    Measured within each ticker's SHARED window -- the SEC history runs back to 2010 and the
    free Sharadar tier starts in 2021, so a plain set comparison would report a decade of
    absent Sharadar rows as mismatches.
    """
    vendor, sec, *_ = sources
    vendor_dates = pd.to_datetime(vendor["date"])
    sec_dates = pd.to_datetime(sec["as_of"])
    total = matched = 0
    misses = []
    for ticker in overlap:
        shar = set(vendor_dates[vendor["ticker"] == ticker])
        rows = sec_dates[sec["ticker"] == ticker]
        window = rows[(rows >= min(shar)) & (rows <= max(shar))]
        if window.empty:
            misses.append((ticker, None, "NO-SEC-ROWS-IN-WINDOW"))
            continue
        total += len(window)
        matched += len(set(window) & shar)
        misses += [(ticker, d.date(), "sec-only") for d in sorted(set(window) - shar)]
        # A Sharadar date with no SEC row inside the window is the DANGEROUS direction: it
        # would mean Sharadar publishes events the SEC layer never saw, and the backward
        # as-of join would then be carrying an older SEC snapshot rather than matching one.
        misses += [(ticker, d.date(), "SHARADAR-ONLY")
                   for d in sorted(shar - set(rows)) if min(window) <= d <= max(window)]

    rate = matched / total
    print(f"\nas_of agreement over {len(overlap)} overlapping ticker(s): "
          f"{matched}/{total} = {rate:.2%}")
    print(f"mismatches ({len(misses)}): {misses if misses else 'none'}")
    assert rate >= AS_OF_MATCH_FLOOR, f"{rate:.2%} of SEC events have no Sharadar row"
    sharadar_only = [m for m in misses if m[2] == "SHARADAR-ONLY"]
    assert not sharadar_only, (
        f"{len(sharadar_only)} Sharadar publication date(s) have NO SEC row inside the "
        f"shared window: {sharadar_only}. Every mismatch must be SEC-only -- the other "
        f"direction means the two sources disagree about what a publication event IS.")


def test_sec_block_joins_at_zero_lag(sources, field_map):
    """How far back the as-of carry actually reaches, on real data.

    The tolerance exists for the rows where the two dates disagree. If it is silently doing
    the work on EVERY row -- a systematic lag rather than an occasional one -- then `as_of`
    is not the same event on both sides and `test_as_of_matches_sec` is measuring the wrong
    thing.
    """
    vendor, sec, employees, actions = sources
    from src.data_extract.utils.fundamentals_sharadar.gap_check import sharadar_history
    shar = sharadar_history(vendor, field_map, actions)
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    joined = join_sec_block(shar.drop(columns=sec_owned, errors="ignore"), sec)
    lag = (joined["as_of"] - joined[SEC_AS_OF]).dt.days.dropna()
    print(f"\nSEC block joined on {len(lag)} of {len(joined)} row(s); "
          f"lag in days: min {lag.min()}, median {lag.median()}, max {lag.max()}")
    print(f"rows carried from an EARLIER SEC filing: {int((lag > 0).sum())}")
    assert (lag >= 0).all(), "a SEC row was joined from the FUTURE -- the join reached forward"


# --------------------------------------------------------------------------- #
# the no-leakage property -- the single most important test in the phase       #
# --------------------------------------------------------------------------- #
def test_sec_block_is_asof_backward():
    """A SEC row filed AFTER a Sharadar publication date must never reach it.

    SYNTHETIC on purpose. Real data cannot prove this: it would only show that no forward
    carry happened to occur, which is a property of the dataset, not of the join. The fixture
    puts a SEC snapshot squarely in the future of two of the three rows and asserts they stay
    empty.
    """
    shar = pd.DataFrame({
        "ticker": ["AAA"] * 3,
        "as_of": pd.to_datetime(["2024-01-31", "2024-04-30", "2024-07-31"]),
        "totalRevenue": [10.0, 20.0, 30.0]})
    sec = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": pd.to_datetime(["2024-01-31", "2024-06-15"]),
        "goodwill": [100.0, 200.0]})

    out = join_sec_block(shar, sec)
    print("\n" + out[["as_of", SEC_AS_OF, "goodwill"]].to_string(index=False))
    assert out.loc[0, "goodwill"] == 100.0, "the same-day SEC row must be visible"
    assert out.loc[1, "goodwill"] == 100.0, "2024-04-30 must NOT see the 2024-06-15 filing"
    assert out.loc[2, "goodwill"] == 200.0, "2024-07-31 sees the latest snapshot before it"
    assert (out["as_of"] >= out[SEC_AS_OF]).all()
    print("no SEC value is dated after the row it lands on -- backward only")


def test_stale_sec_snapshot_is_not_carried_forever():
    """The tolerance is what separates a LAG from a fabricated present.

    Without it, a ticker the SEC producer stopped covering keeps its last snapshot on every
    future row, and a five-year-old goodwill figure reads exactly like a current one.
    """
    shar = pd.DataFrame({"ticker": ["AAA", "AAA"],
                         "as_of": pd.to_datetime(["2024-06-30", "2030-06-30"]),
                         "totalRevenue": [10.0, 20.0]})
    sec = pd.DataFrame({"ticker": ["AAA"], "as_of": pd.to_datetime(["2024-06-01"]),
                        "goodwill": [100.0]})
    out = join_sec_block(shar, sec)
    print(f"\n29 days later -> {out.loc[0, 'goodwill']}; "
          f"6 years later -> {out.loc[1, 'goodwill']}")
    assert out.loc[0, "goodwill"] == 100.0
    assert pd.isna(out.loc[1, "goodwill"]), "a 6-year-old SEC snapshot was carried forward"


# --------------------------------------------------------------------------- #
# the column contract                                                          #
# --------------------------------------------------------------------------- #
def test_column_contract(merged, field_map):
    """The built frame's columns EQUAL the declared list, in order.

    List equality, not a subset check, and in ORDER: a silent column drift is invisible
    downstream because `pit.fundamentals_to_daily` returns an EMPTY FRAME for a column it
    cannot find rather than raising.
    """
    declared = merged_columns(field_map)
    built = tuple(merged.columns)
    print(f"\ndeclared {len(declared)} | built {len(built)}")
    print(f"built but not declared: {[c for c in built if c not in set(declared)]}")
    print(f"declared but not built: {[c for c in declared if c not in set(built)]}")
    assert built == declared
    assert len(built) == 91, f"the merged contract is {len(built)}, not 91"
    assert tuple(Tables.fundamentals_history.read_columns) == declared, \
        "schema.py's read_columns and the field map state the same contract twice; they differ"


def test_no_amendment_columns(merged):
    """The four SEC reconciliation columns are ABSENT, and that is a DECISION (D15).

    Explicit because a future reader will otherwise read their absence as an oversight and
    'fix' it. `is_amendment` / `amended_fiscal_end` / `amended_fields` / `publication_form`
    are properties of the SEC amendment grain, which Sharadar has no equivalent of -- carrying
    them here would produce four permanently-NULL columns that lie about what the table knows.
    They stay on `fundamentals_history_sec`, where the validator uses them.
    """
    present = [c for c in HISTORY_PROVENANCE if c in merged.columns]
    print(f"\nSEC provenance columns: {list(HISTORY_PROVENANCE)}")
    print(f"present in the merged frame: {present or 'none -- as decided (D15)'}")
    assert not present
    assert "source" not in merged.columns, \
        "precedence is per-COLUMN and fixed (D14), so a per-ROW `source` would be a lie"


def test_value_columns_are_float(context, merged):
    """The `ensure_table` TEXT regression, again, on this table.

    `sql/schema.sql` runs only when Postgres INITIALISES a volume, so on a live one
    `store.save` creates the table from the FIRST frame's inferred dtypes. An all-None object
    column becomes TEXT and every later real number is stored as a string -- which is exactly
    how APA's values once landed in `fundamentals_history_sec` as `'1997000000.0'`.
    """
    wrong = [c for c in merged.columns
             if c not in NON_VALUE_COLUMNS and merged[c].dtype != np.float64]
    print(f"\nin-frame: {len(merged.columns) - len(NON_VALUE_COLUMNS)} value column(s); "
          f"non-float64: {wrong or 'none'}")
    assert not wrong

    name = name_of(Tables.fundamentals_history)
    if not context.store.exists(name):
        pytest.skip(f"{name} is not built yet -- run `fundamentals-history-merged`")
    types = {c["name"]: str(c["type"]) for c in inspect(context.store.engine).get_columns(name)}
    texty = {c: t for c, t in types.items()
             if c not in NON_VALUE_COLUMNS
             and any(mark in t.upper() for mark in ("TEXT", "CHAR", "STRING"))}
    print(f"in-DB: {len(types)} column(s); TEXT among the value columns: {texty or 'none'}")
    print(f"{sec_column('regime')} is {types.get(sec_column('regime'))} -- a LABEL, "
          f"so it must stay TEXT")
    assert not texty
    assert "TEXT" in str(types.get(sec_column("regime"), "")).upper()


# --------------------------------------------------------------------------- #
# the override register (D22)                                                  #
# --------------------------------------------------------------------------- #
def test_unapproved_override_is_ignored(sources, field_map, tmp_path):
    """An entry with `approved: null` does not change one cell.

    The governance model in one assertion. `--propose` writes candidates into the same file a
    human approves in, so if a proposal were live on write, running the proposer would BE the
    approval and the review would be theatre.
    """
    vendor, sec, employees, actions = sources
    ticker = "AXP"
    (tmp_path / "sharadar").mkdir()
    entry = {"source": "sec", "reason": "test: proposed, not adjudicated", "approved": None}
    write_overrides({ticker: {"totalRevenue": entry}}, ["test register"],
                    config_dir=str(tmp_path))
    loaded = load_overrides(str(tmp_path))
    print(f"\napproved {len(loaded.approved)} | awaiting decision {len(loaded.pending)}: "
          f"{sorted(f'{t}/{f}' for t, f in loaded.pending)}")
    assert not loaded.approved and len(loaded.pending) == 1

    base = build_frame(vendor, sec, employees, actions, field_map,
                       Overrides(approved={}, pending={}))
    with_pending = build_frame(vendor, sec, employees, actions, field_map, loaded)
    revenue = base.loc[base["ticker"] == ticker, "totalRevenue"]
    print(f"{ticker} totalRevenue unchanged on {len(revenue)} row(s): "
          f"{revenue.head(3).round(0).tolist()}")
    pd.testing.assert_frame_equal(base, with_pending)


def test_approved_override_takes_the_sec_value(context, sources, field_map):
    """The mirror image: an APPROVED entry DOES move the column, and ONLY that column.

    Also pins the coverage cost D14 accepts -- the override reads from a 54-ticker source, so
    for a ticker outside that roster it yields NULL rather than falling back to Sharadar.
    """
    vendor, sec, employees, actions = sources
    ticker, field = "AXP", "totalRevenue"
    entry = {"source": "sec", "reason": "test", "approved": "2026-08-26"}
    overrides = Overrides(approved={(ticker, field): entry}, pending={})

    # ⚠ The SEC projection is built FROM the register, so an approved field the join never
    # brought must RAISE. Writing a NULL over a real Sharadar value instead would be the
    # override silently doing the opposite of what it says.
    with pytest.raises(RuntimeError, match="the SEC block was not loaded"):
        build_frame(vendor, sec, employees, actions, field_map, overrides)
    print(f"\nan override naming a column the SEC projection omitted RAISES rather than "
          f"NULLing {ticker} {field} -- the two cannot drift apart silently")

    values = context.store.load(Tables.fundamentals_history_sec,
                                columns=["ticker", "as_of", field])
    values["as_of"] = pd.to_datetime(values["as_of"]).astype("datetime64[ns]")
    wide = sec.copy()
    wide["as_of"] = pd.to_datetime(wide["as_of"]).astype("datetime64[ns]")
    wide = wide.merge(values.rename(columns={field: f"__sec__{field}"}),
                      on=["ticker", "as_of"], how="left")

    base = build_frame(vendor, sec, employees, actions, field_map,
                       Overrides(approved={}, pending={}))
    out = build_frame(vendor, wide, employees, actions, field_map, overrides)
    rows = out["ticker"] == ticker
    moved, before = out.loc[rows, field], base.loc[rows, field]
    print(f"{ticker} {field}: changed on {int((moved != before).sum())} of {len(moved)} "
          f"row(s); SEC carried it on {int(moved.notna().sum())}, NULL on "
          f"{int(moved.isna().sum())} -- NULL is the coverage cost, not a fallback")
    assert not moved.equals(before), "the approved override changed nothing"
    # Every OTHER ticker is untouched: an override is per (ticker, field), not per field.
    pd.testing.assert_frame_equal(base.loc[~rows], out.loc[~rows])

    # The override RIPPLES, and it must. A ratio of TTM levels whose numerator or denominator
    # just moved to a different source is stale otherwise -- `profitMargins` would still be
    # Sharadar net income over Sharadar revenue while `totalRevenue` reads SEC. The ripple is
    # bounded to the derived columns that READ the overridden field; anything wider would mean
    # `rederive` is re-running formulas it was told not to.
    expected = {field} | {n for n, s in field_map.derived.items()
                          if s.op != "quarter" and field in s.inputs}
    touched = {c for c in base.columns
               if not base[c].equals(out[c])}
    print(f"columns the override rippled into: {sorted(touched)}")
    print(f"derived columns reading {field}: {sorted(expected - {field})}")
    assert touched == expected, f"unexpected ripple: {sorted(touched ^ expected)}"


# --------------------------------------------------------------------------- #
# the gap check                                                                #
# --------------------------------------------------------------------------- #
def test_axp_revenue_gap_is_detected(context):
    """Real: AXP `totalRevenue` is a SYSTEMATIC gap and JPM's is not.

    The two halves are equally load-bearing. AXP's gap is its provision for credit losses --
    Sharadar takes the post-provision caption the repo bans by name -- and it holds on nearly
    every shared date. JPM matched the repo EXACTLY, so a check that flagged it too would be
    flagging the SEC layer against itself.
    """
    gaps = measure_gaps(context, ["AXP", "JPM"], config_dir=str(CONFIG_DIR))
    revenue = gaps[gaps["field"] == "totalRevenue"].set_index("ticker")
    print("\n" + revenue[["n_dates", "n_flagged", "median_pct_gap", "max_pct_gap",
                          "is_systematic"]].to_string())
    assert bool(revenue.loc["AXP", "is_systematic"]), "AXP's provision gap was not detected"
    assert not bool(revenue.loc["JPM", "is_systematic"]), \
        "JPM matched the repo exactly; flagging it means the check compares the wrong things"
    assert revenue.loc["JPM", "n_flagged"] == 0
    found = candidates(gaps)
    print(f"override candidates over AXP+JPM: "
          f"{sorted(zip(found['ticker'], found['field']))}")


def test_expected_forks_are_named_not_rediscovered(context, field_map):
    """The eight phase-3 basis forks are reported as EXPECTED.

    Not cosmetic: a report whose findings are always the same eight rows is a report nobody
    reads, and the real finding -- anything gapping that is NOT one of them -- is what drowns.
    """
    from src.constants.constants import SHARADAR_GAP_EXPECTED_FIELDS
    gaps = measure_gaps(context, ["AXP", "JPM", "WMT"], config_dir=str(CONFIG_DIR))
    systematic = gaps[gaps["is_systematic"]]
    named = sorted(set(systematic.loc[systematic["is_expected"], "field"]))
    print(f"\nsystematic gaps: {len(systematic)}; of which expected forks: {named}")
    print(f"real findings: {sorted(set(candidates(gaps)['field']))}")
    assert set(named) <= SHARADAR_GAP_EXPECTED_FIELDS
    assert not candidates(gaps)["field"].isin(SHARADAR_GAP_EXPECTED_FIELDS).any()


# --------------------------------------------------------------------------- #
# grain and coverage                                                           #
# --------------------------------------------------------------------------- #
def test_same_date_collapse_keeps_the_greatest_period():
    """Sharadar ships no form column, so `FORM_PRECEDENCE` has no analogue: the vendor's own
    rule is the greatest `reportperiod` on a duplicate `(ticker, date)`."""
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "as_of": pd.to_datetime(["2024-02-15", "2024-02-15", "2024-02-15"]),
        "fiscal_end": pd.to_datetime(["2023-09-30", "2023-12-31", "2023-12-31"]),
        "totalRevenue": [1.0, 2.0, 3.0]})
    kept, dropped = collapse_same_date(frame)
    print(f"\nkept:\n{kept.to_string(index=False)}\ndropped:\n{dropped.to_string(index=False)}")
    assert len(kept) == 2 and len(dropped) == 1
    assert kept.loc[kept["ticker"] == "AAA", "totalRevenue"].iloc[0] == 2.0
    assert not kept.duplicated(["ticker", "as_of"]).any()


def test_merged_grain_is_one_row_per_publication(merged):
    """PK (ticker, as_of), asserted on the built frame rather than trusted to the upsert."""
    duplicated = merged[merged.duplicated(["ticker", "as_of"], keep=False)]
    print(f"\n{len(merged)} row(s), {merged['ticker'].nunique()} ticker(s), "
          f"{merged['as_of'].min().date()}..{merged['as_of'].max().date()}; "
          f"duplicate keys: {len(duplicated)}")
    assert duplicated.empty


def test_coverage_asymmetry_is_the_design(merged, overlap):
    """Sharadar columns cover every ticker; SEC-owned ones cover the overlap ONLY.

    Stated as a test because it is the most likely thing to be reported as a bug. A ticker
    outside the SEC roster gets NULL for the 15 SEC-owned columns -- not a Sharadar fallback,
    which would be exactly the mid-series source switch D14 forbids.
    """
    regime = sec_column("regime")
    with_regime = sorted(set(merged.loc[merged[regime].notna(), "ticker"]))
    print(f"\nSharadar-owned `totalAssets`: {int(merged['totalAssets'].notna().sum())} of "
          f"{len(merged)} row(s), {merged.loc[merged['totalAssets'].notna(), 'ticker'].nunique()}"
          f" ticker(s)")
    print(f"SEC-owned `{regime}`: {int(merged[regime].notna().sum())} row(s), "
          f"{len(with_regime)} ticker(s) -> {with_regime}")
    assert set(with_regime) <= set(overlap)
    assert merged["totalAssets"].notna().sum() > merged[regime].notna().sum()


def test_stockholders_equity_incl_nci_is_rederived_at_the_merge(merged, overlap):
    """0/598 out of phase 3 by construction; it only becomes computable HERE.

    Its NCI leg is SEC-owned, so the phase-3 TTM frame cannot evaluate the formula. If this
    column is empty after the merge, `apply_derived(only=...)` never ran and the merge silently
    published a column that exists in name only.
    """
    filled = merged[merged["stockholdersEquityInclNci"].notna()]
    print(f"\nstockholdersEquityInclNci: {len(filled)} of {len(merged)} row(s), "
          f"{filled['ticker'].nunique()} ticker(s)")
    assert not filled.empty, "the merge never re-derived it"
    assert set(filled["ticker"]) <= set(overlap)
    leg = filled["stockholdersEquity"] + filled[sec_column("minorityInterest")]
    assert np.allclose(filled["stockholdersEquityInclNci"], leg, equal_nan=True)


def test_employees_is_forward_filled_from_its_own_table(merged):
    """Headcount is annual 10-K PROSE and was never on the filing cadence, so it comes from
    `fundamentals_employees` carried forward -- NOT from `fundamentals_history_sec`, which has
    no such column at all."""
    employees = sec_column(EMPLOYEES_COLUMN)
    filled = merged[merged[employees].notna()]
    print(f"\n{employees}: {len(filled)} of {len(merged)} row(s), "
          f"{filled['ticker'].nunique()} ticker(s)")
    assert not filled.empty
    per_ticker = filled.groupby("ticker")[employees].nunique()
    print(f"distinct headcounts per ticker (annual disclosure, quarterly rows): "
          f"min {per_ticker.min()}, max {per_ticker.max()}")
    assert (filled.groupby("ticker").size() > per_ticker).any(), \
        "no ticker repeats a headcount -- the annual value is not reaching the interim rows"


@pytest.mark.parametrize("ticker", CIK_CUTOVER)
def test_cik_cutover_continuity(sources, merged, ticker):
    """D19: the join is on `ticker`, so the 3 CIK-cutover names need explicit continuity.

    ⚠ UNVERIFIABLE on this roster and SKIPPED rather than quietly passing. The free tier
    covers the DJIA-30; none of APA / GOOGL / ETN is on it, so there is no data to check. The
    skip message IS the finding -- re-run this on the Full tier before believing D19.
    """
    vendor, *_ = sources
    if ticker not in set(vendor["ticker"]):
        pytest.skip(f"{ticker} is not in the entitled Sharadar roster (DJIA-30) -- D19's "
                    f"CIK-cutover continuity is UNVERIFIABLE here, not verified")
    rows = merged[merged["ticker"] == ticker].sort_values("as_of")
    print(f"\n{ticker}: {len(rows)} row(s), {rows['as_of'].min()}..{rows['as_of'].max()}")
    assert rows["totalAssets"].notna().sum() > 0
    gaps = rows["as_of"].diff().dt.days.dropna()
    print(f"largest gap between publications: {gaps.max()} day(s)")
    assert gaps.max() < 200, "a CIK cutover left a hole in the series"


# --------------------------------------------------------------------------- #
# the register's own file format                                              #
# --------------------------------------------------------------------------- #
def test_reproposing_is_byte_identical(tmp_path):
    """Re-emitting an unchanged register produces the SAME BYTES.

    The property that makes `--propose` safe to re-run. `json.dumps(indent=2)` over the whole
    file reformats every line it touches, so a two-entry proposal would show up as a
    whole-file diff and the review this register exists for becomes impossible.
    """
    (tmp_path / "sharadar").mkdir()
    entries = {"AXP": {"totalRevenue": {"source": "sec", "reason": "r", "approved": None}},
               "GS": {"capex": {"source": "sec", "reason": "r", "approved": "2026-08-26"}}}
    readme = ["line one", "line two"]
    first = write_overrides(entries, readme, config_dir=str(tmp_path)).read_bytes()
    loaded = load_overrides(str(tmp_path))
    again = {t: {} for t, _ in {**loaded.approved, **loaded.pending}}
    for (t, f), e in {**loaded.approved, **loaded.pending}.items():
        again[t][f] = e
    second = write_overrides(again, readme, config_dir=str(tmp_path)).read_bytes()
    print(f"\n{len(first)} bytes; round-trip identical: {first == second}")
    print(json.dumps(json.loads(first.decode()), indent=1)[:200])
    assert first == second
    assert len(loaded.approved) == 1 and len(loaded.pending) == 1


def test_only_sec_is_a_legal_override_direction(tmp_path):
    """Moving a column the OTHER way is a field-BLOCK change (D14), not an override, and
    belongs in `sharadar_field_map.json`. An open vocabulary here would let one silently
    become the other."""
    (tmp_path / "sharadar").mkdir()
    write_overrides({"AXP": {"totalRevenue": {"source": "sharadar", "reason": "r",
                                              "approved": "2026-08-26"}}},
                    ["test"], config_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="ONLY legal direction"):
        load_overrides(str(tmp_path))
    print("\na non-`sec` source is refused at load, not at write")


def test_override_on_a_sec_owned_column_is_refused(sources, field_map):
    """An override moving an already-SEC-owned column to `sec` is a contradiction, not a
    decision -- and a silently destructive one, because the SEC projection would rename the
    column out from under the join and the contract assertion would then report it as
    'missing' rather than as what it is."""
    vendor, sec, employees, actions = sources
    overrides = Overrides(
        approved={("JPM", "goodwill"): {"source": "sec", "reason": "r", "approved": "x"}},
        pending={})
    with pytest.raises(RuntimeError, match="already"):
        build_frame(vendor, sec, employees, actions, field_map, overrides)
    print("\nan override on a SEC-owned column is refused by name, with the reason stated")
