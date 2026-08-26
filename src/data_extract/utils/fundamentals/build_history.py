"""
build_history.py  (src/data_extract/utils/fundamentals/build_history.py)
--------------------------------------------------------------------------------------------
`fundamentals_facts` -> `fundamentals_history` + `fundamentals_reason_codes`, on the
PUBLICATION-EVENT grain.

`as_of` is a FILING DATE, not a period end. That one change is the whole phase: the previous
build computed `as_of` as a median-of-spine heuristic over the concepts that made a period
COMPUTABLE, which pushed 10-Q-derived quarters ~400 days past their own period end and, for
ROP's 2009 year, 59 days BEFORE it -- a look-ahead leak. Under this grain there is nothing to
compute. The five rules, in the order they are applied:

  1. A row is emitted for every `(ticker, date)` on which >=1 extracted value became newly
     public. An ORIGINAL filing always qualifies.
  2. An AMENDMENT emits a row only if it changes >=1 extracted value AND lands <=365 days
     after the original (decision 34). The value test discards the ~88 Part-III/cover-only
     amendments; the cutoff discards the restatements a long/short model cannot learn from,
     because a quarter stays inside some live TTM window for about twelve months.
  3. The row is a COMPLETE SNAPSHOT. Every column carries its latest-known value at that
     date, so a plain `asof` merge needs no reconstruction. This is the property `PitFrames`
     depends on, and the reason `fundamentals_reason_codes` is dense rather than sparse.
  4. Rows are IMMUTABLE once written -- enforced, not asserted: see `diff_against_stored`.
  5. SAME-DAY COLLAPSE by `(ticker, date)`, never by accession. Two filings on one day
     produce one row reflecting both, with provenance resolved by form precedence.

A restated value propagates further than one cell, and that is correct rather than a bug:
the table stores TTM LEVELS, so restating Q1 moves the TTM at Q1, Q2, Q3 and Q4. What stays
frozen is the EARLIER ROWS, which keep their as-filed values forever -- that is where the
no-leakage property lives.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME, Catalogue, load_catalogue)
from src.data_extract.utils.fundamentals.periods import (
    INSTANT, PeriodGuards, build_periods, fiscal_quarter_of_end, fiscal_year_ends,
    load_guards)

#: Form precedence for a same-day collapse (decision 37). Scalar-with-a-precedence-rule
#: rather than a pipe-joined string, so a `publication_form == '10-K'` filter can never
#: silently miss a day on which a 10-K was co-filed.
FORM_PRECEDENCE: tuple[str, ...] = ("10-K", "10-K/A", "10-Q", "10-Q/A")

#: Decision 34. Structural, not tunable -- it encodes "the restated quarter is still inside
#: a live trailing-twelve window", which is a property of the TTM basis and not a knob.
MAX_AMENDMENT_LAG_DAYS = 365

#: THE HARD GUARDS (plan-5b decision 46): `{column: predicate that means the value is
#: IMPOSSIBLE}`. Applied by `_hard_guard` BEFORE the row is written, which is the only place
#: they can live -- the validator never mutates (decision 40), and a post-hoc UPDATE on an
#: append-only table would change a historical row's value after the fact, so yesterday's
#: cube and today's would disagree about the same publication event.
#:
#: FOUR RULES, AND ONLY IMPOSSIBLE ONES. Not "implausible", not "outside a measured band" --
#: impossible, in the sense that no filer could report it and be right. v2 proposed a wider
#: set including a `[-1, 1]` ratio bound; that bound nulls HCA's correct negative
#: `debtToEquity` (its equity IS negative) and every filer whose debt exceeds its equity,
#: which is the 745-correct-rows-nulled-by-an-over-strict-guard failure mode repeating
#: verbatim. Everything else v2 listed is a FLAG-ONLY Tier-1 check (`impossible_value`)
#: that reports the number and leaves it in the table for a human to judge.
#:
#: A negative share count is arithmetically impossible; zero is a filing error rather than a
#: fact (a company with no shares outstanding has no equity to report), so `<= 0` is the
#: test for the three counts. `totalAssets` uses `< 0` and not `<= 0`: a shell in its first
#: period legitimately foots to exactly zero (VRT's pre-merger SPAC), and nulling that would
#: destroy a correct value to catch nothing.
HARD_GUARDS: dict[str, "Callable[[float], bool]"] = {
    "totalAssets": lambda v: v < 0,
    "sharesOutstanding": lambda v: v <= 0,
    "basicShares": lambda v: v <= 0,
    "dilutedShares": lambda v: v <= 0,
}

#: The identity of one as-filed value inside a filing. Two filings that carry the same tuple
#: are reporting the SAME measurement, which is what makes "did this amendment change
#: anything?" answerable by value rather than by fact count -- an amendment can re-tag 200
#: facts to identical values and still be a no-op.
#:
#: Keyed on `period_end`, NOT on the `(fiscal_year, fiscal_period)` LABELS. The labels are not
#: unique inside a filing: AAPL's FY2025 10-K carries FY2023, FY2024 and FY2025 annual revenue,
#: and the label pair collides for 16,340 of them across the roster -- which is exactly why
#: `fundamentals_facts` now keys on `period_end` too. Keeping the labels here would collapse
#: three different measurements to one and then report an amendment as a no-op, or a no-op as
#: an amendment, depending on which one `keep='last'` happened to retain.
_VALUE_KEY: tuple[str, ...] = ("field", "duration_type", "period_end")

#: Every computed column, as `column -> (inputs, formula)`. The formulas are also declared
#: in prose in `fundamentals_kpis.json`; `test_build_history` asserts the two agree, so the
#: config stays the contract and this stays the single implementation of it.
#:
#: Flows are TTM here, because that IS the column (decision 31): `profitMargins` is TTM net
#: income over TTM revenue, never a quarter over a quarter.
_FORMULAS: dict[str, tuple[tuple[str, ...], object]] = {
    "ebitda": (("operatingIncome", "depAmort"), lambda a, b: a + b),
    "freeCashflow": (("operatingCashFlow", "capex"), lambda a, b: a - b),
    "epsDiluted": (("netIncome", "dilutedShares"), lambda a, b: a / b),
    "effectiveTaxRate": (("incomeTaxExpense", "pretaxIncome"), lambda a, b: a / b),
    "grossMargins": (("grossProfit", "totalRevenue"), lambda a, b: a / b),
    "operatingMargins": (("operatingIncome", "totalRevenue"), lambda a, b: a / b),
    "profitMargins": (("netIncome", "totalRevenue"), lambda a, b: a / b),
    "returnOnEquity": (("netIncome", "stockholdersEquity"), lambda a, b: a / b),
    "debtToEquity": (("totalDebt", "stockholdersEquity"), lambda a, b: a / b),
    "optionOverhang": (("dilutedShares", "basicShares"), lambda a, b: a / b - 1),
}

#: Columns taken from the DISCRETE quarter rather than from the trailing twelve. The only
#: two that survive the contract: the legacy table had four `_q` columns and `ebitda_q` /
#: `freeCashflow_q` are declared casualties (Phase 6 §6.1 reconciles them).
_QUARTER_COLUMNS: dict[str, str] = {"revenue_q": "totalRevenue",
                                    "netIncome_q": "netIncome"}

#: Formulas whose second operand is a denominator. A zero denominator is not a ratio, and
#: `x / 0` is an infinity that survives every plausibility check downstream.
_RATIOS: frozenset[str] = frozenset(
    {"epsDiluted", "effectiveTaxRate", "grossMargins", "operatingMargins",
     "profitMargins", "returnOnEquity", "debtToEquity", "optionOverhang"})


@dataclass(frozen=True)
class TickerHistory:
    """One ticker's two frames. Returned together because they are one statement: a null in
    `history` is only legitimate if `reason_codes` explains it, and building them in separate
    passes is how the two would drift."""

    history: pd.DataFrame
    reason_codes: pd.DataFrame


# --------------------------------------------------------------- the event ladder ---

def _same_value(before, after) -> bool:
    """Are two as-filed values the same measurement? NaN == NaN here, deliberately: a
    reason-coded value-less row re-tagged as another value-less row changed nothing."""
    if pd.isna(before) and pd.isna(after):
        return True
    if pd.isna(before) or pd.isna(after):
        return False
    return float(before) == float(after)


def _amended_fields(facts: pd.DataFrame, accession: str,
                    filed: pd.Timestamp) -> list[str]:
    """Which fields this amendment actually MOVED, against everything filed before it.

    Compared by VALUE on `_VALUE_KEY`, not by fact count: 88 of the 246 amendments in the
    legacy dump carry fewer than 10 facts (Part III / cover-page only) and a count threshold
    both admits some of those and rejects a genuine one-number restatement.
    """
    amendment = facts[facts["accession_number"] == accession]
    prior = facts[facts["filing_date"] < filed]
    if prior.empty:
        return sorted(set(amendment["field"]))
    latest = (prior.sort_values("filing_date")
                   .drop_duplicates(subset=list(_VALUE_KEY), keep="last")
                   .set_index(list(_VALUE_KEY))["value"])
    moved: set[str] = set()
    for row in amendment.itertuples():
        key = tuple(getattr(row, c) for c in _VALUE_KEY)
        if key not in latest.index or not _same_value(latest.loc[key], row.value):
            moved.add(row.field)
    return sorted(moved)


def publication_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per `(date)` on which this ticker made new information public.

    Returns `as_of`, `publication_form`, `is_amendment`, `amended_fiscal_end` and
    `amended_fields` -- the four provenance columns, already collapsed to the day.
    """
    if facts.empty:
        return pd.DataFrame(columns=["as_of", *HISTORY_PROVENANCE])
    first_by_period = facts.groupby("period_of_report", dropna=True)["filing_date"].min()
    rows = []
    for accession, group in facts.groupby("accession_number", sort=False):
        filed = pd.Timestamp(group["filing_date"].iloc[0])
        form = str(group["form"].iloc[0])
        period = group["period_of_report"].iloc[0]
        if not bool(group["is_amendment"].iloc[0]):
            rows.append({"as_of": filed, "publication_form": form, "is_amendment": False,
                         "amended_fiscal_end": pd.NaT, "amended_fields": None})
            continue
        moved = _amended_fields(facts, accession, filed)
        if not moved:
            continue                                  # a no-op amendment publishes nothing
        original = first_by_period.get(period)
        if original is not None and pd.notna(original) and pd.notna(period) and (
                filed - pd.Timestamp(original)).days > MAX_AMENDMENT_LAG_DAYS:
            continue                                  # too late to move a published TTM
        rows.append({"as_of": filed, "publication_form": form, "is_amendment": True,
                     "amended_fiscal_end": pd.to_datetime(period, errors="coerce"),
                     "amended_fields": ",".join(moved)})
    if not rows:
        return pd.DataFrame(columns=["as_of", *HISTORY_PROVENANCE])
    return _collapse_same_day(pd.DataFrame(rows))


def _collapse_same_day(events: pd.DataFrame) -> pd.DataFrame:
    """Rule 5. Two filings on one day are ONE publication event; provenance resolves by
    precedence so every column stays scalar and queryable."""
    rank = {form: i for i, form in enumerate(FORM_PRECEDENCE)}
    events = events.assign(_rank=[rank.get(f, len(rank)) for f in events["publication_form"]])
    out = []
    for as_of, group in events.sort_values("_rank").groupby("as_of", sort=True):
        fields = sorted({f for csv in group["amended_fields"].dropna()
                         for f in str(csv).split(",") if f})
        out.append({
            "as_of": as_of,
            "publication_form": group["publication_form"].iloc[0],
            "is_amendment": bool(group["is_amendment"].any()),
            "amended_fiscal_end": group["amended_fiscal_end"].max(),
            "amended_fields": ",".join(fields) or None})
    return pd.DataFrame(out).sort_values("as_of").reset_index(drop=True)


# ------------------------------------------------------------------- the snapshot ---

def carry_latest_known(facts: pd.DataFrame, ends, field: str,
                       on: str = "period_end") -> pd.DataFrame:
    """`field`'s latest known value at each date in `ends`, as a one-column frame.

    The as-of alignment an ANNUAL-ONLY disclosure needs to reach the interim quarters: a
    headcount stated once in the 10-K must populate the following three quarters rather than
    the fiscal-year row alone. It is also the instant path for the wide table -- a
    balance-sheet level's "latest known value" is exactly the last one reported -- which is
    why it is one primitive and not two.

    Ties on `on` are broken by `filing_date`, latest wins: a level is re-tagged as a
    comparative in every later filing, and inside a point-in-time replay the freshest
    visible tagging of a period is the right one.
    """
    # Both sides forced to nanoseconds. A parquet round-trip yields `datetime64[ms]` while a
    # constructed index is `[us]`, and `merge_asof` refuses to join two resolutions rather
    # than silently coercing -- so this normalisation is the difference between working and
    # raising, not a tidiness measure.
    index = pd.DatetimeIndex(pd.to_datetime(ends)).astype("datetime64[ns]").sort_values()
    rows = facts[facts["field"] == field]
    out = pd.DataFrame({on: index})
    if rows.empty:
        out[field] = pd.NA
        return out
    ordered = (rows.assign(**{on: pd.to_datetime(rows[on]).astype("datetime64[ns]")})
                   .sort_values([on, "filing_date"])
                   .drop_duplicates(subset=[on], keep="last")[[on, "value"]]
                   .rename(columns={"value": field})
                   .dropna(subset=[on]))
    if ordered.empty:
        out[field] = pd.NA
        return out
    return pd.merge_asof(out, ordered.sort_values(on), on=on, direction="backward")


#: How far a trailing-twelve window's end may sit from the `fiscal_end` it is reported
#: against before it stops being this period's number. HALF A QUARTER, so the tolerance can
#: only ever admit the SAME fiscal quarter and never the previous one -- which is the whole
#: job, since a one-quarter carry is already a silent basis error.
#:
#: A tolerance rather than exact equality because the two dates come from different columns
#: and legitimately disagree by days: `fiscal_end` is `_latest_period_known`, read off
#: `period_of_report`, while the TTM grid is built on the facts' own `period_end`. A 52/53-week
#: filer's ends walk, and ORCL files a quarter ending 2014-01-31 against a 2014-02-28 calendar
#: -- 28 days apart, unambiguously the same quarter.
TTM_STALENESS_DAYS = 45


def _latest(frame: pd.DataFrame, field: str, column: str = "period_end") -> pd.Series | None:
    """The newest row `frame` holds for `field`, or None. `frame` is a `build_periods`
    output, so "newest" is the newest period end this ticker has reached for that field."""
    if frame is None or frame.empty or "field" not in frame.columns:
        return None
    rows = frame[frame["field"] == field]
    if rows.empty:
        return None
    return rows.loc[pd.to_datetime(rows[column]).idxmax()]


def _is_stale(newest: pd.Series, period: pd.Timestamp) -> bool:
    """Does this trailing-twelve window belong to a different quarter than `period`?

    False when `period` is unknown: a row with no `fiscal_end` has nothing to be stale
    against, and inventing a refusal there would null the first publication event of every
    ticker. Absence of a bound is not a bound of zero.
    """
    if pd.isna(period):
        return False
    end = pd.to_datetime(newest.get("period_end"), errors="coerce")
    return pd.notna(end) and abs((period - end).days) > TTM_STALENESS_DAYS


def _facts_code(visible: pd.DataFrame, field: str) -> str | None:
    """The facts layer's own reason for this field having nothing usable.

    Read off the LATEST filing that mentions the field, never off the whole history: an
    absence explained by a 2011 filing says nothing about a 2024 one, and a code that
    outlives its filing is how a reason table becomes decorative.
    """
    rows = visible[visible["field"] == field]
    if rows.empty:
        return rc.NOT_DISCLOSED
    latest = rows[rows["filing_date"] == rows["filing_date"].max()]
    coded = latest["dc_code"].dropna()
    if not coded.empty:
        return str(coded.iloc[0])
    coded = rows["dc_code"].dropna()
    return str(coded.iloc[-1]) if not coded.empty else None


def _deduced_nci(visible: pd.DataFrame) -> float | None:
    """The non-controlling interest DEDUCED from the filer's own two equity elements.

    Where a filing tags equity on BOTH bases at the same `period_end`, the difference IS the
    NCI -- arithmetic over two filed facts, not an inference about absence. Latest observation
    wins and is carried forward like any other instant, so one deducible date supplies every
    later event until the next observation.

    Measured on the roster: the overlap is RARE (EOG 7 of 63 period ends, TMO 6 of 65, MCD 0 of
    72), so this does not replace the assumed-zero branch, it shrinks it. Where it does fire it
    shows the quantity at stake is tiny -- EOG's two bases are identical to the dollar on 6 of 7
    dates, and TMO's differ by $8-47M against $27-39bn of equity (0.02-0.12%).

    Deliberately NOT written into the `minorityInterest` COLUMN: that column reports what the
    filer tagged for the field, and injecting a computed value would make it non-as-filed. The
    deduction is local to the identity that needs it.
    """
    rows = visible[(visible["field"] == "stockholdersEquity") & visible["value"].notna()]
    if rows.empty:
        return None
    concepts = rows["source_concept"].fillna("").astype(str)
    incl = rows[concepts.str.contains(_EQUITY_INCL_NCI, regex=False)]
    ex = rows[~concepts.str.contains(_EQUITY_INCL_NCI, regex=False)]
    if incl.empty or ex.empty:
        return None
    # One value per period_end per basis, then the latest end carrying both.
    incl_by_end = incl.groupby("period_end")["value"].last()
    ex_by_end = ex.groupby("period_end")["value"].last()
    shared = incl_by_end.index.intersection(ex_by_end.index)
    if shared.empty:
        return None
    latest = max(shared)
    return float(incl_by_end.loc[latest]) - float(ex_by_end.loc[latest])


def _has_valued_fact(visible: pd.DataFrame, field: str) -> bool:
    """Did the filer actually tag a NUMBER for this field in anything visible?

    The difference between "we found nothing" and "we found it and could not use it", which
    `not_disclosed` cannot express and a reader would act on differently.
    """
    rows = visible[visible["field"] == field]
    return bool(len(rows)) and bool(rows["value"].notna().any())


def _qualifiers(visible: pd.DataFrame, field: str) -> list[str]:
    """Codes that describe a value which IS present but is not on the field's nominal basis.

    All three sources are properties of the latest filing that touched the field: a
    `dc_code` in `IS_QUALIFIER` (today only `period_intersection_partial`, riding the
    value-less stub route 3b's strict intersection now emits), and the `adjustment` JSON's
    `basis_qualifier` and `zero_only_retained` keys.
    """
    rows = visible[visible["field"] == field]
    if rows.empty:
        return []
    latest = rows[rows["filing_date"] == rows["filing_date"].max()]
    found = {str(c) for c in latest["dc_code"].dropna() if str(c) in rc.IS_QUALIFIER}
    for blob in latest["adjustment"].dropna():
        try:
            parsed = json.loads(blob)
        except (TypeError, ValueError):
            continue
        if parsed.get("basis_qualifier"):
            found.add(str(parsed["basis_qualifier"]))
        if parsed.get("zero_only_retained"):
            found.add(rc.ZERO_ONLY_RETAINED)
    return sorted(found)


#: The equity concept that ALREADY INCLUDES the non-controlling interest. Where the equity row
#: resolved on this element, adding `minorityInterest` would double-count it.
_EQUITY_INCL_NCI = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


def _total_liabilities_identity(
        row: dict, visible: pd.DataFrame) -> tuple[float | None, str | None]:
    """`totalLiabilities` from the balance sheet's own identity, where no filer tag gave it.

    **§5.1, and the measurement that redirected it.** Register item 8 prescribed exactly this
    identity; the planning interview (decision 30) rejected it as the PRIMARY route on the
    grounds that the filer's own liability legs -- Reg S-X 5-02 captions 21-31 -- had never
    been read, and asked for a route-3b leg-sum measured first. That measurement was run
    (`scripts/measure_total_liabilities_legs.py`, 11 zero-coverage tickers x 4 10-Ks) and it
    refutes the leg-sum, though not for the reason the plan anticipated:

      * **All 44 filings declare a leg-set and NONE declares a `Liabilities` total.** So the
        legs really are the filer's own evidence, exactly as decision 30 argued.
      * **The raw route-3b refusal rate is 68% (30 of 44)**, and it is caused by a single
        element: `us-gaap:CommitmentsAndContingencies`, the Reg S-X 5-02(24) caption filers
        declare under the balance-sheet root as a footnote POINTER and never report a value
        for. Excluding it, 42 of 44 filings (95%) carry a complete leg-set -- the two
        residuals being genuine partials (DUK FY2023 `NotesPayableRelatedPartiesNoncurrent`,
        MCD FY2024 `OperatingLeaseLiabilityNoncurrent`).
      * **But route 3b cannot safely sum them anyway**, and this is the decisive part.
        `_leaf_sum` admits only concepts the catalogue ENUMERATES in `roll_up.any_of`; an
        unlisted sibling is refused only when it is a company EXTENSION, and silently dropped
        when it is us-gaap. The measured leg-sets vary by filer AND by year (2 legs for ETN,
        7 for MCD; EOG and TMO change theirs between 2023 and 2024), so any enumeration is a
        union that cannot be shown to be complete -- and an incomplete union produces a
        balance-sheet total that is SHORT BY A CAPTION and looks entirely plausible. That is
        the `shortTermDebt` defect this whole rebuild exists to remove, re-created on a Tier-1
        field. §B.5 refuted "element names are evidence of what a filer declares" twice; a
        Reg S-X caption list is that reasoning again.

    So the identity wins on the merits, as the plan's own instruction allowed for. It lives
    HERE and not in the facts layer, which is a deliberate departure from §5.1's
    `resolution_method = 'derived_identity'`: `fundamentals_facts` is documented as strictly
    as-filed -- "every row carries a number the filer actually tagged" -- and a derived
    number in that table would contradict the property the publication-event grain rests on.
    The history layer already computes; it stamps `derived_identity` in
    `fundamentals_reason_codes` so the cell never reads as resolved evidence.

    The **NCI bridge** reads the equity ROW rather than the concept priority list, because
    priority-first is not the same as what actually won for a given filing.
    """
    assets, equity = row.get("totalAssets"), row.get("stockholdersEquity")
    if assets is None or equity is None:
        return None, None
    rows = visible[visible["field"] == "stockholdersEquity"]
    concepts = rows[rows["value"].notna()]["source_concept"].dropna()
    incl_nci = bool(len(concepts)) and _EQUITY_INCL_NCI in str(concepts.iloc[-1])
    basis = rc.DERIVED_IDENTITY
    if not incl_nci:
        nci = row.get("minorityInterest")
        if nci is None:
            # Before assuming, try to DEDUCE it: if the filer tags equity on both bases at one
            # period end, their difference is the NCI, and that is two filed facts rather than
            # a claim about absence. Keeps the plain `derived_identity` code, because nothing
            # here rests on interpreting a NULL.
            nci = _deduced_nci(visible)
        if nci is None:
            # Nothing tagged and nothing deducible. A NULL `minorityInterest` conflates "not
            # tagged" with "genuinely zero", and ex-NCI equity plus an UNKNOWN NCI would
            # overstate liabilities -- but a filer that has never tagged an NCI in anything
            # visible has told us it has none, which is its own filing history rather than an
            # asserted rule, and refusing there costs a Tier-1 total for no gain.
            #
            # Read off `visible`, so it is POINT-IN-TIME: at an `as_of` before a filer's first
            # NCI disclosure, zero is what was knowable then. That distinction is load-bearing
            # and easy to get wrong -- measured on LIFETIME facts, TMO looks like it must be
            # refused (38 valued NCI facts); measured point-in-time, its first NCI is filed
            # 2022-02-24 against a history starting 2011-11-04, so a decade of its events had
            # no NCI to know. Only a filer that discloses one in its FIRST filing (LLY, ETN)
            # is refused throughout.
            #
            # What remains on the assumption after the deduction above is therefore: MCD (0 of
            # 72 period ends carry both equity bases, so nothing is ever deducible for it) and,
            # for everyone else, only the events before their first deducible or tagged NCI.
            # Where the quantity IS observable it is negligible -- EOG's two bases agree to the
            # dollar on 6 of 7 dates, TMO's differ by 0.02-0.12% of equity.
            if _has_valued_fact(visible, "minorityInterest"):
                return None, None
            basis = rc.DERIVED_IDENTITY_NCI_ZERO
        else:
            equity = equity + nci
    return float(assets) - float(equity), basis


def _latest_period_known(visible: pd.DataFrame, as_of: pd.Timestamp) -> pd.Timestamp:
    """`fiscal_end`: the latest fiscal period the filer had REPORTED ON by `as_of`.

    Capped at `as_of`, which is what makes the no-look-ahead property structural rather than
    checked. A period end in the future is not knowable, however a filing happens to be
    dated -- and filings ARE occasionally dated ahead of the period they carry: ROP's
    2009-12-31 numbers reached the old builder stamped 2009-11-02, 59 days before the year
    closed, which read as "the full-year figures were public while the year was still
    running". Under the cap the same fixture simply reports the newest period it actually
    could: Q3.

    Monotone non-decreasing by construction -- the visible set only grows and `as_of` only
    advances -- which is the second half of the point-in-time contract. An amendment
    restating an older quarter therefore keeps the LATEST known period here and carries the
    restated one in `amended_fiscal_end`.
    """
    periods = pd.to_datetime(visible["period_of_report"], errors="coerce")
    if periods.isna().all():
        periods = pd.to_datetime(visible["period_end"], errors="coerce")
    known = periods[periods <= as_of]
    return known.max() if not known.empty else pd.NaT


def _instant(instants: pd.DataFrame, field: str, period) -> float | None:
    """A balance-sheet level's latest known value as of `period`.

    NULL only where the field has no instant at all: a level is carried forward because that
    IS its latest known value. Which is also why B.6.6's refused SCHW `cash` period needs a
    qualifier of its own -- there is no null for a gate to find.
    """
    if pd.isna(period):
        return None
    value = carry_latest_known(instants, [period], field)[field].iloc[0]
    return None if pd.isna(value) else float(value)


def _ratio(column: str, numerator, denominator):
    """A formula's value, or None where its inputs cannot produce one. A zero denominator
    is refused rather than allowed to become an infinity that no later check can see."""
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    if column in _RATIOS and float(denominator) == 0.0:
        return None
    return float(_FORMULAS[column][1](float(numerator), float(denominator)))


def _snapshot(ticker: str, visible: pd.DataFrame, event: pd.Series,
              catalogue: Catalogue, guards: PeriodGuards,
              narrow: pd.DataFrame | None = None) -> tuple[dict, list[dict]]:
    """One complete row plus its reason codes, from every fact filed on or before `as_of`.

    `narrow` is the same rows projected to the columns `build_periods` actually reads. It is
    not a micro-optimisation: every filter inside the period engine copies its whole frame,
    and on the full 27-column shape -- half of it Arrow-backed strings -- that copying WAS
    the replay, at ~14 minutes a ticker. See `PERIOD_COLUMNS`.
    """
    refusals: list[dict] = []
    as_of = pd.Timestamp(event["as_of"])
    facts = narrow if narrow is not None else visible
    quarters, ttm, instants = build_periods(facts, catalogue, guards, refusals)
    period = _latest_period_known(visible, as_of)
    regime = visible.sort_values("filing_date")["regime"].dropna()
    regime = str(regime.iloc[-1]) if not regime.empty else None

    # Off the filer's own year ends as of THIS event, not a global calendar: the label has to
    # be knowable from what was filed by `as_of`, and a 52/53-week filer's year ends walk.
    quarter = fiscal_quarter_of_end(period, fiscal_year_ends(facts))

    row: dict = {"ticker": ticker, "as_of": event["as_of"], "fiscal_end": period,
                 "fiscal_quarter": quarter,
                 HISTORY_REGIME: regime,
                 **{c: event[c] for c in HISTORY_PROVENANCE}}
    codes: list[dict] = []

    def code(field: str, dc_code: str) -> None:
        codes.append({"ticker": ticker, "as_of": event["as_of"], "field": field,
                      "dc_code": dc_code,
                      "combined_into": catalogue.combined_into(regime, ticker, field),
                      # Payload of `failed_hard_guard` alone; `_hard_guard` fills it in.
                      "rejected_value": None})

    for field in catalogue.history_fields:
        if field in _FORMULAS:
            continue                                    # computed once the inputs are in
        if catalogue.field(field).kind == INSTANT:
            # Aligned on `as_of`, NOT on `fiscal_end`. The cover-page share count is dated
            # at the filing, days AFTER the period it accompanies -- and it is the only
            # summable count for a multi-class issuer -- so capping instants at `fiscal_end`
            # would delete `sharesOutstanding` for the current period on every filer.
            value, reason = _instant(instants, field, as_of), None
        else:
            # The newest quarter end this field has reached, REQUIRED to be this row's own
            # quarter. `trailing_twelve`'s contract is four discrete quarters or nothing, and
            # carrying the last computable TTM forward is precisely the staircase (1,622 of
            # 26,242 consecutive `totalRevenue` pairs frozen) that this rebuild removed.
            # Coverage drops on purpose.
            #
            # Two ways a TTM goes missing, and only the first used to be handled: a REFUSED
            # window stays NULL with its own code, but a window that stops being COMPUTABLE
            # because its input quarters dried up left `_latest` returning the newest row
            # that had ever existed -- an uncapped forward-fill. `_is_stale` is the cap.
            newest = _latest(ttm, field)
            if newest is not None and _is_stale(newest, period):
                # A window from another quarter is not this row's value. Without this the
                # cell reads as a live measurement forever after the field stops resolving:
                # see `rc.STALE_TTM` for the 27 pairs it was frozen on.
                newest = None
                reason = rc.STALE_TTM
            else:
                reason = (str(newest["dc_code"]) if newest is not None
                          and pd.notna(newest.get("dc_code")) else None)
            value = (None if newest is None or pd.isna(newest["value"])
                     else float(newest["value"]))
        if value is None and reason is None:
            reason = _facts_code(visible, field)
        if value is None and reason is None and _has_valued_fact(visible, field):
            # The facts ARE there and the window still could not be assembled, so
            # `not_disclosed` would be a false statement: the filer disclosed it. The only
            # way a duration field reaches here is a window short of four discrete quarters
            # -- anything the guards refused already carries its own code. Measured on AAPL:
            # 4 cells, all at the FIRST publication event, where one visible filing cannot
            # make a trailing twelve months by construction.
            reason = rc.INSUFFICIENT_QUARTERS
        row[field], gated = _gate(catalogue, regime, field, value)
        if row[field] is None:
            code(field, gated or reason or rc.NOT_DISCLOSED)
        else:
            for qualifier in _qualifiers(visible, field):
                code(field, qualifier)
        _break_code(catalogue, field, period, code)

    # After the field loop, so `minorityInterest` (tier 3) is already resolved -- the NCI
    # bridge needs it and `history_fields` is tier-ordered, which puts tier-1
    # `totalLiabilities` first.
    if row.get("totalLiabilities") is None:
        row["totalLiabilities"], basis = _total_liabilities_identity(row, visible)
        if row["totalLiabilities"] is not None:
            # The absence code the loop just wrote is now false: the cell is not absent, it
            # is derived. Replace rather than accumulate, or the row says both.
            codes[:] = [c for c in codes if c["field"] != "totalLiabilities"
                        or c["dc_code"] in rc.IS_QUALIFIER]
            code("totalLiabilities", basis)

    for column, (inputs, _) in _FORMULAS.items():
        row[column] = _ratio(column, *(row.get(name) for name in inputs))
        if row[column] is None:
            missing = next((n for n in inputs if row.get(n) is None), None)
            code(column, next((c["dc_code"] for c in codes if c["field"] == missing),
                              rc.NOT_DISCLOSED))

    for column, source in _QUARTER_COLUMNS.items():
        newest = _latest(quarters, source)
        row[column] = (None if newest is None or pd.isna(newest["value"])
                       else float(newest["value"]))
        if row[column] is None:
            code(column, _facts_code(visible, source) or rc.NOT_DISCLOSED)

    for refusal in refusals:
        code(refusal["field"], str(refusal["dc_code"]))
    # LAST, so it sees the ratios and the discrete-quarter columns too, and so nothing
    # downstream can put an impossible value back. See `HARD_GUARDS` for why there are four.
    _hard_guard(ticker, event["as_of"], row, codes)
    return row, codes


def _hard_guard(ticker: str, as_of, row: dict, codes: list[dict]) -> None:
    """Null every `HARD_GUARDS` violation in `row`, recording what was thrown away.

    In place, on the row about to be written -- decision 46's "applied before the write".
    The refused number goes onto the reason-code row as `rejected_value` rather than into a
    log line, because a derived cell (a TTM, the `derived_identity` total) has NO fact row
    to go back to and the number would otherwise be unrecoverable. That is what makes "did
    this guard null something correct?" a query instead of an archaeology exercise.

    Replaces any absence code the field already earned: a value that was PRESENT and refused
    is not `not_disclosed`, and a row asserting both would be incoherent.
    """
    for field, is_impossible in HARD_GUARDS.items():
        value = row.get(field)
        if value is None or pd.isna(value) or not is_impossible(float(value)):
            continue
        row[field] = None
        codes[:] = [c for c in codes if c["field"] != field]
        codes.append({"ticker": ticker, "as_of": as_of, "field": field,
                      "dc_code": rc.FAILED_HARD_GUARD, "combined_into": None,
                      "rejected_value": float(value)})


def _gate(catalogue: Catalogue, regime: str | None, field: str,
          value: float | None) -> tuple[float | None, str | None]:
    """Regime gating, applied HERE and not in the facts layer.

    A `regime_gated` field is UNDEFINED for a regime whose register cell says so -- a bank
    has no `AssetsCurrent` because Reg S-X Article 9 has no current/non-current split -- so
    any value that reached us is a resolution accident and is dropped. A field that is merely
    `expected_absent` keeps whatever it resolved (PGR really does tag capex) and only has its
    ABSENCE explained. The facts layer stays regime-agnostic about absence on purpose, so the
    register can be re-measured against it instead of being assumed by it.
    """
    if not regime or not catalogue.expected_absent(regime, field):
        return value, None
    if catalogue.field(field).regime_gated:
        return None, rc.NOT_APPLICABLE_FOR_REGIME
    return value, (None if value is not None else rc.NOT_APPLICABLE_FOR_REGIME)


def _break_code(catalogue: Catalogue, field: str, period, code) -> None:
    """Flag a cell whose own trailing year contains a definitional discontinuity.

    The window is one year because that is the span over which the cell is compared with
    itself: a YoY or TTM read that straddles ASC 842 sees a step that is real accounting and
    not a data defect. Outside the window both sides are internally comparable again.
    """
    effective = catalogue.regime_break_effective(field)
    if effective is None or pd.isna(period):
        return
    if pd.Timestamp(period) - pd.Timedelta(days=365) < effective <= pd.Timestamp(period):
        code(field, rc.REGIME_BREAK)


# ---------------------------------------------------------------------- entry point ---

def build_ticker_history(ticker: str, facts, *, catalogue: Catalogue | None = None,
                         guards: PeriodGuards | None = None) -> pd.DataFrame:
    """One ticker's `fundamentals_history` frame -- 69 columns, one row per publication event.

    The signature the acceptance test has pinned since Phase 1
    (`tests/data_extract/test_fundamentals_point_in_time.py`). `facts` is normally the
    ticker's `fundamentals_facts` rows; a `companyfacts`-shaped mapping is accepted for
    SYNTHETIC FIXTURES ONLY, via `facts_frame_from_companyfacts`.
    """
    return build_ticker(ticker, facts, catalogue=catalogue, guards=guards).history


def build_ticker(ticker: str, facts, *, catalogue: Catalogue | None = None,
                 guards: PeriodGuards | None = None) -> TickerHistory:
    """`build_ticker_history` plus the dense reason-code side table.

    The replay is O(filings): the per-ticker facts frame is loaded ONCE and sliced in
    memory, never re-queried per event (Phase 10 names this explicitly). Every event
    rebuilds the whole snapshot from `filing_date <= as_of`, which is the `as_of_cutoff`
    replay the old builder supported as an audit-only debug path, promoted to the production
    loop -- so the no-leakage property is a consequence of the algorithm rather than a
    guard bolted onto it.
    """
    catalogue = catalogue or load_catalogue()
    guards = guards or load_guards()
    frame = _normalise(facts, catalogue)
    columns = catalogue.history_columns
    assert len(columns) == 69, f"the column contract is {len(columns)}, not 69"
    events = publication_events(frame)
    if events.empty:
        return TickerHistory(pd.DataFrame(columns=columns),
                             pd.DataFrame(columns=list(_CODE_COLUMNS)))

    narrow = _period_projection(frame)
    filed = frame["filing_date"].to_numpy()
    rows, codes = [], []
    for _, event in events.iterrows():
        # A POSITIONAL slice of a filing-date-sorted frame, not a boolean mask: the mask
        # allocates and then fancy-takes all 27 columns once per event, and `iloc[:n]` on a
        # sorted frame is a view. Correct because `_normalise` sorts by `filing_date`, so
        # "filed on or before as_of" is a prefix by construction.
        upto = int(filed.searchsorted(event["as_of"].to_datetime64(), side="right"))
        row, row_codes = _snapshot(ticker, frame.iloc[:upto], event, catalogue, guards,
                                   narrow.iloc[:upto])
        rows.append(row)
        codes.extend(row_codes)

    history = pd.DataFrame(rows).reindex(columns=columns)
    # Pin the date dtypes. An all-NaT `amended_fiscal_end` (a ticker that never amended)
    # infers `datetime64[s]` while a populated one is `[us]`, so two builds of the same
    # ticker compare unequal on dtype alone -- which `diff_against_stored` would then have to
    # forgive, and forgiving a dtype is one step from forgiving a value.
    for column in ("as_of", "fiscal_end", "amended_fiscal_end"):
        history[column] = pd.to_datetime(history[column],
                                         errors="coerce").astype("datetime64[ns]")
    # Nullable Int64, not float: the label is Q1-Q4 and `WHERE fiscal_quarter = 3` should not
    # be a float comparison, but a ticker whose earliest events predate its first annual
    # filing has no fiscal calendar yet and must stay NULL rather than become 0. `sql_type`
    # maps an integer dtype to BIGINT and `copy_load` writes `pd.NA` as an empty CSV field,
    # so this round-trips through both the DDL and the COPY path.
    history["fiscal_quarter"] = history["fiscal_quarter"].astype("Int64")
    # Same reasoning for the text columns: an all-None `amended_fields` infers `object` while
    # a populated one infers `string`.
    for column in ("publication_form", "amended_fields", HISTORY_REGIME):
        history[column] = history[column].astype(object).where(history[column].notna(), None)
    # And every VALUE column to float64, even when it is entirely null for this ticker. This
    # is not cosmetic: `sql/schema.sql` is applied only when Postgres INITIALISES a volume, so
    # on an existing one `store.save` creates the table from the FIRST frame it is handed via
    # `ensure_table`'s dtype inference. An all-None `object` column becomes **TEXT**, and every
    # later ticker's real number is then stored as a string -- measured on the first live run,
    # where VRT (no `minorityInterest`, no `restrictedCash`) created both as TEXT and APA's
    # values came back as `'1997000000.0'`. Caught by `diff_against_stored` on the second run,
    # which is precisely the drift the append-only guard exists to make visible.
    for column in columns:
        if column not in (*HISTORY_KEYS, HISTORY_REGIME, *HISTORY_PROVENANCE):
            history[column] = pd.to_numeric(history[column], errors="coerce").astype(float)
    history["is_amendment"] = history["is_amendment"].astype(bool)
    reason = pd.DataFrame(codes, columns=list(_CODE_COLUMNS)).drop_duplicates(
        subset=["ticker", "as_of", "field", "dc_code"])
    # float64 even when it is entirely null -- which it is for every ticker no guard ever
    # fires on, i.e. almost all of them. `store.ensure_table` infers the column type from the
    # FIRST frame it is handed, and an all-None object column becomes TEXT; that is exactly
    # how a real number once landed in Postgres as the string '1997000000.0'.
    reason["rejected_value"] = pd.to_numeric(reason["rejected_value"],
                                             errors="coerce").astype(float)
    unknown = sorted(set(reason["dc_code"]) - rc.ALL_CODES)
    assert not unknown, f"{ticker}: reason code(s) outside the declared set: {unknown}"
    _assert_grain(ticker, history)
    return TickerHistory(history, reason)


#: `fundamentals_reason_codes`' grain, then its two payloads. Neither payload is a key:
#: `combined_into` names at most one destination per field and `rejected_value` at most one
#: refused number per (field, code), so putting either in the key would let two rows disagree
#: about the same fact rather than making the second one impossible to write.
_CODE_COLUMNS: tuple[str, ...] = ("ticker", "as_of", "field", "dc_code", "combined_into",
                                  "rejected_value")


def _assert_grain(ticker: str, history: pd.DataFrame) -> None:
    """The two invariants §5.0 makes structural, asserted anyway.

    Under this grain neither can fire -- `as_of` IS a filing date and the visible fact set
    only grows -- which is exactly what makes them a good test rather than a redundant one:
    if either ever trips, the grain has been broken somewhere upstream.
    """
    assert not history.duplicated(["ticker", "as_of"]).any(), \
        f"{ticker}: two rows share an (ticker, as_of) -- the same-day collapse failed"
    ends = pd.to_datetime(history["fiscal_end"])
    assert (ends.diff().dropna() >= pd.Timedelta(0)).all(), \
        f"{ticker}: fiscal_end is not monotone non-decreasing in as_of"
    lag = (pd.to_datetime(history["as_of"]) - ends).dt.days.dropna()
    assert (lag >= 0).all(), f"{ticker}: as_of precedes fiscal_end -- look-ahead leak"


#: The only columns `build_periods` reads. Projecting to them before the replay is the single
#: largest cost in this module: profiled on AAPL (7,262 facts / 69 events), every boolean
#: filter inside `quarterize` and `_drop_annual_masquerading_as_quarter` was fancy-taking all
#: 27 columns, and half of those arrive from parquet or Postgres as ARROW-backed strings whose
#: `take` goes through `pyarrow.compute` one call per slice -- 98,646 of them in a single
#: `build_periods`. The engine never looks at `adjustment`, `role_uri` or `roll_up_children`;
#: carrying them through 69 replays cost about ten minutes a ticker.
PERIOD_COLUMNS: tuple[str, ...] = (
    "ticker", "field", "duration_type", "period_start", "period_end", "period_days",
    "value", "filing_date", "source_concept", "fiscal_year", "fiscal_period")


def _period_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` reduced to `PERIOD_COLUMNS`, with its string columns off Arrow.

    Both halves matter. The projection cuts the width; `astype(object)` cuts the per-slice
    Arrow round-trip, which is what actually dominates -- a numpy object take is a pointer
    copy, an Arrow take rebuilds the array through `pyarrow.compute`.
    """
    out = frame[[c for c in PERIOD_COLUMNS if c in frame.columns]].copy()
    for column in ("field", "duration_type", "source_concept", "fiscal_period"):
        if column in out.columns and not pd.api.types.is_object_dtype(out[column]):
            out[column] = out[column].astype(object)
    return out


def _normalise(facts, catalogue: Catalogue) -> pd.DataFrame:
    """The facts frame with its dates as timestamps, and the missing columns a synthetic
    fixture may not carry filled in."""
    if not isinstance(facts, pd.DataFrame):
        facts = facts_frame_from_companyfacts(facts, catalogue)
    out = facts.copy()
    for column in ("filing_date", "period_of_report", "period_start", "period_end"):
        out[column] = pd.to_datetime(out.get(column), errors="coerce")
    for column, default in (("is_amendment", False), ("dc_code", None),
                            ("adjustment", None), ("regime", None), ("form", "10-Q"),
                            ("accession_number", ""), ("source_concept", None)):
        if column not in out.columns:
            out[column] = default
    out["is_amendment"] = out["is_amendment"].fillna(False).astype(bool)
    if "period_of_report" in out and out["period_of_report"].isna().all():
        out["period_of_report"] = out["period_end"]
    return out.sort_values("filing_date")


def facts_frame_from_companyfacts(blob: dict, catalogue: Catalogue) -> pd.DataFrame:
    """A `fundamentals_facts`-shaped frame from a raw `companyfacts` mapping.

    SYNTHETIC FIXTURES ONLY, and never on the production path -- §B.5 measured why:
    companyfacts publishes no company-extension taxonomy and silently drops dimensioned
    facts, so a resolution built on it is a different measurement from the one the linkbase
    walk performs. It exists because the pinned unit test in
    `test_fundamentals_point_in_time.py` builds its ROP-shaped fixture in that layout, and a
    test the build cannot read is a test that passes by returning nothing.

    The concept -> field map comes from the catalogue's OWN declared concepts, so no second
    priority vocabulary is introduced here; a concept no field declares is skipped.
    """
    by_concept: dict[str, str] = {}
    for name in catalogue.extracted_fields:
        spec = catalogue.field(name)
        for concept in [spec.total_concept(), *spec.fallback_concepts()]:
            if concept:
                by_concept.setdefault(concept.split(":")[-1], name)
    rows = []
    for concepts in (blob.get("facts") or {}).values():
        for concept, payload in concepts.items():
            field = by_concept.get(concept.split(":")[-1])
            if field is None:
                continue
            for unit, entries in (payload.get("units") or {}).items():
                for i, entry in enumerate(entries):
                    rows.append({
                        "ticker": "FIXTURE", "accession_number": f"{concept}-{unit}-{i}",
                        "field": field, "fiscal_year": pd.Timestamp(entry["end"]).year,
                        "fiscal_period": entry.get("fp", "NA"),
                        "form": entry.get("form", "10-Q"),
                        "filing_date": entry.get("filed"), "is_amendment": False,
                        "period_of_report": entry["end"], "regime": None,
                        "period_start": entry.get("start"), "period_end": entry["end"],
                        "value": entry.get("val"), "unit": unit,
                        "source_concept": concept, "dc_code": None, "adjustment": None})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # One accession per (filing date, form), so the event ladder sees filings and not facts.
    frame["accession_number"] = (frame["filing_date"].astype(str) + "-"
                                 + frame["form"].astype(str))
    frame["period_days"] = (pd.to_datetime(frame["period_end"])
                            - pd.to_datetime(frame["period_start"])).dt.days
    frame["duration_type"] = [
        INSTANT if pd.isna(d) else ("annual" if d > 300 else "quarterly")
        for d in frame["period_days"]]
    return frame


# ------------------------------------------------------------------- immutability ---

#: The `fundamentals_facts` columns the replay reads. Projected, never `SELECT *`: the table
#: is ~28 columns x ~14k rows per ticker and the replay touches one ticker at a time.
FACT_COLUMNS: tuple[str, ...] = (
    "ticker", "accession_number", "field", "fiscal_year", "fiscal_period", "duration_type",
    "form", "filing_date", "is_amendment", "period_of_report", "regime", "period_start",
    "period_end", "period_days", "value", "unit", "source_concept", "dc_code", "adjustment")


def diff_against_stored(stored: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Every cell an already-stored row would CHANGE if it were rebuilt today.

    The store has no append-only primitive -- `store.save` is `INSERT ... ON CONFLICT DO
    UPDATE` on `(ticker, as_of)` -- so "rows are immutable once written" was, until this
    function, an assertion with nothing behind it: a re-run after a resolution change would
    silently overwrite history, which is the exact failure the publication-event grain exists
    to prevent. Recompute-diff-raise (decision 28) makes immutability PROVABLE, gives Phase
    5b's `pit_leak` check for free, and catches resolution drift the moment it appears
    instead of after a cube has been trained on it. Cost: one ~62-row read per ticker.

    Compared EXACTLY, with no tolerance. If it ever trips on floating-point noise we want to
    find that out rather than to have pre-forgiven it -- `DOUBLE PRECISION` round-trips bit
    for bit, so a difference is a real change until proven otherwise.
    """
    if stored is None or stored.empty or rebuilt.empty:
        return pd.DataFrame(columns=["as_of", "column", "stored", "rebuilt"])
    # DATE columns come back from Postgres as `datetime.date`, which never equals a
    # `Timestamp`; normalise both sides or every row reads as drifted.
    left = _keyed_by_as_of(stored)
    right = _keyed_by_as_of(rebuilt)
    shared_rows = left.index.intersection(right.index)
    shared_cols = [c for c in left.columns if c in right.columns and c != "ticker"]
    rows = []
    for as_of in shared_rows:
        for column in shared_cols:
            was, now = left.at[as_of, column], right.at[as_of, column]
            if pd.isna(was) and pd.isna(now):
                continue
            if pd.isna(was) or pd.isna(now) or was != now:
                rows.append({"as_of": as_of, "column": column,
                             "stored": was, "rebuilt": now})
    return pd.DataFrame(rows, columns=["as_of", "column", "stored", "rebuilt"])


def _keyed_by_as_of(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("as_of", "fiscal_end", "amended_fiscal_end"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.set_index("as_of").sort_index()


def build_fundamentals_history(context, tickers: list[str], *,
                              rebuild_history: bool = False) -> None:
    """`fundamentals_facts` -> `fundamentals_history` + `fundamentals_reason_codes`.

    Append-only in normal operation: a second run over unchanged facts appends **0** rows and
    raises nothing. Where a stored row WOULD change, the run stops on that ticker and prints
    the diff -- because the alternative is publishing a silently different past. Pass
    `rebuild_history=True` (CLI `--rebuild-history`) to delete the ticker's rows from both
    tables and rebuild from the facts already stored; no network is involved, which is the
    whole point of having it separate from `--rebuild`.
    """
    from src.data_store.schema import Tables            # local: avoids a package cycle

    catalogue = load_catalogue()
    guards = load_guards()
    for ticker in tickers:
        facts = context.store.load(Tables.fundamentals_facts, columns=list(FACT_COLUMNS),
                                   where={"ticker": ticker}, optional=True)
        if facts is None:
            context.log.info("history: %s has no stored facts -- skipped", ticker)
            continue
        built = build_ticker(ticker, facts, catalogue=catalogue, guards=guards)
        if built.history.empty:
            continue
        stored = context.store.load(Tables.fundamentals_history, columns=None,
                                    where={"ticker": ticker}, optional=True)
        history, codes = built.history, built.reason_codes
        if rebuild_history:
            deleted = context.store.delete(Tables.fundamentals_history, {"ticker": ticker})
            context.store.delete(Tables.fundamentals_reason_codes, {"ticker": ticker})
            context.log.warning("history: %s REBUILT -- %d row(s) deleted and recomputed. "
                                "Log this in the phase report: a rebuild re-derives numbers "
                                "under whatever model is already trained on them.",
                                ticker, deleted)
        elif stored is not None:
            drift = diff_against_stored(stored, history)
            if not drift.empty:
                context.log.error("history: %s would CHANGE %d already-published cell(s) "
                                  "across %d row(s) -- refusing to overwrite. Re-run with "
                                  "--rebuild-history to accept:\n%s", ticker, len(drift),
                                  drift["as_of"].nunique(), drift.head(20).to_string())
                raise ValueError(
                    f"{ticker}: {len(drift)} stored fundamentals_history cell(s) would "
                    "change; history is append-only (pass --rebuild-history to rebuild)")
            known = set(pd.to_datetime(stored["as_of"]))
            new = ~pd.to_datetime(history["as_of"]).isin(known)
            history, codes = history[new.values], codes[
                pd.to_datetime(codes["as_of"]).isin(set(history[new.values]["as_of"]))]
        if history.empty:
            context.log.info("history: %s already current (0 new events)", ticker)
            continue
        context.store.save(Tables.fundamentals_history, history)
        if not codes.empty:
            context.store.save(Tables.fundamentals_reason_codes, codes)
        context.log.info("history: %s +%d event row(s), %d reason code(s)",
                         ticker, len(history), len(codes))
