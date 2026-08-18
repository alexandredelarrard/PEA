"""
fetch_fundamentals_edgar.py
-----------------------------
edgartools-based, per-filing SEC fundamentals retrieval -> `fundamentals_facts`
(accession-grain, amendment-aware raw facts). Replaces the broken WIP prototype
that used to live at this same path (`fetch_fundamentals_edgartool.py`,
class `FetchFundamentals` -- see git history for the original).

Why per-filing, not the aggregated companyfacts JSON (`fetch_fundamentals.py`):
edgartools' own aggregated-facts convenience path (`Company.get_financials()` /
`EntityFacts`) parses the SAME underlying SEC companyfacts JSON as the old file,
so it would not fix "missing quarters". Walking each filing's OWN XBRL instance
document independently is the only path that is structurally more complete, and
it naturally gives accession-level provenance for amendment tracking.

Reuses `fundamentals_tags.py` (the same tag-candidate lists as `fetch_fundamentals.py`
-- this is what closes the confirmed MAA `capex` gap) and `fundamentals_periods.py`
(fiscal-period resolution + Q1-Q4 decumulation, native-fiscal-year-keyed).

One field is NOT XBRL-sourced: `employees`. Headcount has no GAAP concept, so it
is parsed out of each 10-K's body text by `fundamentals_employees.py` and
appended to that filing's tagged facts as an ordinary instant row (see the loop
in `build_ticker_facts_edgar`). This replaces the separate
`structure/fetch_employees_edgar.py` fetcher and its `employees_history` table:
the 10-K is already open here, so listing and downloading those filings a second
time bought nothing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
from edgar import Company, set_identity

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import manifest_window, record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals.fundamentals_employees import (
    employee_fact_frame, history_by_ticker,
)
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    annual_flow, backfill_fiscal_period_by_filing_order, decumulate_quarterly_flow,
    derive_bank_cash, derive_missing_pretax_income, derive_missing_total_liabilities,
    drop_derived_q4_for_partial_fiscal_years, instant_stock,
    normalize_fiscal_period_label, reassign_misordered_instant_facts,
    resolve_fiscal_year_by_filing_calendar,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    CLASS_OF_STOCK_AXIS_SUFFIX,
    COVER_PAGE_SHARES_MAX_LAG_DAYS, COVER_PAGE_SHARES_TAG, EMPLOYEES_FIELD,
    PARENT_OWNERSHIP_EQUITY_TOLERANCE,
    PARENT_OWNERSHIP_MISMATCH_MIN, PARENT_OWNERSHIP_PERCENTAGE_TAG,
    SHARE_CLASS_CONVERSION_RATIO_TAG, SHARE_CLASS_EQUIVALENT_PERCENTAGE_MARKER,
    EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FIELD_TAG_DENYLIST,
    FINANCIALS_TOPLINE_DOMINANCE,
    FINANCIALS_TOPLINE_MARKERS, FISCAL_YEAR_CONTEXT_DAYS, FLOW_TAGS,
    LATEST_DURATION_TAGS, NON_NEGATIVE_STOCK_FIELDS, PARTIAL_REVENUE_MATERIALITY,
    PARTIAL_REVENUE_TAGS, SGA_GA_ONLY_TAG, SGA_SM_COMPANION_TAG,
    SHARE_CLASS_COMPONENT_FIELDS,
    SHARE_COUNT_MAGNITUDE_FIELDS, SHARE_COUNT_MIN_ABS,
    SHARES_OUTSTANDING_FIELD, SHARES_TAGS, STOCK_TAGS,
    TOTAL_REVENUE_TAG, XBRL_PARSE_ATTEMPTS, XBRL_RETRY_BACKOFF_SECONDS,
)

FLOW_FIELD_TAGS: dict[str, list[str]] = {**FLOW_TAGS, **EXTRA_FLOW_TAGS}
INSTANT_FIELD_TAGS: dict[str, list[str]] = {
    **STOCK_TAGS, **EXTRA_STOCK_TAGS, **SHARES_TAGS, **LATEST_DURATION_TAGS,
}
ALL_FIELD_TAGS: dict[str, list[str]] = {**FLOW_FIELD_TAGS, **INSTANT_FIELD_TAGS}

_FACTS_COLS = ["ticker", "cik", "accession_number", "field", "fiscal_year", "fiscal_period",
              "duration_type", "form", "filing_date", "period_start", "period_end",
              "value", "unit", "source_tag", "is_amendment", "amends_accession",
              "derived", "derived_from_accessions", "fiscal_period_source"]


def _configure_identity() -> None:
    """SEC EDGAR requires a real, descriptive User-Agent -- fail loudly if unset,
    matching this repo's `sec_utils.py` convention. Replaces the WIP file's
    hardcoded placeholder (`set_identity("Jane Doe jdoe@example.com")`)."""
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. SEC EDGAR blocks requests without a "
            "descriptive User-Agent (name + email). Add it to your .env file, e.g.\n"
            '  SEC_USER_AGENT="Your Name your.email@example.com"'
        )
    set_identity(ua)


def _class_of_stock_axis_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(fact carries a class-of-stock axis, fact carries it as its ONLY axis).

    edgartools exposes a fact's dimensions TWICE: one `dim_<prefix>_<Axis>` column per
    axis used anywhere in the filing (NaN for the facts that do not use it), plus a flat
    `dimension`/`member` pair -- which holds only the FIRST axis, so on the flat pair a
    fact dimensioned by share class AND something else is indistinguishable from a
    single-axis one. Counting the `dim_*` columns is the only way to tell them apart,
    and the distinction carries the whole safety property of the summing below:
    confirmed on CME, whose per-class `CommonStockSharesOutstanding` appears BOTH on the
    class axis alone and on (class axis + `StatementEquityComponentsAxis`), so a rule
    reading only `dimension` would count every class twice.

    Frames carrying no `dim_*` column at all (the synthetic unit-test fixtures) fall
    back to the flat pair, where there is no hidden second axis to miss.
    """
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if dim_cols:
        present = df[dim_cols].notna()
        class_cols = [c for c in dim_cols if c[4:].endswith(CLASS_OF_STOCK_AXIS_SUFFIX)]
        has_class = (present[class_cols].any(axis=1) if class_cols
                     else pd.Series(False, index=df.index))
        return has_class, has_class & (present.sum(axis=1) == 1)
    axis = df.get("dimension", pd.Series(index=df.index, dtype=object)).astype(str)
    has_class = df["_dimensioned"] & axis.str.endswith(CLASS_OF_STOCK_AXIS_SUFFIX)
    return has_class, has_class


class ShareClassBasis(NamedTuple):
    """The factors a filing publishes for putting its own class counts onto ONE basis:
    the units of the class the ticker trades as, over the whole consolidated group.

    `multipliers`  member -> base-class shares per share of that class.
    `equivalent_pct` the junior class's economic worth as a fraction of the senior one.
    `parent_pct`   a tagged parent ownership percentage (NOT necessarily group-level --
                   see `equity_parent_share`).
    `equity_parent_share` parent equity / consolidated equity, the INDEPENDENT check that
                   `parent_pct` really describes the whole consolidated group.

    All optional -- every field is empty for the ordinary single-basis filer, which is
    why applying them can only ever refine a total, never remove one. See
    `SHARE_CLASS_CONVERSION_RATIO_TAG` / `PARENT_OWNERSHIP_PERCENTAGE_TAG` in
    `fundamentals_tags.py` for the measured per-filer evidence behind each.
    """
    multipliers: dict[str, float]
    equivalent_pct: float | None
    parent_pct: float | None
    equity_parent_share: float | None


def _share_class_basis(df: pd.DataFrame, sole_class_axis: pd.Series) -> ShareClassBasis:
    """Read the conversion / ownership factors out of a filing's OWN facts.

    Must be called BEFORE the candidate-tag filter: none of these concepts is a candidate
    for any logical field (they are ratios and percentages, not amounts), so they are gone
    from the frame by the time the coalesce runs.
    """
    bare = df["_bare"].astype(str)
    probe = pd.to_numeric(df["_probe"], errors="coerce")
    member = next((df[c] for c in ("member", "dimension_member_label") if c in df.columns),
                  pd.Series(index=df.index, dtype=object))

    ratios = df[(bare == SHARE_CLASS_CONVERSION_RATIO_TAG) & sole_class_axis & (probe > 1.0)]
    multipliers = {str(m): float(v) for m, v in
                   zip(member[ratios.index], probe[ratios.index]) if pd.notna(m)}

    def _scalar(mask: pd.Series) -> float | None:
        """The LATEST-dated qualifying value. Latest, because these facts recur across a
        filing's comparative periods and one of them is the current disclosure -- and, for
        IBKR, because the ownership footnote also restates the ORIGINAL 2007 IPO split
        (10%/90%), which as a stale fact would gross the count up by 10x."""
        sub = df[mask & probe.between(0.0, 1.0, inclusive="neither")]
        if sub.empty:
            return None
        dates = pd.to_datetime(sub["period_end"], errors="coerce")
        # `idxmax` skips NaT; an undated fact only wins when nothing in the group is dated.
        return float(probe[dates.idxmax() if dates.notna().any() else sub.index[-1]])

    def _amount(tag: str) -> float | None:
        """The latest undimensioned value of a balance-sheet concept, read directly rather
        than through the coalesce (which returns only ONE of the two equity bases)."""
        sub = df[(bare == tag) & ~df["_dimensioned"] & probe.notna()]
        if sub.empty:
            return None
        dates = pd.to_datetime(sub["period_end"], errors="coerce")
        return float(probe[dates.idxmax() if dates.notna().any() else sub.index[-1]])

    parent_equity = _amount("StockholdersEquity")
    total_equity = _amount("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    return ShareClassBasis(
        multipliers=multipliers,
        equivalent_pct=_scalar(bare.str.contains(SHARE_CLASS_EQUIVALENT_PERCENTAGE_MARKER)
                               & ~df["_dimensioned"]),
        parent_pct=_scalar(bare == PARENT_OWNERSHIP_PERCENTAGE_TAG),
        equity_parent_share=(parent_equity / total_equity
                             if parent_equity is not None and total_equity else None),
    )


def _to_traded_class_units(values: pd.Series, members: pd.Series,
                           basis: ShareClassBasis) -> pd.Series:
    """Per-class counts rescaled into the traded class's units.

    An explicit per-class ratio wins. Absent one, an `equivalent_pct` scales the SMALLEST
    class -- the senior one, since a share worth ~1,500 of the other is necessarily the
    rarer. Everything else keeps its own units (x1)."""
    mult = members.map(basis.multipliers).astype(float).fillna(1.0)
    if basis.equivalent_pct and not (mult > 1.0).any() and len(values) > 1:
        mult.loc[[values.idxmin()]] = 1.0 / basis.equivalent_pct
    return values * mult


def _grossed_up_to_consolidated(total: float, values: pd.Series,
                                basis: ShareClassBasis) -> float:
    """`total` divided by the parent's ownership percentage -- but only once the filing has
    answered TWO questions about itself, because getting either wrong corrupts the count by
    a multiple (see `PARENT_OWNERSHIP_PERCENTAGE_TAG`).

    1. Does the percentage describe the WHOLE CONSOLIDATED GROUP? `MinorityInterest-
       OwnershipPercentageByParent` is just as often used for one SUBSIDIARY or joint
       venture, and that reading is catastrophic here -- caught on live data, where CMCSA
       tags 0.30 and UHS 0.20 for holdings of theirs, which grossed their counts up by
       3.33x and 5.00x respectively. The test is independent evidence from the same
       filing: the percentage must match the parent's share of consolidated EQUITY. IBKR
       0.266 against 5,363/20,472 = 0.262 agrees; CMCSA's 0.30 against ~0.98 does not.
       No equity evidence at all -> no gross-up, deliberately failing closed.
    2. Does the class sum ALREADY cover the non-controlling holders? CVNA's Class A is
       65.4% of its class sum against a tagged 65%, because each Class B share is paired
       1:1 with an exchangeable LLC unit -- dividing would count the Garcia interest
       twice. IBKR's Class A is 99.99998% against a tagged 26.6%: its LLC members hold no
       paired common stock at all, which is the Up-C signature this corrects.
    """
    if not basis.parent_pct or total <= 0:
        return total
    if basis.equity_parent_share is None or abs(
            basis.equity_parent_share - basis.parent_pct) > PARENT_OWNERSHIP_EQUITY_TOLERANCE:
        return total
    if float(values.max()) / total - basis.parent_pct <= PARENT_OWNERSHIP_MISMATCH_MIN:
        return total
    return total / basis.parent_pct


def _cover_page_class_total(merged: pd.DataFrame, sole_class_axis: pd.Series,
                            component_ok: pd.Series,
                            basis: ShareClassBasis) -> pd.DataFrame:
    """ONE synthetic `sharesOutstanding` row per cover-page date, summing a MULTI-CLASS
    filer's per-class cover-page counts into the company total (see
    `CLASS_OF_STOCK_AXIS_SUFFIX` in `fundamentals_tags.py` for why the cover-page tag is
    the only summable one, with the measured per-class figures).

    Restricted to `COVER_PAGE_SHARES_TAG` facts whose ONLY dimension is the class axis
    and that already pass the per-fact guards (`component_ok`: the sign rule and the
    per-issuer deny-list). The share-count MAGNITUDE floor is deliberately NOT among
    them and is applied by the caller to the TOTAL instead: a real class routinely sits
    below it on its own (BRK-B Class A 505,697 shares, ERIE Class B 2,542, SPG Class B
    8,000, CME Class B-1..B-4 a few hundred each), so screening components would drop
    exactly the small classes the sum exists to add in.

    Per-class counts are rescaled into the TRADED class's units before summing
    (`_to_traded_class_units`) and the sum is put on the consolidated basis after
    (`_grossed_up_to_consolidated`) -- both from factors the filer tags itself, both no-ops
    for a filer that tags none.

    Two ways a group is NOT simply summed:
      * a member whose value equals the sum of all the others is a ROLL-UP of them, not
        a sibling, so it is taken alone -- what stops V's `CommonClassB1B2AndB3` and
        BRK-B's `EquivalentClassA` from double-counting. Two classes of EQUAL size both
        satisfy that arithmetic, so the rule only fires when exactly ONE member does.
      * an already-admitted undimensioned fact for the same (field, date) means the
        filer DID report a total, which always wins -- nothing is summed there.

    Emitted `_dimensioned=False` (it is a company-level total, not one member's slice)
    so the field-level undimensioned override downstream prefers it over the very
    components it was built from, and `source_tag` keeps the real concept name so
    `fundamentals_tag_ledger` still sees where the number came from.
    """
    empty = merged.iloc[0:0]
    eligible = ((merged["field"].isin(SHARE_CLASS_COMPONENT_FIELDS))
                & (merged["_bare"] == COVER_PAGE_SHARES_TAG)
                & sole_class_axis & component_ok)
    if not eligible.any():
        return empty
    has_total = (merged["field"].isin(SHARE_CLASS_COMPONENT_FIELDS)
                 & merged["_admissible"] & ~merged["_dimensioned"])
    dated_totals = set(merged.loc[has_total, "period_end"].dropna())

    rows = merged[eligible].copy()
    member = next((rows[c] for c in ("member", "dimension_member_label") if c in rows.columns),
                  rows["_probe"])
    rows["_member_key"] = member.astype(str)
    rows = rows.drop_duplicates(subset=["field", "period_end", "_member_key"])

    blank = [c for c in rows.columns
             if c.startswith("dim_") or c in ("dimension", "member", "dimension_member_label")]
    out: list[pd.Series] = []
    for (field, date), grp in rows.groupby(["field", "period_end"], dropna=False):
        if date in dated_totals:
            continue
        values = pd.to_numeric(grp["_probe"], errors="coerce")
        if values.isna().any():
            continue
        # Into the TRADED class's units BEFORE summing -- adding a class whose shares are
        # worth 1,500 of the other's straight into the total is what makes a raw sum wrong
        # rather than merely imprecise.
        values = _to_traded_class_units(values, grp["_member_key"], basis)
        total = float(values.sum())
        rollup = grp[np.isclose(values * 2.0, total, rtol=1e-9)]
        base, value = ((rollup.iloc[0], float(values.loc[rollup.index[0]]))
                       if len(grp) > 1 and len(rollup) == 1
                       else (grp.iloc[0], _grossed_up_to_consolidated(total, values, basis)))
        row = base.copy()
        if blank:
            row[blank] = None
        row["_probe"] = row["numeric_value"] = row["value"] = value
        row["_dimensioned"] = False
        row["_admissible"] = True
        out.append(row)
    if not out:
        return empty
    totals = pd.DataFrame(out).drop(columns="_member_key")
    totals["_probe"] = pd.to_numeric(totals["_probe"], errors="coerce")
    return totals


def build_tag_frames(facts_df: pd.DataFrame, tag_map: dict[str, list[str]],
                     *, ticker: str | None = None) -> pd.DataFrame:
    """Per FILING: for each logical field, union candidate tags (by bare concept
    name, namespace-agnostic) restricted to UNDIMENSIONED (or effectively-total
    dimensioned, see below) facts, then per exact (period_start, period_end) keep
    the highest-priority (earliest-listed) candidate that reported it -- the same
    coalescing rule as `fetch_fundamentals.py::_extract_concept`, generalized to
    run per-filing.

    Replaces the WIP file's `_map_facts_to_tags` (which took the literal LAST
    matching row across dimensioned+undimensioned facts with no period/dimension
    filtering -- a real risk of grabbing a segment slice or a YTD-cumulative value
    for a discrete-quarter field). `is_dimensioned` is a direct boolean column on
    `XBRL.facts.to_dataframe()`'s output, confirmed empirically -- filtering on it
    is simpler and faster than a per-tag `.query().exclude_dimensions()` call.

    A dimensioned fact is still admitted as "the total" when it is the value
    with the HIGHEST REPEAT COUNT (>=2) among all dimensioned instances of that
    same (bare concept, period) -- i.e. the dimension is a reporting/table-
    structure technicality, not a genuine multi-value business-segment slice.
    This subsumes the simple "every instance agrees" case (repeat count ==
    group size) and ALSO correctly resolves the harder case where the total is
    re-tagged under >=2 members while OTHER members break it into components
    (see `repeat_count`/`is_modal_repeat` below for the confirmed MAA `Assets`
    example). When NOTHING repeats (a genuine two-way disagreement with no
    redundant tagging, e.g. a dual-registrant filer's Parent vs LP figures that
    are both real but slightly different), falls back to preferring the
    dei:LegalEntityAxis member labeled "parent" -- but only when there is
    exactly one such candidate (see `is_sole_parent_fallback`). A true segment
    axis with genuinely DIFFERENT, unlabeled per-member values and no
    redundancy is still excluded, since no single member there represents the
    whole company -- this is the key safety property: the rule only ever
    admits a number that is either cross-validated by repetition or uniquely
    identifiable as the parent entity, never a guess among genuinely
    disagreeing, unlabeled segment slices.

    `ticker` is optional and consulted ONLY for `FIELD_TAG_DENYLIST`, the per-issuer
    escape hatch for a concept this one filer misuses (see that constant). Omitting it
    resolves purely on the global candidate lists, which is what every caller that has
    no ticker in hand (and every unit test) gets.

    `facts_df` columns expected (edgartools `XBRL.facts.to_dataframe()`): concept,
    value, numeric_value, unit_ref, period_type, period_start, period_end,
    fiscal_year, fiscal_period, is_dimensioned.

    Returns: field, value, unit, period_start, period_end, period_type,
    fiscal_year, fiscal_period, source_tag.
    """
    out_cols = ["field", "value", "unit", "period_start", "period_end",
               "period_type", "fiscal_year", "fiscal_period", "source_tag"]
    if facts_df is None or facts_df.empty or "concept" not in facts_df.columns:
        return pd.DataFrame(columns=out_cols)

    df = facts_df.copy()
    df["_bare"] = df["concept"].astype(str).str.split(":").str[-1]
    # INSTANT facts (period_type == 'instant', e.g. balance-sheet concepts) carry
    # their date in a SEPARATE `period_instant` column -- `period_end`/`period_start`
    # are both NaN for them in edgartools' to_dataframe() output. Without this, every
    # instant/STOCK field (totalAssets, currentLiabilities, ...) silently vanished:
    # the (period_start, period_end) grouping collapsed all of a field's instant
    # facts into one NaN/NaN group, and the current-period filter in
    # `_filing_current_period_rows` (period_end == filing.period_of_report) excluded
    # every one of them since period_end was never populated. Must happen BEFORE the
    # dimension admissibility check below, which groups by (period_start, period_end).
    if "period_instant" in df.columns:
        df["period_end"] = df["period_end"].where(df["period_end"].notna(), df["period_instant"])

    probe = pd.to_numeric(df.get("numeric_value"), errors="coerce")
    probe = probe.where(probe.notna(), pd.to_numeric(df.get("value"), errors="coerce"))
    df = df.assign(_probe=probe)

    if "is_dimensioned" in df.columns:
        dimensioned = df["is_dimensioned"] == True  # noqa: E712
        group_keys = ["_bare", "period_start", "period_end"]

        # An UNDIMENSIONED fact, when present at all for this (concept, period),
        # is the filer's own as-reported consolidated total -- unambiguous,
        # never a guess. The repeat-count / parent-fallback rules below exist
        # ONLY for concepts tagged dimensioned-ONLY (no undimensioned duplicate
        # anywhere); they must never also be consulted when an undimensioned
        # fact already answers the question, or they can OUTVOTE it. Confirmed
        # empirically (ADP `RevenueFromContractWithCustomerExcludingAssessedTax`,
        # FY2019 Q1): ONE undimensioned fact ($3.3232B, the true quarterly
        # revenue) coexists with a `ProductOrServiceAxis`/`StatementBusiness-
        # SegmentsAxis` product/segment breakdown where SEVERAL lines are
        # legitimately $0 that quarter (a discontinued/immaterial line) --
        # $0 ends up the most-REPEATED dimensioned value (6 of ~20 dimensioned
        # rows), so the repeat-count rule alone would prefer it over every
        # real, non-zero product figure, even though the true total was right
        # there, undimensioned, needing no guess at all.
        has_undimensioned = (df.assign(_undim=~dimensioned)
                             .groupby(group_keys, dropna=False)["_undim"].transform("any"))

        # How many OTHER dimensioned rows in the SAME (concept, period) group
        # share this EXACT value. A genuine consolidated total tends to be
        # RE-TAGGED under >=2 members (e.g. both dei:LegalEntityAxis members in
        # a dual-registrant filing agreeing on the same figure); a component/
        # segment slice is normally unique. Confirmed empirically against MAA:
        # FY2014 `Assets` has SIX dimensioned rows for one (concept, period) --
        # $6.83B appears TWICE (Limited Partner + Parent Company, the true
        # total) while FOUR OTHER rows -- ALSO labeled "Parent Company" (a
        # hidden SECOND, uncaptured axis breaking the total into components:
        # $1.25B, $4.51B, $1.00B, $66.6M) -- each appear ONCE. A prior version's
        # "prefer the Parent-labeled member" rule admitted ALL FIVE Parent-
        # labeled rows and let a later `keep="last"` reduction pick ONE
        # ARBITRARILY -- which is how the $66.6M component ended up stored as
        # if it were total assets (a ~100x error, masquerading as a "huge asset
        # drop"). Preferring the value with the HIGHEST repeat count fixes
        # this: it also subsumes the simpler "all members agree" case (repeat
        # count == group size) as one instance of the same rule.
        repeat_count = (df.assign(_dim=dimensioned)
                       .groupby(group_keys + ["_probe"], dropna=False)["_dim"].transform("sum"))
        max_repeat = (df.assign(_repeat=repeat_count.where(dimensioned, 0))
                     .groupby(group_keys, dropna=False)["_repeat"].transform("max"))
        is_modal_repeat = dimensioned & (repeat_count >= 2) & (repeat_count == max_repeat) & (~has_undimensioned)

        # Fallback: NO value repeats at all (a genuine two-way disagreement,
        # e.g. MAA capex Parent=$53.439M vs LP=$53.357M, both real but
        # slightly different consolidation scopes, neither tagged twice) --
        # prefer the member labeled "parent" on one of the KNOWN dual-
        # registrant/combining-entity axes, but ONLY when there is a SINGLE
        # such candidate in the group (if more than one row is ALSO parent-
        # labeled while nothing repeats, there is no safe way to disambiguate
        # -- exclude, matching the safe default). Since the ticker universe is
        # always the PARENT entity (the publicly-traded security, never its
        # operating partnership), this is a reasonable secondary heuristic
        # once the more robust repeat-count signal is unavailable.
        #
        # More than one axis name is checked -- confirmed empirically: MAA
        # tags most dual-registrant (REIT + its operating partnership)
        # concepts under `dei:LegalEntityAxis`, but tags `dividendsPaid`
        # SPECIFICALLY under `us-gaap:ConsolidatedEntitiesAxis` (a distinct,
        # standard taxonomy axis for the same "parent vs. combining entity"
        # concept -- see also `srt:ConsolidatedEntitiesAxis`) -- restricting
        # to one specific axis name silently dropped every dividendsPaid
        # fact for the two fiscal years (2017-2018) where no undimensioned
        # duplicate existed either. A true business-segment axis with
        # genuinely different, unlabeled per-member values is untouched by
        # either rule and stays excluded, exactly as before.
        _PARENT_IDENTIFYING_AXES = {
            "dei:LegalEntityAxis", "us-gaap:ConsolidatedEntitiesAxis", "srt:ConsolidatedEntitiesAxis",
        }
        dimension_col = df.get("dimension", pd.Series(index=df.index, dtype=object))
        member_label_col = df.get("dimension_member_label", pd.Series(index=df.index, dtype=object))
        is_legal_entity_axis = dimension_col.astype(str).isin(_PARENT_IDENTIFYING_AXES)
        is_parent_member = is_legal_entity_axis & member_label_col.astype(str).str.contains(
            "parent", case=False, na=False)
        parent_count = (df.assign(_parent=is_parent_member)
                       .groupby(group_keys, dropna=False)["_parent"].transform("sum"))
        no_repeats = max_repeat <= 1
        is_sole_parent_fallback = (dimensioned & no_repeats & is_parent_member
                                  & (parent_count == 1) & (~has_undimensioned))

        # NOT filtered here (deferred past the tag_map merge below) -- see the
        # field-level override right after the merge for why.
        df = df.assign(_dimensioned=dimensioned,
                       _admissible=(~dimensioned) | is_modal_repeat | is_sole_parent_fallback)
    else:
        df = df.assign(_dimensioned=False, _admissible=True)

    # Some filers (confirmed: ADM) split total revenue between an ASC-606
    # IN-scope concept (`RevenueFromContractWithCustomer(Excluding|Including)
    # AssessedTax`) and an ASC-606 OUT-of-scope companion
    # (`RevenueNotFromContractWithCustomer`, e.g. commodity trading/
    # merchandising revenue that falls outside ASC 606) -- BOTH tagged
    # undimensioned, BOTH genuinely correct, but EACH only PART of total
    # revenue (summing them equals the filer's own `Revenues` total exactly:
    # ADM FY2025, $24.956B + $55.313B == $80.269B). Since both are
    # undimensioned, neither the priority ordering nor the field-level
    # undimensioned-override below (which only guards against a shaky
    # DIMENSIONED admission) has any way to know the priority-0 candidate is
    # only a slice -- picking it as if it were 100% of revenue is wrong even
    # though it is a real, correctly-tagged fact. When the companion
    # co-exists for the EXACT SAME period, the in-scope-only concept is
    # excluded so `totalRevenue`'s candidate list falls through to
    # `Revenues`/`SalesRevenueNet` (the true whole-company total) instead.
    is_partial_revenue = df["_bare"].isin(PARTIAL_REVENUE_TAGS)
    has_revenue_companion = (
        df.assign(_is_companion=(df["_bare"] == "RevenueNotFromContractWithCustomer") & (~df["_dimensioned"]))
        .groupby(["period_start", "period_end"], dropna=False)["_is_companion"].transform("any"))

    # The SAME "correctly tagged, but only a SLICE" failure, in the two forms
    # that dwarf the ADM case above -- confirmed on live data and both silent,
    # since the number looks perfectly well-formed:
    #
    #  (a) `Revenues` co-exists undimensioned for the same period and is
    #      MATERIALLY LARGER. For any filer whose business sits mostly OUTSIDE
    #      ASC 606 -- an insurer (premiums are ASC 944), a REIT (rents are
    #      ASC 842) -- the contract element captures only fee income while
    #      `Revenues` is the real top line. MetLife fiscal 2019-2020: the
    #      priority-0 contract tag reported $313-354M/quarter against a true
    #      $16.3-19.4B, so stored revenue was ~48x too SMALL from 2019 onward
    #      (the "revenue falls off a cliff in 2019" the audit started from);
    #      Regency Centers Q2/Q3-2018: $64.5M/$66.1M against $281.4M/$278.3M.
    #      "Materially" is `PARTIAL_REVENUE_MATERIALITY`, set from the measured
    #      bimodal distribution -- see its note in `fundamentals_tags.py`. An
    #      ASC-606-native filer tagging both at the same value (e.g. GLW), and
    #      an energy/utility filer whose two totals differ only by a reconciling
    #      item, both keep the priority ordering untouched.
    #  (b) NO `Revenues` anywhere, but an interest/premium line
    #      (`FINANCIALS_TOPLINE_MARKERS`) DWARFS the contract element for the
    #      same period. Regions Financial tags the contract element as literally
    #      $0.00 every quarter, so `totalRevenue` was 0 for its ENTIRE history
    #      while real quarterly revenue ran ~$1.7B. There is no whole-company
    #      revenue concept to fall through to for such a filer, so the field is
    #      left NULL -- correct here rather than a guess, because
    #      `fetch_fundamentals._derive_history` already rebuilds the Financials
    #      top line from net interest + noninterest income (banks) or premiums
    #      + net investment income (insurers), which a bogus 0 would only
    #      corrupt.
    #
    #      The DOMINANCE test is what makes this safe, and its absence was a
    #      confirmed bug: the rule used to fire on the mere PRESENCE of a marker
    #      concept anywhere in the filing, but `InterestIncomeExpenseNet` is
    #      also how an ordinary industrial tags NET INTEREST EXPENSE (ZBH
    #      2022-Q2: -$38.8M beside $1,781.8M of revenue). That nulled the whole
    #      top line for ZBH/PKG/BKR/SPGI/WAB -- see the note on
    #      `FINANCIALS_TOPLINE_MARKERS`. Comparing MAGNITUDES per period, rather
    #      than trusting element names to be sector-exclusive, keeps every real
    #      bank/insurer classified while leaving industrials untouched.
    period_keys = ["period_start", "period_end"]
    undimensioned_probe = df["_probe"].abs().where(~df["_dimensioned"])
    revenues_total = (undimensioned_probe.where(df["_bare"] == TOTAL_REVENUE_TAG)
                     .groupby([df[c] for c in period_keys], dropna=False).transform("max"))
    outranked_by_total = (revenues_total.notna()
                         & (df["_probe"].abs() * PARTIAL_REVENUE_MATERIALITY < revenues_total))
    # Largest undimensioned marker value in the SAME period. Only POSITIVE values
    # count: an industrial's marker is a net interest EXPENSE (negative), which is
    # evidence AGAINST it being a top line, never for it.
    marker_probe = df["_probe"].where(~df["_dimensioned"] & df["_bare"].isin(FINANCIALS_TOPLINE_MARKERS))
    marker_total = (marker_probe.where(marker_probe > 0)
                   .groupby([df[c] for c in period_keys], dropna=False).transform("max"))
    outranked_by_marker = (marker_total.notna()
                          & (df["_probe"].abs() * FINANCIALS_TOPLINE_DOMINANCE < marker_total))

    df["_admissible"] = df["_admissible"] & ~(is_partial_revenue & (
        has_revenue_companion | outranked_by_total | outranked_by_marker))

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # ONE vectorized pass over every field at once, instead of looping
    # `tag_map.items()` (~209 fields) and re-scanning the WHOLE per-filing
    # frame on each iteration. Confirmed by profiling a large, heavily-
    # dimensioned filer (JPM, ~7,000-8,000 facts/filing): the per-field loop
    # cost ~3.8s/filing on its own (comparable to the network fetch of the
    # filing itself), almost entirely pandas call overhead from ~200 repeated
    # full-frame boolean-mask filters + copies, not real computation.
    tag_rows = [{"_bare": tag, "field": field, "_prio": prio}
               for field, candidates in tag_map.items()
               for prio, tag in enumerate(candidates)]
    if not tag_rows:
        return pd.DataFrame(columns=out_cols)
    tag_lookup = pd.DataFrame(tag_rows)

    # The conversion / ownership factors a multi-class or Up-C filer publishes for its OWN
    # classes. Read HERE, while the frame still holds every concept: none of them is a
    # candidate tag for any field (they are ratios and percentages, not amounts), so the
    # filter on the next line would discard them.
    share_class_basis = _share_class_basis(df, _class_of_stock_axis_flags(df)[1])

    # A single filter to the union of every field's candidate tags (not one
    # filter per field), then a single merge onto (field, priority) -- a bare
    # tag shared by multiple fields' candidate lists correctly produces one
    # row per field, exactly like the old per-field loop did independently.
    df = df[df["_bare"].isin(set(tag_lookup["_bare"]))]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    merged = df.merge(tag_lookup, on="_bare", how="inner")

    # A balance-sheet MAGNITUDE (a debt, an asset, a share count -- see
    # `NON_NEGATIVE_STOCK_FIELDS`) reported NEGATIVE is a filer tagging defect, so the
    # fact is made inadmissible and the field's coalesce falls through to its next
    # candidate tag. Must be applied HERE, after the tag_map merge: the sign is only
    # disqualifying relative to the logical FIELD the concept was mapped onto, which
    # the per-concept `_admissible` above cannot see. Confirmed on DTE, whose FY2011/
    # FY2012 10-Ks tag the priority-0 `shortTermDebt` candidate `us-gaap:DebtCurrent`
    # as -$355M / -$634M (the negated "Less amount due within one year" deduction row
    # of the long-term-debt footnote) -- both undimensioned, so nothing else here could
    # reject them, and stored short-term debt came out negative for both years.
    negative_stock = merged["field"].isin(NON_NEGATIVE_STOCK_FIELDS) & (merged["_probe"] < 0)
    merged["_admissible"] = merged["_admissible"] & ~negative_stock

    # Per-issuer deny-list (`FIELD_TAG_DENYLIST`): a concept THIS filer misuses for a
    # different measure, removed from its field's candidates so the coalesce continues to
    # the next one. Same placement rationale as the sign guard above -- the verdict is on
    # a (field, concept) pair, which only exists after the merge. A ticker with no entry
    # (and `ticker=None`) is untouched, so this can never change global resolution.
    denied_pairs = {(field, tag)
                    for field, tags in FIELD_TAG_DENYLIST.get(ticker or "", {}).items()
                    for tag in tags}
    is_denied = (pd.Series(list(zip(merged["field"], merged["_bare"])),
                           index=merged.index).isin(denied_pairs) if denied_pairs
                 else pd.Series(False, index=merged.index))
    merged["_admissible"] = merged["_admissible"] & ~is_denied

    # A share count broken out BY SHARE CLASS is one class's count -- a COMPONENT of the
    # company total, never the total -- so for `SHARE_CLASS_COMPONENT_FIELDS` it is
    # rejected outright, ahead of the repeat-count / parent-member heuristics above.
    # Those heuristics have no way to know that (they read redundancy, not meaning) and
    # were confirmed admitting a single class as if it were the whole company: CME's
    # per-class facts are each tagged twice (class axis alone, then class + equity-
    # components axis), so BOTH Class A and Class B reach a repeat count of 2 and
    # `drop_duplicates(keep="last")` decides between 359,275,000 and 3,000 shares on
    # frame order alone; CVNA resolves the same way. Rejecting is what lets the
    # cover-page sum below become the answer instead.
    has_class_axis, sole_class_axis = _class_of_stock_axis_flags(merged)
    is_share_class_component = merged["field"].isin(SHARE_CLASS_COMPONENT_FIELDS) & has_class_axis
    merged["_admissible"] = merged["_admissible"] & ~is_share_class_component

    # ... and the total those components add up to, for the multi-class filers that
    # report NO undimensioned share count anywhere (36 of 498 tickers were >=60% NULL on
    # `shares_outstanding` for exactly this reason). Placed BEFORE the magnitude floor so
    # the floor screens the TOTAL and not the individual classes -- see
    # `_cover_page_class_total`.
    class_totals = _cover_page_class_total(merged, sole_class_axis,
                                           ~negative_stock & ~is_denied, share_class_basis)
    if not class_totals.empty:
        merged = pd.concat([merged, class_totals], ignore_index=True)

    # A share-count-shaped field (`SHARE_COUNT_MAGNITUDE_FIELDS`) reporting a magnitude
    # below `SHARE_COUNT_MIN_ABS` is a filer scale defect (confirmed: MCD tags its weighted-
    # average share count as `721.8` instead of `721,800,000`, a 1,000,000x error baked into
    # the raw XBRL instance -- see that constant's docstring in `fundamentals_tags.py`), never
    # a real business fact. Same mechanism as the sign guard above -- reject so the field's
    # candidate coalesce falls through, never rescale (no way to distinguish a genuine
    # 1e3x-vs-1e6x scale error from here, and silently multiplying would be exactly the
    # "silently rewrite the filer's number" risk `NON_NEGATIVE_STOCK_FIELDS`'s own docstring
    # rejects for the sign case).
    implausible_share_scale = (merged["field"].isin(SHARE_COUNT_MAGNITUDE_FIELDS)
                               & (merged["_probe"].abs() < SHARE_COUNT_MIN_ABS))
    merged["_admissible"] = merged["_admissible"] & ~implausible_share_scale

    # A field's candidate tags are checked in priority order, but the
    # per-TAG admissibility above (`_admissible`, computed independently per
    # bare concept) has no visibility into a DIFFERENT, lower-priority
    # candidate tag for the SAME field. If any candidate tag reports an
    # undimensioned fact for this exact (field, period) -- the unambiguous
    # as-reported total -- that must win outright for the WHOLE field, even
    # when a higher-priority candidate was otherwise let through only by the
    # modal-repeat/parent-fallback heuristics. Confirmed empirically (ABBV
    # Q1-2019 totalRevenue): priority-0 `RevenueFromContractWithCustomer-
    # ExcludingAssessedTax` has 31 purely-dimensioned facts where two
    # (out of ~30 unrelated product/geography slices) coincidentally share
    # the value $25M -- a repeat count of 2 is enough to pass the tag-local
    # modal-repeat guard -- while the true total ($7.828B) sits right there,
    # undimensioned, under priority-2 `Revenues`. Without this override the
    # shaky priority-0 admission wins outright and the real value is never
    # even considered.
    field_group_keys = ["field", "period_start", "period_end"]
    field_has_undimensioned = (merged.assign(_undim=~merged["_dimensioned"])
                               .groupby(field_group_keys, dropna=False)["_undim"].transform("any"))
    # `_admissible` is ANDed in even in the undimensioned branch so the
    # revenue-companion exclusion above (a real, undimensioned, but
    # PARTIAL-revenue fact) is still honored -- being undimensioned is
    # necessary but not sufficient once a known-partial companion is present.
    merged = merged[np.where(field_has_undimensioned,
                             merged["_admissible"] & ~merged["_dimensioned"],
                             merged["_admissible"])]
    if merged.empty:
        return pd.DataFrame(columns=out_cols)

    merged["_min_prio"] = merged.groupby(field_group_keys, dropna=False)["_prio"].transform("min")
    merged = merged[merged["_prio"] == merged["_min_prio"]]
    merged = merged.drop_duplicates(subset=["field", "period_start", "period_end"], keep="last")

    # numeric_value is edgartools' own coerced float; `value` can be the raw XBRL
    # string (thousands separators, or non-numeric for a mis-matched fact) -- coerce
    # explicitly and drop anything that still isn't a real number rather than let a
    # raw string reach the period-decumulation math downstream.
    merged["_val"] = pd.to_numeric(merged.get("numeric_value"), errors="coerce")
    fallback = pd.to_numeric(merged.get("value"), errors="coerce")
    merged["_val"] = merged["_val"].where(merged["_val"].notna(), fallback)
    merged = merged[merged["_val"].notna()]
    if merged.empty:
        return pd.DataFrame(columns=out_cols)

    # SG&A companion-tag summing (see `SGA_GA_ONLY_TAG`/`SGA_SM_COMPANION_TAG`'s
    # docstring in fundamentals_tags.py): a filer that tags ONLY
    # `GeneralAndAdministrativeExpense` for `sellingGeneralAdmin` -- i.e. the combined
    # tag was absent, so G&A won by priority alone -- may ALSO tag a genuinely
    # additive `SellingAndMarketingExpense` companion for the exact same period. Add
    # it in, undimensioned-only, so the field reflects the filer's FULL SG&A rather
    # than just its G&A component.
    is_ga_only = (merged["field"] == "sellingGeneralAdmin") & (merged["_bare"] == SGA_GA_ONLY_TAG)
    if is_ga_only.any():
        companion = df[(df["_bare"] == SGA_SM_COMPANION_TAG) & (~df["_dimensioned"])]
        companion_by_period = (companion.drop_duplicates(subset=["period_start", "period_end"], keep="last")
                               .set_index(["period_start", "period_end"])["_probe"])
        keys = pd.MultiIndex.from_arrays(
            [merged.loc[is_ga_only, "period_start"], merged.loc[is_ga_only, "period_end"]])
        addend = pd.Series(companion_by_period.reindex(keys).values, index=merged.index[is_ga_only]).fillna(0.0)
        merged.loc[is_ga_only, "_val"] = merged.loc[is_ga_only, "_val"] + addend

    out = pd.DataFrame({
        "field": merged["field"], "value": merged["_val"],
        "unit": merged.get("unit_ref"),
        "period_start": merged.get("period_start"), "period_end": merged.get("period_end"),
        "period_type": merged.get("period_type"),
        "fiscal_year": merged.get("fiscal_year"),
        # Normalized here, the EARLIEST point fiscal_period is captured -- so
        # `backfill_fiscal_period_from_filing` (which borrows whichever OTHER
        # row in the same filing has native fy/fp to fill an instant fact's
        # blank one) never propagates a raw 'YTDn' label onto a balance-sheet
        # field; every downstream consumer (backfill, decumulation, annual,
        # instant) sees only 'Q1'..'Q4'/'FY'.
        "fiscal_period": merged["fiscal_period"].map(normalize_fiscal_period_label)
                        if "fiscal_period" in merged.columns else None,
        "source_tag": merged.get("concept"),
    })
    return out[out_cols].reset_index(drop=True)


def backfill_fiscal_period_from_filing(tagged: pd.DataFrame) -> pd.DataFrame:
    """INSTANT (balance-sheet) facts carry no native fiscal_year/fiscal_period at
    all in edgartools' output (confirmed empirically against real filings: only
    DURATION facts -- e.g. the dei cover-page tags -- have it). Without this
    backfill, every STOCK field (totalAssets, currentLiabilities, ...) is silently
    dropped later by `instant_stock()`'s dropna. Backfills any row missing
    fiscal_year/fiscal_period from whichever OTHER row in the SAME (already
    current-period-filtered) frame does carry it -- duration facts in the same
    filing always do, since they share this filing's one reporting period."""
    if tagged.empty:
        return tagged
    tagged = tagged.copy()
    have_native = tagged.dropna(subset=["fiscal_year", "fiscal_period"])
    if not have_native.empty:
        fy, fp = have_native.iloc[0][["fiscal_year", "fiscal_period"]]
        missing = tagged["fiscal_year"].isna() | tagged["fiscal_period"].isna()
        tagged.loc[missing, "fiscal_year"] = fy
        tagged.loc[missing, "fiscal_period"] = fp
    return tagged


def _header_period_of_report(filing) -> pd.Timestamp | None:
    """The EDGAR submission header's period-of-report, as a LAST-RESORT fallback
    behind the XBRL cover page (see `_resolve_period_of_report`). Wrapped because
    it is not a plain attribute: it lazily fetches the filing homepage and has been
    observed raising (`TypeError` out of `edgar.attachments.get_filing_dates`) --
    which, on the bare-attribute read this replaces, propagated all the way out of
    `build_ticker_facts_edgar` and cost the ENTIRE ticker its extraction via the
    per-ticker handler in `fetch_fundamentals_edgartools`."""
    try:
        por = filing.period_of_report
    except Exception:                              # noqa: BLE001 - homepage fetch/parse
        return None
    return pd.Timestamp(por).normalize() if por else None


def _resolve_period_of_report(filing, xb) -> pd.Timestamp | None:
    """The filing's OWN reporting-period end, preferring the XBRL cover page's
    `dei:DocumentPeriodEndDate` over the EDGAR submission header.

    The header value is filer-keyed metadata and is sometimes simply wrong, while
    the cover-page tag is part of the audited instance document and was correct in
    every disagreement found on live data. Because the caller filters facts to
    `period_end == this date` EXACTLY, a wrong date does not degrade the result --
    it discards the entire filing, or worse keeps the prior-year comparatives:

      * KeyCorp Q1-2013 10-Q: header 2013-03-**15**, cover page 2013-03-31
        -> 0 of 63 current-period rows kept, fiscal 2013 Q1 missing outright.
      * Packaging Corp fiscal-2016 10-K: header **2017-02-28** (its own FILING
        date), cover page 2016-12-31 -> whole fiscal year lost.
      * Cboe Q3-2013 10-Q: header 2013-11-**04**, cover page 2013-09-30.
      * Baker Hughes Q1-2018 10-Q: header **2017**-03-31, cover page 2018-03-31
        -> the 36 rows that matched were the PRIOR-YEAR comparatives, stored as if
        they were fiscal 2017 Q1 while the real Q1-2018 was dropped. The only one
        of the four where a wrong header produced data rather than none.
    """
    cover = None
    try:
        cover = (xb.entity_info or {}).get("document_period_end_date")
    except Exception:                              # noqa: BLE001 - malformed instance
        cover = None
    if cover:
        try:
            return pd.Timestamp(cover).normalize()
        except (ValueError, TypeError):
            pass
    return _header_period_of_report(filing)


def _filing_xbrl(filing, *, log: logging.Logger | None = None):
    """This filing's parsed XBRL instance, or None when it genuinely has none.

    Retries `XBRL_PARSE_ATTEMPTS` times because the previous single bare
    `except Exception: return empty` conflated the two outcomes it must NOT
    conflate -- "this filing has no XBRL" (a pre-2009 filing, or a Part-III-only
    10-K/A: nothing to do, silence is right) and "the fetch/parse failed this
    time" (retryable, and if it is not retried the filing is silently absent from
    the table forever). Measured on the live table: 8 filings carrying XBRL
    (`isXBRL=1`) had contributed ZERO rows -- AEP fiscal 2021, AFL Q3-2013, AIZ
    fiscal 2018, C Q1-2023, ORLY fiscal 2013, PKG Q3-2013, REG fiscal 2011, SYK
    fiscal 2013 -- yet all 8 re-parse cleanly today (78-107 tagged rows each), so
    each was a transient failure that no log line anywhere recorded."""
    for attempt in range(1, XBRL_PARSE_ATTEMPTS + 1):
        try:
            return filing.xbrl()
        except Exception as e:                     # noqa: BLE001 - network or parse
            if attempt == XBRL_PARSE_ATTEMPTS:
                if log is not None:
                    log.warning(
                        "fundamentals: %s %s (%s) XBRL unavailable after %d attempts (%s)",
                        getattr(filing, "form", "?"), getattr(filing, "accession_number", "?"),
                        getattr(filing, "filing_date", "?"), XBRL_PARSE_ATTEMPTS, e)
                return None
            time.sleep(XBRL_RETRY_BACKOFF_SECONDS * attempt)
    return None


def _filing_current_period_rows(filing, tag_map: dict[str, list[str]],
                                *, ticker: str | None = None,
                                log: logging.Logger | None = None) -> pd.DataFrame:
    """One filing's tagged facts, restricted to its OWN current period (period_end
    == the filing's period_of_report) -- excludes multi-year comparatives a filing
    routinely re-discloses (a 10-K shows 3 years of income-statement history; those
    prior years were already captured as THEIR OWN filing's current period). Both
    the discrete-quarter and YTD-cumulative variant of a duration concept share the
    same period_end (only period_start differs), so this filter keeps both.

    Also attaches the filing's cover-page `dei:DocumentFiscalYearFocus` as
    `cover_fiscal_year` -- one of the two labels
    `fundamentals_periods.resolve_fiscal_year_by_filing_calendar` votes on to
    reconstruct the issuer's fiscal calendar (edgartools' per-fact `fiscal_year`
    being the other, and both carrying typos).

    `ticker` is optional and used only to look up `FIELD_TAG_DENYLIST` -- passing None
    resolves purely on the global candidate lists."""
    xb = _filing_xbrl(filing, log=log)
    if xb is None:
        return pd.DataFrame()
    try:
        facts_df = xb.facts.to_dataframe()
    except Exception as e:                         # noqa: BLE001 - malformed instance
        if log is not None:
            log.warning("fundamentals: %s facts unreadable (%s)",
                        getattr(filing, "accession_number", "?"), e)
        return pd.DataFrame()
    tagged = build_tag_frames(facts_df, tag_map, ticker=ticker)
    if tagged.empty:
        return tagged
    por_ts = _resolve_period_of_report(filing, xb)
    if por_ts is None:
        return pd.DataFrame()
    period_end = pd.to_datetime(tagged["period_end"], errors="coerce").dt.normalize()
    current = tagged[period_end == por_ts]
    current = _cover_page_shares_fallback(tagged, current, por_ts)
    tagged = current
    if tagged.empty:
        # XBRL parsed and matched candidate tags, yet NOTHING sits on the filing's
        # own reporting period -- the signature of a period_of_report that is still
        # wrong (see `_resolve_period_of_report`). Never silent: this is the one
        # failure that looks identical to a legitimately empty filing.
        if log is not None:
            log.warning("fundamentals: %s has no facts at period_of_report %s (%d tagged)",
                        getattr(filing, "accession_number", "?"), por_ts.date(), len(period_end))
        return tagged
    tagged = tagged.assign(cover_fiscal_year=_cover_fiscal_year(xb))
    return backfill_fiscal_period_from_filing(tagged)


def _cover_page_shares_fallback(all_periods: pd.DataFrame, current: pd.DataFrame,
                                por_ts: pd.Timestamp) -> pd.DataFrame:
    """Recover `sharesOutstanding` from the filing's COVER PAGE for the filers that
    tag no balance-sheet share count, re-stamped onto the filing's own period. Serves
    two populations: the single-class filers below, and every MULTI-CLASS filer, whose
    total arrives here as the per-class cover-page sum `_cover_page_class_total` builds
    (that function decides WHAT the value is; this one decides WHEN it applies, and the
    fill-only rule below is the same for both).

    `dei:EntityCommonStockSharesOutstanding` is stated as of a date AFTER the period
    it reports on -- the date the cover page was signed -- so the current-period
    filter above discards it every time. Confirmed on live filings: GPC
    137,622,108 @ 2026-02-17, Chubb 390,156,552 @ 2026-02-20 and J.B. Hunt
    94,604,083 @ 2026-02-17, all against a 2025-12-31 period of report. With the
    diluted-average concept no longer standing in for it (see `SHARES_TAGS`), those
    filings would simply have no share count at all -- hence this recovery rather
    than a null.

    FILL-ONLY: an as-reported balance-sheet count for the period is always
    preferred and never overridden, so the two concepts can never both produce a
    row and be picked between arbitrarily (which is the defect `SHARES_TAGS`
    documents). Only the EARLIEST qualifying cover date is used -- an amended or
    re-filed cover page can carry more than one -- and only within
    `COVER_PAGE_SHARES_MAX_LAG_DAYS`, so a stray fact from an unrelated period
    cannot be pulled in.

    `period_end` is re-stamped to the period of report, deliberately: it is the
    period this filing reports on, it is how every other instant fact in the frame
    is keyed, and `fundamentals_derive` keys instant series on period_end. The true
    as-of date is a few weeks later, i.e. the value is marginally FRESHER than
    period_end claims -- immaterial for a panel that is lagged by filing_date
    anyway, and the alternative (a share count keyed to a date no other field in
    the filing shares) would not join to anything."""
    if all_periods.empty or (not current.empty
                             and (current["field"] == SHARES_OUTSTANDING_FIELD).any()):
        return current
    dates = pd.to_datetime(all_periods["period_end"], errors="coerce").dt.normalize()
    lag = (dates - por_ts).dt.days
    candidates = all_periods[
        (all_periods["field"] == SHARES_OUTSTANDING_FIELD)
        & all_periods["source_tag"].astype(str).str.endswith(COVER_PAGE_SHARES_TAG)
        & lag.between(0, COVER_PAGE_SHARES_MAX_LAG_DAYS)
    ]
    if candidates.empty:
        return current
    recovered = candidates.loc[[dates[candidates.index].idxmin()]].assign(period_end=por_ts)
    return pd.concat([current, recovered], ignore_index=True)


def _cover_fiscal_year(xb) -> int | None:
    """`dei:DocumentFiscalYearFocus` off the filing's cover page, as an int."""
    try:
        return int((xb.entity_info or {}).get("fiscal_year"))
    except (TypeError, ValueError):
        return None


def populate_amends_accession(facts: pd.DataFrame) -> pd.DataFrame:
    """amends_accession = the immediately-prior accession (by filing_date) sharing
    (ticker, field, fiscal_year, fiscal_period, duration_type) -- chains correctly
    even when an amendment amends an earlier amendment. A financial-restatement-free
    10-K/A (filed only to add Part III proxy items, the most common amendment
    reason) carries no new financial facts, so it produces zero rows here and never
    enters this function at all -- only genuine restatements ever show up."""
    if facts.empty:
        facts["amends_accession"] = None
        return facts
    key = ["ticker", "field", "fiscal_year", "fiscal_period", "duration_type"]
    facts = facts.sort_values("filing_date").copy()
    facts["amends_accession"] = (
        facts.groupby(key)["accession_number"].shift(1).where(facts["is_amendment"] == 1.0)
    )
    return facts


def _apply_fiscal_year_calendar(raw: pd.DataFrame, ticker: str,
                                *, log: logging.Logger | None = None) -> pd.DataFrame:
    """`resolve_fiscal_year_by_filing_calendar`, plus a log line naming how many
    facts it re-keyed and drop of the `cover_fiscal_year` helper column (a
    per-filing label used only to vote, never persisted)."""
    resolved = resolve_fiscal_year_by_filing_calendar(raw)
    if log is not None:
        before = pd.to_numeric(raw["fiscal_year"], errors="coerce")
        after = pd.to_numeric(resolved["fiscal_year"], errors="coerce")
        changed = int((before != after).sum())
        if changed:
            log.info("fundamentals: %s re-keyed %d facts onto its own fiscal calendar "
                     "(years %s -> %s)", ticker, changed,
                     sorted(before[before != after].dropna().unique().astype(int).tolist()),
                     sorted(after[before != after].dropna().unique().astype(int).tolist()))
    return resolved.drop(columns="cover_fiscal_year", errors="ignore")


def _filings_to_parse(sorted_filings: list, done_accessions: frozenset[str]) -> list:
    """The filings to actually open: every not-yet-extracted one, PLUS the
    already-extracted filings reporting within `FISCAL_YEAR_CONTEXT_DAYS` of one --
    i.e. the rest of the same fiscal year.

    A fiscal year's facts are not independent. `decumulate_quarterly_flow` derives
    Q4 as FY - (Q1 + Q2 + Q3), `drop_derived_q4_for_partial_fiscal_years` votes on
    a year cross-field, and `resolve_fiscal_year_by_filing_calendar` needs 10-K
    period ends to place a quarter -- all off the ONE in-memory frame this
    function builds. So an incremental run that opened only the new filing could
    never produce a Q4 at all: the year's other three quarters were already in the
    DB and therefore absent from the frame. Re-opening the siblings costs at most
    a handful of extra parses (a routine run adds one filing and re-reads its own
    year), and every re-emitted row upserts onto its own unchanged primary key."""
    pending = [f for f in sorted_filings if f.accession_number not in done_accessions]
    if not pending or len(pending) == len(sorted_filings):
        return sorted_filings if pending else []
    window = pd.Timedelta(days=FISCAL_YEAR_CONTEXT_DAYS)
    pending_dates = [pd.Timestamp(f.filing_date) for f in pending]
    return [f for f in sorted_filings
            if any(abs(pd.Timestamp(f.filing_date) - d) <= window for d in pending_dates)]


def build_ticker_facts_edgar(
    ticker: str,
    *,
    cik: str | None = None,
    forms: tuple[str, ...] = tuple(FUNDAMENTALS_FORMS),
    done_accessions: frozenset[str] = frozenset(),
    tag_map: dict[str, list[str]] | None = None,
    since: pd.Timestamp | None = None,
    employee_history: list[int] | None = None,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=list(forms))`, skips accessions
    already in `done_accessions` or filed before `since`, extracts each filing's
    current-period facts, then decumulates/derives quarters (FLOW_FIELD_TAGS) and
    takes point-in-time values (INSTANT_FIELD_TAGS) via `fundamentals_periods`.
    Returns rows matching the `fundamentals_facts` schema (see schema_registry.py).

    `since=None` (default) pulls the issuer's ENTIRE available filing history --
    callers wanting the repo's usual `years_history` scoping should pass an
    explicit cutoff (see `fetch_fundamentals_edgartools`).

    `employee_history` seeds the headcount continuity guard with the values
    already stored for this ticker (see `fundamentals_employees.history_by_ticker`)
    -- without it an incremental run, which re-parses only the single new 10-K,
    would judge that filing against an empty history and so not guard it at all."""

    tag_map = tag_map or ALL_FIELD_TAGS
    flow_fields = set(FLOW_FIELD_TAGS) & set(tag_map)
    instant_fields = set(INSTANT_FIELD_TAGS) & set(tag_map)

    company = Company(ticker)
    filings = company.get_filings(form=list(forms))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]
    to_parse = _filings_to_parse(sorted_filings, done_accessions)

    # Grows as the (chronologically ordered) filings are parsed, so each 10-K's
    # headcount is checked against every value accepted before it -- the stored
    # history plus this run's own earlier years.
    accepted_headcounts = list(employee_history or [])

    raw_frames: list[pd.DataFrame] = []
    for filing in to_parse:
        tagged = _filing_current_period_rows(filing, tag_map, ticker=ticker, log=log)
        if tagged.empty:
            continue
        # HEADCOUNT: no XBRL concept exists, so it is parsed out of this filing's
        # body text (10-K only) and appended as a ready-made instant fact -- from
        # here on it is indistinguishable from a tagged field and rides the same
        # period-resolution, amendment and persistence machinery. Appending it
        # AFTER `_filing_current_period_rows` (rather than inside `build_tag_frames`)
        # is what keeps the XBRL path free of a special case; re-running
        # `backfill_fiscal_period_from_filing` is what gives the new row this
        # filing's fiscal_year/fiscal_period, borrowed from the duration facts
        # above it exactly as every genuine instant fact does. That function is
        # idempotent -- rows it already filled carry native values now and are
        # left alone.
        #
        # A filing whose XBRL yielded NOTHING (`tagged.empty`, i.e. a pre-2009
        # 10-K or a Part-III-only 10-K/A) is skipped before this point on
        # purpose: with no duration fact anywhere in it there is nothing to
        # borrow a fiscal period FROM, and a headcount that cannot be placed in a
        # fiscal year is unusable -- "null, never guess wrong", as elsewhere.
        if EMPLOYEES_FIELD in tag_map:
            employees = employee_fact_frame(filing, accepted_headcounts, log=log)
            if employees is not None:
                tagged = backfill_fiscal_period_from_filing(
                    pd.concat([tagged, employees], ignore_index=True))
                accepted_headcounts.append(int(employees["value"].iloc[0]))
        tagged["accession_number"] = filing.accession_number
        tagged["form"] = filing.form
        tagged["filing_date"] = pd.Timestamp(filing.filing_date)
        tagged["is_amendment"] = 1.0 if str(filing.form).upper().endswith("/A") else 0.0
        raw_frames.append(tagged)

    if not raw_frames:
        return pd.DataFrame(columns=_FACTS_COLS)
    
    raw = pd.concat(raw_frames, ignore_index=True)
    raw["period_start"] = pd.to_datetime(raw["period_start"], errors="coerce")
    raw["period_end"] = pd.to_datetime(raw["period_end"], errors="coerce")
    native_mask = raw["fiscal_year"].notna() & raw["fiscal_period"].notna()
    # Some filers (confirmed: KR) tag a filing's facts with NO native
    # fiscal_period anywhere in it -- backfill_fiscal_period_from_filing (above,
    # per-filing) has nothing to borrow from in that case. This second,
    # cross-filing pass resolves those from the ticker's OWN filing order
    # (see its docstring); computed against native_mask captured BEFORE this
    # call so fiscal_period_source below still reflects the true origin tier.
    raw = backfill_fiscal_period_by_filing_order(raw)
    raw["fiscal_period_source"] = np.where(native_mask, "native", "date_arithmetic_fallback")
    # BEFORE any period grouping: a wrong fiscal_year silently merges two different
    # fiscal years' facts into one bucket (see the function's docstring for the
    # confirmed CSCO/JCI/SJM cases), which no later step can undo.
    raw = _apply_fiscal_year_calendar(raw, ticker, log=log)

    out_rows: list[pd.DataFrame] = []
    for field in sorted(set(raw["field"])):
        fdf = raw[raw["field"] == field]
        # `decumulate_quarterly_flow`/`annual_flow`/`instant_stock` each already carry
        # their OWN row's provenance (source_tag, fiscal_period_source, is_amendment)
        # straight from the SAME raw fact that produced that row's value/accession_number
        # -- no separate lookup needed. A prior version joined provenance back on by
        # (fiscal_year, fiscal_period) alone via a `groupby(...).last()`, which could
        # silently borrow is_amendment/source_tag from a DIFFERENT accession than the
        # one the row's own value/accession_number came from whenever more than one
        # filing reported the same period (confirmed empirically: JPM's original
        # 10-Q for fiscal 2012 Q1 came back tagged is_amendment=1.0, contaminated from
        # a later 10-Q/A restating the same quarter with the same value) -- provenance
        # must always come from the SAME row as the value, never a re-derived lookup.
        def _finish(frame: pd.DataFrame, duration_type: str) -> pd.DataFrame | None:
            if frame is None or frame.empty:
                return None
            return frame.assign(field=field, duration_type=duration_type)

        if field in flow_fields:
            for frame, dtype in ((decumulate_quarterly_flow(fdf), "quarterly"),
                                 (annual_flow(fdf), "annual")):
                finished = _finish(frame, dtype)
                if finished is not None:
                    out_rows.append(finished)
        else:
            finished = _finish(instant_stock(fdf), "instant")
            if finished is not None:
                out_rows.append(finished)

    if not out_rows:
        return pd.DataFrame(columns=_FACTS_COLS)

    facts = pd.concat(out_rows, ignore_index=True)
    facts["ticker"] = ticker
    facts["cik"] = cik
    if "derived" not in facts.columns:
        facts["derived"] = False
    if "derived_from_accessions" not in facts.columns:
        facts["derived_from_accessions"] = None
    # annual_flow/instant_stock rows carry neither column (always as-reported, never
    # computed) -- pd.concat fills them NaN; make that explicit as derived=0.0/None.
    facts["derived"] = facts["derived"].fillna(False).astype(float)
    facts["derived_from_accessions"] = facts["derived_from_accessions"].apply(
        lambda v: ",".join(v) if isinstance(v, list) else (None if pd.isna(v) else v))
    facts["is_amendment"] = facts["is_amendment"].fillna(0.0)
    facts["fiscal_period_source"] = facts["fiscal_period_source"].fillna("native")
    # BEFORE the cross-field derivations, so nothing is built on a Q4 that came
    # from an FY anchor the year's other fields already prove is partial.
    facts = drop_derived_q4_for_partial_fiscal_years(facts)
    facts = derive_missing_total_liabilities(facts)
    facts = derive_missing_pretax_income(facts)
    facts = derive_bank_cash(facts)
    facts = reassign_misordered_instant_facts(facts)
    facts = populate_amends_accession(facts)
    for c in _FACTS_COLS:
        if c not in facts.columns:
            facts[c] = None
    return facts[_FACTS_COLS].reset_index(drop=True)


def fetch_fundamentals_edgartools(context: Context, tickers: list[str]) -> pd.DataFrame:
    """Public entry point (mirrors every sibling fetcher's plain-function
    convention, e.g. `fetch_8k_items`) -- persists per-ticker to `fundamentals_facts`
    via `context.store.save(..., pk=[...])`, NOT the WIP file's nonexistent
    `context.store.upsert(...)`. Per-ticker try/except so one bad ticker (a
    `Company(ticker)`/`.get_filings()` failure) cannot abort the batch -- the WIP
    file only guarded the inner per-filing loop, not this outer one.

    Saves EACH ticker's rows immediately after its own extraction finishes,
    rather than accumulating every ticker in memory and saving once at the
    end -- a single large filer (e.g. JPM) can take minutes per ticker, so an
    interrupted run (or one bad ticker further down the list) must never lose
    the already-extracted tickers' work (this repo's "save per entity"
    incremental convention).

    Tickers are walked CONCURRENTLY on a thread pool (`run_per_ticker`), same as
    every other edgartools per-filing fetcher (8-K, 13D, DEF 14A -- see
    `parallel_fetch.py`'s module docstring). This one was the sole holdout still
    walking tickers one at a time, which is the confirmed reason a from-scratch
    pull took ~24h: the per-filing walk is network-I/O bound (SEC XBRL downloads),
    and edgartools' own client already rate-limits *request starts* globally
    (thread-safe, ~9 req/sec) -- a sequential loop never has more than one request
    in flight and so never comes close to saturating that budget.
    """
    _configure_identity()
    de = context.config.data_extract
    years = int(getattr(de, "fundamentals_years_history", de.years_history))
    full_since = pd.Timestamp.today() - pd.DateOffset(years=years)

    pk = ["ticker", "accession_number", "field", "fiscal_year", "fiscal_period", "duration_type"]
    existing_accessions: dict[str, frozenset[str]] = {}
    existing = context.store.load("fundamentals_facts",
                                 columns=["ticker", "accession_number"], optional=True)
    if existing is not None:
        for t, grp in existing.groupby("ticker"):
            existing_accessions[t] = frozenset(grp["accession_number"].unique())

    # Stored headcounts, so the continuity guard has a per-ticker anchor even on an
    # incremental run that re-parses a single 10-K (see `build_ticker_facts_edgar`).
    try:
        stored_employees = context.store.load(
            "fundamentals_facts", columns=["ticker", "filing_date", "value"],
            where={"field": EMPLOYEES_FIELD})
        employee_histories = history_by_ticker(stored_employees)
    except Exception:                              # noqa: BLE001 - table may not exist yet
        employee_histories = {}

    # `run_per_ticker` needs a (ticker, cik) frame, but this fetcher's contract is
    # "process every ticker it's given" regardless of `sp500_tickers` coverage --
    # unlike the sibling fetchers that filter to `load_cik_mapping`'s rows, so a
    # ticker missing a resolved CIK (still None here, matching this function's
    # pre-existing behavior) is never silently dropped from the run.
    try:
        cik_by_ticker = load_cik_mapping(context).set_index("ticker")["cik"].to_dict()
    except Exception:                              # noqa: BLE001 - table may not exist yet
        cik_by_ticker = {}
    cik_map = pd.DataFrame({"ticker": tickers, "cik": [cik_by_ticker.get(t) for t in tickers]})

    rescan_days = int(getattr(de, "manifest_full_rescan_days", 30))
    since, is_full_rescan = manifest_window(
        context, "fundamentals_facts", len(cik_map), fallback_since=full_since,
        full_rescan_days=rescan_days)

    def _worker(ticker: str, cik: str | None) -> pd.DataFrame:
        try:
            done = existing_accessions.get(ticker, frozenset())
            ticker_facts = build_ticker_facts_edgar(
                ticker, cik=cik, done_accessions=done, since=since,
                employee_history=employee_histories.get(ticker), log=context.log)
        except Exception as e:                      # noqa: BLE001
            context.log.warning("fetch_fundamentals_edgartools: %s failed (%s)", ticker, e)
            return pd.DataFrame(columns=_FACTS_COLS)
        if ticker_facts.empty:
            return ticker_facts
        context.store.save("fundamentals_facts", ticker_facts, pk=pk)
        context.log.info("fetch_fundamentals_edgartools: saved %d fundamentals_facts rows for %s.",
                         len(ticker_facts), ticker)
        return ticker_facts

    results = run_per_ticker(cik_map, _worker, desc="Fundamentals (edgartools)")
    all_frames = [f for f in results if not f.empty]
    total_rows = sum(len(f) for f in all_frames)

    if not all_frames:
        context.log.info("fetch_fundamentals_edgartools: no new fundamentals_facts rows.")
        out = pd.DataFrame(columns=_FACTS_COLS)
    else:
        out = pd.concat(all_frames, ignore_index=True)

    record_run(context, "fundamentals_facts", len(cik_map), total_rows, is_full_rescan=is_full_rescan)
    return out
