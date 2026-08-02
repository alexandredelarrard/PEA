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
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from edgar import Company, set_identity

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.fundamentals.fundamentals_employees import (
    employee_fact_frame, history_by_ticker,
)
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    annual_flow, backfill_fiscal_period_by_filing_order, decumulate_quarterly_flow,
    derive_bank_cash, derive_missing_pretax_income, derive_missing_total_liabilities,
    drop_derived_q4_for_partial_fiscal_years, instant_stock,
    normalize_fiscal_period_label, reassign_misordered_instant_facts,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    EMPLOYEES_FIELD, EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FINANCIALS_TOPLINE_DOMINANCE,
    FINANCIALS_TOPLINE_MARKERS, FLOW_TAGS, LATEST_DURATION_TAGS,
    PARTIAL_REVENUE_MATERIALITY, PARTIAL_REVENUE_TAGS, SHARES_TAGS, STOCK_TAGS,
    TOTAL_REVENUE_TAG,
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


def build_tag_frames(facts_df: pd.DataFrame, tag_map: dict[str, list[str]]) -> pd.DataFrame:
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

    # A single filter to the union of every field's candidate tags (not one
    # filter per field), then a single merge onto (field, priority) -- a bare
    # tag shared by multiple fields' candidate lists correctly produces one
    # row per field, exactly like the old per-field loop did independently.
    df = df[df["_bare"].isin(set(tag_lookup["_bare"]))]
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    merged = df.merge(tag_lookup, on="_bare", how="inner")

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


def _filing_current_period_rows(filing, tag_map: dict[str, list[str]]) -> pd.DataFrame:
    """One filing's tagged facts, restricted to its OWN current period (period_end
    == the filing's period_of_report) -- excludes multi-year comparatives a filing
    routinely re-discloses (a 10-K shows 3 years of income-statement history; those
    prior years were already captured as THEIR OWN filing's current period). Both
    the discrete-quarter and YTD-cumulative variant of a duration concept share the
    same period_end (only period_start differs), so this filter keeps both."""
    try:
        xb = filing.xbrl()
        facts_df = xb.facts.to_dataframe()
    except Exception:
        return pd.DataFrame()
    tagged = build_tag_frames(facts_df, tag_map)
    if tagged.empty:
        return tagged
    por = filing.period_of_report
    if not por:
        return pd.DataFrame()
    por_ts = pd.Timestamp(por).normalize()
    period_end = pd.to_datetime(tagged["period_end"], errors="coerce").dt.normalize()
    tagged = tagged[period_end == por_ts]
    return backfill_fiscal_period_from_filing(tagged)


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

    # Grows as the (chronologically ordered) filings are parsed, so each 10-K's
    # headcount is checked against every value accepted before it -- the stored
    # history plus this run's own earlier years.
    accepted_headcounts = list(employee_history or [])

    raw_frames: list[pd.DataFrame] = []
    for filing in sorted_filings:
        if filing.accession_number in done_accessions:
            continue
        tagged = _filing_current_period_rows(filing, tag_map)
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

    Scoped by `data_extract.fundamentals_years_history` if set, else the global
    `data_extract.years_history` (same `getattr(..., default)` idiom as
    `fetch_financial_notes.py`'s `notes_years_history`) -- the WIP file had no
    years scoping at all, re-pulling an issuer's ENTIRE history every run."""
    _configure_identity()
    de = context.config.data_extract
    years = int(getattr(de, "fundamentals_years_history", de.years_history))
    since = pd.Timestamp.today() - pd.DateOffset(years=years)

    pk = ["ticker", "accession_number", "field", "fiscal_year", "fiscal_period", "duration_type"]
    existing_accessions: dict[str, frozenset[str]] = {}
    try:
        existing = context.store.load("fundamentals_facts", columns=["ticker", "accession_number"])
        if not existing.empty:
            for t, grp in existing.groupby("ticker"):
                existing_accessions[t] = frozenset(grp["accession_number"].unique())
    except Exception:                              # noqa: BLE001 - table may not exist yet
        existing_accessions = {}

    # Stored headcounts, so the continuity guard has a per-ticker anchor even on an
    # incremental run that re-parses a single 10-K (see `build_ticker_facts_edgar`).
    try:
        stored_employees = context.store.load(
            "fundamentals_facts", columns=["ticker", "filing_date", "value"],
            where={"field": EMPLOYEES_FIELD})
        employee_histories = history_by_ticker(stored_employees)
    except Exception:                              # noqa: BLE001 - table may not exist yet
        employee_histories = {}

    random.shuffle(tickers)

    all_frames: list[pd.DataFrame] = []
    for ticker in tqdm(tickers, "Extract filings edgartools"):
        try:
            done = existing_accessions.get(ticker, frozenset())
            ticker_facts = build_ticker_facts_edgar(
                ticker, done_accessions=done, since=since,
                employee_history=employee_histories.get(ticker), log=context.log)
        except Exception as e:                      # noqa: BLE001
            context.log.warning("fetch_fundamentals_edgartools: %s failed (%s)", ticker, e)
            continue
        if ticker_facts.empty:
            continue
        context.store.save("fundamentals_facts", ticker_facts, pk=pk)
        context.log.info("fetch_fundamentals_edgartools: saved %d fundamentals_facts rows for %s.",
                         len(ticker_facts), ticker)
        all_frames.append(ticker_facts)

    if not all_frames:
        context.log.info("fetch_fundamentals_edgartools: no new fundamentals_facts rows.")
        return pd.DataFrame(columns=_FACTS_COLS)

    return pd.concat(all_frames, ignore_index=True)
