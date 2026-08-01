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
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from edgar import Company, set_identity

from src.constants.constants import FUNDAMENTALS_FORMS
from src.context import Context
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    annual_flow, backfill_fiscal_period_by_filing_order, decumulate_quarterly_flow,
    instant_stock, normalize_fiscal_period_label, reassign_misordered_instant_facts,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FLOW_TAGS, LATEST_DURATION_TAGS,
    SHARES_TAGS, STOCK_TAGS,
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

    if "is_dimensioned" in df.columns:
        dimensioned = df["is_dimensioned"] == True  # noqa: E712
        probe = pd.to_numeric(df.get("numeric_value"), errors="coerce")
        probe = probe.where(probe.notna(), pd.to_numeric(df.get("value"), errors="coerce"))
        df = df.assign(_probe=probe)
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

        admissible = (~dimensioned) | is_modal_repeat | is_sole_parent_fallback
        df = df[admissible]
        
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

    merged["_min_prio"] = merged.groupby(
        ["field", "period_start", "period_end"], dropna=False)["_prio"].transform("min")
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
) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=list(forms))`, skips accessions
    already in `done_accessions` or filed before `since`, extracts each filing's
    current-period facts, then decumulates/derives quarters (FLOW_FIELD_TAGS) and
    takes point-in-time values (INSTANT_FIELD_TAGS) via `fundamentals_periods`.
    Returns rows matching the `fundamentals_facts` schema (see schema_registry.py).

    `since=None` (default) pulls the issuer's ENTIRE available filing history --
    callers wanting the repo's usual `years_history` scoping should pass an
    explicit cutoff (see `fetch_fundamentals_edgartools`)."""

    tag_map = tag_map or ALL_FIELD_TAGS
    flow_fields = set(FLOW_FIELD_TAGS) & set(tag_map)
    instant_fields = set(INSTANT_FIELD_TAGS) & set(tag_map) 

    company = Company(ticker)
    filings = company.get_filings(form=list(forms))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    raw_frames: list[pd.DataFrame] = []
    for filing in sorted_filings:
        if filing.accession_number in done_accessions:
            continue
        tagged = _filing_current_period_rows(filing, tag_map)
        if tagged.empty:
            continue
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

    all_frames: list[pd.DataFrame] = []
    for ticker in tqdm(tickers, "Extract filings edgartools"):
        try:
            done = existing_accessions.get(ticker, frozenset())
            ticker_facts = build_ticker_facts_edgar(ticker, done_accessions=done, since=since)
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
