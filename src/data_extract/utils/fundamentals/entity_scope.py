"""
entity_scope.py (src/data_extract/utils/fundamentals/entity_scope.py)
------------------------------------------------------------------------
Reduce one filing's XBRL facts to **the consolidated registrant's own numbers**, which is
the population every KPI in the catalogue is defined over.

The rule is one line -- *keep the dimensionally-unqualified (default-member) facts* -- and
the reason it is that line rather than a member deny-list is measured:

  * **MAA** (Up-C REIT) files two `dei:EntityCentralIndexKey` facts, parent and operating
    partnership, but all 1,417 `xbrli:identifier` values are the PARENT's CIK. The LP's
    full primary statements are carried under `dei:LegalEntityAxis` with a **company
    extension member**, `maa:LimitedPartnershipMember` -- confirmed live: 315
    LegalEntityAxis facts, members `maa:LimitedPartnershipMember` and
    `maa:TemporaryEquityMember`. A fixed us-gaap member list cannot catch an extension
    member, so filtering on the MEMBER fails by construction; filtering on the AXIS works.
  * **Southern Company** carries six registrant CIKs and 3,579 `LegalEntityAxis`
    occurrences in one instance, again all identifiers = parent.
  * **JPM** scopes its bank subsidiary the same way (`jpm:JpmorganChaseBankNAMember`).

So `entity_identifier` is NOT a usable discriminator -- it is the parent's CIK on every
fact, including the subsidiary's. Dimensional qualification is.

**The deliberate cost.** Regulatory capital is reachable *only* dimensioned: ASC
942-505-50-1 requires the disclosure for the holding company AND each significant bank
subsidiary, so every CET1 fact is qualified by `LegalEntityAxis`. That is also why SEC's
`companyconcept` returns 404 for `CommonEquityTierOneCapitalRatio` on JPM, USB and BAC --
`companyfacts` and `frames` publish only unqualified facts. `DIMENSIONED_EXCEPTIONS` names
that hook so the exclusion is a recorded decision rather than a silent loss; nothing in
this pass uses it.
"""
from __future__ import annotations

import pandas as pd

#: Axes that scope a fact to something other than the consolidated registrant. Kept for
#: documentation and for the `dimensioned_facts` escape hatch -- the default path does not
#: consult it, because ANY dimensional qualification already disqualifies a fact.
ENTITY_AXES: frozenset[str] = frozenset({
    "dei:LegalEntityAxis",
    "srt:ConsolidatedEntitiesAxis",
    "us-gaap:StatementBusinessSegmentsAxis",
    "srt:StatementGeographicalAxis",
})

#: Fields that can only ever be read from DIMENSIONED facts, with the reason. Declared,
#: not used: adding one means accepting that the value is scoped to a named legal entity
#: and is therefore not on the same basis as every other column in the table.
DIMENSIONED_EXCEPTIONS: dict[str, str] = {
    "tier1CapitalRatio": (
        "ASC 942-505-50-1 requires regulatory capital for the holding company and each "
        "significant bank subsidiary, so every fact carries dei:LegalEntityAxis. This is "
        "why SEC companyconcept 404s for CET1 on JPM/USB/BAC. Not extracted in this pass."),
}

#: edgartools names each axis column `dim_<prefix>_<AxisName>`.
_DIM_PREFIX = "dim_"

#: The only columns anything downstream of scoping reads. Everything else in the 70-column
#: facts frame -- the per-axis `dim_*` columns, labels, statement roles, footnote keys -- is
#: either all-NaN once the undimensioned filter has run or is not consulted at all.
_KEPT_COLUMNS: tuple[str, ...] = (
    "concept", "numeric_value", "unit_ref", "decimals",
    "period_type", "period_start", "period_end", "period_instant",
    "fiscal_year", "fiscal_period", "balance",
)

#: Units that are not a money/count amount this pipeline can store as a float. Per-share
#: amounts are excluded here rather than downstream because a per-share figure is NOT
#: additive: summing four quarterly EPS drifts from annual EPS as the share count moves,
#: which is why `epsDiluted` is COMPUTED from netIncome/dilutedShares (decision #9) and
#: the as-reported tag is kept only as the validator's independent cross-check.
#: Substrings marking a `unit_ref` as a PER-SHARE amount rather than a currency total.
#: Matched case-insensitively, because the spelling is the filer's.
#:
#: **This filter matched nothing until 2026-08-23.** It looked for `perShare`/`PerShare`
#: while edgartools emits `iso4217_USD_per_shares` -- lower case, underscore-separated -- so
#: every per-share fact reached the resolver. That is how AXP's `EarningsPerShareBasic`
#: became a discovered revenue root and stored a **top line of $3.40**.
#:
#: It is a partial guard by necessity and must not be relied on alone. `unit_ref` is a
#: FILER-AUTHORED ID, not a unit: across 3,538 swept filings the values include `usd`, `USD`,
#: `iso4217_USD`, `U_iso4217USD`, `Unit_USD`, `USDollar`, `u000`, `Unit1` and `Unit12`. An
#: opaque `Unit12` may well be a per-share unit and no substring test can know. Measured, only
#: **5 of 265,786** valued rows carry a self-describing per-share unit -- so `NOT_A_TOP_LINE`
#: carries the real weight and this closes the self-describing cases.
_NON_AMOUNT_UNIT_HINTS = ("per_share", "per_shares", "pershare")


def consolidated_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """The filing's facts, reduced to numeric, undimensioned, consolidated amounts.

    Returns an empty frame rather than raising when a filing yields nothing usable: a
    filing with no readable XBRL is a fact about that filing, and one bad filing must not
    abort a 490-ticker walk.
    """
    if facts is None or facts.empty:
        return pd.DataFrame()

    keep = facts
    # 1. Undimensioned only -- the entity-scope rule. Verified on XOM: of 2,503 facts, 778
    #    are undimensioned and NONE of them carries a non-null value in any of the 39
    #    `dim_*` columns, so the flag and the columns agree.
    if "is_dimensioned" in keep.columns:
        keep = keep[~keep["is_dimensioned"].fillna(False).astype(bool)]
    dim_cols = [c for c in keep.columns if c.startswith(_DIM_PREFIX)]
    if dim_cols:
        keep = keep[keep[dim_cols].isna().all(axis=1)]

    # 2. Numeric only. Cover-page strings, extensible enumerations and text blocks share
    #    the frame with the amounts.
    if "numeric_value" not in keep.columns:
        return pd.DataFrame()
    keep = keep[keep["numeric_value"].notna()]

    # 3. Drop per-share units (see `_NON_AMOUNT_UNIT_HINTS`).
    if "unit_ref" in keep.columns:
        unit = keep["unit_ref"].fillna("").astype(str).str.lower()
        for hint in _NON_AMOUNT_UNIT_HINTS:
            keep = keep[~unit.loc[keep.index].str.contains(hint, regex=False)]

    # 4. Project. A filing's facts frame is one column per DIMENSION AXIS plus ~30 others --
    #    39 axes on XOM, 70 columns total -- and after step 1 every `dim_*` column is
    #    all-NaN by construction. Carrying them costs real memory at scale: a 26-ticker
    #    x 15-year sweep held them and grew past 9 GB before being killed, and the full
    #    490-ticker backfill is ~20x that walk. Keep only what the resolver reads.
    wanted = [c for c in _KEPT_COLUMNS if c in keep.columns]
    return keep[wanted].reset_index(drop=True)


def bare_concept(namespaced: str) -> str:
    """`us-gaap:Assets` -> `Assets`.

    The facts frame namespaces `concept`; the calculation linkbase splits the same thing
    into `concept` + `concept_taxonomy`. The catalogue is written in bare names, so this
    is the join key between all three. (The legacy `fundamentals_facts` table learned the
    same lesson the hard way: its `concept` is namespaced and its `bare_tag` is not.)
    """
    return namespaced.split(":", 1)[-1] if namespaced else namespaced


def us_gaap_only(facts: pd.DataFrame) -> pd.DataFrame:
    """Facts tagged with a STANDARD element. A company extension is legitimate reporting
    but has no cross-filer meaning, and cross-filer comparability is the entire point of
    the catalogue -- an extension can still foot into a standard total via the linkbase,
    which is how its amount reaches a KPI."""
    if facts.empty or "concept" not in facts.columns:
        return facts
    return facts[facts["concept"].str.startswith("us-gaap:", na=False)]


def reported_concepts(facts: pd.DataFrame) -> frozenset[str]:
    """Bare names of every concept this filing reports a usable consolidated fact for.

    This is the resolver's `available` set. It is what stops the linkbase climb at a node
    the filer declared structurally but never reported -- selecting one of those would
    produce a confident NULL that looks exactly like a coverage regression.
    """
    if facts.empty or "concept" not in facts.columns:
        return frozenset()
    return frozenset(bare_concept(c) for c in facts["concept"].dropna().unique())


def duration_concepts(facts: pd.DataFrame) -> frozenset[str]:
    """Bare names this filing reports ONLY as duration (flow) facts.

    `discover_root`'s parentless-root search needs this: without it a balance-sheet total
    is a perfectly good "root with all-positive children", and the 26-ticker sweep duly
    stored `Assets` (18 rows) and `LiabilitiesAndStockholdersEquity` (16) as revenue.
    `period_type` is the reliable axis for that test -- `balance` (debit/credit), the
    obvious alternative, is EMPTY for GS's `RevenuesNetOfInterestExpense` and DTE's
    `RegulatedAndUnregulatedOperatingRevenue`, so it would reject the correct answers.

    "Only" is deliberate: a concept a filer tags both ways in one filing is not a clean
    flow and should not anchor a revenue block.
    """
    if facts.empty or "period_type" not in facts.columns:
        return frozenset()
    kinds = (facts.assign(_bare=[bare_concept(c) for c in facts["concept"]])
             .groupby("_bare")["period_type"].nunique(dropna=True))
    only_one = kinds[kinds == 1].index
    first = (facts.assign(_bare=[bare_concept(c) for c in facts["concept"]])
             .groupby("_bare")["period_type"].first())
    return frozenset(str(name) for name in only_one if first.get(name) == "duration")


def zero_only_concepts(facts: pd.DataFrame) -> frozenset[str]:
    """Bare names this filing reports as **exactly 0 in every period it reports at all**.

    The discriminator between a filer's tagging artefact and a real zero, and it is a
    filing-level property, not a period-level one: measured across 26 tickers x 15 years,
    every zero-valued `totalRevenue` came from a concept that was zero in EVERY period of
    its filing. There is no one-bad-quarter case, so the resolver never needs a value at
    resolution time and stays period-agnostic.

    Being zero-only is NOT itself an error -- `xbrl_linkbase.resolve_field` withholds these
    on a first pass and restores them on a second, so a filer whose whole answer is zero
    (VRT's pre-merger SPAC filings) keeps it, flagged.
    """
    if facts.empty or "numeric_value" not in facts.columns:
        return frozenset()
    grouped = (facts.assign(_bare=[bare_concept(c) for c in facts["concept"]])
               .groupby("_bare")["numeric_value"])
    extremes = grouped.agg(["min", "max"])
    zero = extremes[(extremes["min"] == 0) & (extremes["max"] == 0)]
    return frozenset(str(name) for name in zero.index)


def dimensioned_facts(facts: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Facts qualified by exactly `axis` -- the escape hatch `DIMENSIONED_EXCEPTIONS`
    documents. Separate from the default path so that reading a legal-entity-scoped value
    is always an explicit act."""
    column = _DIM_PREFIX + axis.replace(":", "_")
    if facts.empty or column not in facts.columns:
        return pd.DataFrame()
    return facts[facts[column].notna()].reset_index(drop=True)
