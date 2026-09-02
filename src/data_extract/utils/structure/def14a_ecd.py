"""
def14a_ecd.py (src/data_extract/utils/structure/def14a_ecd.py)
--------------------------------------------------------------------------------
The Pay-versus-Performance / Executive Compensation Disclosure (ECD) block of a
proxy, read straight from the filing's inline-XBRL facts.

These are facts the FILER tagged, so they are the one part of a DEF 14A worth
reading deterministically -- an LLM would be strictly worse. Everything else
(compensation tables, director fees, ownership, audit fees, board
recommendations) is HTML prose and belongs to the LLM path.

Regulatory scope: Item 402(v) applies to fiscal years ending on or after
2022-12-16, so a proxy covering an earlier year carries no `ecd:` facts at all
and must produce NO ROW. AAPL's 2023-01-12 proxy is the clean example -- its
FY2022 ended 2022-09-24, and `filing.xbrl()` returns nothing.

WHY THIS BYPASSES `ProxyStatement` (measured, edgartools 5.51.0)
----------------------------------------------------------------
`_get_concept_value` / `_get_concept_series` filter on `concept ==` only, sort by
`period_end` and take `.iloc[0]`. They never look at a dimension, so on a
co-PEO year DOCUMENT ORDER decides which executive survives:

  * BA's 2025 proxy covers FY2024 with TWO PEOs -- Ortberg 18,388,629 and
    Calhoun 15,050,812 (compensation actually paid +19,904,513 and
    **-23,875,735**). The library keeps one and silently drops the other.
  * NKE's 2025 proxy covers FY2025 with Hill 26,018,068 and Donahoe 28,442,712
    (CAP 17,010,238 and **-10,924,243**).
  * SBUX tags a full individual x year MATRIX with 0.0 in every non-applicable
    cell, so an undimensioned read of FY2025 can return 0.0 while the real value
    (Niccol, 30,992,773) sits in the same column one row down.

THE DIMENSION FILTER IS CONDITIONAL, AND THAT IS NOT A STYLE CHOICE
-------------------------------------------------------------------
Filers discriminate PEO facts in two incompatible ways, and a fixed
`dim_ecd_ExecutiveCategoryAxis == 'ecd:PeoMember'` filter returns **zero rows on
every filing measured**:

  * BA / NKE / SBUX put the executive on `ecd:IndividualAxis` ONLY. Of BA's 6
    `ecd:PeoTotalCompAmt` facts, **0** carry an ExecutiveCategoryAxis.
  * AAPL tags `ecd:PeoName` on BOTH axes (26 facts, 5 of them `ecd:PeoMember`,
    the other 21 `ecd:NonPeoNeoMember` -- this is the tag the library reads
    undimensioned) while its `ecd:PeoTotalCompAmt` is fully UNDIMENSIONED,
    5 facts, one per covered year.

So the axis is applied only when the concept's own facts actually carry it, and
the name and the amounts are resolved independently -- on AAPL they live in
different dimensional spaces and cannot be joined on an individual key.

Units are NOT normalised by edgartools (`instance.py` is `float(value)`
verbatim, no sign/scale handling), so a filer tagging in millions arrives in
millions. `net_income` therefore keeps its plausibility floor; see `net_income`.

**Negative `peo_actually_paid_comp` is legitimate.** CAP subtracts prior-year
unvested fair value, so a share-price fall makes it negative -- 28.5% of 2023
and 33.7% of 2025 S&P 500 proxies report at least one. Never take an absolute
value here.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: The frame column edgartools builds for each dimension it finds on the instance.
_CATEGORY_AXIS = "dim_ecd_ExecutiveCategoryAxis"
_INDIVIDUAL_AXIS = "dim_ecd_IndividualAxis"
#: Measured on BA / NKE / SBUX / AAPL 2026: the axis only ever takes these two values.
_PEO_MEMBER = "ecd:PeoMember"

#: Undimensioned ECD scalars -- one fact per covered fiscal year. `ecd:PeoTotalComp` (no `Amt`)
#: does NOT exist; the real concept names all carry the suffix.
_PEO_TOTAL = "ecd:PeoTotalCompAmt"
_PEO_CAP = "ecd:PeoActuallyPaidCompAmt"
_PEO_NAME = "ecd:PeoName"

#: (row column, concept) for the facts that are one-per-year and need no dimension work.
_SCALARS: tuple[tuple[str, str], ...] = (
    ("neo_avg_total_comp", "ecd:NonPeoNeoAvgTotalCompAmt"),
    ("neo_avg_actually_paid_comp", "ecd:NonPeoNeoAvgCompActuallyPaidAmt"),
    ("total_shareholder_return", "ecd:TotalShareholderRtnAmt"),
    ("peer_group_tsr", "ecd:PeerGroupTotalShareholderRtnAmt"),
    ("company_selected_measure_value", "ecd:CoSelectedMeasureAmt"),
)
_TEXT_SCALARS: tuple[tuple[str, str], ...] = (
    ("company_selected_measure_name", "ecd:CoSelectedMeasureName"),
)
#: Item 402(x) boolean flags -> 1.0 / 0.0 / NaN.
_FLAGS: tuple[tuple[str, str], ...] = (
    ("insider_trading_policy_adopted", "ecd:InsiderTrdPoliciesProcAdoptedFlag"),
    ("award_timing_mnpi_considered", "ecd:AwardTmgMnpiCnsdrdFlag"),
    ("award_dates_predetermined", "ecd:AwardTmgPredtrmndFlag"),
    ("mnpi_disclosure_timed_for_comp_value", "ecd:MnpiDiscTimedForCompValFlag"),
)
#: PVP column (h). `ecd:NetIncLossAmt` does NOT exist -- the taxonomy points at the us-gaap
#: concept, and it is wired `priority="-1"` so a filer may override it with its own.
_NET_INCOME = "us-gaap:NetIncomeLoss"

#: A PEO's total compensation is never exactly zero; that value is the SBUX matrix's
#: "not applicable this year" cell, not a disclosure.
_ZERO_TOL = 1e-9


def ecd_facts(filing) -> pd.DataFrame | None:
    """The filing's inline-XBRL facts, or None when it has none.

    This is the same frame `ProxyStatement._facts_dataframe` builds, so a caller that already
    holds the filing downloads nothing extra.
    """
    try:
        xbrl = filing.xbrl()
    except Exception as e:                                  # noqa: BLE001 -- best-effort
        logger.info("%s: filing.xbrl() failed (%s: %s)",
                    getattr(filing, "accession_number", "?"), type(e).__name__, e)
        return None
    if xbrl is None:
        return None
    try:
        df = xbrl.facts.to_dataframe()
    except Exception as e:                                  # noqa: BLE001
        logger.info("%s: facts.to_dataframe() failed (%s: %s)",
                    getattr(filing, "accession_number", "?"), type(e).__name__, e)
        return None
    if df is None or df.empty or "concept" not in df.columns:
        return None
    return df


def has_ecd_block(facts: pd.DataFrame | None) -> bool:
    """True when the filing carries Item 402(v) facts at all. A proxy covering a fiscal year
    that ended before 2022-12-16 has none, and must not produce a row."""
    if facts is None or facts.empty:
        return False
    return bool(facts["concept"].astype(str).str.startswith("ecd:").any())


def _peo_only(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep the PEO facts, using whichever discriminator this filer actually used.

    The axis filter is applied ONLY when the concept's own facts carry the axis: BA / NKE / SBUX
    tag `ecd:PeoTotalCompAmt` on `IndividualAxis` alone, so an unconditional
    `== 'ecd:PeoMember'` would drop all 6 of BA's facts and return an empty frame -- on exactly
    the co-PEO filing the filter exists to fix.
    """
    if _CATEGORY_AXIS in sub.columns and sub[_CATEGORY_AXIS].notna().any():
        return sub[sub[_CATEGORY_AXIS] == _PEO_MEMBER]
    return sub


def _prefer_dimensioned(sub: pd.DataFrame) -> pd.DataFrame:
    """Drop undimensioned duplicates of a dimensioned fact within the same period.

    NKE's 2026 proxy tags FY2026's PEO total TWICE -- once on `nke:ElliottHillMember` and once
    with no dimension at all, same 36,340,876. Counting both reads as two PEOs, one of them
    nameless.
    """
    if _INDIVIDUAL_AXIS not in sub.columns:
        return sub
    out = []
    for _period, g in sub.groupby("period_end", dropna=False):
        dim = g[g[_INDIVIDUAL_AXIS].notna()]
        out.append(dim if not dim.empty else g)
    return pd.concat(out) if out else sub


def _concept(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Every fact for `concept`, PEO-filtered and de-duplicated."""
    sub = facts[facts["concept"].astype(str) == concept]
    return sub if sub.empty else _prefer_dimensioned(_peo_only(sub))


def _numeric(sub: pd.DataFrame) -> pd.Series:
    """`value` as floats. NOT abs() and NOT rescaled -- both would corrupt real disclosures."""
    return pd.to_numeric(sub["value"], errors="coerce")


def latest_period(facts: pd.DataFrame) -> pd.Timestamp | None:
    """The most recent fiscal year the PVP table covers.

    Taken from the PEO comp facts rather than `filing.period_of_report`, which for a proxy is
    the MEETING date, not a fiscal year end. Without it nothing in the row says which year
    `peo_total_comp` belongs to.
    """
    sub = _concept(facts, _PEO_TOTAL)
    if sub.empty:
        sub = _concept(facts, _PEO_CAP)
    if sub.empty or "period_end" not in sub.columns:
        return None
    p = pd.to_datetime(sub["period_end"], errors="coerce").dropna()
    return None if p.empty else p.max()


def _at_period(sub: pd.DataFrame, period: pd.Timestamp | None) -> pd.DataFrame:
    if period is None or sub.empty or "period_end" not in sub.columns:
        return sub
    return sub[pd.to_datetime(sub["period_end"], errors="coerce") == period]


def peo_block(facts: pd.DataFrame) -> dict:
    """The PEO facts for the latest covered fiscal year.

    Shape decision: `sec_def14a` stays one row per `(ticker, accession_number)`, so a co-PEO
    year cannot put both executives in the `peo_*` columns. The individual with the LARGEST
    `peo_total_comp` is kept, and `n_peos` + `peo_names_all` make the co-PEO year visible rather
    than silently halved. A `(ticker, accession, individual)` child table was rejected --
    reintroducing a child table one phase after deleting four of them, for ~1% of rows.

    Note the measured consequence of "largest total": on NKE's 2025 proxy that selects Donahoe
    (28,442,712), the DEPARTING CEO, over the incumbent Hill (26,018,068). `n_peos = 2` is what
    tells a consumer not to read the retained name as "the CEO".
    """
    period = latest_period(facts)
    totals = _at_period(_concept(facts, _PEO_TOTAL), period)
    caps = _at_period(_concept(facts, _PEO_CAP), period)

    # every PEO named anywhere in the table, not just in the covered year -- this is the
    # co-PEO breadcrumb, and BA tags both names at a single period_end regardless of the year
    # their compensation belongs to
    names_all = _concept(facts, _PEO_NAME)
    all_names = sorted({str(v).strip() for v in names_all.get("value", pd.Series(dtype=object))
                        if isinstance(v, str) and v.strip()})

    out = {
        "ecd_period_end": period,
        "peo_name": None,
        "peo_total_comp": float("nan"),
        "peo_actually_paid_comp": float("nan"),
        "n_peos": float("nan"),
        "peo_names_all": ",".join(all_names) or None,
    }
    if totals.empty and caps.empty:
        return out

    # the SBUX zero-matrix: drop "not applicable this year" cells BEFORE selecting, which
    # recovers the correct value instead of the NULL a post-hoc 0.0 guard would leave
    vals = _numeric(totals)
    keep = totals[vals.abs() > _ZERO_TOL]
    kept_vals = _numeric(keep)
    out["n_peos"] = float(len(keep))

    if not keep.empty:
        winner = keep.loc[kept_vals.idxmax()]
        out["peo_total_comp"] = float(kept_vals.max())
        ind = winner.get(_INDIVIDUAL_AXIS) if _INDIVIDUAL_AXIS in keep.columns else None
        label = winner.get("dimension_member_label")
        if isinstance(label, str) and label.strip():
            out["peo_name"] = label.strip()
        # the CAP for the SAME individual, not the largest CAP -- they can disagree in sign
        cap = caps
        if ind is not None and not pd.isna(ind) and _INDIVIDUAL_AXIS in caps.columns:
            same = caps[caps[_INDIVIDUAL_AXIS] == ind]
            cap = same if not same.empty else caps
        cap_vals = _numeric(cap)
        cap_vals = cap_vals[cap_vals.abs() > _ZERO_TOL]
        if not cap_vals.empty:
            out["peo_actually_paid_comp"] = float(cap_vals.iloc[0])
    elif not caps.empty:
        cap_vals = _numeric(caps)
        cap_vals = cap_vals[cap_vals.abs() > _ZERO_TOL]
        if not cap_vals.empty:
            out["peo_actually_paid_comp"] = float(cap_vals.iloc[0])

    # AAPL's amounts are undimensioned while its names are not, so fall back to the
    # PeoMember-filtered name when the winning fact carried no individual label
    if out["peo_name"] is None:
        named = _at_period(names_all, period)
        vals = [str(v).strip() for v in named.get("value", pd.Series(dtype=object))
                if isinstance(v, str) and v.strip()]
        if not vals and len(all_names) == 1:
            vals = all_names
        if vals:
            out["peo_name"] = vals[0]
    return out


def _scalar(facts: pd.DataFrame, concept: str, period: pd.Timestamp | None) -> float:
    sub = _at_period(_concept(facts, concept), period)
    vals = _numeric(sub).dropna()
    return float(vals.iloc[0]) if not vals.empty else float("nan")


def _text(facts: pd.DataFrame, concept: str, period: pd.Timestamp | None) -> str | None:
    sub = _at_period(_concept(facts, concept), period)
    if sub.empty:                              # the measure name is often tagged once, not per year
        sub = _concept(facts, concept)
    for v in sub.get("value", pd.Series(dtype=object)):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _flag(facts: pd.DataFrame, concept: str) -> float:
    """Item 402(x) flags are tagged once per filing, not per covered year."""
    sub = _concept(facts, concept)
    for v in sub.get("value", pd.Series(dtype=object)):
        if isinstance(v, bool):
            return float(v)
        s = str(v).strip().lower()
        if s in ("true", "1", "yes"):
            return 1.0
        if s in ("false", "0", "no"):
            return 0.0
    return float("nan")


def net_income(facts: pd.DataFrame, period: pd.Timestamp | None) -> float:
    """PVP column (h), read from `us-gaap:NetIncomeLoss`.

    Kept raw. The plausibility floor lives in `def14a_validate.repair_main_row`, because a value
    like SBUX FY2025's `1856.4` (tagged in $ millions, `decimals='1'`, `unit_ref='usd'`) is
    indistinguishable from a real number HERE -- nothing in the fact disambiguates millions from
    units, and the trustworthy figure is in `fundamentals_history` anyway.
    """
    sub = _at_period(_concept(facts, _NET_INCOME), period)
    if sub.empty:
        sub = _concept(facts, _NET_INCOME)
    vals = _numeric(sub).dropna()
    return float(vals.iloc[0]) if not vals.empty else float("nan")


def ecd_row(facts: pd.DataFrame) -> dict:
    """Every ECD column for one filing. Caller must have checked `has_ecd_block` first."""
    period = latest_period(facts)
    row = peo_block(facts)
    for col, concept in _SCALARS:
        row[col] = _scalar(facts, concept, period)
    for col, concept in _TEXT_SCALARS:
        row[col] = _text(facts, concept, period)
    for col, concept in _FLAGS:
        row[col] = _flag(facts, concept)
    row["net_income"] = net_income(facts, period)
    row["has_individual_executive_data"] = float(
        _INDIVIDUAL_AXIS in facts.columns and bool(facts[_INDIVIDUAL_AXIS].notna().any()))
    return row


__all__ = ["ecd_facts", "ecd_row", "has_ecd_block", "latest_period", "net_income", "peo_block"]
