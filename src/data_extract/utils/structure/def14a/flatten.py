"""
flatten.py  (src/data_extract/utils/structure/def14a/flatten.py)
-----------------------------------------------------------------
A filled `Def14AExtract` -> the `def14a_llm` parent row and the four child tables.

Every builder here is a PURE function of `(ticker, filing, Def14AExtract)`. That is what
makes them replayable over the already-stored `def14a_json` for free: the tokens were paid
when those filings were first extracted, so the flatten can be verified over 445 real
filings without a single LLM call.

`_prepare_frame` here takes `(rows, numeric, pk)`. `votes/flatten.py` has a different
function of the same name taking `(rows)`; they are NOT interchangeable.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.data_extract.utils.common.frame_sanitize import strip_nul
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.validate import (
    DEF14A_AUDIT_FEE_MIN_PLAUSIBLE, clean_holder_name, clean_person_name, clean_text,
    is_subtotal_holder, repair_pay_ratio, rescale_block, sum_fee_total,
)
from src.data_store.schema import Table, Tables
from src.gpt_extract.utils.schemas_gpt import LlmResult

logger = logging.getLogger(__name__)

# flattened output columns that must be stored numeric (float) in the DB
# (governance booleans are surfaced as 1.0/0.0 flags so they are usable features)
_NUMERIC_COLS = [
    "fiscal_year_extract",
    # board / directors
    "n_directors", "board_size", "avg_director_age", "avg_board_tenure",
    "pct_independent_directors", "pct_female_directors", "avg_other_public_boards",
    "pct_gender_stated", "n_women_directors_vs_inferred",
    # CEO
    "ceo_age", "ceo_since_year", "ceo_is_founder", "ceo_is_board_chair",
    "ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
    "ceo_non_equity_incentive", "ceo_all_other_comp", "ceo_total_comp", "ceo_equity_pay_pct",
    # NEO aggregate
    "n_neos", "total_neo_comp", "sct_years",
    # child-table row counts (so a silent recall regression is visible on the parent row)
    "n_director_comp_rows", "n_ownership_rows",
    # ownership -- the ECONOMIC leg and the VOTING leg are separate columns on purpose. The
    # schema used to ask the model to read the first and suppress the second, and it returned
    # voting power as ownership on 59 filings; both are now extracted and code picks the leg.
    "insider_ownership_pct", "insider_voting_pct", "insider_shares",
    "ceo_ownership_pct", "ceo_voting_pct", "n_five_percent_holders",
    # governance provisions
    "independent_chair", "lead_independent_director", "classified_board",
    "dual_class_shares", "poison_pill", "majority_voting", "say_on_pay_support_pct",
    "ceo_pay_ratio", "median_employee_pay", "auditor_fees",
    # auditor name is TEXT; the rest of the fee block is numeric
    "auditor_since_year", "audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax",
    "audit_fees_other", "auditor_fees_prior",
]

# The task-tailored prompt lives in `src/gpt_extract/prompt_templates/def14a_*.md`. It is
# precise about WHERE each field lives and how to normalise it, which materially lifts the
# fill rate versus a generic "extract structured data" instruction, and it is cached per
# (model, schema) so that precision stays cheap.

def _latest_sct_year(extract: Def14AExtract) -> int | None:
    """The most recent fiscal year present in the SCT rows, or None when no row carries one."""
    years = [c.fiscal_year for c in extract.compensation if c.fiscal_year is not None]
    return max(years) if years else None


def _latest_sct_rows(extract: Def14AExtract) -> list:
    """The SCT rows for the most recent fiscal year only.

    The schema now asks for every (NEO x year) cell, so three derived values MUST be restricted
    to one year or they silently change meaning: the CEO row would come from an arbitrary year,
    `n_neos` would triple (making the `n_neos == 1` metric meaningless), and `total_neo_comp`
    would sum three years of pay. Rows with no `fiscal_year` are kept as a fallback -- a filing
    whose Year column did not parse still has to yield a CEO row.
    """
    latest = _latest_sct_year(extract)
    if latest is None:
        return list(extract.compensation)
    return [c for c in extract.compensation if c.fiscal_year == latest]


def _ceo_from_compensation(extract: Def14AExtract) -> "ExecutiveCompensation | None":  # noqa: F821
    """The CEO's Summary-Compensation-Table row for the MOST RECENT fiscal year: match on the
    extracted CEO name first, else on a CEO-like title, else the first (usually highest-paid)
    NEO of that year."""
    rows = _latest_sct_rows(extract)
    ceo_name = (extract.ceo_name or "").strip().lower()
    if ceo_name:
        for c in rows:
            if (c.name or "").strip().lower() == ceo_name:
                return c
    ceo_kws = ("chief executive", "ceo", "president and chief")
    for c in rows:
        if any(kw in (c.title or "").lower() for kw in ceo_kws):
            return c
    return rows[0] if rows else None


def _ceo_age(extract: Def14AExtract) -> int | None:
    """CEO age: the explicit top-level field, else looked up by CEO name in the
    directors list (the CEO is almost always a director nominee)."""
    if extract.ceo_age is not None:
        return extract.ceo_age
    name = (extract.ceo_name or "").strip().lower()
    if name:
        for person in extract.directors:
            if (person.name or "").strip().lower() == name and person.age is not None:
                return person.age
    return None


def _bnum(x: bool | None) -> float | None:
    """Bool -> 1.0/0.0 (numeric flag for the feature store); None stays None."""
    return None if x is None else float(bool(x))


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _flatten(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    dirs = extract.directors
    ages = [d.age for d in dirs if d.age is not None]
    tenures = [d.tenure_years for d in dirs if d.tenure_years is not None]
    genders = [(d.gender or "").strip().lower() for d in dirs if d.gender]
    other_boards = [d.other_public_company_boards for d in dirs
                    if d.other_public_company_boards is not None]
    ceo = _ceo_from_compensation(extract)
    latest_rows = _latest_sct_rows(extract)
    gov = extract.governance
    g = lambda a: getattr(gov, a, None) if gov is not None else None  # noqa: E731
    ceo_name = (extract.ceo_name or (ceo.name if ceo else None) or "")

    ceo_equity = None
    if ceo and ceo.total_compensation_usd:
        equity = (ceo.stock_awards_usd or 0) + (ceo.option_awards_usd or 0)
        ceo_equity = round(equity / ceo.total_compensation_usd, 3)

    board_size = g("board_size") or (len(dirs) or None)
    # board composition: prefer the DIRECT governance-highlights counts (robust to text
    # trimming), fall back to computing from the per-director list.
    n_indep = g("n_independent_directors")
    if n_indep is not None and board_size:
        pct_independent = round(n_indep / board_size, 3)
    elif dirs and any(d.is_independent is not None for d in dirs):
        pct_independent = round(sum(bool(d.is_independent) for d in dirs) / len(dirs), 3)
    else:
        pct_independent = None
    # gender provenance: `stated` and `honorific` are DOCUMENT evidence, `name` is a bare
    # prior. Only 17.4% of proxies state gender at all, so this ratio is what makes
    # `pct_female_directors` auditable instead of silently inferred.
    bases = [(d.gender_basis or "").strip().lower() for d in dirs if d.gender]
    pct_gender_stated = (round(sum(b in ("stated", "honorific") for b in bases) / len(bases), 3)
                         if bases else None)
    n_female_inferred = sum(x.startswith("f") for x in genders)

    n_women = g("n_women_directors")
    if n_women is not None and board_size:
        pct_female = round(n_women / board_size, 3)
    elif genders:
        pct_female = round(sum(x.startswith("f") for x in genders) / len(genders), 3)
    else:
        pct_female = None

    row = {
        "ticker": ticker,
        "as_of": filing["filing_date"],
        "period": pd.to_datetime(filing.get("period_of_report"), errors="coerce"),
        "accession_number": filing["accession_number"],
        "company_name": extract.company_name,
        "fiscal_year_extract": extract.fiscal_year,
        # ---- Board / directors ----
        "n_directors": len(dirs) or None,
        "board_size": board_size,
        "avg_director_age": _mean(ages),
        "avg_board_tenure": _mean(tenures),
        "pct_independent_directors": pct_independent,
        "pct_female_directors": pct_female,
        "avg_other_public_boards": _mean(other_boards),
        # ---- CEO ----
        "ceo_name_proxy": ceo_name or None,
        "ceo_age": _ceo_age(extract),
        "ceo_since_year": extract.ceo_since_year,
        "ceo_is_founder": _bnum(extract.ceo_is_founder),
        "ceo_is_board_chair": _bnum(extract.ceo_is_board_chair
                                    if extract.ceo_is_board_chair is not None else g("ceo_is_board_chair")),
        "ceo_salary": ceo.salary_usd if ceo else None,
        "ceo_bonus": ceo.bonus_usd if ceo else None,
        "ceo_stock_awards": ceo.stock_awards_usd if ceo else None,
        "ceo_option_awards": ceo.option_awards_usd if ceo else None,
        "ceo_non_equity_incentive": ceo.non_equity_incentive_usd if ceo else None,
        "ceo_all_other_comp": ceo.all_other_comp_usd if ceo else None,
        "ceo_total_comp": ceo.total_compensation_usd if ceo else None,
        "ceo_equity_pay_pct": ceo_equity,
        # ---- NEO aggregate (most recent fiscal year only -- see `_latest_sct_rows`) ----
        "n_neos": len({(c.name or "").strip().lower() for c in latest_rows if c.name}) or None,
        "total_neo_comp": sum(c.total_compensation_usd for c in latest_rows
                              if c.total_compensation_usd is not None) or None,
        # how many fiscal years the SCT actually yielded. Item 402(c) requires three, so a 1
        # here is the "the carve missed most of the table" pathology -- still measurable after
        # the schema change that stops `n_neos` from tripling.
        "sct_years": len({c.fiscal_year for c in extract.compensation
                          if c.fiscal_year is not None}) or None,
        # ---- Ownership / alignment (direct from the beneficial-ownership summary) ----
        # ⚠ FOUR COLUMNS, TWO PAIRS, NO DERIVATION HERE. `*_ownership_pct` is the percent of
        # CLASS (economic) and `*_voting_pct` the percent of total voting power, each stored
        # exactly as extracted. A null voting leg means "this filing's ownership table printed
        # no voting-power column", which is the ordinary single-class case -- it is NOT
        # back-filled from the ownership leg here, because that would make the stored table a
        # mixture of what a filer disclosed and what we inferred. The single-class identity
        # (voting == ownership) is applied in the cube, where `f_control_wedge` is built and
        # the inference can be labelled as one.
        "insider_ownership_pct": g("insider_ownership_pct"),
        "insider_voting_pct": g("insider_voting_pct"),
        # The group's SHARE COUNT, which is always disclosed and exact, unlike a combined
        # economic percentage -- Alphabet's table prints per-class percentages and total voting
        # power and no combined column at all, so for that filer shape the percentage is
        # COMPUTABLE (shares / shares outstanding) and not extractable.
        "insider_shares": g("insider_shares"),
        "ceo_ownership_pct": g("ceo_ownership_pct"),
        "ceo_voting_pct": g("ceo_voting_pct"),
        "n_five_percent_holders": g("n_five_percent_holders"),
        # ---- Governance provisions ----
        "independent_chair": _bnum(g("independent_chair")),
        "lead_independent_director": _bnum(g("lead_independent_director")),
        "classified_board": _bnum(g("classified_board")),
        "dual_class_shares": _bnum(g("dual_class_shares")),
        "poison_pill": _bnum(g("poison_pill")),
        "majority_voting": _bnum(g("majority_voting_for_directors")),
        "say_on_pay_support_pct": g("say_on_pay_support_pct"),
        "ceo_pay_ratio": g("ceo_pay_ratio"),
        "median_employee_pay": g("median_employee_pay_usd"),
        # ---- auditor: name, tenure, fee breakdown (2.05% fill on the retired edgar path) ----
        "auditor_name": clean_text(g("auditor_name") or "") or None,
        "auditor_since_year": g("auditor_since_year"),
        "auditor_fees": g("auditor_fees_usd"),
        "audit_fees_audit": g("audit_fees_audit_usd"),
        "audit_fees_audit_related": g("audit_fees_audit_related_usd"),
        "audit_fees_tax": g("audit_fees_tax_usd"),
        "audit_fees_other": g("audit_fees_other_usd"),
        "auditor_fees_prior": g("auditor_fees_prior_usd"),
        # ---- child-table row counts + gender provenance ----
        "n_director_comp_rows": len(extract.director_compensation) or None,
        "n_ownership_rows": len(extract.ownership_holders) or None,
        # share of the board whose gender came from the DOCUMENT rather than a first-name prior
        "pct_gender_stated": pct_gender_stated,
        # the filing's own women-director count minus the count derived from per-director
        # gender. 0 when they agree; a persistent non-zero is the honest error bar on the
        # inference. Never used to overwrite anything.
        "n_women_directors_vs_inferred": (None if n_women is None or not genders
                                          else n_women - n_female_inferred),
        # Full JSON for downstream access. Keeping it is what makes RE-flattening free: the
        # four row builders and every derived column above are pure functions of this blob, so
        # a schema change can be replayed over 8,667 stored filings without an LLM call.
        "def14a_json": extract.model_dump_json(),
    }
    # Safety net BEHIND the prompt's unit instruction, not instead of it. A filer reports every
    # cell of its fee table in one unit, so the block is rescaled together or not at all --
    # rescaling cell-by-cell would invent a table whose categories no longer sum to the total.
    # Fires only when the LARGEST fee in the block is still implausibly small for an S&P 500
    # audit, which means an "(in thousands)" / "($ in millions)" note was missed (measured on 8
    # of the 10 smallest values: MS 57.6 = $57.6M, TSLA 10,919 = $10.9M).
    rescale_block(row, list(_FEE_COLS), DEF14A_AUDIT_FEE_MIN_PLAUSIBLE)
    # AFTER the rescale, so both sides of the comparison are in whole dollars. Recovers the total
    # on the fee tables that have no Total row, where the model reports the `Audit Fees` line as
    # the total (BA 39.1M -> 43.6M, T 34.2M -> 38.9M).
    sum_fee_total(row, "auditor_fees", list(_FEE_CATEGORY_COLS))
    # The WITHIN-ROW half of the CEO-pay sanity step (D4). It belongs here because it compares
    # only this row's own three pay-ratio columns; the rest of that step needs the same CEO's
    # OTHER filings for its neighbour reference, which one flattened filing cannot see, so it
    # runs as a batch over stored rows instead (`scripts/def14a_sct_sanity.py`).
    row = repair_pay_ratio(row)
    return row


# --------------------------------------------------------------------------- #
# Child-table row builders                                                    #
#                                                                             #
# All four are PURE functions of (ticker, filing, Def14AExtract), which is what makes them      #
# replayable over the already-stored `def14a_json` for free -- the tokens are paid, so the      #
# flatten can be verified over 445 real filings without a single LLM call.                      #
# --------------------------------------------------------------------------- #
#: `total - sum(components)` within this many dollars counts as reconciling. A FLAG, not a
#: filter (D10): values are kept either way, and the failure rate becomes measurable over time.
#: $10 absorbs the filers who round each component to the nearest dollar independently.
_RECONCILE_TOLERANCE_USD = 10.0

#: The seven SCT components (Item 402(c)) and the six director-comp components (Item 402(k)),
#: in the flat column vocabulary. Deliberately the SAME names the retired edgar child tables
#: used, so the Phase-6 before/after comparison is column-for-column.
_EXEC_COMPONENT_COLS = ("salary", "bonus", "stock_awards", "option_awards",
                        "non_equity_incentive", "pension_change", "other_compensation")
_DIRECTOR_COMPONENT_COLS = ("fees_earned", "stock_awards", "option_awards",
                            "non_equity_incentive", "pension_change", "other_compensation")
#: The fee block is rescaled TOGETHER or not at all -- a filer reports every cell of one table
#: in one unit, so a cell-by-cell rescale would invent a table whose parts no longer sum.
_FEE_COLS = ("auditor_fees", "audit_fees_audit", "audit_fees_audit_related",
             "audit_fees_tax", "audit_fees_other", "auditor_fees_prior")
#: The four Item 9(e) categories that make up the current-year total — `_FEE_COLS` minus the
#: total itself and minus the prior year, whose categories this schema does not carry.
_FEE_CATEGORY_COLS = ("audit_fees_audit", "audit_fees_audit_related",
                      "audit_fees_tax", "audit_fees_other")


def _keys(ticker: str, filing: pd.Series) -> dict:
    """The point-in-time key stamp every child row carries. `as_of` is the FILING date, never a
    period end -- that is what keeps the tables leak-free."""
    return {
        "ticker": ticker,
        "cik": str(filing.get("cik") or "") or None,
        "accession_number": filing["accession_number"],
        "as_of": filing["filing_date"],
    }


def _reconciles(row: dict, components: tuple[str, ...]) -> float | None:
    """1.0 when the components sum to `total` within `_RECONCILE_TOLERANCE_USD`, else 0.0; None
    when `total` is absent.

    A FLAG rather than a repair. The old edgar path filled a single missing component from the
    residual, which is no longer sound: with `pension_change` now in the schema the residual is
    not an unattributable gap, and filling it would overwrite a real column.
    """
    total = row.get("total")
    if total is None or not isinstance(total, (int, float)):
        return None
    parts = [row.get(c) for c in components]
    if not any(isinstance(v, (int, float)) for v in parts):
        return None
    summed = sum(float(v) for v in parts if isinstance(v, (int, float)))
    return 1.0 if abs(float(total) - summed) <= _RECONCILE_TOLERANCE_USD else 0.0


def _exec_comp_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per (NEO x fiscal year) of the Summary Compensation Table.

    34,741 such rows already sat unqueryable inside `def14a_llm.def14a_json`, against the
    retired edgar table's 2,378 on 25 tickers -- and on every measurable axis the LLM rows are
    better: title 100% vs 45.4%, stock awards 93.8% vs 45.4%, option awards 88.0% vs 27.5%, and
    2 rows above $1e9 versus 109.
    """
    rows = []
    for c in extract.compensation:
        name = clean_person_name(c.name)
        if not name or c.fiscal_year is None:
            # `fiscal_year` is Optional on the Pydantic model but PART OF THIS TABLE'S PRIMARY
            # KEY, so a null aborts the whole Postgres insert -- not one row. It is also a
            # useless row: comp that cannot be placed in time. 0 of 1,849 replayed rows lack
            # one, but that is evidence, not a guarantee, so the guard is structural.
            continue
        row = {
            **_keys(ticker, filing),
            "name": name,
            "title": clean_text(c.title or "") or None,
            "fiscal_year": c.fiscal_year,
            "salary": c.salary_usd,
            "bonus": c.bonus_usd,
            "stock_awards": c.stock_awards_usd,
            "option_awards": c.option_awards_usd,
            "non_equity_incentive": c.non_equity_incentive_usd,
            "pension_change": c.pension_change_usd,
            "other_compensation": c.all_other_comp_usd,
            "total": c.total_compensation_usd,
        }
        row["reconciles"] = _reconciles(row, _EXEC_COMPONENT_COLS)
        rows.append(row)
    return rows


def _director_comp_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per non-employee director (Item 402(k)).

    Single-year BY REGULATION -- 402(k) requires only the last completed fiscal year -- and
    membership here IS the definition of an outside director, which Phase 5's vote role map
    depends on. Absent from the Pydantic contract entirely before this phase.
    """
    rows = []
    for d in extract.director_compensation:
        name = clean_person_name(d.name)
        if not name:
            continue
        row = {
            **_keys(ticker, filing),
            "name": name,
            "fiscal_year": d.fiscal_year,
            "fees_earned": d.fees_earned_usd,
            "stock_awards": d.stock_awards_usd,
            "option_awards": d.option_awards_usd,
            "non_equity_incentive": d.non_equity_incentive_usd,
            "pension_change": d.pension_change_usd,
            "other_compensation": d.all_other_comp_usd,
            "total": d.total_compensation_usd,
        }
        row["reconciles"] = _reconciles(row, _DIRECTOR_COMPONENT_COLS)
        rows.append(row)
    return rows


def _ownership_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per beneficial holder (Item 403).

    Knowingly redundant with 13F / SC 13D-G / Forms 3-4-5, which stay the preferred sources.
    Two row shapes are dropped rather than stored: an "as a group" subtotal (that aggregate is
    already the `insider_ownership_pct` scalar) and a cell that is only a street address.
    """
    rows = []
    for h in extract.ownership_holders:
        if is_subtotal_holder(h.holder_name):
            continue
        name = clean_holder_name(h.holder_name)
        if not name:
            continue
        holder_type = (h.holder_type or "").strip().lower() or None
        if holder_type not in ("5pct_holder", "director_officer", None):
            holder_type = "5pct_holder" if (h.percent_of_class or 0) >= 0.05 else "director_officer"
        rows.append({
            **_keys(ticker, filing),
            "holder_name": name,
            "holder_type": holder_type or "director_officer",
            "shares": h.shares,
            "percent_of_class": h.percent_of_class,
            "percent_of_voting_power": h.percent_of_voting_power,
        })
    return rows


def _director_rows(ticker: str, filing: pd.Series, extract: Def14AExtract) -> list[dict]:
    """One row per director per filing -- the `directors[]` array, flattened.

    The most trustworthy block in the extract: 99.74% of names appear verbatim in the source,
    93% of ages and 98% of tenures are confirmable, and a full hand-check of HUBB 2022 was 27/27
    correct including the public-vs-private board judgements. `gender_basis` is what makes the
    gender field auditable, and the cross-filing consensus pass CANNOT be written without this
    table -- it needs a GROUP BY over people, across tickers and years.
    """
    rows = []
    for d in extract.directors:
        name = clean_person_name(d.name)
        if not name:
            continue
        gender = (d.gender or "").strip().lower() or None
        basis = (d.gender_basis or "").strip().lower() or None
        rows.append({
            **_keys(ticker, filing),
            "name": name,
            "age": d.age,
            "tenure_years": d.tenure_years,
            "is_independent": _bnum(d.is_independent),
            "gender": gender,
            # never leave the provenance blank when a gender is set: an unlabelled value is
            # indistinguishable from the first-name prior this upgrade exists to expose
            "gender_basis": basis or ("name" if gender else None),
            "other_public_company_boards": d.other_public_company_boards,
        })
    return rows


def _child_frames(ticker: str, filing: pd.Series, extract: Def14AExtract) -> dict:
    """All four child frames for one filing, keyed by table name."""
    return {
        "def14a_executive_comp": _exec_comp_rows(ticker, filing, extract),
        "def14a_director_comp": _director_comp_rows(ticker, filing, extract),
        "def14a_ownership": _ownership_rows(ticker, filing, extract),
        "def14a_directors": _director_rows(ticker, filing, extract),
    }

#: The `=== LABEL ===` block the carve emits for the Item 402(k) table. A populated section with
#: zero extracted rows is a RECALL failure and nothing else -- see `_log_director_comp_recall`.
_DIRECTOR_COMP_SECTION = "=== DIRECTOR COMPENSATION TABLE ==="


def _log_director_comp_recall(ticker: str, filing: pd.Series, payload: str,
                              n_rows: int) -> None:
    """Warn when the model was SHOWN a director-compensation table and returned no rows.

    ⚠ THIS IS THE CHECK WHOSE ABSENCE LET 1,097 FILINGS FAIL SILENTLY across 272 companies --
    12.45% of every post-2007 proxy in the archive yielding zero Item 402(k) rows, with IBM, WMB
    and LNT at 20 of 20 filings each and nothing anywhere saying so. `n_director_comp_rows` was
    already written to the parent row, so the number was in the database the whole time; what
    was missing was anything that read it.

    The distinction it draws is the one that matters, and it is the same A/B/C triage the
    diagnostic uses. A zero row count is only evidence of a defect when the table REACHED the
    payload: if the carve emitted no section the fault is the classifier (fixed in `tables.py`,
    and silent here because there is nothing to blame the model for), and if the filing predates
    the 2007 proxy season there is no table to find at all. So this fires only on the narrow
    case the message names, which is what stops it becoming noise nobody reads.
    """
    if n_rows or _DIRECTOR_COMP_SECTION not in (payload or ""):
        return
    body = (payload or "").split(_DIRECTOR_COMP_SECTION, 1)[1]
    body = body.split("\n=== ", 1)[0]
    if len(body.strip()) < 200:                 # an emitted but empty/truncated section
        return
    logger.warning(
        "%s %s (%s): the carve supplied a %d-char DIRECTOR COMPENSATION TABLE and the extract "
        "returned ZERO director-comp rows — extraction RECALL failure, not a carve miss",
        ticker, filing.get("filing_date", ""), filing.get("accession_number", ""),
        len(body.strip()))


def _result_frames(result: LlmResult) -> dict[Table, pd.DataFrame]:
    """One answer -> the five frames it fans out to, save-ready.

    CHILDREN FIRST, parent last: `run_extraction` saves in this order, so a crash between
    the two leaves a child row without a parent (recoverable -- the accession dedup keys on
    `def14a_llm`) rather than a parent that claims children it lacks.
    """
    ticker = str(result.task.meta["ticker"])
    filing = result.task.meta["filing"]
    extract = result.parsed

    _log_director_comp_recall(ticker, filing, result.task.payload,
                              len(_director_comp_rows(ticker, filing, extract)))

    frames: dict[Table, pd.DataFrame] = {}
    for name, child_rows in _child_frames(ticker, filing, extract).items():
        if child_rows:
            numeric, pk = _CHILD_SPEC[name]
            frames[_CHILD_TABLES[name]] = _prepare_frame(child_rows, numeric, pk)
    frames[Tables.def14a_llm] = _prepare_frame(
        [_flatten(ticker, filing, extract)], tuple(_NUMERIC_COLS),
        ["ticker", "accession_number"])
    return frames


#: match `schema.py`'s registration exactly -- a mismatch here silently drops rows on a filing
#: that lists the same person twice.
_CHILD_SPEC = {
    "def14a_executive_comp": (
        ("fiscal_year", "salary", "bonus", "stock_awards", "option_awards",
         "non_equity_incentive", "pension_change", "other_compensation", "total", "reconciles"),
        ["ticker", "accession_number", "name", "fiscal_year"]),
    "def14a_director_comp": (
        ("fiscal_year", "fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
         "pension_change", "other_compensation", "total", "reconciles"),
        ["ticker", "accession_number", "name"]),
    "def14a_ownership": (
        ("shares", "percent_of_class", "percent_of_voting_power"),
        ["ticker", "accession_number", "holder_name", "holder_type"]),
    "def14a_directors": (
        ("age", "tenure_years", "is_independent", "other_public_company_boards"),
        ["ticker", "accession_number", "name"]),
}
_CHILD_TABLES = {
    "def14a_executive_comp": Tables.def14a_executive_comp,
    "def14a_director_comp": Tables.def14a_director_comp,
    "def14a_ownership": Tables.def14a_ownership,
    "def14a_directors": Tables.def14a_directors,
}


def _prepare_frame(rows: list[dict], numeric: tuple[str, ...], pk: list[str]) -> pd.DataFrame:
    """Rows -> a save-ready frame: numeric columns coerced, NULs stripped, `as_of` normalised,
    duplicates collapsed on the table's own primary key."""
    df = pd.DataFrame(rows)
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = strip_nul(df)                          # Postgres TEXT rejects NUL (\x00)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.normalize()
    return df.drop_duplicates(subset=[c for c in pk if c in df.columns], keep="last")


