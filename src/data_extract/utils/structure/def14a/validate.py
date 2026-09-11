"""
validate.py (src/data_extract/utils/structure/def14a/validate.py)
--------------------------------------------------------------------------------
Row cleaner shared by the two DEF 14A paths: text normalisation, the name and
holder primary keys, the auditor-fee unit rescale, and the `sec_def14a` (ECD)
row repair.

Guiding rule: NEVER fabricate. A value is written only when it is
deterministically recoverable (a unit rescale a sibling filing confirms).
Anything else that fails a sanity check is set to NaN, because a NULL is honest
and a wrong number is not.

WHAT USED TO BE HERE, AND WHY IT IS GONE
----------------------------------------
This module began as a repair layer for edgartools' proxy HTML parser, whose
defects it neutralised row by row: a hardcoded `0.5` standing in for the "*"
("less than 1%") ownership footnote, a missed "(in thousands)" fee header that
made KO's audit fee 30,587 one year and 32,104,000 the next for the SAME fee,
the Summary Compensation Table's Total column duplicated into a component slot,
and pay-ratio triplets completed from an identity. That whole HTML block was
deleted -- a parser that returns values which are silently WRONG rather than
absent cannot be repaired into a source, only replaced -- so the repairs went
with it.

What survives is what BOTH paths still need:

- `clean_person_name` / `clean_holder_name` are the primary keys. The footnote
  strip is the load-bearing part: without it the same director keys as
  "Emma N. Walmsley11" one year and "Emma N. Walmsley10" the next, silently
  duplicating the row instead of updating it. `clean_holder_name` returns None
  for a cell that is only a street address, so the caller can drop the row
  rather than store a street as a shareholder.
  ⚠ `clean_person_name` and `clean_text` no longer LIVE here -- they moved to
  `src/utils/` so `src/data_aggregate/` can share the one person key without
  cross-importing this package, and are re-exported below so every call site
  here is unchanged. Edit them there, not by adding a second copy here.
- `is_subtotal_holder` rejects "Total" / "as a group (16 people)" lines. An LLM
  returns these just as readily as a grid parser did, and storing one
  double-counts the insiders it aggregates.
- `rescale_block` is the safety net behind the LLM's own unit conversion, and
  fires on the whole fee block at once -- rescaling cell by cell would invent a
  table whose components no longer sum to its total.
- `repair_main_row` now handles only the ECD row (see its docstring).

The pay-ratio identity did NOT disappear: it lives on the LLM side in
`def14a_impute._reconcile_rows`, where it holds on 95.0% of 3,838 rows.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Re-exported, not defined: the person key is shared vocabulary between this package and
# `src/data_aggregate/`'s governance features, so it lives in `src/utils/names.py` (see D26 /
# AGENTS.md "no cross-imports between `src/` subfolders"). Keeping the names importable from
# here is what makes that a MOVE rather than a refactor: zero call sites change.
from src.utils.names import clean_person_name, person_key  # noqa: F401
from src.utils.string import clean_text  # noqa: F401

# Sanity bounds separating "implausible for an S&P 500 issuer, therefore mis-scaled or
# fabricated by the parser" from "small but real" -- the repair layer only fires on the former.
DEF14A_AUDIT_FEE_MIN_PLAUSIBLE = 1e5     # a sub-$100k TOTAL auditor fee => block is in thousands
DEF14A_NET_INCOME_MIN_PLAUSIBLE = 1e4    # a sub-$10k net income => figure is in millions/billions
#: Dollars of slack when comparing a fee total against its categories. Fee tables are printed to
#: the dollar or to $0.1M, so $10 is below any real rounding step while still absorbing float
#: error -- the two real defects miss by $4.5M and $4.7M, not by cents.
DEF14A_FEE_SUM_TOL = 10.0

__all__ = [
    "clean_text", "clean_person_name", "clean_holder_name", "is_subtotal_holder",
    "rescale_block", "sum_fee_total", "DEF14A_AUDIT_FEE_MIN_PLAUSIBLE",
    "DEF14A_NET_INCOME_MIN_PLAUSIBLE", "DEF14A_FEE_SUM_TOL", "repair_main_row",
    "repair_pay_ratio", "sct_reference", "sanity_check_exec_comp", "DEF14A_SCT_PART_COLS",
    "DEF14A_SCT_PARTS_LO", "DEF14A_SCT_PARTS_HI", "DEF14A_SCT_KEEP_BAND",
    "DEF14A_SCT_REPAIR_BAND", "DEF14A_CEO_TOTAL_MIN_PLAUSIBLE", "DEF14A_PAY_RATIO_TOL",
    "DEF14A_MEDIAN_PAY_JUMP_MAX",
]

_NAN = float("nan")

# A street address glued onto (or standing in for) an institutional holder name:
# "The Vanguard Group 100 Vanguard Blvd. Malvern, PA 19355" / "50 Hudson Yards, New York, NY 10001".
_ADDRESS_TAIL_RE = re.compile(r"\s+\d+\s+[A-Z][\w.]*(?:\s+[\w.]+)*?,?\s*[A-Z]{2}\s+\d{5}.*$")
_ADDRESS_ONLY_RE = re.compile(r"^\d+\s+.*\b[A-Z]{2}\s+\d{5}\b")

# Aggregate / subtotal pseudo-holders that must never be stored as a holder row.
_SUBTOTAL_HOLDER_RE = re.compile(
    r"^\s*(?:sub)?total\b|\bas\s+a\s+group\b|\ball\s+(?:current\s+)?(?:directors|executive)",
    re.I,
)


def _isnum(x: Any) -> bool:
    """True only for a real, finite number -- `pd.notna` alone still lets `inf` through."""
    try:
        return x is not None and pd.notna(x) and float(x) not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


def is_subtotal_holder(value: Any) -> bool:
    """True for an aggregate pseudo-holder that must never be stored as a holder row:
    "Total", "as a group", "All current directors and executive officers".

    That aggregate is already carried as the `insider_ownership_pct` SCALAR on `def14a_llm`, so
    storing it again as a holder would double-count the insiders against the real per-person
    rows. An LLM returns these lines just as readily as a grid parser did."""
    return bool(value) and bool(_SUBTOTAL_HOLDER_RE.search(str(value)))


def clean_holder_name(value: Any) -> str | None:
    """Institutional holder name with the mailing address stripped off the tail. Returns None when
    the cell is ONLY an address (edgartools grabbed the wrong line -- JPM's proxy), so the caller
    can drop the row rather than store a street as a shareholder."""
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    if _ADDRESS_ONLY_RE.match(cleaned):
        return None
    cleaned = _ADDRESS_TAIL_RE.sub("", cleaned).strip()
    cleaned = cleaned.rstrip(",;:-").strip()
    return cleaned or None


def rescale_block(row: dict, cols: list[str], min_plausible: float) -> None:
    """Rescale a whole fee block to dollars IN PLACE. edgartools reports every cell of a given
    table in one unit, so the block is rescaled together or not at all -- rescaling cell-by-cell
    would invent a table where the components no longer sum to the total. Fires only when the
    LARGEST value in the block is still implausibly small, which for an S&P 500 auditor fee means
    the "(in thousands)" header was missed."""
    values = [float(row[c]) for c in cols if c in row and _isnum(row[c])]
    values = [v for v in values if v != 0.0]
    if not values:
        return
    largest = max(abs(v) for v in values)
    if largest >= min_plausible:
        return
    factor = 1_000.0 if largest * 1_000.0 >= min_plausible else 1_000_000.0
    for c in cols:
        if c in row and _isnum(row[c]):
            row[c] = float(row[c]) * factor


def sum_fee_total(row: dict, total_col: str, part_cols: list[str],
                  tol: float = DEF14A_FEE_SUM_TOL) -> bool:
    """Rebuild a fee TOTAL from its categories IN PLACE when the total is really a category.

    Not every fee table has a Total row. BA's and T's do not, and on both the model put the
    `Audit Fees` LINE into the total: BA 39.1M against a real 43.6M (39.1 + 4.5 + 0 + 0), T
    34.2M against 38.9M (34.2 + 1.3 + 3.4 + 0). That understates the total by 10-12%, and the
    field is the fee-growth signal's base.

    Three conditions, all necessary, so this is the defect's shape and not a general override:
      * all four categories present -- a PARTIAL set sums to less than the truth, and the four
        filings here with 3 categories (INCY, JPM, KLAC, PFE) all have a correct stated total
        already, so a partial sum would corrupt them;
      * the total EQUALS one of the categories within `tol` -- the fingerprint of a copied line.
        Without this the filer's own Total row would lose to our sum, and the filer's row is
        authoritative: it may legitimately include a category this schema does not model;
      * the sum EXCEEDS the total by more than `tol` -- not merely differs. A sum BELOW a stated
        total means exactly that unmodelled-category case, where the total is the better number.

    Measured over 22 filings: 16 of the 18 complete blocks already sum to their stated total to
    the dollar, so this fires on BA and T alone. Returns whether it fired.
    """
    if not _isnum(row.get(total_col)):
        return False
    parts = [row.get(c) for c in part_cols]
    if not all(_isnum(p) for p in parts):
        return False
    total = float(row[total_col])
    parts_f = [float(p) for p in parts]
    if not any(abs(total - p) <= tol for p in parts_f):
        return False
    if sum(parts_f) - total <= tol:
        return False
    row[total_col] = sum(parts_f)
    return True


def repair_main_row(row: dict) -> dict:
    """Repair one `sec_def14a` (ECD) row. Mutates and returns a COPY.

    Only three things are left to do here. The fee-block rescale, the fee-table year labels and
    the pay-ratio identity went with the HTML block they existed for; the pay-ratio identity is
    NOT lost -- it lives on the LLM side in `def14a_impute._reconcile_rows`, where it holds on
    95.0% of 3,838 rows, and duplicating it here would give two places to edit.
    """
    row = dict(row)
    for col in ("company_name", "peo_name", "company_selected_measure_name", "peo_names_all"):
        if col in row:
            row[col] = clean_text(row.get(col))

    # net_income cannot be rescaled the way a fee block can. SBUX FY2025 arrives as `1856.4`
    # (raw value '1856.4', decimals='1', unit_ref='usd' -- tagged in $ millions), and that is
    # equally consistent with millions and billions: NOTHING in the fact disambiguates them,
    # unlike a fee block whose factor a sibling filing's overlapping year confirms. So an
    # implausible figure is dropped rather than guessed; `fundamentals_history` has the real one.
    if _isnum(row.get("net_income")) and 0 < abs(float(row["net_income"])) < DEF14A_NET_INCOME_MIN_PLAUSIBLE:
        row["net_income"] = _NAN

    # Belt-and-braces behind `def14a_ecd`'s selection-time zero drop: a PEO is never paid exactly
    # $0. On the ECD path that value is SBUX's individual x year matrix marking a year the person
    # was not PEO, and dropping it at selection time recovers the CORRECT value instead of this
    # NULL -- so if a zero still reaches here, the selection missed something.
    for col in ("peo_total_comp", "peo_actually_paid_comp", "neo_avg_total_comp",
                "neo_avg_actually_paid_comp"):
        if _isnum(row.get(col)) and float(row[col]) == 0.0:
            row[col] = _NAN
    return row



# --------------------------------------------------------------------------------------------
# THE SUMMARY-COMPENSATION-TABLE SANITY STEP (D5 / D4 / D2)
# --------------------------------------------------------------------------------------------
# Pure arithmetic on columns `def14a_llm` already stores, run AFTER the LLM has written its
# rows. Nothing here reads a filing or calls a model: every rule is an identity the stored row
# either satisfies or does not.
#
# ⚠ WHY THIS IS SPLIT IN TWO, AND WHY IT IS NOT ALL IN THE WRITE PATH. The plan for this step
# asked for one function called from `flatten._result_frames`. That is impossible for half of
# it: `_result_frames` flattens ONE filing (`[_flatten(ticker, filing, extract)]`), and the
# neighbour reference below needs the same CEO's OTHER filings. So the rules divide by what
# they can see:
#   * `repair_pay_ratio` is WITHIN-ROW -- it compares a row's own three pay-ratio columns -- and
#     runs in the write path on every future extraction.
#   * `sanity_check_exec_comp` is CROSS-ROW and runs as a batch step over stored rows
#     (`scripts/def14a_sct_sanity.py`), which is re-runnable after any extraction, not a
#     one-off backfill.
#
# ⚠ THE NEIGHBOUR BAND DECIDES WHICH LEG TO TRUST -- IT IS NOT DIRECTIONAL. The plan wrote two
# opposed rules: "total below its components -> replace the total with the components' sum" and
# "total above its components -> keep the total". Measured over all 56 failing filings, that
# pair deletes correct filed values. GE's 2001 proxy is the case that settles it: John F.
# Welch's FY2000 total is $16,754,019 = salary $4,000,000 + bonus $12,700,000 + other $54,019,
# which is exactly what the PRE-2006 Summary Compensation Table's Total column was -- equity
# awards were disclosed in their own columns and were NOT summed into it. So Sigma(parts)
# reaches $67,973,744 by adding ~$51.2M of equity the filed Total legitimately excludes, and
# the directional rule nulls a total that Welch's own other years corroborate (reference
# $10,104,944, ratio 1.66). One rule replaces both: score EACH leg against the reference and
# keep the leg the neighbours support.
#
# ⚠ TWO BANDS, BECAUSE KEEPING AND WRITING ARE NOT THE SAME ACT. Keeping a filed value needs
# only corroboration, so it uses the plan's wide 0.33x-3.0x. REPLACING a total with
# Sigma(parts) writes a NEW number and needs a near-match, so it uses 0.67x-1.5x. The plan
# stated 0.33x-3.0x for the repair too, but its own verified table then rejected Sigma(parts)
# at 0.34x (RCL 2021), 0.36x (F 2011) and 0.44x (ADBE 2001) while accepting only 0.96x
# (FAST 2000) -- the narrow band is what its evidence actually describes. Measured, the two
# bands reproduce that table exactly: of 56 failing filings, 38 KEEP, 9 NULL, 8 have no
# reference, and FAST 2000 is the single repair.
#
# ⚠ A NEGATIVE COMPONENT IS FILED EVIDENCE, NOT A PARSE ERROR, so only the TOTAL is tested for
# sign. The plan asked to blank negative components as well ("a Summary Compensation Table
# total is a sum of non-negative awards"). It is not: for fiscal years 2006-2008 the SCT
# reported Stock and Option Awards as the FAS 123R expense RECOGNISED in the year, which goes
# negative when a performance award is reversed. All 10 stored rows with a negative component
# are that regime or a negative pension line -- AEP 2009, COF 2008, EQT 2009, PH 2009, SPGI
# 2009, URI 2009, MTD 2010/2015/2017, ON 2002 -- and 3 of them (EQT, MTD 2017, URI) have
# `total == Sigma(parts)` TO THE DOLLAR with the negative included, so blanking the component
# would break a row that currently reconciles exactly. `def14a_executive_comp` corroborates
# independently: EQT's FY2008 rows for Murry Gerber (-8,920,166) and Philip Conti (-346,545)
# both carry `reconciles = 1`.
#
# The one negative TOTAL is still blanked, and EQT 2009 is it. That value is arithmetically
# real, but a negative pay LEVEL cannot serve any consumer of this column -- it has no
# logarithm, it inverts every growth rate through it, and it is a denominator in the pay ratio.
# It is dropped as unusable, NOT as wrong; the components that produce it stay on the row.

#: The six stored Summary Compensation Table components. `pension_change` is deliberately absent
#: -- `def14a_llm` does not carry it (only the per-NEO `def14a_executive_comp` does), which is
#: why the two identities below are loose BANDS and not `total == Sigma(parts)`. A real SCT also
#: carries a deferred-compensation-earnings line this schema does not model, and an exact
#: identity would fire on thousands of correct filings.
DEF14A_SCT_PART_COLS = (
    "ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
    "ceo_non_equity_incentive", "ceo_all_other_comp",
)
#: A total below half, or above twice, the sum of its own components has lost or gained a whole
#: component. Median `total / Sigma(parts)` is 1.0000 in every disclosure regime (pre-2006,
#: 2007-09, 2010+), so the identity holds throughout and only its FAILURE RATE moves with the
#: regime: 2.33% of pre-2006 filings fall below the band against 0.06% of 2010-and-later ones.
DEF14A_SCT_PARTS_LO = 0.5
DEF14A_SCT_PARTS_HI = 2.0
#: Corroborating a value the filer wrote (wide) versus writing a new one ourselves (narrow).
DEF14A_SCT_KEEP_BAND = (0.33, 3.0)
DEF14A_SCT_REPAIR_BAND = (0.67, 1.5)
#: A CEO total of $1 or less is a placeholder, not a package, so a pay ratio computed FROM it is
#: meaningless -- GOOGL 2018/2019 ($1 against a $197,274 median) and SMCI 2023 (a stated ratio
#: of 0.1). It is NOT a test of whether the total itself is real: TSLA's $0 is real, and it
#: needs no exemption here because its ratio and its recomputation agree at 0.
DEF14A_CEO_TOTAL_MIN_PLAUSIBLE = 1.0
#: How far a disclosed pay ratio may sit from `ceo_total_comp / median_employee_pay` before the
#: row is called UNRECONCILED. 123 of 3,845 rows carrying both legs and a plausible total (3.2%)
#: disagree by more than this. It is a REPORTING threshold, not a repair trigger -- see
#: `repair_pay_ratio` for the measurement that took the repair away.
DEF14A_PAY_RATIO_TOL = 0.25
#: `median_employee_pay` year on year is otherwise wage inflation -- p05/p50/p95 = 0.808 /
#: 1.036 / 1.292 over 3,480 consecutive pairs -- so a move past this factor is worth reporting.
#: It is REPORTED, never blanked: of the 6 filings that breach it, CCL 2022 (0.32x) and LYV
#: 2021/2022 (3.12x then 0.28x) are the real composition effect of furloughing low-paid staff
#: through COVID, not a parse error.
DEF14A_MEDIAN_PAY_JUMP_MAX = 3.0


def _f(x: Any) -> float | None:
    """`float(x)` for a real finite number, else None -- the row-dict counterpart of `_isnum`."""
    return float(x) if _isnum(x) else None


def repair_pay_ratio(row: dict) -> dict:
    """Drop one row's `ceo_pay_ratio` when the row proves it unusable. Mutates, returns a COPY.

    WITHIN-ROW, so this is the half of the sanity step that can run in the write path. Only two
    mechanisms survived measurement, and both DELETE rather than rewrite:

    1. **The columns are swapped.** `ceo_pay_ratio == median_employee_pay` exactly is not a
       coincidence at these magnitudes -- GOOGL 2018 (197,274) and 2019 (246,804) hold the
       identical number in both columns against a $1 total. One of the two is definitely wrong
       and neither can be recovered from the other, so the ratio goes.
    2. **The total is a placeholder and the ratio disagrees with it.** A ratio cannot be
       reconciled against a `ceo_total_comp` of $1 or less, and where the two disagree the row
       is internally broken with no recoverable leg: GOOGL's recomputation would be 0.0 and
       SMCI 2023's 1.24e-5 against a stated 0.1. 5 rows, and TSLA's 2022 filing is one of them
       -- a stated 18,043 against a $0 total, where `def14a_executive_comp` independently
       carries Musk's FY2021 total as 0 with `reconciles = 1` and TSLA's four other $0 filings
       all disclose a ratio of 0.

    ⚠ THE PLAN'S THIRD RULE WAS REMOVED BECAUSE IT CORRUPTS THE COLUMN. It asked that a
    disagreement beyond `DEF14A_PAY_RATIO_TOL` be resolved by PREFERRING
    `ceo_total_comp / median_employee_pay`, "because both its legs are independently
    checkable". Measured over all 123 disagreeing rows, that rewrite is wrong more often than
    it is right, because three different mechanisms produce the same arithmetic symptom:

      * **the disclosed ratio is right and OUR TOTAL is under-extracted.** JPM's 2022 proxy
        discloses 917 against a stored total of $34,500,000; 917 x $92,112 = $84,466,704, which
        is James Dimon's real FY2021 Summary Compensation Table total -- our row is missing his
        $52.6M special option award (`ceo_option_awards` is 0). GS 2023 is the same shape.
      * **the disclosed ratio is stale.** AMZN 2023 discloses 6,474, which is Andrew Jassy's
        FY2021 figure read out of a comparative discussion; 6,474 x $32,855 = $212,703,270 is
        his one-time 2021 RSU grant, and the stored $1,298,723 total is correct for FY2022.
        This is the case the plan generalised from, and it is the MINORITY.
      * **neither column is wrong.** Item 402(u) lets a filer annualise a partial-year CEO's
        pay or use the person serving at year end, so after a mid-year change the disclosed
        ratio legitimately does not reconcile with the incoming CEO's SCT total -- MDLZ 2018
        (Dirk Van de Put from November), LOW 2019, HWM 2019.

    That last mechanism dominates: **87 of the 123 disagreements (71%) sit on a filing where
    the CEO changed**, and the disagreement rate is **21.0% across a transition against 1.0%
    for the same CEO -- a 21x elevation**. So the arithmetic cannot tell a defect from a
    disclosure rule, and `ceo_total_comp` is not the trustworthy leg the plan assumed: it is
    the leg the 56 failing identities in `sanity_check_exec_comp` are about. Agreement between
    the total and its own components is NOT independent corroboration either -- both come from
    one model read of one table, which is why 122 of 123 stored totals sit closer to
    Sigma(parts) than the ratio-implied total does while JPM's is still the wrong number.

    The 34 same-CEO disagreements are the genuinely suspect residue and are the right input for
    a targeted re-extraction (phase 5), not for arithmetic. They are counted, not touched.

    ⚠ A ZERO RATIO IS NOT REPAIRED AND MUST NOT BE. Four of the five stored zeros are TSLA,
    where `ceo_total_comp` is 0, `ceo_salary` is 0 and `median_employee_pay` is populated and
    plausible ($34,084-$57,243): Elon Musk takes no pay and `0 / 46,150` is the correct ratio.
    No exemption is written for it -- the recomputation independently returns 0, the two agree,
    and rule 2 never fires. `test_a_zero_pay_ratio_survives` pins that.
    """
    row = dict(row)
    ratio = _f(row.get("ceo_pay_ratio"))
    median = _f(row.get("median_employee_pay"))
    total = _f(row.get("ceo_total_comp"))
    if ratio is None:
        return row
    # 1. the swap
    if median is not None and median > 0 and ratio == median:
        row["ceo_pay_ratio"] = _NAN
        return row
    if median is None or median <= 0 or total is None:
        return row
    recomputed = total / median
    # `recomputed == 0` makes a relative comparison undefined; agreement there means `ratio` is
    # 0 too, which is the TSLA case and is left exactly as filed.
    agrees = (ratio == 0.0) if recomputed == 0.0 else (
        abs(ratio - recomputed) <= DEF14A_PAY_RATIO_TOL * abs(recomputed))
    # 2. a disagreement over a placeholder total leaves nothing to keep. A disagreement over a
    # REAL total is reported by `sanity_check_exec_comp` and left alone -- see above.
    if not agrees and total <= DEF14A_CEO_TOTAL_MIN_PLAUSIBLE:
        row["ceo_pay_ratio"] = _NAN
    return row


def _leave_one_out_median(values: pd.Series) -> pd.Series:
    """Per element, the median of every OTHER non-null element of `values`. NaN where there is
    no other element. An element that is itself null gets the median of all of them.

    Written out rather than done with `transform("median")` because that median INCLUDES the row
    being tested, and the row being tested is the one suspected of being wrong.
    """
    arr = values.to_numpy(dtype="float64", copy=True)
    known = np.flatnonzero(~np.isnan(arr))
    pool = arr[known]
    out = np.full(arr.shape, np.nan)
    if pool.size == 0:
        return pd.Series(out, index=values.index)
    for position in range(arr.size):
        if np.isnan(arr[position]):
            rest = pool                      # contributes nothing, so nothing to take out
        else:
            rest = np.delete(pool, int(np.searchsorted(known, position)))
        if rest.size:
            out[position] = np.median(rest)
    return pd.Series(out, index=values.index)


def sct_reference(rows: pd.DataFrame) -> pd.Series:
    """Per row, the median POSITIVE `ceo_total_comp` across the same CEO's OTHER filings at the
    same ticker. NaN where that CEO has no other filing. Indexed like `rows`.

    ⚠ THE KEY IS `person_key`, NOT `clean_person_name`. The plan specified the latter, which
    preserves casing and so splits one CEO in two: COHR files "FRANCIS J. KRAMER" in caps
    against "Francis J. Kramer" elsewhere, and a split reference is a MISSING reference, which
    silently turns a repairable row into an untestable one. `person_key` folds case, drops
    generational suffixes and keys on the first initial, and is the same key
    `governance.names.ceo_identity` uses -- so this step and the CEO-turnover guard agree about
    who the CEO is rather than disagreeing at the margins.

    ⚠ THE REFERENCE EXCLUDES THE ROW BEING TESTED, AND IT IS A TRUE LEAVE-ONE-OUT. Otherwise a
    CEO with a single filing scores a perfect 1.00 against themselves and every rule passes
    vacuously. A plain group median is NOT good enough either -- it contains the row under test,
    which is precisely the row suspected of being wrong, and that contamination bites hardest on
    the `total > 2 x Sigma(parts)` cases where the suspect total is large and positive. So the
    median is taken over the OTHER rows only, one row at a time.

    8 of the 56 failing filings have no reference at all (AIG 2001, ATO 2018, CSX 1999, IFF
    2006, INCY 2000, NEM 2007, PNW 2009, TSN 2003) and are left untouched.
    """
    if "ceo_total_comp" not in rows.columns:
        return pd.Series(_NAN, index=rows.index, dtype="float64")
    names = rows.get("ceo_name_proxy", pd.Series(None, index=rows.index, dtype="object"))
    keys = names.map(person_key)
    tickers = rows.get("ticker", pd.Series(None, index=rows.index, dtype="object"))
    positive = pd.to_numeric(rows["ceo_total_comp"], errors="coerce")
    positive = positive.where(positive > 0)
    if positive.empty:
        return pd.Series(_NAN, index=rows.index, dtype="float64")
    return (positive.groupby([tickers, keys], dropna=False)
            .transform(_leave_one_out_median).astype("float64"))


def sanity_check_exec_comp(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """The full CEO-pay sanity step over a frame of stored `def14a_llm` rows. Returns a COPY and
    a tally of every decision taken.

    CROSS-ROW: the neighbour reference needs a ticker's whole filing history, so this runs as a
    batch over stored rows and not per filing. Idempotent -- a row it has already settled
    satisfies the identities and is not tested again.

    The rules, in order:

    1. **A negative total is dropped** as unusable (see the module comment: it is real
       arithmetic off a real negative FAS 123R expense, but a negative pay level has no
       logarithm and inverts every growth rate through it). Negative COMPONENTS are kept --
       they are the filer's own numbers and three rows reconcile exactly with them.
    2. **A total that fails an identity is scored against the neighbour reference** -- below its
       own salary, below `DEF14A_SCT_PARTS_LO` x Sigma(parts), or above `DEF14A_SCT_PARTS_HI` x
       Sigma(parts). Then, and only then, one of four things happens:
         * the total sits in `DEF14A_SCT_KEEP_BAND` of the reference -> **KEEP** it; the
           components are the suspect leg (38 of 56 rows, including every "keep" the plan
           verified: JPM 2025, TEL 2021, ELV 2002, SPG 2014, MSI 2026);
         * else Sigma(parts) sits in the narrower `DEF14A_SCT_REPAIR_BAND` -> **REPAIR**
           `ceo_total_comp := Sigma(parts)` (FAST 2000: $0 becomes $117,000 against a $122,500
           reference, and it is the only repair in the table);
         * else -> **NULL** (9 rows: A 2011, ADBE 2001, COHR 2013, DOW 2021, EQT 2009, F 2011,
           FAST 2004, RCL 2021, VRTX 2003). Agilent is why the band is load-bearing: without it
           its $0 becomes $160,091, which is 1.7% of its reference and far more dangerous as a
           pay-ratio denominator than the zero was, because it looks plausible;
         * no reference at all -> **leave the row alone** and count it.
    3. **`repair_pay_ratio` runs last**, so it judges the ratio against a total that has
       already been repaired or dropped rather than against the raw one. It only ever DELETES an
       unusable ratio; the 123 rows whose ratio disagrees with its own legs are counted and left
       alone, because 87 of them are mid-year CEO transitions where Item 402(u) permits exactly
       that and no arithmetic separates them from the rest.

    ⚠ A CLEAN ROW IS NEVER TESTED AGAINST ITS NEIGHBOURS. CEO pay is genuinely lumpy and a real
    mega-grant year sits well outside 3x its neighbours; the band gates only rows that have
    ALREADY failed an arithmetic identity, which is what keeps a wide band safe.
    """
    out = rows.copy()
    tally: dict[str, int] = {}

    def _count(key: str, n: int) -> None:
        tally[key] = tally.get(key, 0) + int(n)

    if "ceo_total_comp" not in out.columns:
        return out, tally
    total = pd.to_numeric(out["ceo_total_comp"], errors="coerce")
    present = [c for c in DEF14A_SCT_PART_COLS if c in out.columns]
    parts_df = out[present].apply(pd.to_numeric, errors="coerce") if present else None
    parts = parts_df.fillna(0.0).sum(axis=1) if present else pd.Series(0.0, index=out.index)
    n_parts = parts_df.notna().sum(axis=1) if present else pd.Series(0, index=out.index)

    # 1. the negative total
    negative = total < 0
    _count("nulled_negative_total", negative.sum())
    _count("negative_component_rows_KEPT", int(
        (parts_df.min(axis=1) < 0).sum()) if present else 0)
    total = total.mask(negative)

    # 2. the identities. Both band tests need a POSITIVE Sigma(parts) to be meaningful, which a
    # row carrying a negative component may not have.
    salary = (pd.to_numeric(out["ceo_salary"], errors="coerce")
              if "ceo_salary" in out.columns else pd.Series(_NAN, index=out.index))
    testable = total.notna() & (n_parts > 0) & (parts > 0)
    below_salary = total.notna() & salary.notna() & (total < salary)
    below_parts = testable & (total < DEF14A_SCT_PARTS_LO * parts)
    above_parts = testable & (total > DEF14A_SCT_PARTS_HI * parts)
    failing = below_salary | below_parts | above_parts
    _count("failed_total_below_salary", below_salary.sum())
    _count("failed_total_below_parts", below_parts.sum())
    _count("failed_total_above_parts", above_parts.sum())

    ref = sct_reference(out)
    no_ref = failing & ref.isna()
    _count("left_alone_no_reference", no_ref.sum())

    def _in(value: pd.Series, band: tuple[float, float]) -> pd.Series:
        lo, hi = band
        return value.notna() & ref.notna() & (value >= lo * ref) & (value <= hi * ref)

    keep = failing & _in(total, DEF14A_SCT_KEEP_BAND)
    repair = failing & ~keep & _in(parts.where(testable), DEF14A_SCT_REPAIR_BAND)
    null = failing & ~keep & ~repair & ref.notna()
    _count("kept_total_components_suspect", keep.sum())
    _count("repaired_total_from_components", repair.sum())
    _count("nulled_total_no_plausible_leg", null.sum())
    total = total.mask(repair, parts).mask(null)
    out["ceo_total_comp"] = total

    # 3. the pay ratio, against the total as it now stands
    if "ceo_pay_ratio" in out.columns:
        before = pd.to_numeric(out["ceo_pay_ratio"], errors="coerce")
        repaired = pd.DataFrame(
            [repair_pay_ratio(r) for r in out.to_dict("records")], index=out.index)
        out["ceo_pay_ratio"] = pd.to_numeric(repaired["ceo_pay_ratio"], errors="coerce")
        after = out["ceo_pay_ratio"]
        _count("pay_ratio_nulled_unusable", int((before.notna() & after.isna()).sum()))
        _count("pay_ratio_rewritten", int(
            (before.notna() & after.notna() & (before != after)).sum()))
        # REPORTED, never repaired: three mechanisms share this symptom and arithmetic cannot
        # separate them (`repair_pay_ratio`). This is the phase-5 re-extraction worklist.
        med = pd.to_numeric(out["median_employee_pay"], errors="coerce")
        recomputed = (pd.to_numeric(out["ceo_total_comp"], errors="coerce") / med).where(med > 0)
        unreconciled = (after.notna() & recomputed.notna() & (recomputed > 0)
                        & ((after - recomputed).abs() > DEF14A_PAY_RATIO_TOL * recomputed))
        _count("pay_ratio_UNRECONCILED_reported_only", unreconciled.sum())

    # a report, never a repair -- two of the six breaches are a real COVID composition effect
    if {"median_employee_pay", "ticker", "as_of"} <= set(out.columns):
        med = pd.to_numeric(out["median_employee_pay"], errors="coerce")
        ordered = out.assign(_m=med.where(med > 0)).sort_values(["ticker", "as_of"])
        prev = ordered.groupby("ticker", sort=False)["_m"].shift(1)
        step = (ordered["_m"] / prev).dropna()
        jump = (step > DEF14A_MEDIAN_PAY_JUMP_MAX) | (step < 1.0 / DEF14A_MEDIAN_PAY_JUMP_MAX)
        _count("median_employee_pay_jump_REPORTED_only", jump.sum())
    return out, tally
