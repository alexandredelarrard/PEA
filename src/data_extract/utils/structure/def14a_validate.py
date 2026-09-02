"""
def14a_validate.py (src/data_extract/utils/structure/def14a_validate.py)
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

import pandas as pd

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
]

_NAN = float("nan")

# Trailing footnote markers on a person cell: "(3)", "*", "†", or bare digits glued to the surname
# ("Daniel Pinto7", "Emma N. Walmsley11"). Bare digits are only stripped after a letter, so a name
# is never confused with a numbered list item.
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?:\(\d+\)|[*†‡§]|(?<=[a-z])\d{1,2})+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")

# Titles that edgartools glues onto the name when the source cell has the name and the position on
# two visual lines. The leading modifier group matters: without it "Luca Maestri Former Senior Vice
# President" splits at "Senior Vice" and leaves "Luca Maestri Former" as the name (and likewise
# "Bob De Lange Group"), so the modifier is consumed into the TITLE where it belongs.
_GLUED_TITLE_RE = re.compile(
    r"\s*(?:Former\s+|Group\s+|Interim\s+|Acting\s+|Co-)?(?:"
    r"Chairman\b|Chief\s|President\b|Senior\s+Vice\b|Executive\s+Vice\b|Vice\s+Chair\b|"
    r"General\s+Counsel\b|Co-CEO\b|\bCEO\b|\bCFO\b|\bCOO\b"
    r").*$"
)

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


def clean_text(value: Any) -> str | None:
    """Collapse the whitespace runs edgartools preserves from the source HTML
    ("Free                cash flow" -> "Free cash flow"). None for empty."""
    if value is None or not isinstance(value, str):
        return None
    cleaned = _WHITESPACE_RE.sub(" ", value.replace("\xa0", " ")).strip()
    return cleaned or None


def clean_person_name(value: Any) -> str | None:
    """Normalise a person cell into a STABLE primary key: collapse whitespace, strip the glued-on
    title and any trailing footnote marker. Casing is left alone (it is source-faithful and
    lower-casing would fight the rest of the repo), but the footnote strip is what actually
    matters -- without it the same director keys as "Emma N. Walmsley11" one year and
    "Emma N. Walmsley10" the next, silently duplicating the row instead of updating it."""
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    cleaned = _GLUED_TITLE_RE.sub("", cleaned).strip()
    cleaned = _FOOTNOTE_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.rstrip(",;:-").strip()
    return cleaned or None


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

