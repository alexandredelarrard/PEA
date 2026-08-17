"""
def14a_validate.py (src/data_extract/utils/structure/def14a_validate.py)
--------------------------------------------------------------------------------
Repair layer sitting between edgartools' `ProxyStatement` and the `def14a_edgar*`
tables. `fetch_def14a_edgar.py` is a faithful pass-through of the library; this
module is where the library's KNOWN defects are neutralised before anything is
persisted.

Why this exists (all reproduced against edgartools 5.44.1's
`edgar/proxy/html_extractor.py` on live filings):

- `_parse_percent` HARDCODES `return 0.5` for the "*" / "less than 1%" ownership
  footnote (html_extractor.py:958-960). Half of every ownership row we stored was
  this fabricated number -- Apple's CEO and its 12-person director group both came
  back as exactly 0.5%.
- `_detect_multiplier` misses "(in thousands)" on some fee tables, so KO's 2026
  proxy yielded an audit fee of 30,587 where its 2025 proxy yielded 32,104,000 for
  the SAME fee -- a silent 1000x break WITHIN one ticker.
- `AuditFees.current_year` / `.prior_year` are fiscal YEAR LABELS (2025, 2024), not
  fees. They were being written into columns named `audit_fee_*_year`.
- The Summary Compensation Table's Total column gets duplicated into a neighbouring
  component column (CAT: `pension_change == total` on every row, while the other
  components already sum to `total` exactly).
- Person cells arrive with the title glued on ("James DimonChairman and CEO"),
  footnote indices glued on ("Emma N. Walmsley11" -- which becomes "...10" the next
  year, so the NAME PRIMARY KEY does not survive year-over-year), or as an outright
  address ("100 Vanguard Blvd, Malvern, PA 19355" as a JPM holder_name).
- Subtotal rows ("Total", "... as a group (16 people)") are emitted as if they were
  holders, double-counting anything that aggregates the table.

Guiding rule: NEVER fabricate. A value is only written when it is deterministically
recoverable (the third leg of the pay-ratio identity, a single missing component in
an otherwise-reconciling comp row, a unit rescale a sibling filing confirms).
Anything else that fails a sanity check is set to NaN, because a NULL is honest and
a wrong number is not. Defects that are NOT recoverable here (edgartools returning
an empty table, or every component of a table NULL) stay empty by design -- those
remain the LLM path's job (`fetch_def14a_llm.py`).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.constants.constants import (
    DEF14A_AUDIT_FEE_MIN_PLAUSIBLE, DEF14A_COMP_RECONCILE_TOLERANCE, DEF14A_FISCAL_YEAR_MIN,
    DEF14A_NET_INCOME_MIN_PLAUSIBLE, DEF14A_PAY_RATIO_TOLERANCE, DEF14A_PLACEHOLDER_PERCENT,
)

__all__ = [
    "clean_text", "clean_person_name", "repair_main_row", "repair_exec_comp_rows",
    "repair_director_comp_rows", "repair_ownership_rows",
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

# A title modifier stranded at the END of the name because edgartools split the cell BETWEEN the
# modifier and the noun ("Bob De Lange Group" / "President"). _GLUED_TITLE_RE cannot see this one:
# by the time we get the row the modifier is already in a different column from its title.
_ORPHAN_MODIFIER_RE = re.compile(r"\s+(Former|Group|Interim|Acting)\s*$", re.I)

# Generational suffix and/or footnote left at the FRONT of a title when the name/title split landed
# mid-suffix ("D. James Umpleby" / "III(7) Chairman and CEO").
_TITLE_LEADING_JUNK_RE = re.compile(r"^(?:[IVX]+|Jr\.?|Sr\.?|\d+)?\s*(?:\(\d+\))?[\s,.-]*", re.I)

_EXEC_COMP_COMPONENTS = ["salary", "bonus", "stock_awards", "option_awards",
                         "non_equity_incentive", "pension_change", "other_compensation"]
_DIRECTOR_COMP_COMPONENTS = ["fees_earned", "stock_awards", "option_awards",
                             "non_equity_incentive", "pension_change", "other_compensation"]
_AUDIT_FEE_COLS = [
    "audit_fees_current", "audit_fees_prior", "audit_related_fees_current",
    "audit_related_fees_prior", "tax_fees_current", "tax_fees_prior",
    "other_fees_current", "other_fees_prior", "total_fees_current", "total_fees_prior",
]


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


def _clean_holder_name(value: Any) -> str | None:
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


def _rescale_block(row: dict, cols: list[str], min_plausible: float) -> None:
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


def _repair_pay_ratio(row: dict) -> None:
    """Complete or invalidate the CEO pay-ratio triplet IN PLACE, using the identity
    `ratio = ceo_comp / median_comp`. Any one missing leg is recoverable from the other two
    (GE discloses the ratio and the median but not the CEO figure); when all three are present
    but do not reconcile, the whole triplet is dropped rather than picking a winner."""
    ceo, med, ratio = (row.get("ceo_pay_ratio_ceo_comp"), row.get("ceo_pay_ratio_median_employee_comp"),
                       row.get("ceo_pay_ratio"))
    has = (_isnum(ceo) and ceo != 0, _isnum(med) and med != 0, _isnum(ratio) and ratio != 0)

    if all(has):
        if abs(float(ceo) / float(med) - float(ratio)) > float(ratio) * DEF14A_PAY_RATIO_TOLERANCE:
            row["ceo_pay_ratio_ceo_comp"] = _NAN
            row["ceo_pay_ratio_median_employee_comp"] = _NAN
            row["ceo_pay_ratio"] = _NAN
        return
    if has[1] and has[2] and not has[0]:
        row["ceo_pay_ratio_ceo_comp"] = float(med) * float(ratio)
    elif has[0] and has[2] and not has[1]:
        row["ceo_pay_ratio_median_employee_comp"] = float(ceo) / float(ratio)
    elif has[0] and has[1] and not has[2]:
        row["ceo_pay_ratio"] = float(ceo) / float(med)


def repair_main_row(row: dict) -> dict:
    """Repair one `def14a_edgar` row. Mutates and returns a COPY."""
    row = dict(row)
    for col in ("company_name", "peo_name", "company_selected_measure_name", "auditor_name"):
        row[col] = clean_text(row.get(col))

    # Fee-table year labels, not fees -- reject anything outside a sane window so a stray parse
    # cannot masquerade as a fiscal year.
    max_year = pd.Timestamp.today().year + 1
    for col in ("audit_fiscal_year_current", "audit_fiscal_year_prior"):
        val = row.get(col)
        row[col] = float(val) if _isnum(val) and DEF14A_FISCAL_YEAR_MIN <= float(val) <= max_year else _NAN

    _rescale_block(row, _AUDIT_FEE_COLS, DEF14A_AUDIT_FEE_MIN_PLAUSIBLE)

    # net_income cannot be rescaled the way the fee block can. PG's proxy yields 16.1 where every
    # other issuer yields whole dollars, but 16.1 is equally consistent with "$ in millions" and
    # "$ in billions" and NOTHING in the row disambiguates them -- unlike the fee block, whose
    # factor a sibling filing's overlapping year confirms. So an implausible figure is dropped
    # rather than guessed; the real value is available from `fundamentals_history` downstream.
    if _isnum(row.get("net_income")) and 0 < abs(float(row["net_income"])) < DEF14A_NET_INCOME_MIN_PLAUSIBLE:
        row["net_income"] = _NAN

    # A PEO is never paid exactly $0; that is a failed XBRL read, not a disclosure.
    for col in ("peo_total_comp", "peo_actually_paid_comp", "neo_avg_total_comp",
                "neo_avg_actually_paid_comp"):
        if _isnum(row.get(col)) and float(row[col]) == 0.0:
            row[col] = _NAN

    _repair_pay_ratio(row)
    return row


def _reconcile_components(row: dict, components: list[str]) -> dict:
    """Reconcile a compensation row against its reported `total` IN PLACE (returns a copy).

    Two deterministic repairs, in order:
    1. Duplicated-Total column: when a component equals `total` AND the remaining components
       already sum to `total`, the parser wrote the Total column into that component's slot too
       (CAT's `pension_change`). The true value is 0, so set it to 0.
    2. Single missing component: when exactly one component is NULL and the others fall short of
       `total`, the residual IS that component -- write it.

    A row that still does not reconcile is LEFT AS-IS: `total` is the number the filer actually
    printed and is the trustworthy field; spreading an unattributable residual across several NULL
    components would be a guess."""
    row = dict(row)
    total = row.get("total")
    if not _isnum(total):
        return row
    total = float(total)

    present = {c: float(row[c]) for c in components if _isnum(row.get(c))}
    if not present:
        return row

    for col, val in list(present.items()):
        others = sum(v for c, v in present.items() if c != col)
        if val == total and abs(others - total) <= DEF14A_COMP_RECONCILE_TOLERANCE:
            row[col] = 0.0
            present[col] = 0.0
            return row

    missing = [c for c in components if not _isnum(row.get(c))]
    if len(missing) == 1:
        residual = total - sum(present.values())
        if residual > DEF14A_COMP_RECONCILE_TOLERANCE:
            row[missing[0]] = residual
    return row


def repair_exec_comp_rows(rows: list[dict]) -> list[dict]:
    """Repair `def14a_edgar_executive_comp` rows: recover the title edgartools glued into the name,
    stabilise the name key, and reconcile components against `total`. Rows where EVERY component
    and the total are NULL (JPM -- names parsed, values all dropped) are removed: an all-NULL row
    carries nothing but still occupies a primary key, blocking a later good extraction."""
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        raw_name = clean_text(row.get("name")) or ""
        glued = _GLUED_TITLE_RE.search(raw_name)
        if glued and not clean_text(row.get("title")):
            title = clean_text(_FOOTNOTE_SUFFIX_RE.sub("", glued.group(0).strip()))
        else:
            title = clean_text(row.get("title"))
        name = clean_person_name(raw_name)
        if not name:
            continue

        # Re-attach a modifier edgartools stranded on the name ("Bob De Lange Group" + "President")
        # and drop a generational suffix it stranded on the title ("III(7) Chairman and CEO").
        orphan = _ORPHAN_MODIFIER_RE.search(name)
        if orphan and title:
            name = name[: orphan.start()].strip()
            title = f"{orphan.group(1)} {title}"
        if title:
            title = clean_text(_TITLE_LEADING_JUNK_RE.sub("", title)) or title
        row["name"], row["title"] = name, title
        if not row["name"]:
            continue
        if not any(_isnum(row.get(c)) for c in _EXEC_COMP_COMPONENTS + ["total"]):
            continue
        out.append(_reconcile_components(row, _EXEC_COMP_COMPONENTS))
    return out


def repair_director_comp_rows(rows: list[dict]) -> list[dict]:
    """Repair `def14a_edgar_director_comp` rows: stabilise the name key, drop subtotal pseudo-rows,
    and reconcile components against `total`."""
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        raw_name = clean_text(row.get("name")) or ""
        if _SUBTOTAL_HOLDER_RE.search(raw_name):
            continue
        row["name"] = clean_person_name(raw_name)
        if not row["name"]:
            continue
        if not any(_isnum(row.get(c)) for c in _DIRECTOR_COMP_COMPONENTS + ["total"]):
            continue
        out.append(_reconcile_components(row, _DIRECTOR_COMP_COMPONENTS))
    return out


def repair_ownership_rows(rows: list[dict], insider_names: set[str]) -> list[dict]:
    """Repair `def14a_edgar_ownership` rows.

    - Drops address-only holder names (JPM) and aggregate/subtotal rows (GE's "Total", the
      "... as a group (16 people)" line), which are not holders and double-count the table.
    - Strips the mailing address off institutional names (PG's Vanguard/BlackRock rows).
    - NULLs `percent_of_class` when it is edgartools' fabricated 0.5 placeholder for a "*" cell.
      The "*" footnote means "less than 1%", which is a BOUND, not a measurement -- storing 0.5
      asserts a precision the filing never gave and made half our ownership rows fiction.
    - Re-types a holder as `director_officer` when the name matches a known insider of the SAME
      filing (its PEO or anyone in its comp tables). edgartools tagged GE's and XOM's CEO as a
      `5pct_holder`; nobody appears in both roles, so the comp-table match settles it.
    """
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        raw_name = clean_text(row.get("holder_name")) or ""
        if _SUBTOTAL_HOLDER_RE.search(raw_name):
            continue
        holder = _clean_holder_name(raw_name)
        if not holder:
            continue
        row["holder_name"] = holder

        if _isnum(row.get("percent_of_class")) and float(row["percent_of_class"]) == DEF14A_PLACEHOLDER_PERCENT:
            row["percent_of_class"] = _NAN

        if row.get("holder_type") == "5pct_holder" and clean_person_name(holder) in insider_names:
            row["holder_type"] = "director_officer"
        out.append(row)
    return out
