"""
fundamentals_employees.py
--------------------------
Employee headcount as an ordinary `fundamentals_facts` field, parsed out of the
10-K BODY TEXT. Replaces the standalone `structure/fetch_employees_edgar.py`
fetcher (and its dedicated `employees_history` table): headcount is disclosed in
the SAME 10-K the fundamentals walk already opens, so there is no reason to list,
download and date those filings a second time in a separate pass.

Why text and not a tag: there is NO GAAP concept for headcount (`dei:Entity-
NumberOfEmployees` exists but US filers essentially never tag it), so the
`field: [candidate tags]` machinery in `fundamentals_tags.py` has nothing to
resolve. `employees` is therefore declared there with an EMPTY candidate list --
`build_tag_frames` skips it entirely, while every DOWNSTREAM consumer
(`_assemble_base`'s latest-value ffill, `_derive_history`'s output column,
`fundamentals_derive`'s per-field series) picks it up from the same dict as any
other field. This module only produces the FACT ROW; nothing else changes.

Everything that made the old fetcher accurate is preserved, only re-shaped:
  * the in-document scoring parser (`edgar_extract.extract_employee_count`),
    unchanged and still unit-tested on its own;
  * the per-ticker CONTINUITY guard (`is_continuous`), the last line of defence
    against a parse artifact that survives every in-document heuristic.

What is DROPPED as redundant: the filing listing, the per-ticker `as_of` cutoff,
the meta sidecar and the annual-cadence gate. `fetch_fundamentals_edgar` already
skips accessions present in `fundamentals_facts`, so a re-run never re-opens a
10-K it has parsed -- the same incremental property, from one mechanism instead
of two.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.constants.constants import HEADCOUNT_CONTINUITY_MAX, HEADCOUNT_CONTINUITY_MIN
from src.data_extract.utils.common.edgar_extract import extract_employee_count, html_to_text
from src.data_extract.utils.fundamentals.fundamentals_tags import EMPLOYEES_FIELD

# Only the ANNUAL report states a headcount ("As of December 31, we had
# approximately N employees"); a 10-Q never does. The amendment is included
# because a 10-K/A that restates the year restates the workforce disclosure
# with it -- it produces its own row (is_amendment=1.0) and
# `fundamentals_derive._resolve_latest_per_period` prefers it from its own
# filing date onward, exactly like a restated financial figure.
HEADCOUNT_FORMS: tuple[str, ...] = ("10-K", "10-K/A")
# Not a real XBRL unit (headcount is dimensionless) -- a marker that makes the
# scale of `value` obvious to anyone reading `fundamentals_facts` directly, where
# every neighbouring row is USD or shares.
EMPLOYEES_UNIT = "employees"
# Provenance in place of an XBRL concept name: `source_tag` is what every other
# row uses to record WHERE its number came from, and "this was parsed out of the
# filing's prose" is exactly the kind of thing a later audit needs to see.
EMPLOYEES_SOURCE_TAG = "text:10-K"


def is_headcount_form(form: str | None) -> bool:
    """Does this form type carry a workforce disclosure to parse? Keeps the
    (expensive) body-text download off the ~75% of filings that are 10-Qs."""
    return str(form or "").upper() in HEADCOUNT_FORMS


def is_continuous(count: int, history: list[int]) -> bool:
    """Is `count` continuous with a ticker's own headcount history?

    Headcount is a SLOW-MOVING series: a real company does not multiply or divide its
    workforce by five between two annual filings, so this catches the text-extraction
    misses that survive every in-document heuristic. The 2026-07 audit measured 6.3% of
    year-over-year transitions at >2x or <0.5x, and the verification run caught CSGP
    picking up "2.3 million" (2,300,000) against a stored 1,155.

    Anchored on the MEDIAN of the accepted history, not the previous value: a median
    cannot be dragged by one bad reading, so a single wrong row does not then reject the
    correct ones after it (WRB's 4,502,942 would have done exactly that). A ticker's
    first filing has no anchor and is always accepted.
    """
    if not history:
        return True
    anchor = float(sorted(history)[len(history) // 2])
    if anchor <= 0:
        return True
    return HEADCOUNT_CONTINUITY_MIN <= count / anchor <= HEADCOUNT_CONTINUITY_MAX


def history_by_ticker(facts: pd.DataFrame | None) -> dict[str, list[int]]:
    """Already-accepted headcounts per ticker, in FILING-DATE order -- the anchor
    `is_continuous` compares a newly-parsed value against.

    Seeded from the `employees` rows ALREADY in `fundamentals_facts` (columns
    ticker / filing_date / value), so an incremental run -- which by construction
    re-parses only the one new 10-K -- guards it against the ticker's whole
    stored history rather than against an empty list. Sorted by filing date so
    the median reflects the series, not the row order the DB happened to return.
    """
    required = {"ticker", "filing_date", "value"}
    if facts is None or facts.empty or not required.issubset(facts.columns):
        return {}
    s = facts[["ticker", "filing_date", "value"]].copy()
    s["filing_date"] = pd.to_datetime(s["filing_date"], errors="coerce")
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    s = s.dropna(subset=["filing_date", "value"]).sort_values("filing_date")
    return {t: g["value"].astype(int).tolist() for t, g in s.groupby("ticker")}


def filing_body_text(filing) -> str:
    """The filing's primary document as plain text.

    Routed through this repo's own `html_to_text` rather than edgartools'
    renderer so the parser sees BYTE-FOR-BYTE what the retired `sec_get(doc_url)`
    path fed it -- every regex, score and threshold in `edgar_extract` was tuned
    against that exact flattening, and its audit cases (AMZN/MCD/XOM/KO/CF/C/AES)
    are only reproducible on it. `filing.text()` is the fallback for the rare
    filing edgartools exposes no HTML for (pre-2001 plain `.txt` submissions).
    """
    raw = filing.html()
    if raw:
        return html_to_text(raw)
    return filing.text() or ""


def employee_fact_frame(
    filing,
    history: list[int] | None = None,
    log: logging.Logger | None = None,
) -> pd.DataFrame | None:
    """One 10-K's headcount as a single `build_tag_frames`-shaped row, or None
    when the filing is not an annual report, states no trustworthy headcount, or
    reports one that is discontinuous with `history`.

    Shaped as an INSTANT fact (`period_type='instant'`, no `period_start`): a
    headcount is a point-in-time level stated as of the fiscal year end, exactly
    like a balance-sheet line, and the missing `period_start` is what tells
    `fundamentals_periods.instant_stock` to normalize this filing's native 'FY'
    label to 'Q4' (the year-end snapshot) instead of treating it as a duration
    measure that legitimately has both an FY and a Q4 flavour.

    `fiscal_year`/`fiscal_period` are deliberately left EMPTY: the caller re-runs
    `backfill_fiscal_period_from_filing` after appending this row, which borrows
    them from the tagged duration facts of the SAME filing -- the identical route
    every genuine instant fact takes, since instant facts carry no native fy/fp
    either.
    """
    if not is_headcount_form(getattr(filing, "form", None)):
        return None
    period_of_report = getattr(filing, "period_of_report", None)
    if not period_of_report:
        return None

    try:
        count = extract_employee_count(filing_body_text(filing))
    except Exception as e:                                  # noqa: BLE001
        if log:
            log.warning("employees: %s %s text fetch/parse failed (%s)",
                        getattr(filing, "accession_number", "?"), filing.form, e)
        return None
    if count is None:
        return None

    accepted = list(history or [])
    if not is_continuous(count, accepted):
        if log:
            log.warning(
                "employees: %s %s headcount %d is discontinuous with its own history "
                "(median %d) -- dropped as a parse artifact",
                getattr(filing, "accession_number", "?"), filing.form, count,
                sorted(accepted)[len(accepted) // 2])
        return None

    return pd.DataFrame([{
        "field": EMPLOYEES_FIELD,
        "value": float(count),
        "unit": EMPLOYEES_UNIT,
        "period_start": pd.NaT,
        "period_end": pd.Timestamp(period_of_report).normalize(),
        "period_type": "instant",
        "fiscal_year": None,
        "fiscal_period": None,
        "source_tag": EMPLOYEES_SOURCE_TAG,
    }])
