"""
fetch_8k_edgar.py (src/data_extract/utils/institutionals/fetch_8k_edgar.py)
------------------------------------------------------------------------
SEC Form 8-K filings -> `sec_8k`, one row per (ticker, accession, item code).
Item codes come from the filing index; `has_earnings` / `has_press_release` and
the per-item text come from edgartools' typed `CurrentReport` (`sec_io.filing_obj`).
A transient SEC failure raises and fails the filing; a parse failure keeps a best-effort row.
Financial statements in attached earnings releases are out of scope.
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any

from src.constants.constants import SEC_8K_FORMS
from src.data_extract.utils.common.edgar_driver import EdgarFetch, FilingStamp, build_filing_rows
from src.data_extract.utils.common.sec_io import TransientReadError, filing_obj, filing_text
from src.data_extract.utils.structure.votes.guard import has_vote_table
from src.data_store.schema import Tables

_COLS = [
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "period_of_report",
    "n_items",
    "is_amendment",
    "has_earnings",
    "has_press_release",
    "primary_document",
    "item",
    "item_tag",
    "item_text",
]

# Curated leading distress/governance codes for the feature layer; any other code is
# tagged `other_unclassified_item` rather than dropped.
_HIGH_SIGNAL_ITEMS = {
    # 1: registrant's business and operations
    "1.01": "material_agreement_entered",
    "1.02": "material_agreement_terminated",
    "1.03": "bankruptcy_or_receivership",
    "1.04": "mine_safety_reporting",
    "1.05": "cybersecurity_incidents",
    # 2: financial information
    "2.01": "completion_acquisition_or_disposition",
    "2.02": "results_of_operations_and_financial_condition",
    "2.03": "creation_of_direct_financial_obligation",
    "2.04": "triggering_events_accelerating_financial_obligation",
    "2.05": "restructuring_costs",
    "2.06": "impairment",
    # 3: securities and trading markets
    "3.01": "delisting_or_covenant",
    "3.02": "unregistered_sales_of_equity",
    "3.03": "material_modification_to_security_rights",
    # 4: accountants and financial statements
    "4.01": "auditor_change",
    "4.02": "non_reliance_restatement",
    # 5: corporate governance and management
    "5.01": "change_in_control",
    "5.02": "exec_or_director_change",
    "5.03": "bylaw_change",
    "5.04": "employee_benefit_plan_trading_suspension",
    "5.05": "code_of_ethics_amendment_or_waiver",
    "5.06": "change_in_shell_company_status",
    "5.07": "vote_of_security_holders",
    "5.08": "shareholder_director_nominations",
    # 6: asset-backed securities
    "6.01": "abs_informational_computational_material",
    "6.02": "change_of_servicer_or_trustee",
    "6.03": "change_in_credit_enhancement",
    "6.04": "failure_to_make_required_distribution",
    "6.05": "securities_act_updating_disclosure",
    # 7: Regulation FD
    "7.01": "regulation_fd_disclosure",
    # 8: other events
    "8.01": "other_events",
    # 9: financial statements and exhibits
    "9.01": "financial_statements_and_exhibits",
}

_RESULTS_FOLLOW_RE = re.compile(
    r"(?is)\b(?:results?|votes?)\b.{0,160}\b(?:below|following|as follows|set forth)\b"
    r"|\b(?:below|following)\b.{0,160}\b(?:results?|votes?)\b"
)
_ITEM_507_HEADING_RE = re.compile(r"(?im)^\s*Item\s+5\.07\b[^\n]*")
_NEXT_8K_SECTION_RE = re.compile(r"(?im)^\s*(?:Item\s+(?!5\.07\b)\d\.\d{2}\b[^\n]*|SIGNATURES?)\s*$")


def _recover_item_507_from_primary(filing: Any, item_text: str) -> str:
    """Replace a table-less Item 5.07 slice that announces results "below" with the longer
    primary-document section, only when that section has a vote table; else keep it unchanged.
    """
    if not _RESULTS_FOLLOW_RE.search(item_text) or has_vote_table(item_text):
        return item_text
    try:
        primary_text = str(filing_text(filing) or "")
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- best-effort filing recovery
        return item_text
    for heading in _ITEM_507_HEADING_RE.finditer(primary_text):
        following = _NEXT_8K_SECTION_RE.search(primary_text, heading.end())
        candidate = primary_text[heading.start() : following.start() if following else len(primary_text)].strip()
        if len(candidate) > len(item_text) and has_vote_table(candidate):
            return candidate
    return item_text


def _filing_row(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One 8-K -> one row per item code. A failed parse keeps the item rows with both flags NaN
    (not None, so a cold table never infers the column as TEXT); a transient read raises."""
    filing = stamp.filing
    # Item codes come off the filing index, read before the parse so a code-less filing skips it.
    items = getattr(filing, "items", "") or ""
    item_list = [i.strip() for i in str(items).split(",") if i.strip()]
    if not item_list:
        return []

    has_earnings = has_press_release = float("nan")
    obj = None
    try:
        obj = filing_obj(filing)
        has_earnings = float(bool(obj.has_earnings))
        has_press_release = float(bool(obj.has_press_release))
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- best-effort only (a parse failure or a missing flag)
        pass

    base = {
        "ticker": ticker,
        # The filing registrant's CIK (not the roster's), so a registrant boundary stays visible.
        "cik": stamp.cik,
        "accession_number": stamp.accession_number,
        "form": stamp.form,
        "filing_date": stamp.filed,
        "period_of_report": stamp.period_of_report,
        "n_items": len(item_list),
        "is_amendment": float(stamp.is_amendment),
        "has_earnings": has_earnings,
        "has_press_release": has_press_release,
        "primary_document": stamp.primary_document,
    }

    rows = []
    for item_code in item_list:
        item_text = None
        if obj is not None:
            try:
                item_text = obj["Item " + item_code]
            except Exception:  # noqa: BLE001 -- best-effort only
                item_text = None
        if item_code == "5.07":
            item_text = _recover_item_507_from_primary(filing, str(item_text or ""))
        rows.append(
            {**base, "item": item_code, "item_tag": _HIGH_SIGNAL_ITEMS.get(item_code, "other_unclassified_item"), "item_text": item_text or ""}
        )
    return rows


#: `build_filing_rows` collapses a repeated item code ("5.02,5.02") into one PK row.
SEC_8K_FETCH = EdgarFetch(
    desc="8-K (edgartools)",
    tables=(Tables.sec_8k,),
    build=partial(build_filing_rows, forms=SEC_8K_FORMS, table=Tables.sec_8k, columns=_COLS, row_fn=_filing_row),
)
