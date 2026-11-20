"""SEC 8-K rows at (ticker, accession, item), using submissions metadata and CurrentReport text.
Bounded Item 5.07 bodies and explicitly referenced voting exhibits supplement the structured slices.
Unresolved metadata and transient reads remain pending; parse failures retain best-effort item rows.
Resolved filings with no reported items become empty markers; attached earnings statements are out of scope.
"""

from __future__ import annotations

import logging
import re
import warnings
from functools import partial
from typing import Any, cast

import pandas as pd

from src.data_extract.utils.common import sec_io
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_obj, filing_text
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, FilingStamp, parse_filing_rows
from src.data_extract.utils.edgar_extract.html_text import html_to_text
from src.data_extract.utils.edgar_extract.item_carve import carve_spans
from src.data_extract.utils.structure.votes.guard import has_vote_table
from src.data_store.schema import Resume, Tables

logger = logging.getLogger(__name__)

# An item code listed in the filing index but absent from the parsed body is routine (items are
# optional per filer); `_item_text` already maps that None to an empty text. Process-wide filter,
# because `warnings.catch_warnings` is not thread-safe and the fetch runs threaded.
warnings.filterwarnings("ignore", message=r"CurrentReport\[.*found no such item in this filing", category=FutureWarning)


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

_ITEM_507_HEADING_RE = re.compile(r"(?im)^[ \t]*(?:i\s*tem[.\s]*5\.07\b[.:\t ]*|5\.07[ \t]*[-–—][ \t]*)")
_NEXT_8K_SECTION_RE = re.compile(r"(?im)^[ \t]*(?:i\s*tem[.\s]*\d\.\d{2}\b|\d\.\d{2}[ \t]*[-–—]|SIGNATURES?\b)")


def _item_text(obj: Any, item_code: str) -> Any:
    """The parsed report's text for one item code; None without a parsed report or when the lookup fails."""
    if obj is None:
        return None
    try:
        return obj["Item " + item_code]
    except Exception:  # noqa: BLE001 -- best-effort only
        return None


def _filing_row(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One row per reported item, supplementing voting slices from the primary and referenced exhibits.
    Failed CurrentReport parsing retains item rows with NaN flags; failed source reads remain pending."""
    # Read the primary once; a blank structured slice or incomplete metadata cannot prove no voting results.
    filing = stamp.filing
    if not hasattr(filing, "items"):
        raise ParseFailureError(f"{stamp.accession_number}: 8-K item metadata was not resolved")
    primary_text = str(filing_text(filing) or "")
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

    # Keep internal proposal headings inside the bounded Item 5.07 body.
    item_list = [item.strip() for item in str(filing.items or "").split(",") if item.strip()]
    item_texts = {item: _item_text(obj, item) for item in item_list}
    sections = [primary_text[start:end].strip() for start, end in carve_spans(primary_text, _ITEM_507_HEADING_RE, (_NEXT_8K_SECTION_RE,))]
    tables = [section for section in sections if has_vote_table(section)]
    primary_vote = max(tables or sections, key=len, default="")
    vote_text = primary_vote if tables else str(item_texts.get("5.07") or primary_vote)

    # Follow explicit EX-99 voting supplements, excluding plan and charter documents.
    references = {f"EX-{number}" for number in re.findall(r"\bexhibits?\s+(99\.\d+)\b", primary_vote or vote_text, re.I)}
    attachments = sec_io.filing_attachments(filing) if references else ()
    exhibits = []
    for attachment in attachments or ():
        kind = str(attachment.document_type or "").strip().upper()
        if kind not in references or attachment.is_binary():
            continue
        raw = sec_io.sec_call(getattr, attachment, "content", label=f"{stamp.accession_number} {kind}")
        text = html_to_text(raw or "")
        if has_vote_table(text):
            exhibits.append(f"Exhibit {kind.removeprefix('EX-')} ({attachment.document})\n{text}")
    if exhibits:
        vote_text = "\n\n".join([primary_vote or vote_text, *exhibits])
    if primary_vote and "5.07" not in item_list:
        item_list.append("5.07")
    item_texts["5.07"] = vote_text

    # Preserve the filing's identifiers, publication time and flags for every item row.
    base = {
        "ticker": ticker,
        # The filing registrant's CIK (not the roster's), so a registrant boundary stays visible; the raw filing time.
        **stamp.columns(Tables.sec_8k, normalise=False),
        "n_items": len(item_list),
        "is_amendment": float(stamp.is_amendment),
        "has_earnings": has_earnings,
        "has_press_release": has_press_release,
        "primary_document": stamp.primary_document,
    }

    return [
        {**base, "item": item, "item_tag": _HIGH_SIGNAL_ITEMS.get(item, "other_unclassified_item"), "item_text": item_texts[item] or ""}
        for item in item_list
    ]


def submission_filings(ticker: str, df_units: pd.DataFrame) -> dict[str, Any]:
    """The listed 8-Ks as edgartools `EntityFiling`s (item codes, primary document), by accession.

    One `Company` listing per CIK holding listed units; an unresolvable CIK is warned and its units
    fall back to the plain index filing, whose missing item metadata keeps it pending. A transient SEC failure raises."""
    wanted = set(df_units["accession"].astype(str))
    out: dict[str, Any] = {}
    for cik in df_units["cik"].astype(str).unique():
        try:
            company = sec_io.company(int(cik))
        except TransientReadError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a dead CIK, not a bug
            logger.warning("8-K: %s CIK %s could not be resolved (%s)", ticker, cik, exc)
            continue
        out.update(
            {
                str(f.accession_number): f
                for f in sec_io.company_filings(company, cast(Resume, Tables.sec_8k.resume).forms)
                if str(f.accession_number) in wanted
            }
        )
    if missing := sorted(wanted - out.keys()):
        logger.warning(
            "8-K: %s %d indexed filing(s) absent from the submissions listing, kept pending: %s",
            ticker,
            len(missing),
            ", ".join(missing[:20]),
        )
    return out


#: `parse_filing_rows` collapses a repeated item code ("5.02,5.02") into one PK row.
SEC_8K_FETCH = EdgarFetch(
    tables=(Tables.sec_8k,),
    parse=partial(parse_filing_rows, table=Tables.sec_8k, row_fn=_filing_row),
    filings=submission_filings,
)
