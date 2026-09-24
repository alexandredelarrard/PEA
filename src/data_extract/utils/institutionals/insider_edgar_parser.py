"""Pure SEC ownership-XML parser for live Forms 3, 4, and 5.

Live filings and quarterly ZIP rows have the same economics but incompatible source row
identifiers. This adapter assigns an XML-order sequence and never invents a bulk
`transaction_sk`.
"""

from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pandas as pd

TRANSACTION_COLUMNS = (
    "source_row_sequence",
    "security_type",
    "issuer_cik",
    "issuer_name",
    "ticker",
    "owner_cik",
    "owner_name",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "period_of_report",
    "is_10b5_1",
    "security_title",
    "transaction_date",
    "transaction_code",
    "acquired_disposed",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "direct_indirect",
    "transaction_form_type",
    "equity_swap_involved",
    "deemed_execution_date",
    "nature_of_ownership",
    "transaction_timeliness",
    "exercise_price",
    "exercise_date",
    "expiration_date",
    "underlying_security_title",
    "underlying_shares",
    "underlying_value",
    "footnote_ids",
)
FOOTNOTE_COLUMNS = ("footnote_id", "footnote_text")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: ElementTree.Element | None, name: str) -> ElementTree.Element | None:
    if parent is None:
        return None
    return next((node for node in parent if _local_name(node.tag) == name), None)


def _children(parent: ElementTree.Element | None, name: str) -> list[ElementTree.Element]:
    if parent is None:
        return []
    return [node for node in parent if _local_name(node.tag) == name]


def _text(parent: ElementTree.Element | None, name: str) -> str | None:
    node = _child(parent, name)
    if node is None:
        return None
    value = _child(node, "value")
    target = value if value is not None else node
    text = "".join(target.itertext()).strip()
    return text or None


def _number(parent: ElementTree.Element | None, name: str) -> float:
    raw = _text(parent, name)
    if raw is None:
        return float("nan")
    try:
        return float(raw.replace(",", "").replace("$", ""))
    except ValueError:
        return float("nan")


def _date(parent: ElementTree.Element | None, name: str) -> pd.Timestamp:
    return pd.to_datetime(_text(parent, name), errors="coerce")


def _flag(parent: ElementTree.Element | None, name: str) -> float:
    raw = (_text(parent, name) or "").strip().lower()
    if raw in {"1", "true", "y", "yes"}:
        return 1.0
    if raw in {"0", "false", "n", "no"}:
        return 0.0
    return float("nan")


def _role_flag(parent: ElementTree.Element | None, name: str) -> float:
    """Ownership-role checkboxes are false when omitted from a valid relationship block."""
    value = _flag(parent, name)
    return 0.0 if pd.isna(value) and parent is not None else value


def _footnote_ids(node: ElementTree.Element) -> str:
    ids = [str(item.attrib.get("id", "")).strip() for item in node.iter() if _local_name(item.tag) == "footnoteId"]
    return ",".join(dict.fromkeys(item for item in ids if item))


def _owner(root: ElementTree.Element) -> dict[str, object]:
    """First reporting owner, matching the quarterly adapter's accession-level rule."""
    reporting_owner = next((node for node in root.iter() if _local_name(node.tag) == "reportingOwner"), None)
    owner_id = _child(reporting_owner, "reportingOwnerId")
    relation = _child(reporting_owner, "reportingOwnerRelationship")
    return {
        "owner_cik": _text(owner_id, "rptOwnerCik"),
        "owner_name": _text(owner_id, "rptOwnerName"),
        "is_director": _role_flag(relation, "isDirector"),
        "is_officer": _role_flag(relation, "isOfficer"),
        "is_ten_pct_owner": _role_flag(relation, "isTenPercentOwner"),
        "is_other": _role_flag(relation, "isOther"),
        "officer_title": _text(relation, "officerTitle"),
    }


def _base(root: ElementTree.Element) -> dict[str, object]:
    issuer = _child(root, "issuer")
    return {
        "issuer_cik": _text(issuer, "issuerCik"),
        "issuer_name": _text(issuer, "issuerName"),
        "ticker": (_text(issuer, "issuerTradingSymbol") or "").strip().upper(),
        "document_type": _text(root, "documentType"),
        "period_of_report": _date(root, "periodOfReport"),
        "is_10b5_1": _flag(root, "aff10b5One"),
        **_owner(root),
    }


def _transaction(
    node: ElementTree.Element,
    *,
    security_type: str,
    sequence: int,
    base: dict[str, object],
) -> dict[str, object]:
    coding = _child(node, "transactionCoding")
    amounts = _child(node, "transactionAmounts")
    post = _child(node, "postTransactionAmounts")
    ownership = _child(node, "ownershipNature")
    underlying = _child(node, "underlyingSecurity")
    shares = _number(amounts, "transactionShares")
    price = _number(amounts, "transactionPricePerShare")
    stated_value = _number(amounts, "transactionTotalValue")
    value = stated_value if np.isfinite(stated_value) else shares * price if np.isfinite(shares) and np.isfinite(price) else float("nan")
    return {
        **base,
        "source_row_sequence": sequence,
        "security_type": security_type,
        "security_title": _text(node, "securityTitle"),
        "transaction_date": _date(node, "transactionDate"),
        "transaction_code": _text(coding, "transactionCode"),
        "acquired_disposed": _text(amounts, "transactionAcquiredDisposedCode"),
        "shares": shares,
        "price_per_share": price,
        "value_usd": value,
        "shares_owned_after": _number(post, "sharesOwnedFollowingTransaction"),
        "direct_indirect": _text(ownership, "directOrIndirectOwnership"),
        "transaction_form_type": _text(coding, "transactionFormType"),
        "equity_swap_involved": _text(coding, "equitySwapInvolved"),
        "deemed_execution_date": _date(node, "deemedExecutionDate"),
        "nature_of_ownership": _text(ownership, "natureOfOwnership"),
        # SEC ownership XML places this beside `transactionCoding`; keep the fallback for
        # older/non-standard filings that nested it inside the coding block.
        "transaction_timeliness": (_text(node, "transactionTimeliness") or _text(coding, "transactionTimeliness")),
        "exercise_price": _number(node, "conversionOrExercisePrice"),
        "exercise_date": _date(node, "exerciseDate"),
        "expiration_date": _date(node, "expirationDate"),
        "underlying_security_title": _text(underlying, "underlyingSecurityTitle"),
        "underlying_shares": _number(underlying, "underlyingSecurityShares"),
        "underlying_value": _number(underlying, "underlyingSecurityValue"),
        "footnote_ids": _footnote_ids(node),
    }


def parse_ownership_xml(xml: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized transactions and accession-local footnotes from ownership XML."""
    root = ElementTree.fromstring(xml)
    if _local_name(root.tag) != "ownershipDocument":
        raise ValueError("ownership XML has no ownershipDocument root")

    base = _base(root)
    rows: list[dict[str, object]] = []
    specs = (
        ("nonDerivativeTable", "nonDerivativeTransaction", "nonderiv"),
        ("derivativeTable", "derivativeTransaction", "deriv"),
    )
    for table_name, row_name, security_type in specs:
        table = _child(root, table_name)
        for sequence, node in enumerate(_children(table, row_name), start=1):
            rows.append(
                _transaction(
                    node,
                    security_type=security_type,
                    sequence=sequence,
                    base=base,
                )
            )

    notes = []
    for note in _children(_child(root, "footnotes"), "footnote"):
        note_id = str(note.attrib.get("id", "")).strip()
        if note_id:
            notes.append(
                {
                    "footnote_id": note_id,
                    "footnote_text": "".join(note.itertext()).strip(),
                }
            )
    return (
        pd.DataFrame(rows, columns=TRANSACTION_COLUMNS),
        pd.DataFrame(notes, columns=FOOTNOTE_COLUMNS),
    )
