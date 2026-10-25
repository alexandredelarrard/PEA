"""EDGAR ownership XML (Forms 3, 4, 5) -> the canonical insider string frames.

Field paths come from `insider_common.INSIDER_FIELDS`; typing is `build_insider_frame`. Transaction
rows carry a 1-based XML-order `row_sequence` per security table; every `reportingOwner` node gives
one owner row.
"""

from __future__ import annotations

from math import isfinite
from xml.etree import ElementTree

import pandas as pd

from src.data_extract.utils.institutionals.insider_common import FLAG_TRUE, FOOTNOTE_COLUMNS, INSIDER_FIELDS, OWNER_STRING_COLUMNS, InsiderField

#: (table element, row element, security_type) of the two transaction tables.
XML_TABLES = (
    ("nonDerivativeTable", "nonDerivativeTransaction", "nonderiv"),
    ("derivativeTable", "derivativeTransaction", "deriv"),
    ("nonDerivativeTable", "nonDerivativeHolding", "nonderiv_holding"),
    ("derivativeTable", "derivativeHolding", "deriv_holding"),
)
#: Relationship checkbox -> the role name the bulk data sets write in `RPTOWNER_RELATIONSHIP`.
ROLE_CHECKBOXES = (("isDirector", "Director"), ("isOfficer", "Officer"), ("isTenPercentOwner", "TenPercentOwner"), ("isOther", "Other"))
XML_STRING_COLUMNS = (
    "accession_number",
    "row_sequence",
    "security_type",
    *(field.name for field in INSIDER_FIELDS if field.xml and field.scope != "owner"),
    "footnote_ids",
)


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
    """Stripped text of child `name` (or of its `<value>`), None when absent or blank."""
    node = _child(parent, name)
    if node is None:
        return None
    value = _child(node, "value")
    target = value if value is not None else node
    text = "".join(target.itertext()).strip()
    return text or None


def _path_text(node: ElementTree.Element | None, path: str) -> str | None:
    """`_text` at a `parent/child/leaf` path below `node`."""
    *parents, leaf = path.split("/")
    for name in parents:
        node = _child(node, name)
    return _text(node, leaf)


def _relationship(relation: ElementTree.Element | None) -> str | None:
    """Checked role names joined by commas; "" when none is checked, None without a relationship block."""
    if relation is None:
        return None
    return ",".join(role for tag, role in ROLE_CHECKBOXES if (_text(relation, tag) or "").lower() in FLAG_TRUE)


def _field_text(node: ElementTree.Element | None, field: InsiderField) -> str | None:
    """One canonical string field: the first non-empty of its XML paths."""
    if field.kind == "role":
        return _relationship(_child(node, field.xml[0]))
    value = next((value for value in (_path_text(node, path) for path in field.xml) if value), None)
    if field.kind == "number":
        value = _number_text(value, field.name)
    if value is not None and field.kind == "date" and pd.isna(pd.to_datetime(value, errors="coerce")):
        raise ValueError(f"invalid ownership date field {field.name}")
    return value


def _number_text(value: str | None, name: str) -> str | None:
    if value is not None and not isfinite(float(value.replace(",", "").replace("$", ""))):
        raise ValueError(f"nonfinite ownership numeric field {name}")
    return value


def _scope_strings(node: ElementTree.Element | None, scope: str) -> dict[str, str | None]:
    return {field.name: _field_text(node, field) for field in INSIDER_FIELDS if field.scope == scope and field.xml}


def _row_strings(node: ElementTree.Element, security_type: str) -> dict[str, str | None]:
    fields = _scope_strings(node, "transaction")
    if security_type.endswith("_holding"):
        fields["total_value"] = _number_text(_path_text(node, "postTransactionAmounts/valueOwnedFollowingTransaction"), "holding_value")
    return fields


def _footnote_ids(node: ElementTree.Element) -> str:
    ids = [str(item.attrib.get("id", "")).strip() for item in node.iter() if _local_name(item.tag) == "footnoteId"]
    return ",".join(dict.fromkeys(item for item in ids if item))


def _footnotes(root: ElementTree.Element, accession_number: str) -> pd.DataFrame:
    """(accession_number, footnote_id, footnote_text) of every footnote carrying an id."""
    notes = []
    for note in _children(_child(root, "footnotes"), "footnote"):
        note_id = str(note.attrib.get("id", "")).strip()
        if not note_id or any(note["footnote_id"] == note_id for note in notes):
            raise ValueError("missing or repeated ownership footnote key")
        notes.append({"accession_number": accession_number, "footnote_id": note_id, "footnote_text": "".join(note.itertext()).strip()})
    return pd.DataFrame(notes, columns=FOOTNOTE_COLUMNS)


def _validate_xml_payload(root: ElementTree.Element, notes: pd.DataFrame) -> None:
    """Recognize every ownership row and bind every note reference before any empty marker."""
    for table_name in ("nonDerivativeTable", "derivativeTable"):
        allowed = {row for table, row, _ in XML_TABLES if table == table_name}
        table = _child(root, table_name)
        if table is not None and any(_local_name(node.tag) not in allowed for node in table):
            raise ValueError("unrecognized ownership payload row")
    available = set(notes["footnote_id"])
    references = [str(node.attrib.get("id", "")).strip() for node in root.iter() if _local_name(node.tag) == "footnoteId"]
    if any(not value or value not in available for value in references):
        raise ValueError("unresolved ownership footnote reference")


def _owner_strings(root: ElementTree.Element, accession_number: str) -> pd.DataFrame:
    """One owner string row per `reportingOwner` node, in document order."""
    rows = [{"accession_number": accession_number, **_scope_strings(owner, "owner")} for owner in _children(root, "reportingOwner")]
    return pd.DataFrame(rows, columns=OWNER_STRING_COLUMNS)


def extract_xml_strings(xml: str, accession_number: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership XML -> (transaction strings, owner strings, footnotes), all keyed on `accession_number`."""
    root = ElementTree.fromstring(xml)
    if _local_name(root.tag) != "ownershipDocument":
        raise ValueError("ownership XML has no ownershipDocument root")
    base = {"accession_number": accession_number, **_scope_strings(root, "filing")}
    if not str(base.get("issuer_cik") or "").isdigit():
        raise ValueError("ownership XML lacks its actual issuer CIK")
    notes = _footnotes(root, accession_number)
    _validate_xml_payload(root, notes)
    rows: list[dict[str, object]] = []
    for table_name, row_name, security_type in XML_TABLES:
        for sequence, node in enumerate(_children(_child(root, table_name), row_name), start=1):
            rows.append(
                {
                    **base,
                    "row_sequence": sequence,
                    "security_type": security_type,
                    **_row_strings(node, security_type),
                    "footnote_ids": _footnote_ids(node),
                }
            )
    if not rows:
        rows.append({**base, "row_sequence": 0, "security_type": "_empty"})
    return pd.DataFrame(rows, columns=XML_STRING_COLUMNS), _owner_strings(root, accession_number), notes
