"""EDGAR ownership XML (Forms 3, 4, 5) -> the canonical insider string frames.

Field paths come from `insider_common.INSIDER_FIELDS`; typing is `build_insider_frame`. Transaction
rows carry a 1-based XML-order `row_sequence` per security table; every `reportingOwner` node gives
one owner row.
"""

from __future__ import annotations

from xml.etree import ElementTree

import pandas as pd

from src.data_extract.utils.institutionals.insider_common import FLAG_TRUE, FOOTNOTE_COLUMNS, INSIDER_FIELDS, OWNER_STRING_COLUMNS, InsiderField

#: (table element, row element, security_type) of the two transaction tables.
XML_TABLES = (
    ("nonDerivativeTable", "nonDerivativeTransaction", "nonderiv"),
    ("derivativeTable", "derivativeTransaction", "deriv"),
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
    return next((value for value in (_path_text(node, path) for path in field.xml) if value), None)


def _scope_strings(node: ElementTree.Element | None, scope: str) -> dict[str, str | None]:
    return {field.name: _field_text(node, field) for field in INSIDER_FIELDS if field.scope == scope and field.xml}


def _footnote_ids(node: ElementTree.Element) -> str:
    ids = [str(item.attrib.get("id", "")).strip() for item in node.iter() if _local_name(item.tag) == "footnoteId"]
    return ",".join(dict.fromkeys(item for item in ids if item))


def _footnotes(root: ElementTree.Element, accession_number: str) -> pd.DataFrame:
    """(accession_number, footnote_id, footnote_text) of every footnote carrying an id."""
    notes = []
    for note in _children(_child(root, "footnotes"), "footnote"):
        note_id = str(note.attrib.get("id", "")).strip()
        if note_id:
            notes.append({"accession_number": accession_number, "footnote_id": note_id, "footnote_text": "".join(note.itertext()).strip()})
    return pd.DataFrame(notes, columns=FOOTNOTE_COLUMNS)


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
    rows: list[dict[str, object]] = []
    for table_name, row_name, security_type in XML_TABLES:
        for sequence, node in enumerate(_children(_child(root, table_name), row_name), start=1):
            rows.append(
                {
                    **base,
                    "row_sequence": sequence,
                    "security_type": security_type,
                    **_scope_strings(node, "transaction"),
                    "footnote_ids": _footnote_ids(node),
                }
            )
    return pd.DataFrame(rows, columns=XML_STRING_COLUMNS), _owner_strings(root, accession_number), _footnotes(root, accession_number)
