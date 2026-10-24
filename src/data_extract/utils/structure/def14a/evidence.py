"""Conservative disclosure checks: headers alone never justify a paid repair."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data_extract.utils.structure.def14a.flatten import _CHILD_SPEC
from src.data_extract.utils.structure.def14a.retry import Section
from src.data_extract.utils.structure.def14a.validate import clean_holder_name, clean_person_name


def table_has_gap(section: Section, payload: str, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Match a disclosed numeric cell to its named row and output field."""
    # Only explicit numeric cells count; bounds, dashes and footnote symbols remain unknown.
    aliases = {
        "salary": r"salary",
        "bonus": r"bonus",
        "stock_awards": r"stock.*award",
        "option_awards": r"option.*award",
        "non_equity_incentive": r"non.?equity.*incentive",
        "pension_change": r"pension",
        "other_compensation": r"(?:all )?other.*compensation",
        "total": r"total",
        "fees_earned": r"fees.*(?:earned|cash)|cash.*fees",
        "shares": r"shares|amount.*beneficial|number.*beneficial",
        "percent_of_class": r"percent.*class|percent.*shares|%.*class",
        "percent_of_voting_power": r"voting.*(?:power|percent)|percent.*voting",
        "age": r"age",
        "tenure_years": r"director since|since",
    }
    allowed = set(_CHILD_SPEC[section.table.name][0]) if section.key != "audit" else set()
    fields: dict[int, str] = {}
    name_column, year_column = 0, None
    last_name = ""
    for line in payload.splitlines():
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2:
            continue
        if section.key == "audit":
            label = cells[0].lower()
            fee = next(
                (
                    field
                    for pattern, field in (
                        (r"^audit.related", "audit_fees_audit_related"),
                        (r"^audit fees", "audit_fees_audit"),
                        (r"^tax", "audit_fees_tax"),
                        (r"^(?:all )?other", "audit_fees_other"),
                        (r"^total", "auditor_fees"),
                    )
                    if re.search(pattern, label)
                ),
                None,
            )
            numbers = [cell for cell in cells[1:] if re.fullmatch(r"\$?-?\d[\d,]*(?:\.\d+)?", cell)]
            if fee and numbers and pd.isna(parent.get(fee)):
                return True
            if fee == "auditor_fees" and len(numbers) > 1 and pd.isna(parent.get("auditor_fees_prior")):
                return True
            continue
        header = {
            index: field
            for index, cell in enumerate(cells)
            for field, pattern in aliases.items()
            if field in allowed and re.search(pattern, cell, re.I)
        }
        if header:
            fields = header
            name_column = next((index for index, cell in enumerate(cells) if re.search(r"name|principal position", cell, re.I)), 0)
            year_column = next((index for index, cell in enumerate(cells) if re.fullmatch(r"(?:fiscal )?year", cell, re.I)), None)
            continue
        if not fields or name_column >= len(cells):
            continue

        # A real person's row may continue onto another fiscal year with a blank name.
        name = cells[name_column] or last_name
        clean = clean_holder_name(name) if section.key == "ownership" else clean_person_name(name)
        if not clean or not re.search(r"[A-Za-z].*\s+[A-Za-z]", clean) or clean.lower() in {"all directors", "name and principal position"}:
            continue
        last_name = name
        name_key = "holder_name" if section.key == "ownership" else "name"
        rows = saved.loc[saved[name_key].astype(str).str.casefold().eq(clean.casefold())] if name_key in saved else saved.iloc[:0]
        if section.key == "executive_pay":
            year = cells[year_column] if year_column is not None and year_column < len(cells) else ""
            if not re.fullmatch(r"(?:19|20)\d{2}", year):
                continue
            if "fiscal_year" in rows:
                rows = rows.loc[pd.to_numeric(rows.fiscal_year, errors="coerce").eq(int(year))]
        for index, field in fields.items():
            cell = cells[index].replace("%", "").strip() if index < len(cells) else ""
            if re.fullmatch(r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?", cell) and (rows.empty or field not in rows or rows[field].isna().all()):
                return True
    return False


def prose_has_gap(section: Section, payload: str, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Use explicit labelled values and named biographies; ignore optional silence."""
    if section.key == "audit":
        labels = {
            "audit_fees_audit": r"\baudit fees",
            "audit_fees_audit_related": r"\baudit.related fees",
            "audit_fees_tax": r"\btax fees",
            "audit_fees_other": r"\b(?:all )?other fees",
            "auditor_fees": r"\btotal (?:audit(?:or)? )?fees",
        }
        for field, label in labels.items():
            if pd.isna(parent.get(field)) and re.search(label + r"[^\n]{0,100}\$\s*\d[\d,.]*", payload, re.I):
                return True
        return pd.isna(parent.get("auditor_name")) and bool(
            re.search(r"\b(?:Deloitte|PricewaterhouseCoopers|Ernst\s*(?:&|and)\s*Young|KPMG|Grant Thornton|BDO)\b", payload, re.I)
        )
    if section.key == "ownership":
        group = re.search(r"(?:all\s+)?directors\s+and\s+(?:executive\s+)?officers\s+as\s+a\s+group[^\n]{0,100}", payload, re.I)
        if group:
            return (pd.isna(parent.get("insider_shares")) and bool(re.search(r"\b\d{1,3}(?:,\d{3})+\b", group.group()))) or (
                pd.isna(parent.get("insider_ownership_pct")) and bool(re.search(r"\d+(?:\.\d+)?\s*%", group.group()))
            )
        return False
    if section.key == "director_pay":
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[^\n]{0,80}?\$\s*\d[\d,.]*", payload):
            rows = saved.loc[saved.name.astype(str).str.casefold().eq(match[1].casefold())] if "name" in saved else saved.iloc[:0]
            if rows.empty or "fees_earned" not in rows or rows.fees_earned.isna().all():
                return True
        return False
    if section.key != "governance":
        return False

    # A named CEO disclosure is a key even when every director row is already complete.
    role = r"\b(?:Chief\s+Executive\s+Officer|CEO)\b"
    named_role = re.search(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+)){1,3}[^.\n]{0,100}" + role, payload)
    ceo = parent.get("ceo_name_proxy")
    if (not isinstance(ceo, str) or not ceo.strip()) and named_role:
        return True
    if isinstance(ceo, str) and pd.isna(parent.get("ceo_since_year")):
        bios = re.finditer(re.escape(ceo) + r"[^.\n]{0,250}", payload, re.I)
        if any(re.search(role + r"[^.\n]{0,60}\bsince\s+(?:19|20)\d{2}\b", bio.group(), re.I) for bio in bios):
            return True

    # A number must be tied to a named biography or an explicit board/CEO label.
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[, ]+\s*(?:age[d]?\s*:?\s*)(\d{2})\b", payload):
        rows = saved.loc[saved.name.astype(str).str.casefold().eq(match[1].casefold())] if "name" in saved else saved.iloc[:0]
        if rows.empty or "age" not in rows or rows.age.isna().all():
            return True
        ceo = parent.get("ceo_name_proxy")
        if isinstance(ceo, str) and ceo.casefold() == match[1].casefold() and pd.isna(parent.get("ceo_age")):
            return True
    labels = {
        "board_size": r"(?:board\s+(?:has|consists of|comprises)\s+\d+|\d+\s+(?:members|directors)\s+on\s+(?:the|our)\s+board)",
        "ceo_pay_ratio": r"(?:pay\s+ratio|ratio\s+of)[^\n]{0,100}\d+(?:\.\d+)?\s*(?::|to)\s*1",
        "say_on_pay_support_pct": r"(?:say.on.pay|compensation)[^\n]{0,100}\d+(?:\.\d+)?\s*%",
    }
    return any(pd.isna(parent.get(field)) and re.search(pattern, payload, re.I) for field, pattern in labels.items())
