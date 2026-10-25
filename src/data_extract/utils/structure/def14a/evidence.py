"""Conservative disclosure checks: headers alone never justify a paid repair."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data_extract.utils.structure.def14a.flatten import _CHILD_SPEC
from src.data_extract.utils.structure.def14a.retry import Section
from src.data_extract.utils.structure.def14a.validate import clean_holder_name, clean_person_name


def table_has_gap(section: Section, payload: str, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Match explicit numeric or independence cells to named rows and output fields."""
    # Literal Yes/No and numeric cells count; bounds, dashes and footnotes remain unknown.
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
        "is_independent": r"\bindependen(?:ce|t)\b",
    }
    allowed = set(_CHILD_SPEC[section.table.name][0]) if section.key != "audit" else set()
    fields: dict[int, str] = {}
    name_column, year_columns = 0, []
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
            if field in allowed
            and re.search(pattern, cell, re.I)
            and not (field == "option_awards" and re.search(r"#|\bnumber\b|\bsecurities\b|\bshares\b", cell, re.I))
        }
        if header:
            fields = header
            name_column = next((index for index, cell in enumerate(cells) if re.search(r"name|principal position", cell, re.I)), 0)
            year_columns = [index for index, cell in enumerate(cells) if re.search(r"\byear$", cell, re.I)]
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
            year = next((cells[index] for index in year_columns if index < len(cells) and cells[index]), "")
            label = re.fullmatch(r"((?:19|20)\d{2})(?:-(\d{2}|\d{4}))?", year)
            if not label or (label[2] and not str(int(label[1]) + 1).endswith(label[2])):
                continue
            fiscal_year = int(label[1]) + bool(label[2])
            if "fiscal_year" in rows:
                rows = rows.loc[pd.to_numeric(rows.fiscal_year, errors="coerce").eq(fiscal_year)]
        for index, field in fields.items():
            cell = cells[index].strip() if index < len(cells) else ""
            cell = cell if field == "is_independent" else cell.replace("%", "")
            pattern = r"yes|no" if field == "is_independent" else r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?"
            if re.fullmatch(pattern, cell, re.I) and (rows.empty or field not in rows or rows[field].isna().all()):
                return True
    return False


def layout_has_gap(section: Section, payload: str, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Recognize explicit legacy salary rows and a year-labelled transposed fee matrix."""
    # A transposed fee matrix needs its complete category order, dollar basis and adjacent years.
    if section.key == "audit":
        header = re.search(
            r"\(IN\s+(?:MILLIONS|THOUSANDS|DOLLARS)\)\s+AUDIT(?: FEES)?\s+AUDIT\s*-\s*RELATED(?: FEES)?\s+"
            r"TAX(?: FEES)?\s+ALL\s+OTHER(?: FEES)?\s+TOTAL(?: FEES)?\s+((?:19|20)\d{2})\s*\n",
            payload,
            re.I,
        )
        if not header:
            return False
        cells = [line.strip() for line in payload[header.end() :].splitlines() if line.strip()]
        if len(cells) < 11 or cells[5] != str(int(header[1]) - 1):
            return False
        amounts = cells[:5] + cells[6:11]
        if not all(re.fullmatch(r"\$-?\d[\d,]*(?:\.\d+)?|[-—]+|N/?A", cell, re.I) for cell in amounts):
            return False
        fields = ("audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax", "audit_fees_other", "auditor_fees", "auditor_fees_prior")
        return any(pd.isna(parent.get(field)) and cell.startswith("$") for field, cell in zip(fields, cells[:5] + [cells[10]], strict=True))
    if section.key != "executive_pay":
        return False

    # Plain SCT recognition is limited to the explicitly first monetary column, salary.
    for block in re.split(r"(?im)^.*summary compensation table[^\n]*\n", payload)[1:]:
        header = re.search(r"(?im)^[^\n\t]*\byear\s+salary\b[^\n]*$|^[^\n\t]*\bsalary\b[^\n]*\nPOSITION YEAR \(\$\)[^\n]*$", block)
        if not header:
            continue
        for line in block[header.end() :].splitlines():
            if re.fullmatch(r"(?:\s*\(\d+\))+\s*", line):
                continue
            if re.fullmatch(r"[A-Z][A-Z /&-]*", line.strip()) or re.match(r"\s*(?:NAME|PRINCIPAL POSITION)\b", line, re.I):
                break
            if re.match(r"\s*(?:NOTES?\b|\(?\d+\)|(?:STOCK )?OPTIONS?\b|ALL OTHER COMPENSATION)", line, re.I):
                break
            row = re.fullmatch(
                r"([A-Za-z][A-Za-z .,'-]+?)\s+((?:19|20)\d{2})\s+(\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?)"
                r"(?:\(\d+\))?\s+(?:\$?\s*-?\d|--?|N/A).*",
                line.strip(),
            )
            if not row:
                continue
            name = clean_person_name(re.sub(r"\.{2,}", "", row[1]))
            if not name or len(re.findall(r"[A-Za-z]+", name)) < 2:
                continue
            rows = saved.loc[saved.name.astype(str).str.casefold().eq(name.casefold())] if "name" in saved else saved.iloc[:0]
            if "fiscal_year" in rows:
                rows = rows.loc[pd.to_numeric(rows.fiscal_year, errors="coerce").eq(int(row[2]))]
            if rows.empty or "salary" not in rows or rows.salary.isna().all():
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

    # A named explicit independence statement supports either True or False.
    for match in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\s+(?:not\s+)?(?:an?\s+)?independent(?:\s+director)?\s*(?:[.;]|$)", payload
    ):
        rows = saved.loc[saved.name.astype(str).str.casefold().eq(match[1].casefold())] if "name" in saved else saved.iloc[:0]
        if rows.empty or "is_independent" not in rows or rows.is_independent.isna().all():
            return True

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
