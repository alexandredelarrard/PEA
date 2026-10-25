"""Verify narrow ownership/pay conflicts against the same filing's labelled cells."""

from __future__ import annotations

import math
import re

import pandas as pd

from src.data_extract.utils.structure.def14a.holders import _holder_name
from src.data_extract.utils.structure.def14a.retry import Section
from src.data_extract.utils.structure.def14a.validate import _leave_one_out_median


def _source_rows(source: str) -> list[tuple[str, int | None, dict[str, str]]]:
    """Read only tab-separated named source rows, retaining blank cells and explicit years."""
    result = []
    header: list[str] = []
    name_column, year_column = 0, None
    last_name = ""
    for line in source.splitlines():
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2:
            header = []
            continue
        names = [index for index, cell in enumerate(cells) if re.search(r"\bname\b", cell, re.I)]
        if names:
            header, name_column = cells, names[0]
            year_column = next((index for index, cell in enumerate(header) if re.fullmatch(r"(?:Fiscal\s+)?Year", cell, re.I)), None)
            last_name = ""
            continue
        if not header or len(cells) != len(header):
            continue
        name = re.sub(r"\s*\(\d+\)", "", cells[name_column]) or last_name
        last_name = name
        years = re.findall(r"\b(?:19|20)\d{2}\b", cells[year_column] if year_column is not None else " ".join(header))
        year = int(years[0]) if len(set(years)) == 1 else None
        result.append((name, year, dict(zip(header, cells, strict=True))))
    return result


def _source_values(cells: dict[str, str], ownership: bool) -> dict[str, float | None]:
    """Map explicit category labels; retain identified incompatible categories separately."""
    values: dict[str, float | None] = {}
    headers = " ".join(cells).lower()
    currencies = [value.strip().upper() for label, value in cells.items() if re.fullmatch(r"currency(?: code)?", label.strip(), re.I)]
    if not ownership and any(currency not in {"USD", "US DOLLARS", "U.S. DOLLARS"} for currency in currencies):
        return values
    if not ownership and not re.search(r"\$|\bUSD\b|U\.?S\.? dollars", headers + " " + " ".join(cells.values()), re.I):
        return values
    for label, raw in cells.items():
        label = label.lower()
        if not ownership and re.search(r"\b(?:EUR|GBP|CAD|AUD|CHF|JPY)\b|[€£¥]", label, re.I):
            continue
        field = None
        if ownership and re.search(r"percent|%|voting", label):
            continue
        quantity = re.search(r"#|\b(?:shares?|units?|securities|number|count)\b", label)
        monetary = re.search(r"\$|\bUSD\b|dollar|fair.?value", label, re.I)
        if not ownership and quantity and not monetary and re.search(r"stock|option|securities|sars", label):
            field = "_wrong_stock" if "stock" in label and "option" not in label else "_wrong_option"
        elif ownership:
            if re.search(r"additional underlying|cash.settled", label):
                field = "_extra"
            elif "total beneficial ownership" in label:
                field = "shares"
            elif "total" in label and re.search(r"additional underlying|cash.settled", headers):
                field = "_inclusive_total"
            elif (
                "shares" not in values
                and re.search(r"common stock|shares beneficially owned|amount and nature|^total$", label)
                and not re.search(r"percent|voting|sole|shared", label)
            ):
                field = "shares"
        else:
            patterns = {
                "_wrong_total": r"total annual|total all other",
                "_wrong_bonus": r"other annual",
                "_wrong_option": r"accounting expense|securities.*underlying|options?[/ ]+sars|number.*options",
                "salary": r"\bsalary\b",
                "bonus": r"\bbonus\b",
                "fees_earned": r"fees.*(?:earned|cash)",
                "stock_awards": r"stock.*award",
                "option_awards": r"option.*award",
                "non_equity_incentive": r"non.equity.*incentive",
                "pension_change": r"pension",
                "other_compensation": r"all other.*compensation",
                "total": r"\btotal\b",
            }
            field = next((field for field, pattern in patterns.items() if re.search(pattern, label)), None)
        if field is None:
            continue
        numeric = re.sub(r"\s+\(\d+\)$", "", raw).replace(",", "").replace("$", "").replace(" ", "")
        value = float(numeric.strip("()")) * (-1 if numeric.startswith("(") else 1) if re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", numeric) else None
        scale = re.search(r"\bin\s+(thousands|millions)\b", label, re.I)
        factor = {"thousands": 1000.0, "millions": 1000000.0}[scale[1].lower()] if scale else 1.0
        if field not in values or value is not None:
            values[field] = value * factor if value is not None else None
    return values


def _identity(value: object) -> str:
    """Use the same address cleanup as holder reconciliation, without losing suffixes."""
    return re.sub(r"[\W_]+", " ", _holder_name(str(value)).casefold()).strip()


def correct_basis(
    section: Section, merged: pd.DataFrame, new: pd.DataFrame, source: str, ceo_name: str | None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float | None]]]:
    """Correct only exact same-source conflicts; a median alone never changes an amount."""
    if not source or merged.empty or section.key not in ("executive_pay", "director_pay", "ownership"):
        return merged, {}
    result = merged.copy()
    ownership = section.key == "ownership"
    name_column = "holder_name" if ownership else "name"
    records = []
    for name, year, cells in _source_rows(source):
        if not ownership:
            name = re.split(r"\s+(?:Chairman|Chief (?:Executive|Financial|Operating)|President|CEO|CFO|COO)\b", name, maxsplit=1, flags=re.I)[0]
        if re.search(r"\bin\s+(?:thousands|millions)\b", source, re.I) and not re.search(r"\bin\s+(?:thousands|millions)\b", " ".join(cells), re.I):
            continue
        item = (_identity(name), year, _source_values(cells, ownership))
        if item not in records:
            records.append(item)
    groups = [
        values
        for name, _, values in records
        if " as a group" in name and values.get("shares") is not None and values.get("_inclusive_total") is not None
    ]
    parent: dict[str, tuple[float, float | None]] = (
        {"insider_shares": (float(groups[0]["_inclusive_total"]), float(groups[0]["shares"]))} if ownership and len(groups) == 1 else {}
    )
    ceo_rows = result.loc[result[name_column].map(_identity).eq(_identity(ceo_name))] if ceo_name else result.iloc[:0]
    latest = pd.to_numeric(ceo_rows.get("fiscal_year", pd.Series(dtype=float)), errors="coerce").max()

    # Literal source identity/year prevents cross-person and cross-period correction.
    for index, row in result.iterrows():
        identity = _identity(row[name_column])
        matches = [(name, year, values) for name, year, values in records if name == identity and (ownership or year == row.get("fiscal_year"))]
        if len(matches) != 1:
            continue
        name, year, values = matches[0]
        candidates = new.loc[new[name_column].map(_identity).eq(identity)] if name_column in new else new.iloc[:0]
        if not ownership and "fiscal_year" in candidates:
            candidates = candidates.loc[candidates.fiscal_year.eq(year)]
        candidate = candidates.iloc[0] if len(candidates) == 1 else pd.Series(dtype=object)
        for field, raw in values.items():
            if field.startswith("_") or field not in result or raw is None:
                continue
            value = raw
            proposed = candidate.get(field)
            if pd.notna(proposed) and math.isclose(float(proposed), value):
                result.at[index, field] = value
                continue
            if ownership and field == "shares" and pd.notna(row.get(field)) and row[field] == values.get("_inclusive_total"):
                result.at[index, field] = value
                continue

            # Both thousandfold directions need the source cell and other comparable years.
            peers = [
                other[field]
                for person, other_year, other in records
                if person == name and other_year != year and other.get(field) is not None and other[field] > 0
            ]
            median = _leave_one_out_median(pd.Series([float("nan"), *peers], dtype=float)).iloc[0] if peers else float("nan")
            actual = row.get(field)
            if ownership or pd.isna(actual) or actual <= 0 or pd.isna(median) or median <= 0:
                continue
            correction = 1000.0 if math.isclose(actual * 1000.0, value) else 0.001 if math.isclose(actual * 0.001, value) else None
            if correction is not None and value > 0 and abs(math.log(value / median)) < abs(math.log(actual / median)):
                result.at[index, field] = value

        # Clear only a value tied to an explicitly incompatible source column.
        wrong = {
            "bonus": "_wrong_bonus",
            "total": "_wrong_total",
            "option_awards": "_wrong_option",
            "stock_awards": "_wrong_stock",
            "non_equity_incentive": "bonus",
        }
        for field, incompatible in wrong.items():
            if not (
                field in result
                and values.get(field) is None
                and values.get(incompatible) is not None
                and pd.notna(row.get(field))
                and math.isclose(float(row[field]), float(values[incompatible]))
            ):
                continue
            result.at[index, field] = float("nan")
            if section.key == "executive_pay" and ceo_name and identity == _identity(ceo_name) and year == latest:
                scalar = "ceo_total_comp" if field == "total" else "ceo_" + field
                parent[scalar] = (float(row[field]), None)
    return result, parent
