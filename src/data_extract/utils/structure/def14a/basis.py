"""Verify narrow ownership/pay conflicts against the same filing's labelled cells."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from src.data_extract.utils.structure.def14a.holders import _holder_name, normalize_name
from src.data_extract.utils.structure.def14a.retry import Section
from src.data_extract.utils.structure.def14a.validate import _leave_one_out_median


def _source_rows(source: str) -> list[tuple[str, int | None, dict[str, str]]]:
    """Read only tab-separated named source rows, retaining blank cells and explicit years."""
    result = []
    header: list[str] = []
    previous_header: list[str] = []
    name_column, year_column = 0, None
    last_name = ""
    caption_year, table_year = None, None
    for line in source.splitlines():
        caption_years = set(re.findall(r"director\s+compensation\s+table\s+for\s+((?:19|20)\d{2})", line, re.I))
        local_fiscal = (
            r"following table(?: and footnotes)?.{0,180}?compensation.{0,180}?Directors?\s+in\s+FY\s+((?:19|20)\d{2}(?:-(?:\d{2}|\d{4}))?)(?![-\d])"
        )
        for fiscal in re.findall(local_fiscal, line, re.I):
            label = re.fullmatch(r"((?:19|20)\d{2})(?:-(\d{2}|\d{4}))?", fiscal)
            if label and (not label[2] or str(int(label[1]) + 1).endswith(label[2])):
                caption_years.add(str(int(label[1]) + bool(label[2])))
        if caption_years:
            caption_year = int(next(iter(caption_years))) if len(caption_years) == 1 else None
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2:
            header = []
            previous_header = []
            if line.strip() and not caption_years and not line.strip().startswith("Source table (tab-separated columns):"):
                caption_year = None
            continue
        names = [index for index, cell in enumerate(cells) if re.search(r"\bname\b", cell, re.I)]
        if names:
            header, name_column = cells, names[0]
            missing_names = len(cells) - len(previous_header)
            if (
                missing_names in (1, 2)
                and names == [0]
                and not any(cells[1:missing_names])
                and all(re.search(r"fees|awards|compensation|pension|total", label, re.I) for label in previous_header)
            ):
                header = [
                    *cells[:missing_names],
                    *(f"{label} {unit}".strip() for label, unit in zip(previous_header, cells[missing_names:], strict=True)),
                ]
            previous_header = []
            year_column = next((index for index, cell in enumerate(header) if re.fullmatch(r"(?:Fiscal\s+)?Year", cell, re.I)), None)
            table_year = caption_year if any(re.search(r"fees.*(?:earned|cash)", cell, re.I) for cell in header) else None
            caption_year = None
            last_name = ""
            continue
        if not header:
            previous_header = cells
            continue
        if len(cells) != len(header):
            continue
        name = re.sub(r"\s*\(\d+\)", "", cells[name_column]) or last_name
        last_name = name
        years = re.findall(r"\b(?:19|20)\d{2}\b", cells[year_column] if year_column is not None else " ".join(header))
        if year_column is not None and len(years) > 1:
            vectors = [cell.split() if cell else [""] * len(years) for cell in cells]
            vectors[name_column] = [name] * len(years)
            if len(set(years)) != len(years) or any(len(vector) != len(years) for vector in vectors):
                continue
            result.extend(
                (name, int(year), dict(zip(header, [vector[index] for vector in vectors], strict=True))) for index, year in enumerate(years)
            )
            continue
        year = int(years[0]) if len(set(years)) == 1 else table_year if not years and year_column is None else None
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
        if not ownership and quantity and not monetary and re.search(r"stock|option|securities|sars?", label):
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
                "option_awards": r"option.*award" + (r"|\bsars?\b.*award" if monetary else ""),
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
        if field == "stock_awards" and raw.strip() in {"", "-", "\u2013", "\u2014"}:
            values["_blank_stock"] = 1.0
        if field == "option_awards" and monetary and re.search(r"\bsars?\b", label) and not re.search(r"\boption", headers):
            values["_sar_awards"] = values[field]
        if field == "option_awards" and monetary and not re.search(r"#|\b(?:number|count|units|securities)\b", label):
            values["_option_money"] = values[field]
    return values


def _identity(value: object) -> str:
    """Use the same address cleanup as holder reconciliation, without losing suffixes."""
    return normalize_name(_holder_name(str(value)))


def _source_records(source: str, section: Section) -> list[tuple[str, int | None, dict[str, float | None]]]:
    """Bind explicit source categories to complete normalized names and fiscal years."""
    records = []
    ownership = section.key == "ownership"
    for name, year, cells in _source_rows(source):
        if section.key == "executive_pay" and any(re.search(r"fees.*(?:earned|cash)", cell, re.I) for cell in cells):
            continue
        if not ownership:
            role = r"Chairman|Chief (?:Executive|Financial|Operating)|(?:(?:Executive|Senior) )?Vice President|President|CEO|CFO|COO"
            name = re.split(r"\s+(?:" + role + r")\b", name, maxsplit=1, flags=re.I)[0]
        if re.search(r"\bin\s+(?:thousands|millions)\b", source, re.I) and not re.search(r"\bin\s+(?:thousands|millions)\b", " ".join(cells), re.I):
            continue
        item = (_identity(name), year, _source_values(cells, ownership))
        if item not in records:
            records.append(item)
    return records


def complete_director_years(section: Section, payload: dict[str, Any], source: str) -> dict[str, Any]:
    """Fill only missing director years on normalized copies using their own labelled annual rows."""
    if section.key != "director_pay":
        return payload
    records = _source_records(source, section)
    for row in payload.get("compensation", []):
        if row.get("fiscal_year") is not None:
            continue
        matches = [
            year
            for name, year, values in records
            if name == _identity(row.get("name"))
            and year is not None
            and all(
                row.get(field) is not None and (source_amount := values.get(field)) is not None and math.isclose(float(row[field]), source_amount)
                for field in ("fees_earned", "stock_awards", "total")
            )
        ]
        if len(matches) == 1:
            row["fiscal_year"] = matches[0]
    return payload


def _correct_director_categories(rows: pd.DataFrame, records: list[tuple[str, int | None, dict[str, float | None]]]) -> pd.DataFrame:
    """Move an explicitly misplaced option amount without assigning or changing a fiscal year."""
    result = rows.copy()
    for index, row in result.iterrows():
        identity = _identity(row["name"])
        same_person = result.loc[result.name.map(_identity).eq(identity)]
        source_rows = [(year, values) for name, year, values in records if name == identity]
        if len(same_person) != 1 or len(source_rows) != 1 or pd.isna(row.get("fiscal_year")):
            continue
        year, values = source_rows[0]
        if year is not None and year != row["fiscal_year"]:
            continue
        option = values.get("_option_money")
        anchors_match = all(
            pd.notna(row.get(field)) and (source_amount := values.get(field)) is not None and math.isclose(float(row[field]), source_amount)
            for field in ("fees_earned", "total")
        )
        if (
            option is not None
            and anchors_match
            and values.get("_blank_stock") == 1.0
            and pd.notna(row.get("stock_awards"))
            and math.isclose(float(row["stock_awards"]), option)
            and (pd.isna(row.get("option_awards")) or math.isclose(float(row["option_awards"]), option))
        ):
            result.at[index, "stock_awards"] = float("nan")
            result.at[index, "option_awards"] = option
    return result


def _unsupported_group_percentage(source: str) -> dict[str, tuple[float, None]]:
    """Reject a common-class group percentage as all-class ownership only with actual preferred holdings."""
    aggregate = (
        r"(?:all[- ]classes?|combined classes?).{0,80}(?:beneficial|economic)?\s*ownership|(?:beneficial|economic)\s+ownership.{0,40}all classes"
    )
    if re.search(aggregate, source, re.I):
        return {}
    preferred = [
        _source_values(cells, True).get("shares")
        for name, _, cells in _source_rows(source)
        if "as a group" in _identity(name) and re.search(r"preferred stock", " ".join(cells), re.I)
    ]
    if not any(value is not None and value > 0 for value in preferred):
        return {}
    common = (
        r"COMMON STOCK\s+NUMBER OF SHARES(?:/OPTIONS)?\s+Amount and Nature of Beneficial Ownership\s+Name(?:(?!PREFERRED STOCK).){0,400}?Percent of Class"
        r"(?:(?!PREFERRED STOCK).){0,6000}?Directors and executive officers,?\s+as a group\s+[\d,.\s]+?\s+(\.\d+|\d+(?:\.\d+)?)%"
    )
    percentages = {float(value) * 0.01 for value in re.findall(common, source, re.I | re.S)}
    return {"insider_ownership_pct": (next(iter(percentages)), None)} if len(percentages) == 1 else {}


def _correct_security_rows(rows: pd.DataFrame, source: str, company_name: str | None) -> pd.DataFrame:
    """Replace foreign-class quantities only under the explicit disjoint ownership layout."""
    header = (
        r"SOLE VOTING\s+SOLE VOTING\s+SHARED VOTING\s+AND LIMITED\s+AND\s+AND\s+OR NO\s+"
        r"INVESTMENT\s+INVESTMENT\s+INVESTMENT\s+PERCENT\s+TITLE OF CLASS\s+NAME\s+POWER(?:\(\d+\))*\s+POWER\s+POWER(?:\(\d+\))*\s+OF CLASS"
    )
    if not company_name or not re.search(header, source, re.I):
        return rows
    identities = set(rows.holder_name.map(_identity))
    labels = [""]
    holdings: list[dict[str, tuple[float, ...]]] = [{}]

    # Literal class prefixes distinguish the filing issuer from the separately disclosed parent.
    for line in source.splitlines():
        if re.match(r"\s*\(\d+\)|\(See notes", line, re.I):
            break
        match = re.match(r"^(.*?)\.{2,}\s+(.+)$", line)
        if match is None:
            continue
        before = _identity(match[1])
        names = [name for name in identities if before == name or before.endswith(" " + name)]
        if len(names) != 1:
            continue
        name = names[0]
        prefix = before[: -len(name)].strip()
        if prefix and labels[-1].endswith("common stock"):
            labels.append("")
            holdings.append({})
        labels[-1] = (labels[-1] + " " + prefix).strip()
        tokens = re.sub(r"\(\d+\)", "", match[2]).split()
        if not tokens or not re.fullmatch(r"\*|\d+\.\d+%?", tokens[-1]):
            continue
        components = tokens[:-1]
        if 1 <= len(components) <= 3 and all(re.fullmatch(r"\d[\d,]*", cell) for cell in components):
            values = tuple(float(cell.replace(",", "")) for cell in components)
            holdings[-1][name] = values if name not in holdings[-1] or holdings[-1][name] == values else ()
    issuer = [index for index, label in enumerate(labels) if label == _identity(company_name) + " common stock"]
    if len(issuer) != 1:
        return rows

    # Keep an already saved issuer component; only an identified foreign quantity is wrong.
    result = rows.copy()
    for index, row in result.iterrows():
        name = _identity(row.holder_name)
        parts = holdings[issuer[0]].get(name, ())
        foreign = [
            sum(values[name])
            for label, values in zip(labels, holdings, strict=True)
            if label.endswith("common stock") and label != labels[issuer[0]] and values.get(name)
        ]
        if parts and row.get("shares") in foreign and row["shares"] not in parts:
            result.at[index, "shares"] = sum(parts)
    return result


def correct_basis(
    section: Section, merged: pd.DataFrame, new: pd.DataFrame, source: str, ceo_name: str | None, company_name: str | None = None
) -> tuple[pd.DataFrame, dict[str, tuple[float, float | None]]]:
    """Correct only exact same-source conflicts; a median alone never changes an amount."""
    if not source or merged.empty or section.key not in ("executive_pay", "director_pay", "ownership"):
        return merged, {}
    result = merged.copy()
    ownership = section.key == "ownership"
    name_column = "holder_name" if ownership else "name"
    if ownership:
        result = _correct_security_rows(result, source, company_name)
    records = _source_records(source, section)
    if section.key == "director_pay":
        result = _correct_director_categories(result, records)
    groups = [
        (float(total), float(shares))
        for name, _, values in records
        if " as a group" in name and (shares := values.get("shares")) is not None and (total := values.get("_inclusive_total")) is not None
    ]
    parent: dict[str, tuple[float, float | None]] = {"insider_shares": groups[0]} if ownership and len(groups) == 1 else {}
    if ownership:
        parent.update(_unsupported_group_percentage(source))
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
        if not ownership and "fiscal_year" in candidates and year is not None:
            candidates = candidates.loc[candidates.fiscal_year.eq(year)]
        candidate = candidates.iloc[0] if len(candidates) == 1 else pd.Series(dtype=object)
        for field, raw in values.items():
            if field.startswith("_") or field not in result or raw is None:
                continue
            value = raw
            proposed = candidate.get(field)
            if (pd.notna(proposed) and math.isclose(float(proposed), value)) or (
                field == "option_awards" and pd.isna(row.get(field)) and values.get("_sar_awards") == value
            ):
                result.at[index, field] = value
                continue
            if ownership and field == "shares" and pd.notna(row.get(field)) and row[field] == values.get("_inclusive_total"):
                result.at[index, field] = value
                continue

            # Both thousandfold directions need the source cell and other comparable years.
            peers = [
                peer_value
                for person, other_year, other in records
                if person == name and other_year != year and (peer_value := other.get(field)) is not None and peer_value > 0
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
            incompatible_value = values.get(incompatible)
            if not (
                field in result
                and values.get(field) is None
                and incompatible_value is not None
                and pd.notna(row.get(field))
                and math.isclose(float(row[field]), incompatible_value)
            ):
                continue
            result.at[index, field] = float("nan")
            if section.key == "executive_pay" and ceo_name and identity == _identity(ceo_name) and year == latest:
                scalar = "ceo_total_comp" if field == "total" else "ceo_" + field
                parent[scalar] = (float(row[field]), None)
    return result, parent
