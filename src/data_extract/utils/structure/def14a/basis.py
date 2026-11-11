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
        caption_years = _caption_years(line)
        if caption_years:
            caption_year = int(next(iter(caption_years))) if len(caption_years) == 1 else None
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2:
            header = []
            previous_header = []
            prose = line.strip() and not caption_years and not line.strip().startswith("Source table (tab-separated columns):")
            caption_year = None if prose else caption_year
            continue
        names = [index for index, cell in enumerate(cells) if re.search(r"\bname\b", cell, re.I)]
        if names:
            header, name_column = _header(cells, names, previous_header), names[0]
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
            result.extend(_year_rows(name, cells, header, name_column, years))
            continue
        year = int(years[0]) if len(set(years)) == 1 else table_year if not years and year_column is None else None
        result.append((name, year, dict(zip(header, cells, strict=True))))
    return result


def _caption_years(line: str) -> set[str]:
    """The fiscal years a director-compensation caption on `line` names (an `FY 2019-20` label as its end year)."""
    caption_years = set(re.findall(r"director\s+compensation(?:\s+table)?\s+for\s+(?:fiscal\s*)?((?:19|20)\d{2})", line, re.I))
    local_fiscal = (
        r"following table(?: and footnotes)?.{0,180}?compensation.{0,180}?Directors?\s+in\s+FY\s+((?:19|20)\d{2}(?:-(?:\d{2}|\d{4}))?)(?![-\d])"
    )
    for fiscal in re.findall(local_fiscal, line, re.I):
        label = re.fullmatch(r"((?:19|20)\d{2})(?:-(\d{2}|\d{4}))?", fiscal)
        if label and (not label[2] or str(int(label[1]) + 1).endswith(label[2])):
            caption_years.add(str(int(label[1]) + bool(label[2])))
    return caption_years


def _header(cells: list[str], names: list[int], previous_header: list[str]) -> list[str]:
    """The header row, its pay labels prefixed by the label row above it when that row lacks the name column(s)."""
    missing_names = len(cells) - len(previous_header)
    if (
        missing_names in (1, 2)
        and names == [0]
        and not any(cells[1:missing_names])
        and all(re.search(r"fees|awards|compensation|pension|total", label, re.I) for label in previous_header)
    ):
        return [
            *cells[:missing_names],
            *(f"{label} {unit}".strip() for label, unit in zip(previous_header, cells[missing_names:], strict=True)),
        ]
    return cells


def _year_rows(name: str, cells: list[str], header: list[str], name_column: int, years: list[str]) -> list[tuple[str, int, dict[str, str]]]:
    """One row per year of a multi-year source row whose cells stack one value per year; none when they do not."""
    vectors = [cell.split() if cell else [""] * len(years) for cell in cells]
    vectors[name_column] = [name] * len(years)
    if len(set(years)) != len(years) or any(len(vector) != len(years) for vector in vectors):
        return []
    return [(name, int(year), dict(zip(header, [vector[index] for vector in vectors], strict=True))) for index, year in enumerate(years)]


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
        if ownership and re.search(r"percent|%|voting", label):
            continue
        quantity = re.search(r"#|\b(?:shares?|units?|securities|number|count)\b", label)
        monetary = re.search(r"\$|\bUSD\b|dollar|fair.?value", label, re.I)
        field = _ownership_field(label, headers, values) if ownership else _pay_field(label, quantity is not None, monetary is not None)
        if field is None:
            continue
        numeric = re.sub(r"\s+\(\d+\)$", "", raw).replace(",", "").replace("$", "").replace(" ", "")
        value = float(numeric.strip("()")) * (-1 if numeric.startswith("(") else 1) if re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", numeric) else None
        scale = re.search(r"\bin\s+(thousands|millions)\b", label, re.I)
        factor = {"thousands": 1000.0, "millions": 1000000.0}[scale[1].lower()] if scale else 1.0
        if field not in values or value is not None:
            values[field] = value * factor if value is not None else None
        if field == "other_compensation" and monetary and not quantity and raw.strip() in {"-", "\u2013", "\u2014"}:
            values["_omitted_other"] = 1.0
        if field == "stock_awards" and raw.strip() in {"", "-", "\u2013", "\u2014"}:
            values["_blank_stock"] = 1.0
        if field == "option_awards" and monetary and re.search(r"\bsars?\b", label) and not re.search(r"\boption", headers):
            values["_sar_awards"] = values[field]
        if field == "option_awards" and monetary and not re.search(r"#|\b(?:number|count|units|securities)\b", label):
            values["_option_money"] = values[field]
    return values


def _ownership_field(label: str, headers: str, values: dict[str, float | None]) -> str | None:
    """The ownership field a lower-cased column label names: the share count, its inclusive total or an extra
    underlying/cash-settled column; None for any other column."""
    if re.search(r"additional underlying|cash.settled", label):
        return "_extra"
    if "total beneficial ownership" in label:
        return "shares"
    if "total" in label and re.search(r"additional underlying|cash.settled", headers):
        return "_inclusive_total"
    if (
        "shares" not in values
        and re.search(r"common stock|shares beneficially owned|amount and nature|^total$", label)
        and not re.search(r"percent|voting|sole|shared", label)
    ):
        return "shares"
    return None


def _pay_field(label: str, quantity: bool, monetary: bool) -> str | None:
    """The pay field a lower-cased column label names; an award counted in shares or options, or an incompatible
    category, maps to its `_wrong_*` field. None for an unrecognised column."""
    if quantity and not monetary and re.search(r"stock|option|securities|sars?", label):
        return "_wrong_stock" if "stock" in label and "option" not in label else "_wrong_option"
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
    return next((field for field, pattern in patterns.items() if re.search(pattern, label)), None)


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


def _correct_director_categories(
    rows: pd.DataFrame, records: list[tuple[str, int | None, dict[str, float | None]]], new: pd.DataFrame
) -> pd.DataFrame:
    """Correct an explicit option-category error or an omitted other-pay zero in its own annual row."""
    # compute: each person-year row matched to its single source row and the new answer's candidate
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
        candidates = (
            new.loc[new.name.map(_identity).eq(identity) & new.fiscal_year.eq(row["fiscal_year"])]
            if "name" in new and "fiscal_year" in new
            else new.iloc[:0]
        )

        # A literal omitted amount is unknown, while a new explicit zero or valid saved amount remains intact.
        if (
            year == row["fiscal_year"]
            and values.get("_omitted_other") == 1.0
            and anchors_match
            and row.get("other_compensation") == 0
            and len(candidates) == 1
            and pd.isna(candidates.iloc[0].get("other_compensation"))
        ):
            result.at[index, "other_compensation"] = float("nan")
        if (
            option is not None
            and anchors_match
            and values.get("_blank_stock") == 1.0
            and values.get("stock_awards") is None
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
    # clean: only a source with the disjoint sole/shared voting layout, and the holder identities to look for
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
    # clean: the layout corrections, the labelled source records, the group totals and the CEO's latest year
    if not source or merged.empty or section.key not in ("executive_pay", "director_pay", "ownership"):
        return merged, {}
    result = merged.copy()
    ownership = section.key == "ownership"
    name_column = "holder_name" if ownership else "name"
    if ownership:
        result = _correct_security_rows(result, source, company_name)
    records = _source_records(source, section)
    if section.key == "director_pay":
        result = _correct_director_categories(result, records, new)
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
        stated = {field: value for field, value in values.items() if not field.startswith("_") and field in result and value is not None}
        accepted = [
            field
            for field, value in stated.items()
            if _confirmed(field, value, candidate.get(field), row, values, ownership)
            or (not ownership and _thousandfold(field, value, row.get(field), name, year, records))
        ]
        for field in accepted:
            result.at[index, field] = stated[field]

        # Clear only a value tied to an explicitly incompatible source column.
        cleared = _incompatible_fields(row, values, result.columns)
        for field in cleared:
            result.at[index, field] = float("nan")
        if section.key == "executive_pay" and ceo_name and identity == _identity(ceo_name) and year == latest:
            parent.update({("ceo_total_comp" if field == "total" else "ceo_" + field): (float(row[field]), None) for field in cleared})
    return result, parent


def _confirmed(field: str, value: float, proposed: Any, row: pd.Series, values: dict[str, float | None], ownership: bool) -> bool:
    """Whether the source cell confirms the new answer's value, fills an empty option cell from a SAR award, or is
    the share count an ownership row took from the inclusive total column."""
    if pd.notna(proposed) and math.isclose(float(proposed), value):
        return True
    if field == "option_awards" and pd.isna(row.get(field)) and values.get("_sar_awards") == value:
        return True
    return bool(ownership and field == "shares" and pd.notna(row.get(field)) and row[field] == values.get("_inclusive_total"))


def _thousandfold(
    field: str, value: float, actual: Any, name: str, year: int | None, records: list[tuple[str, int | None, dict[str, float | None]]]
) -> bool:
    """Whether the saved amount is the source cell off by a thousandfold, judged against the person's other years:
    the source value must sit closer to their leave-one-out median."""
    peers = [
        peer_value
        for person, other_year, other in records
        if person == name and other_year != year and (peer_value := other.get(field)) is not None and peer_value > 0
    ]
    median = _leave_one_out_median(pd.Series([float("nan"), *peers], dtype=float)).iloc[0] if peers else float("nan")
    if pd.isna(actual) or actual <= 0 or pd.isna(median) or median <= 0:
        return False
    correction = 1000.0 if math.isclose(actual * 1000.0, value) else 0.001 if math.isclose(actual * 0.001, value) else None
    return correction is not None and value > 0 and abs(math.log(value / median)) < abs(math.log(actual / median))


def _incompatible_fields(row: pd.Series, values: dict[str, float | None], columns: pd.Index) -> list[str]:
    """The saved fields whose value equals an explicitly incompatible source column while the source states none."""
    wrong = {
        "bonus": "_wrong_bonus",
        "total": "_wrong_total",
        "option_awards": "_wrong_option",
        "stock_awards": "_wrong_stock",
        "non_equity_incentive": "bonus",
    }
    return [
        field
        for field, incompatible in wrong.items()
        if field in columns
        and values.get(field) is None
        and (incompatible_value := values.get(incompatible)) is not None
        and pd.notna(row.get(field))
        and math.isclose(float(row[field]), incompatible_value)
    ]
