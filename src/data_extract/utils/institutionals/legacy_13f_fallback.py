"""
legacy_13f_fallback.py (src/data_extract/utils/institutionals/legacy_13f_fallback.py)
--------------------------------------------------------------------------------------
Source-accounted pre-XML information tables. Holding and voting lines stay separate;
present cover declarations use the shared tolerance, absent declarations remain unknown.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import Decimal
from typing import Any

import pandas as pd

from src.data_extract.utils.institutionals.fetch_cusip_map import normalize_cusip
from src.utils.institutional_vintages import literal_13f_cusip, reconcile_13f_source

_TABLE = re.compile(r"<TABLE[^>]*>(.*?)</TABLE>", re.I | re.S)
_CUSIP = re.compile(r"(?<![0-9])([A-Z0-9*@#]{6}-[A-Z0-9*@#]{2}-[0-9]|[A-Z0-9*@#]{8,9})(?![A-Z0-9*@#])", re.I)
_NUMBER = re.compile(r"(?<![\w.,/])(?:\$\s*)?(?P<number>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<type>SH|PRN)?(?=\s|$)", re.I)
_ENTRY_COUNT = re.compile(r"Form\s*13\s*F\s+(?:Information\s+)?Table\s+Entry\s+Total\s*:?\s*([\d,]+)\b", re.I)
_VALUE_TOTAL = re.compile(
    r"Form\s*13\s*F\s+(?:Information\s+)?Table\s+Value\s+Total(?:\s*\([^)]*\)|\s*x\s*\$?\s*1000)?\s*:\s*(?:\([^)]*\)\s*)?\$?\s*([\d,]+(?:\.\d+)?)"
    r"(?:\s*(Billion|Million|\(?(?:in\s+)?thousands?\)?|dollars?))?",
    re.I,
)
_FOOTER_TOTAL = re.compile(r"REPORT SUMMARY\s+\d+\s+DATA RECORDS\s+([\d,]+)", re.I)
_NON_HOLDING_LINE = re.compile(
    r"\s*(?:(?:GRAND|FINAL)\s+TOTALS?\b|(?:S\s+)?REPORT SUMMARY\b)"
    r"|\s{2,}(?:SOLE|DEFINED|SHARED|OTHER)\s+\d"
    r"|(?:\s*COLUMN\s+\d+)+\s*$"
    r"|\s*FORM\s*13\s*F\b.*\bPAGE\s+\d+\s*$",
    re.I,
)


def _valid_cusip(token: str) -> bool:
    """True for a 9-character CUSIP whose check digit verifies."""
    if len(token) != 9 or not token[-1].isdigit():
        return False
    codes = {"*": 36, "@": 37, "#": 38}
    if not re.fullmatch(r"[A-Z0-9*@#]{8}[0-9]", token):
        return False
    values = [int(char) if char.isdigit() else codes[char] if char in codes else ord(char) - ord("A") + 10 for char in token[:8]]
    total = sum(sum(divmod(value * (2 if index % 2 else 1), 10)) for index, value in enumerate(values))
    return (10 - total % 10) % 10 == int(token[-1])


def _source_rows(raw: str) -> list[tuple[str, str, str]]:
    """Logical holding lines, carrying a layout only across compatible continuation pages."""
    rows: list[tuple[str, str, str]] = []
    _check_boundaries(raw)
    header = ""
    columns: list[int] = []
    blocks = _TABLE.findall(raw)
    if blocks and any(_holding_shaped(line) for line in _TABLE.sub("\n", raw).splitlines()):
        raise ValueError("Unaccounted legacy holdings outside tagged tables")
    if not blocks and re.search(r"<TEXT>", raw, re.I):
        blocks = [raw]
    for block in blocks:
        lines = block.splitlines()
        headers = [i for i, line in enumerate(lines) if "CUSIP" in line.upper()]
        markers = [i for i, line in enumerate(lines) if "<S>" in line.upper()]
        positions = [match.start() for match in re.finditer(r"<[SC]>", lines[markers[0]], re.I)] if markers else []
        if headers:
            header = lines[headers[0]]
            columns = positions
            start = max(headers[0], markers[0] if markers else 0) + 1
        elif not header or not positions or positions[:5] != columns[:5]:
            if any(_holding_shaped(line) for line in lines):
                raise ValueError("Unaccounted legacy holding page: missing or incompatible columns")
            continue
        else:
            start = markers[0] + 1
        if any(_holding_shaped(line) for line in lines[:start]):
            raise ValueError("Unaccounted legacy holdings before the table header/column marker")
        pending = ""
        i = start
        while i < len(lines):
            line = lines[i]
            i += 1
            upper = line.upper()
            if "CUSIP" in upper:
                header = line
                continue
            if _NON_HOLDING_LINE.match(line):
                continue
            if "<" in line:
                if _holding_shaped(line):
                    raise ValueError(f"Unaccounted marked-up holding line: {line[:120]}")
                continue
            if re.fullmatch(r"[\s\-=._]+", line):
                continue
            if len(columns) >= 6 and not line[columns[3] : columns[5]].strip():
                has_identifier = any(re.search(r"[0-9*@#]", m.group(1)) for m in _CUSIP.finditer(line[: columns[3]]))
                if not has_identifier and line[columns[5] :].strip():
                    continue  # Class/voting subdivisions have no economic cells or new identifier.
            match = _cusip_match(header, line)
            # A blank value cell with a populated shares cell may wrap onto the next
            # physical line. Read only that value column; a warrant date stays in class.
            if match and len(columns) >= 5 and abs(match.start() - columns[2]) <= 2 and not line[match.end() : columns[4]].strip() and i < len(lines):
                continuation = lines[i]
                value = continuation[columns[3] :].strip()
                if not continuation[: columns[1]].strip() and not continuation[columns[2] : columns[3]].strip() and _NUMBER.fullmatch(value):
                    fragment = continuation[columns[1] : columns[2]].strip()
                    prefix = line[: match.start()].rstrip() + (" " + fragment if fragment else "")
                    line = f"{prefix}  {match.group(1)} {value} {line[match.end() :].lstrip()}"
                    i += 1
            if len(_NUMBER.findall(line)) < 2:
                if match:
                    raise ValueError(f"Unsplit source value/shares: {line[:120]}")
                if line.strip() and not any(word in upper for word in ("INVSTMT", "DSCRTN", "VOTING", "FORM 13F")):
                    pending = (pending + " " + line.strip()).strip()
                continue
            rows.append((header, line, pending))
            pending = ""
    return rows


def _holding_shaped(line: str) -> bool:
    """Economic-looking content cannot disappear because its page/header was unsupported."""
    plain = re.sub(r"<[^>]*>", " ", line)
    if _NON_HOLDING_LINE.match(plain):
        return False
    if len(_NUMBER.findall(plain)) < 2:
        return False
    if re.search(r"\b(?:SH|SHS|PRN|SHARES|PRINCIPAL|CALL|PUT)\b", plain, re.I):
        return True
    identifiers = [match for match in _CUSIP.finditer(plain) if re.search(r"[0-9*@#]", match.group(1))]
    return any(re.search(r"[A-Z]", plain[: m.start()], re.I) and _NUMBER.match(plain[m.end() :].lstrip()) for m in identifiers)


def _check_boundaries(raw: str) -> None:
    """Reject cut source tables/documents before their surviving rows can self-certify."""
    for tag in ("TABLE", "TEXT", "DOCUMENT", "SEC-DOCUMENT"):
        depth = 0
        for token in re.finditer(rf"<(/?){tag}(?:\s[^>]*|)>", raw, re.I):
            depth += -1 if token.group(1) else 1
            if depth < 0:
                raise ValueError(f"Unmatched legacy {tag} boundary")
        if depth:
            raise ValueError(f"Unclosed legacy {tag} boundary")


def _cusip_match(header: str, line: str) -> re.Match[str] | None:
    """Find the source identifier before amounts; joined class text needs a valid checksum."""
    upper = line.upper()
    matches = [
        match
        for match in _CUSIP.finditer(upper)
        if re.search(r"[0-9*@#]", match.group(1))
        and (not match.start() or not line[match.start() - 1].isalnum() or _valid_cusip(literal_13f_cusip(match.group(1)) or ""))
    ]
    eligible = [match for match in matches if re.search(r"[A-Z]", line[: match.start()], re.I) and _NUMBER.match(line[match.end() :].lstrip())]
    hint = re.search("CUSIP", header, re.I)
    if hint:
        aligned = [match for match in eligible if abs(match.start() - hint.start()) <= 3 and _valid_cusip(literal_13f_cusip(match.group(1)) or "")]
        if len(aligned) == 1:
            return aligned[0]
    if not eligible:
        candidates = []
        if hint:
            token = re.compile(r"([A-Z0-9*@#]{8}[0-9])(?![A-Z0-9*@#])", re.I)
            for offset in range(max(0, hint.start() - 3), hint.start() + 4):
                match = token.match(upper, offset)
                if match and _valid_cusip(match.group(1)) and _NUMBER.match(line[match.end() :].lstrip()):
                    candidates.append(match)
        if len(candidates) == 1:
            return candidates[0]
        return None
    first = eligible[0]
    if any(match.start() < first.start() and _valid_cusip(literal_13f_cusip(match.group(1)) or "") for match in matches):
        raise ValueError(f"Ambiguous source CUSIP: {line[:120]}")
    return first


def _name_and_class(prefix: str) -> tuple[str, str]:
    """`(issuer, title of class)` from the text before the CUSIP."""
    prefix = re.sub(r"^D[\t ]+", "", prefix.strip(), flags=re.I)
    parts = [part.strip() for part in re.split(r"\s{2,}|\t+", prefix) if part.strip()]
    if len(parts) >= 2:
        return " ".join(parts[:-1]), parts[-1]
    match = re.match(r"(.+?)\s+(COM|ADR|CS|CONV|WTS|COMMON STOCK)\s*$", prefix, re.I)
    return (match.group(1), match.group(2)) if match else (prefix, "")


def _holding(
    header: str, line: str, pending: str, prior: tuple[str, str, str] | None, value_factor: float = 1000.0
) -> tuple[dict[str, Any], tuple[str, str, str], bool]:
    """One holding row (value in dollars from the $1000 column); a CUSIP-less line continues `prior`."""
    match = _cusip_match(header, line)
    if match:
        cusip = literal_13f_cusip(match.group(1))
        if cusip is None or (len(cusip) == 9 and cusip.isdigit() and not _valid_cusip(cusip)):
            raise ValueError(f"Invalid source CUSIP {match.group(1)!r}")
        issuer, title = _name_and_class(f"{pending} {line[: match.start()]}")
        if not issuer:
            raise ValueError(f"Missing issuer for {cusip}")
        tail = line[match.end() :]
    else:
        if prior is None:
            raise ValueError(f"Continuation without a prior CUSIP: {line[:100]}")
        cusip, issuer, title = prior
        tail = line.lstrip()
    tail = _separate_adjacent_amounts(tail)
    numbers = []
    offset = len(tail) - len(tail.lstrip())
    for _ in range(2):
        token = _NUMBER.match(tail, offset)
        if token is None:
            raise ValueError(f"Unsplit source value/shares for {cusip}: {line[:120]}")
        numbers.append(token)
        offset = token.end() + len(tail[token.end() :]) - len(tail[token.end() :].lstrip())
    first, second = (Decimal(token.group("number").replace(",", "")) for token in numbers)
    reverse = "SHARES" in header.upper() and "VALUE" in header.upper() and header.upper().find("SHARES") < header.upper().find("VALUE")
    value, amount = (second, first) if reverse else (first, second)
    rest = tail[numbers[1].end() :]
    amount_token = numbers[0] if reverse else numbers[1]
    amount_type = "PRN" if (amount_token.group("type") or "").upper() == "PRN" or re.search(r"\b(?:PRN|PR|PRINCIPAL)\b", rest, re.I) else "SH"
    option = re.search(r"\b(?:PUT|CALL)\b", rest, re.I)
    if value < 0 or amount < 0:
        raise ValueError(f"Invalid source amounts for {cusip}: {line[:120]}")
    row = {
        "CUSIP": cusip,
        "NAMEOFISSUER": issuer,
        "TITLEOFCLASS": title,
        "VALUE": float(value * Decimal(str(value_factor))),
        "SSHPRNAMT": float(amount),
        "SSHPRNAMTTYPE": amount_type,
        "PUTCALL": option.group().upper() if option else "",
    }
    return row, (cusip, issuer, title), match is not None


def _separate_adjacent_amounts(tail: str) -> str:
    """Split touching digits only when the three voting cells independently prove quantity."""
    first = _NUMBER.match(tail.lstrip())
    if first is None or not re.fullmatch(r"\d+", first.group("number")):
        return tail
    rest = tail.lstrip()[first.end() :]
    if not re.match(r"\s+(?:SH|SHS|PRN|PR)\b", rest, re.I):
        return tail
    discretion = re.search(r"\b(?:SOLE|DEFINED|OTHER|SHARED)\b", rest, re.I)
    votes = _NUMBER.findall(rest[discretion.end() :]) if discretion else []
    if len(votes) < 3:
        return tail
    quantity = sum(Decimal(value.replace(",", "")) for value, _ in votes[-3:])
    suffix = str(int(quantity))
    digits = first.group("number")
    if quantity <= 0 or quantity != int(quantity) or not digits.endswith(suffix) or len(digits) <= len(suffix):
        return tail
    return f"{digits[: -len(suffix)]} {suffix}{rest}"


def _holding_rows(raw: str, source_rows: list[tuple[str, str, str]], value_factor: float) -> list[dict[str, Any]]:
    """Keep economic continuations; remove unlabelled subtotals only with exact source count proof."""
    rows: list[dict[str, Any]] = []
    subtotals: set[int] = set()
    prior: tuple[str, str, str] | None = None
    value_sum = amount_sum = 0.0
    group_rows = 0
    for header, line, pending in source_rows:
        row, prior, named = _holding(header, line, pending, prior, value_factor)
        if named:
            value_sum = amount_sum = 0.0
            group_rows = 0
        plain = not named and bool(re.fullmatch(r"\s*[\d,.]+\s+[\d,.]+\s*", line))
        # Integer VALUE cells and their subtotal may each be independently rounded.
        if (
            plain
            and math.isclose(row["VALUE"], value_sum, rel_tol=1e-12, abs_tol=(group_rows + 1) * value_factor / 2)
            and math.isclose(row["SSHPRNAMT"], amount_sum, rel_tol=1e-12)
        ):
            subtotals.add(len(rows))
            value_sum = amount_sum = 0.0
            group_rows = 0
        else:
            value_sum += row["VALUE"]
            amount_sum += row["SSHPRNAMT"]
            group_rows += 1
        rows.append(row)
    count = _ENTRY_COUNT.search(raw)
    if subtotals:
        if count is None:
            raise ValueError("Ambiguous unlabelled subtotal/continuation without source entry count")
        if len(rows) - len(subtotals) == int(count.group(1).replace(",", "")):
            rows = [row for i, row in enumerate(rows) if i not in subtotals]
    return rows


def _value_total_error(raw: str, values_usd: list[Any], *, default_factor: float = 1000.0) -> str | None:
    """Compare USD to explicit cover magnitude/precision; unlabelled legacy covers are $1000s."""
    summary = _VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)
    if summary is None:
        return None
    total = Decimal(summary.group(1).replace(",", ""))
    unit = (summary.group(2) or "").lower() if summary.re is _VALUE_TOTAL else ""
    explicit_thousands = "thousand" in unit or bool(re.search(r"\bx\s*1000\s*:", summary.group(0), re.I))
    scale = Decimal(
        10**9 if unit == "billion" else 10**6 if unit == "million" else 1 if unit == "dollars" else 1000 if explicit_thousands else default_factor
    )
    parsed = sum((Decimal(str(value)) for value in values_usd), Decimal(0))
    tolerance = (
        scale * Decimal(10) ** total.as_tuple().exponent / 2 if unit in ("billion", "million") else max(Decimal(15000), total * scale / 100000)
    )
    if unit in ("billion", "million"):
        if abs(parsed - total * scale) > tolerance:
            return f"Source value total differs materially: {parsed} versus {total}"
        return None
    try:
        reconcile_13f_source(len(values_usd), None, float(parsed), float(total * scale), source_accounted=True)
    except ValueError as error:
        return f"Source value total differs materially: {parsed} versus {total}; {error}"
    return None


def _check_totals(raw: str, rows: list[dict[str, Any]], *, value_factor: float = 1000.0, verify_values: bool = True) -> None:
    """Check stated Summary Page totals without inventing missing declarations."""
    count = _ENTRY_COUNT.search(raw)
    declared = int(count.group(1).replace(",", "")) if count else None
    try:
        reconcile_13f_source(len(rows), declared, 0.0, None, source_accounted=True)
    except ValueError as error:
        raise ValueError(f"Source has {len(rows)} holding lines, cover declares {count.group(1) if count else None}; {error}") from error
    error = _value_total_error(raw, [row["VALUE"] for row in rows], default_factor=value_factor) if verify_values else None
    if error:
        raise ValueError(error)


def needs_legacy_fallback(raw: str, parsed: pd.DataFrame | None) -> bool:
    """True when EdgarTools' text parse is short, malformed, or off the source: entry count (when
    stated), logical holding facts, or (when stated) the Summary Page value total. A clean book is
    not reparsed. A filing with no Summary Page keeps EdgarTools' book only when it matches the
    independent, structurally accounted source parse row for row."""
    if parsed is None or parsed.empty:
        return True
    count = _ENTRY_COUNT.search(raw)
    if count is not None and len(parsed) != int(count.group(1).replace(",", "")):
        return True
    columns = {name.lower(): name for name in parsed.columns}
    column = columns.get("cusip")
    if column is None:
        return True
    parsed_cusips = [normalize_cusip(value) for value in parsed[column]]
    if any(not _valid_cusip(value or "") for value in parsed_cusips):
        return True
    values = columns.get("value")
    amounts = columns.get("sshprnamt", columns.get("sharesprnamount"))
    if values is None or amounts is None:
        return True
    parsed_values = pd.to_numeric(parsed[values], errors="coerce")
    if _value_total_error(raw, parsed_values.fillna(0).tolist()):
        return True
    parsed_amounts = pd.to_numeric(parsed[amounts], errors="coerce")
    amount_type = columns.get("sshprnamttype", columns.get("type"))
    option = columns.get("putcall")
    parsed_types = parsed[amount_type] if amount_type else pd.Series("", index=parsed.index)
    parsed_options = parsed[option] if option else pd.Series("", index=parsed.index)
    parsed_types = parsed_types.astype("string").str.strip().str.upper().fillna("").replace({"": "SH", "SHARES": "SH", "PRINCIPAL": "PRN"})
    parsed_options = parsed_options.astype("string").str.strip().str.upper().fillna("")
    source_facts = []
    try:
        for row in _holding_rows(raw, _source_rows(raw), 1000.0):
            source_facts.append((row["CUSIP"], row["VALUE"], row["SSHPRNAMT"], row["SSHPRNAMTTYPE"], row["PUTCALL"]))
    except ValueError:
        return True
    return Counter(source_facts) != Counter(zip(parsed_cusips, parsed_values, parsed_amounts, parsed_types, parsed_options, strict=True))


def parse_legacy_information_table(raw: str, *, value_factor: float = 1000.0, verify_values: bool = True) -> pd.DataFrame:
    """Return source-supported rows in the shape consumed by the existing 13F classifier."""
    source_rows = _source_rows(raw)
    if not source_rows:
        raise ValueError("No legacy 13F information-table lines")
    rows = _holding_rows(raw, source_rows, value_factor)
    _check_totals(raw, rows, value_factor=value_factor, verify_values=verify_values)
    frame = pd.DataFrame(rows)
    frame.attrs["source_accounted"] = True
    return frame
