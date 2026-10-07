"""
legacy_13f_fallback.py (src/data_extract/utils/institutionals/legacy_13f_fallback.py)
--------------------------------------------------------------------------------------
Source-checked parser for pre-XML fixed-width 13F information tables. Every line needs a
check-digit-valid CUSIP and two numbers, and the whole book must match the SEC Summary Page
entry count and value total; anything else raises `ValueError`, so an unverifiable book is
never returned. Output columns are the XML names `fetch_13f._book_frame` reads.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal
from typing import Any

import pandas as pd

from src.data_extract.utils.institutionals.fetch_cusip_map import normalize_cusip

_TABLE = re.compile(r"<TABLE[^>]*>(.*?)</TABLE>", re.I | re.S)
_CUSIP = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{8,9})(?![A-Z0-9])", re.I)
_NUMBER = re.compile(r"(?<![\w.,/])(?P<number>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<type>SH|PRN)?(?=\s|$)", re.I)
_ENTRY_COUNT = re.compile(r"Form\s+13F\s+Information\s+Table\s+Entry\s+Total[ \t]*:?[ \t]*([\d,]+)\b", re.I)
_VALUE_TOTAL = re.compile(
    r"Form\s+13F\s+Information\s+Table\s+Value\s+Total(?:\s*x\s*1000)?:\s*\$?\s*([\d,]+(?:\.\d+)?)"
    r"(?:\s*(Billion|Million|\(?(?:in\s+)?thousands?\)?|dollars?))?",
    re.I,
)
_FOOTER_TOTAL = re.compile(r"REPORT SUMMARY\s+\d+\s+DATA RECORDS\s+([\d,]+)", re.I)


def _valid_cusip(token: str) -> bool:
    """True for a 9-character CUSIP whose check digit verifies."""
    if len(token) != 9 or not token[-1].isdigit():
        return False
    values = [int(char) if char.isdigit() else ord(char) - ord("A") + 10 for char in token[:8]]
    total = sum(sum(divmod(value * (2 if index % 2 else 1), 10)) for index, value in enumerate(values))
    return (10 - total % 10) % 10 == int(token[-1])


def _source_rows(raw: str) -> list[tuple[str, str, str]]:
    """Logical holding lines, carrying a layout only across compatible continuation pages."""
    rows: list[tuple[str, str, str]] = []
    header = ""
    columns: list[int] = []
    for block in _TABLE.findall(raw):
        lines = block.splitlines()
        headers = [i for i, line in enumerate(lines) if "CUSIP" in line.upper()]
        markers = [i for i, line in enumerate(lines) if "<S>" in line.upper()]
        positions = [match.start() for match in re.finditer(r"<[SC]>", lines[markers[0]], re.I)] if markers else []
        if headers:
            header = lines[headers[0]]
            columns = positions
            start = max(headers[0], markers[0] if markers else 0) + 1
        elif not header or not positions or positions[:5] != columns[:5]:
            continue
        else:
            start = markers[0] + 1
        pending = ""
        i = start
        while i < len(lines):
            line = lines[i]
            i += 1
            upper = line.upper()
            if "CUSIP" in upper:
                header = line
                continue
            if "<" in line or "REPORT SUMMARY" in upper or re.fullmatch(r"[\s\-=._]+", line):
                continue
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


def _cusip_match(header: str, line: str) -> re.Match[str] | None:
    """Find the checksum-valid identifier before amounts; header spacing is only a hint."""
    matches = [match for match in _CUSIP.finditer(line.upper()) if _valid_cusip(normalize_cusip(match.group(1)) or "")]
    eligible = [match for match in matches if re.search(r"[A-Z]", line[: match.start()], re.I) and _NUMBER.match(line[match.end() :].lstrip())]
    if not eligible:
        return None
    first = eligible[0]
    if any(match.start() < first.start() for match in matches):
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


def _holding(header: str, line: str, pending: str, prior: tuple[str, str, str] | None) -> tuple[dict[str, Any], tuple[str, str, str]]:
    """One holding row (value in dollars from the $1000 column); a CUSIP-less line continues `prior`."""
    match = _cusip_match(header, line)
    if match:
        cusip = normalize_cusip(match.group(1))
        if cusip is None or not _valid_cusip(cusip):
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
    amount_type = "PRN" if (amount_token.group("type") or "").upper() == "PRN" or re.search(r"\b(?:PRN|PRINCIPAL)\b", rest, re.I) else "SH"
    option = re.search(r"\b(?:PUT|CALL)\b", rest, re.I)
    if value < 0 or amount < 0 or value * 1000 > 1e12 or (amount_type == "SH" and option is None and value > 0 and amount == 0):
        raise ValueError(f"Invalid source amounts for {cusip}: {line[:120]}")
    row = {
        "CUSIP": cusip,
        "NAMEOFISSUER": issuer,
        "TITLEOFCLASS": title,
        "VALUE": float(value * 1000),
        "SSHPRNAMT": float(amount),
        "SSHPRNAMTTYPE": amount_type,
        "PUTCALL": option.group().upper() if option else "",
    }
    return row, (cusip, issuer, title)


def _value_total_error(raw: str, values_usd: list[Any]) -> str | None:
    """Compare USD to explicit cover magnitude/precision; unlabelled legacy covers are $1000s."""
    summary = _VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)
    if summary is None:
        return None
    total = Decimal(summary.group(1).replace(",", ""))
    unit = (summary.group(2) or "").lower() if summary.re is _VALUE_TOTAL else ""
    scale = Decimal(10**9 if unit == "billion" else 10**6 if unit == "million" else 1 if unit == "dollars" else 1000)
    parsed = sum((Decimal(str(value)) for value in values_usd), Decimal(0))
    tolerance = (
        scale * Decimal(10) ** total.as_tuple().exponent / 2 if unit in ("billion", "million") else max(Decimal(15000), total * scale / 100000)
    )
    if abs(parsed - total * scale) > tolerance:
        return f"Source value total differs materially: {parsed} versus {total}"
    return None


def _check_totals(raw: str, rows: list[dict[str, Any]]) -> None:
    """Raise unless the rows match the Summary Page entry count and, when stated, its value total."""
    count = _ENTRY_COUNT.search(raw)
    if count is None:
        raise ValueError("SEC Summary Page entry count is missing")
    if len(rows) != int(count.group(1).replace(",", "")):
        raise ValueError(f"Source has {len(rows)} holding lines, cover declares {count.group(1)}")
    error = _value_total_error(raw, [row["VALUE"] for row in rows])
    if error:
        raise ValueError(error)


def needs_legacy_fallback(raw: str, parsed: pd.DataFrame | None) -> bool:
    """True when EdgarTools' text parse is short, malformed, or off the source: entry count,
    CUSIP multiset, or (when stated) the Summary Page value total. A clean book is not reparsed."""
    if parsed is None or parsed.empty:
        return True
    count = _ENTRY_COUNT.search(raw)
    if count is None or len(parsed) != int(count.group(1).replace(",", "")):
        return True
    column = next((name for name in parsed.columns if name.lower() == "cusip"), None)
    if column is None:
        return True
    parsed_cusips = [normalize_cusip(value) for value in parsed[column]]
    if any(not _valid_cusip(value or "") for value in parsed_cusips):
        return True
    values = next((name for name in parsed.columns if name.lower() == "value"), None)
    if values is None or _value_total_error(raw, pd.to_numeric(parsed[values], errors="coerce").fillna(0).tolist()):
        return True
    source_cusips = []
    prior: tuple[str, str, str] | None = None
    try:
        for header, line, pending in _source_rows(raw):
            row, prior = _holding(header, line, pending, prior)
            source_cusips.append(row["CUSIP"])
    except ValueError:
        return True
    return Counter(source_cusips) != Counter(parsed_cusips)


def parse_legacy_information_table(raw: str) -> pd.DataFrame:
    """Return source-supported rows in the shape consumed by the existing 13F classifier."""
    source_rows = _source_rows(raw)
    if not source_rows:
        raise ValueError("No legacy 13F information-table lines")
    rows: list[dict[str, Any]] = []
    prior: tuple[str, str, str] | None = None
    for header, line, pending in source_rows:
        row, prior = _holding(header, line, pending, prior)
        rows.append(row)
    _check_totals(raw, rows)
    return pd.DataFrame(rows)
