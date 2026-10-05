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
_NUMBER = re.compile(r"(?<![A-Z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Z0-9])", re.I)
_ENTRY_COUNT = re.compile(r"Form\s+13F\s+Information\s+Table\s+Entry\s+Total[ \t]*:?[ \t]*([\d,]+)\b", re.I)
_VALUE_TOTAL = re.compile(r"Form\s+13F\s+Information\s+Table\s+Value\s+Total(?:\s*x\s*1000)?:\s*\$?\s*([\d,]+(?:\.\d+)?)", re.I)
_FOOTER_TOTAL = re.compile(r"REPORT SUMMARY\s+\d+\s+DATA RECORDS\s+([\d,]+)", re.I)


def _valid_cusip(token: str) -> bool:
    """True for a 9-character CUSIP whose check digit verifies."""
    if len(token) != 9 or not token[-1].isdigit():
        return False
    values = [int(char) if char.isdigit() else ord(char) - ord("A") + 10 for char in token[:8]]
    total = sum(sum(divmod(value * (2 if index % 2 else 1), 10)) for index, value in enumerate(values))
    return (10 - total % 10) % 10 == int(token[-1])


def _source_rows(raw: str) -> list[tuple[str, str, str]]:
    """`(header, line, pending issuer text)` for every holding line of every `<TABLE>` block."""
    rows: list[tuple[str, str, str]] = []
    for block in _TABLE.findall(raw):
        lines = block.splitlines()
        headers = [i for i, line in enumerate(lines) if "CUSIP" in line.upper()]
        if not headers:
            continue
        markers = [i for i, line in enumerate(lines) if "<S>" in line.upper()]
        start = max([*markers, *headers]) + 1
        header = lines[max(headers)]
        pending = ""
        for line in lines[start:]:
            upper = line.upper()
            if "<" in line or "REPORT SUMMARY" in upper or re.fullmatch(r"[\s\-=._]+", line):
                continue
            if len(_NUMBER.findall(line)) < 2:
                if line.strip() and not any(word in upper for word in ("INVSTMT", "DSCRTN", "VOTING", "FORM 13F")):
                    pending = (pending + " " + line.strip()).strip()
                continue
            rows.append((header, line, pending))
            pending = ""
    return rows


def _cusip_match(header: str, line: str) -> re.Match[str] | None:
    """The line's CUSIP token: the first valid one in a tabbed table, else the one nearest the header column."""
    column = header.upper().find("CUSIP")
    matches = list(_CUSIP.finditer(line.upper()))
    if "\t" in header:
        eligible = [match for match in matches if _valid_cusip(normalize_cusip(match.group(1)) or "")]
        return eligible[0] if eligible else None
    eligible = [match for match in matches if column - 8 <= match.start() <= column + 12]
    return min(eligible, key=lambda match: abs(match.start() - column)) if eligible else None


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
        column = header.upper().find("CUSIP")
        tail = line if "\t" in line else line[max(column + 6, 0) :]
    numbers = list(_NUMBER.finditer(tail))
    if len(numbers) < 2:
        raise ValueError(f"Unsplit source value/shares for {cusip}: {line[:120]}")
    first, second = (Decimal(token.group().replace(",", "")) for token in numbers[:2])
    reverse = "SHARES" in header.upper() and "VALUE" in header.upper() and header.upper().find("SHARES") < header.upper().find("VALUE")
    value, amount = (second, first) if reverse else (first, second)
    rest = tail[numbers[1].end() :]
    amount_type = "PRN" if re.search(r"\b(?:PRN|PRINCIPAL)\b", rest, re.I) else "SH"
    option = re.search(r"\b(?:PUT|CALL)\b", rest, re.I)
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
    """Why dollar `values_usd` disagree with the stated Summary Page value total ($1000s or
    dollars), or None when they agree or no total is stated."""
    summary = _VALUE_TOTAL.search(raw) or _FOOTER_TOTAL.search(raw)
    if summary is None:
        return None
    total = Decimal(summary.group(1).replace(",", ""))
    parsed = sum((Decimal(str(value)) / 1000 for value in values_usd), Decimal(0))
    difference = min(abs(parsed - total), abs(parsed * 1000 - total))
    if difference > max(Decimal(15), total / 100000):
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
    previous: str | None = None
    for header, line, _ in _source_rows(raw):
        match = _cusip_match(header, line)
        if match:
            previous = normalize_cusip(match.group(1))
        if not _valid_cusip(previous or ""):
            return True
        source_cusips.append(previous)
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
