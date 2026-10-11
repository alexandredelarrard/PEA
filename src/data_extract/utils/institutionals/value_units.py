"""Source-declared 13F VALUE units, independent of prices and share magnitudes."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import SEC_13F_DOLLAR_UNITS_FROM, SEC_13F_SCHEMA_VALUE_FACTORS

_TABLE = re.compile(r"<TABLE[^>]*>(.*?)</TABLE>", re.I | re.S)
_VALUE_CAPTION = re.compile(r"\bVALUE\b|x\s*\$", re.I)
_MULTIPLIER = re.compile(r"(?:\bx\s*\$?\s*|\$\s*)(\d[\d,]*(?:\.\d+)?)\b", re.I)


@dataclass(frozen=True)
class ValueUnits:
    """Keep prescribed and stated evidence separate; a conflict has no USD factor."""

    expected_factor: float | None
    stated_factor: float | None
    factor: float | None
    evidence: str


def _caption_factors(raw: str | None) -> set[float]:
    """Read information-table headings; Summary Page display units are independent."""
    factors: set[float] = set()
    for block in _TABLE.findall(raw or ""):
        lines = block.splitlines()[:15]
        if not re.search(r"\bCUSIP\b", " ".join(lines), re.I):
            continue
        for index, line in enumerate(lines):
            if not _VALUE_CAPTION.search(line) or re.search(r"\bTOTAL\b|SUMMARY", line, re.I):
                continue
            following = lines[index + 1 : index + 2]
            caption = line + " " + (" ".join(following) if not following or not _VALUE_CAPTION.search(following[0]) else "")
            caption = re.sub(r"<[^>]*>", " ", caption)
            matches = _MULTIPLIER.findall(caption)
            factors.update(float(value.replace(",", "")) for value in matches if float(value.replace(",", "")) > 0)
            for pattern, factor in ((r"\bthousands?\b", 1000.0), (r"\bmillions?\b", 1e6), (r"\bbillions?\b", 1e9)):
                if re.search(pattern, caption, re.I):
                    factors.add(factor)
            if not matches and not re.search(r"\b(?:thousands?|millions?|billions?)\b", caption, re.I) and re.search(r"\bdollars?\b", caption, re.I):
                factors.add(1.0)
    return factors


def resolve_value_units(
    filing_date: str | pd.Timestamp | None,
    *,
    schema_version: str | None = None,
    information_table_text: str | None = None,
) -> ValueUnits:
    """Prefer recognized schema, then actual filing date; always inspect VALUE captions.

    The report quarter never selects units. An unsupported schema, multiple captions or
    disagreement with prescribed units withholds dollars, while raw shares remain usable.
    """
    schema = (schema_version or "").strip().upper()
    factors = _caption_factors(information_table_text)
    stated = next(iter(factors)) if len(factors) == 1 else None
    if schema:
        if schema not in SEC_13F_SCHEMA_VALUE_FACTORS:
            return ValueUnits(None, stated, None, "unsupported_schema")
        expected = SEC_13F_SCHEMA_VALUE_FACTORS[schema]
        evidence = "schema:" + schema
    else:
        filed = pd.to_datetime(filing_date, errors="coerce")
        expected = None if pd.isna(filed) else 1.0 if filed.date() >= pd.Timestamp(SEC_13F_DOLLAR_UNITS_FROM).date() else 1000.0
        evidence = "filing_date" if expected is not None else "caption" if stated is not None else "unknown"
    if len(factors) > 1 or (expected is not None and stated is not None and expected != stated):
        return ValueUnits(expected, stated, None, "conflicting_caption")
    return ValueUnits(expected, stated, expected if expected is not None else stated, evidence)


def normalize_information_values(infotable: pd.DataFrame, units: ValueUnits) -> pd.DataFrame:
    """Normalize raw VALUE once, retaining raw values and every source share quantity."""
    if "raw_value" in infotable:
        raise ValueError("13F information-table values already normalized")
    out = infotable.copy()
    value = next((column for column in out if column.lower() == "value"), None)
    if value is None:
        raise ValueError("13F information table has no VALUE column")
    out["raw_value"] = pd.to_numeric(out[value], errors="coerce")
    out[value] = out["raw_value"] * units.factor if units.factor is not None else float("nan")
    return out
