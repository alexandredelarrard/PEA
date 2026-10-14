"""13F VALUE units: source declarations, and the per-filing evidence that decides the cell-to-USD factor."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import SEC_13F_DOLLAR_UNITS_FROM, SEC_13F_SCHEMA_VALUE_FACTORS, SEC_13F_SOURCE_RATIO_BOUNDS

logger = logging.getLogger(__name__)

_TABLE = re.compile(r"<TABLE[^>]*>(.*?)</TABLE>", re.I | re.S)
_VALUE_CAPTION = re.compile(r"\bVALUE\b|x\s*\$", re.I)
# The multiplier ends at its last digit group, so `(X$1,000PRN AMOUNT` reads as x1000.
_MULTIPLIER = re.compile(r"(?:\bx\s*\$?\s*|\$\s*)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\d|,\d)", re.I)

#: Plausible median USD price per share of a book's share rows.
_PRICE_BAND = (1.0, 1000.0)
#: Fewest share rows whose median implied price outranks a stated unit; fewer rows still beat no evidence.
_MIN_PRICE_ROWS = 3
#: Evidence label of a unique price fit over fewer than `_MIN_PRICE_ROWS` share rows.
FEW_ROWS_PRICE = "implied_price:few_rows"
#: Fewest share rows whose median may read $millions cells from the price alone.
_MIN_MILLIONS_ROWS = 5
#: Evidence label of $millions cells read from the implied price alone.
MILLIONS_PRICE = "implied_price:millions"
#: Cell units the implied price chooses between; $millions only when another source says so.
_PRICE_FACTORS = (1.0, 1000.0)
_MILLIONS = 1e6
#: Unlabelled cover-total / cell-sum ratio -> cell factor: a $1000s cover over dollar cells, or a
#: dollar cover over $1000s or $millions cells. A ratio of 1 says nothing about the unit.
_RATIO_FACTORS = {1e-3: 1.0, 1e3: 1000.0, 1e6: 1e6}
#: A stated-unit cover converted to dollars over the cell sum is the cell factor itself.
_STATED_COVER_FACTORS = (1.0, 1000.0, 1e6)
#: Evidence label of the last resort: no evidence, cells stored as dollars.
NO_EVIDENCE = "no_evidence:assumed_x1"


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


def caption_factor(raw: str | None) -> float | None:
    """The single cell unit the VALUE captions state; NaN when they state several, None when none."""
    factors = _caption_factors(raw)
    return None if not factors else next(iter(factors)) if len(factors) == 1 else math.nan


def declared_units(filing_date: object, *, schema_version: str | None = None, information_table_text: str | None = None) -> ValueUnits:
    """Source declarations only: the SEC-prescribed unit (schema, else filing date) and the filer's own
    statement (a recognized XML schema, else the VALUE caption). Neither is a decided factor."""
    schema = (schema_version or "").strip().upper()
    if schema in SEC_13F_SCHEMA_VALUE_FACTORS:
        factor = SEC_13F_SCHEMA_VALUE_FACTORS[schema]
        return ValueUnits(factor, factor, None, "stated_schema")
    filed = pd.to_datetime(filing_date, errors="coerce")
    by_date = None if schema or pd.isna(filed) else 1.0 if filed.date() >= pd.Timestamp(SEC_13F_DOLLAR_UNITS_FROM).date() else 1000.0
    stated = caption_factor(information_table_text)
    return ValueUnits(by_date, stated, None, "stated_caption" if stated is not None else "")


def _within(ratio: float, target: float) -> bool:
    """True when `ratio` is `target` within the shared source ratio band."""
    lower, upper = SEC_13F_SOURCE_RATIO_BOUNDS
    return lower <= ratio / target <= upper


def cover_cell_factor(cell_total: float, cover_total: float | None, cover_unit: float | None) -> float | None:
    """The cell factor a cover total implies: with a stated cover unit, the dollar cover over the cell sum;
    unlabelled, `_RATIO_FACTORS`. None when the ratio says nothing (e.g. 1) or aligns with no unit."""
    if cover_total is None or not math.isfinite(cover_total) or cover_total <= 0 or cell_total <= 0:
        return None
    if cover_unit is not None:
        if not math.isfinite(cover_unit):
            return None
        ratio = cover_total * cover_unit / cell_total
        return next((factor for factor in _STATED_COVER_FACTORS if _within(ratio, factor)), None)
    ratio = cover_total / cell_total
    return next((factor for target, factor in _RATIO_FACTORS.items() if _within(ratio, target)), None)


def implied_price_factor(prices: pd.Series, *, millions: bool = False) -> tuple[float | None, float | None, int]:
    """`(factor, median raw VALUE/share, share rows)`: the one candidate factor that puts the median inside
    `_PRICE_BAND`, else None; the caller weighs it by the row count."""
    finite = pd.to_numeric(prices, errors="coerce").replace([math.inf, -math.inf], math.nan).dropna()
    if finite.empty:
        return None, None, 0
    median = float(finite.median())
    candidates = (*_PRICE_FACTORS, _MILLIONS) if millions else _PRICE_FACTORS
    fits = [factor for factor in candidates if _PRICE_BAND[0] <= median * factor <= _PRICE_BAND[1]]
    return (fits[0] if len(fits) == 1 else None), median, len(finite)


def decide_value_units(
    prices: pd.Series,
    declared: ValueUnits,
    *,
    cell_total: float,
    cover_total: float | None,
    cover_unit: float | None = None,
    accession: str = "",
) -> ValueUnits:
    """The cell-to-USD factor from per-filing evidence: the median implied share price over at least
    `_MIN_PRICE_ROWS` share rows (x1e6 from the price alone needs `_MIN_MILLIONS_ROWS` and a sub-dollar median at
    x1000), then the stated unit (caption or schema, and a stated-unit cover), then the
    unlabelled cover ratio, then a price fit over fewer rows, else x1 with a WARNING. Disagreeing evidence is
    logged and the price wins; the filing date never decides."""
    statements: dict[str, float] = {}
    conflicting = declared.stated_factor is not None and math.isnan(declared.stated_factor)
    if declared.stated_factor is not None and not conflicting:
        statements[declared.evidence or "stated_caption"] = declared.stated_factor
    if cover_unit is not None:
        conflicting = conflicting or math.isnan(cover_unit)
        from_cover = cover_cell_factor(cell_total, cover_total, cover_unit)
        if from_cover is not None:
            statements["stated_cover"] = from_cover
    ratio = cover_cell_factor(cell_total, cover_total, None) if cover_unit is None else None
    agreed = set(statements.values())
    stated = next(iter(agreed)) if len(agreed) == 1 and not conflicting else None
    factor, median, rows = implied_price_factor(prices, millions=_MILLIONS in {*agreed, ratio})
    price_evidence = "implied_price"
    if factor is None and rows >= _MIN_MILLIONS_ROWS and median is not None and median * 1000 < _PRICE_BAND[0]:
        # Sub-dollar at x1000 over enough rows: $millions cells when x1e6 is the one fit.
        factor = implied_price_factor(prices, millions=True)[0]
        price_evidence = MILLIONS_PRICE if factor == _MILLIONS else price_evidence
        factor = factor if factor == _MILLIONS else None
    recorded = declared.stated_factor if declared.stated_factor is not None and not math.isnan(declared.stated_factor) else stated

    if factor is not None and rows >= _MIN_PRICE_ROWS:
        others = {**statements, **({"cover_total_ratio": ratio} if ratio is not None else {})}
        for label, other in others.items():
            if other != factor:
                logger.warning("13F %s: %s says x%g but the median implied price %.4g/share picks x%g", accession, label, other, median, factor)
        return ValueUnits(declared.expected_factor, recorded, factor, price_evidence)
    if factor is None and rows >= _MIN_PRICE_ROWS:
        logger.warning("13F %s: median raw VALUE/share %.4g over %s share rows fits no unit in %s", accession, median, rows, _PRICE_BAND)
    if stated is not None:
        return ValueUnits(
            declared.expected_factor, recorded, stated, next((label for label in statements if label != "stated_cover"), "stated_cover")
        )
    if statements or conflicting:
        logger.warning("13F %s: stated units conflict (%s); not used", accession, statements or "several captions or cover units")
    if ratio is not None:
        return ValueUnits(declared.expected_factor, recorded, ratio, "cover_total_ratio")
    if factor is not None:
        logger.warning("13F %s: unit x%g from the implied price over only %s share row(s) (median %.4g/share)", accession, factor, rows, median)
        return ValueUnits(declared.expected_factor, recorded, factor, FEW_ROWS_PRICE)
    if cell_total > 0:
        logger.warning(
            "13F %s: no unit evidence (%s share rows, no stated unit, no usable cover ratio); cells stored as dollars (x1)", accession, rows
        )
    return ValueUnits(declared.expected_factor, recorded, 1.0, NO_EVIDENCE)
