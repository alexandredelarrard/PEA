"""Small string normalisers shared across packages.

`pad_cik` / `pad_cik_series` are the one CIK spelling every package writes and joins on;
`normalise_ticker` is the one ticker spelling, `yahoo_symbol` Yahoo's spelling of a share class and `squash` the
separator-free key two symbol spellings are compared on; `clean_text` is the whitespace half of the
person key in `src/utils/names.py`.
"""

import re
from typing import Any, cast

import pandas as pd

#: Whitespace runs edgartools preserves from the source HTML; non-breaking spaces are replaced first.
_WHITESPACE_RE = re.compile(r"\s+")


def camel_to_snake(x: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()


def clean_text(value: Any) -> str | None:
    """Collapse the whitespace runs edgartools preserves from the source HTML
    ("Free                cash flow" -> "Free cash flow"). None for empty."""
    if value is None or not isinstance(value, str):
        return None
    cleaned = _WHITESPACE_RE.sub(" ", value.replace("\xa0", " ")).strip()
    return cleaned or None


def pad_cik(x: object) -> str:
    """Canonical 10-digit zero-padded CIK. Tolerates ints, '123', '123.0' and already-padded strings;
    '' when there is no digit at all."""
    s = re.sub(r"\D", "", str(x).strip().split(".")[0])
    return s.zfill(10) if s else ""


def pad_cik_series(values: pd.Series) -> pd.Series:
    """Vectorised `pad_cik`: the same string per element ('' for a null), object dtype, index kept."""
    digits = values.astype("string").str.strip().str.split(".", n=1).str[0].str.replace(r"\D", "", regex=True).fillna("")
    return digits.str.zfill(10).where(digits != "", "").astype(object)


def normalise_ticker(value: object) -> str:
    """Canonical ticker spelling: stripped and upper-case."""
    return str(value).strip().upper()


def yahoo_symbol(symbol: object) -> str:
    """Yahoo's spelling of a share-class symbol: a `/` or `.` class separator becomes `-` (`BRK/A`, `BRK.A` -> `BRK-A`)."""
    return normalise_ticker(symbol).replace("/", "-").replace(".", "-")


def squash(symbol: object) -> str:
    """The comparison key of a symbol: upper case without class separators (`BRK-B`, `BRK/B`, `BRK.B` -> `BRKB`)."""
    if symbol is None or (not isinstance(symbol, str) and pd.isna(cast(Any, symbol))):
        return ""
    return re.sub(r"[\s\-./]", "", str(symbol)).upper()
