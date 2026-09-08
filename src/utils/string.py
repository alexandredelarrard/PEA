"""Small string normalisers shared across packages.

`pad_cik` lives here rather than in either package that needs it: `data_extract`
(fetch_superinvestors) writes the padded CIK, `data_aggregate` (superinvestor_features)
joins on it, and the two had grown byte-identical private copies precisely because
cross-importing between `src/` subfolders is not allowed. One definition means the
write side and the read side can never pad differently.

`clean_text` is here for the same reason one step removed: it is the whitespace half of
`src/utils/names.py`'s person key, which both packages now share. Leaving it in
`data_extract/.../def14a/validate.py` would have left `names.py` importing back into
`data_extract` -- the very cross-import the move exists to remove -- and inlining a second
copy of one regex substitution is how two normalisers drift apart. `validate.py` re-exports
it, so every extraction call site is unchanged.
"""
from typing import Any

import re

#: edgartools preserves the source HTML's whitespace runs verbatim, and a non-breaking space
#: is not `\s` to `str.strip` -- both have to go before any key is built on the value.
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
    """Canonical 10-digit zero-padded CIK (as stored in sp500_tickers /
    sec13f_hr / the superinvestor roster JSON). Tolerates ints, '123',
    '123.0' and already-padded strings; '' when there is no digit at all."""
    s = re.sub(r"\D", "", str(x).strip().split(".")[0])
    return s.zfill(10) if s else ""
