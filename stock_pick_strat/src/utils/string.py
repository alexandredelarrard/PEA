"""Small string normalisers shared across packages.

`pad_cik` lives here rather than in either package that needs it: `data_extract`
(fetch_superinvestors) writes the padded CIK, `data_aggregate` (superinvestor_features)
joins on it, and the two had grown byte-identical private copies precisely because
cross-importing between `src/` subfolders is not allowed. One definition means the
write side and the read side can never pad differently.
"""
import re


def camel_to_snake(x: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()


def pad_cik(x: object) -> str:
    """Canonical 10-digit zero-padded CIK (as stored in sp500_tickers /
    institutional_holdings / the superinvestor roster JSON). Tolerates ints, '123',
    '123.0' and already-padded strings; '' when there is no digit at all."""
    s = re.sub(r"\D", "", str(x).strip().split(".")[0])
    return s.zfill(10) if s else ""
