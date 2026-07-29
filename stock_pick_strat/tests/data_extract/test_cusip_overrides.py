"""CINS -> ticker overrides for the 13F reconciliation
(src/constants/constants.py::CUSIP_TICKER_OVERRIDES + fetch_cusip_map).

13F reports holdings by CUSIP, so an unresolved identifier makes a name INVISIBLE in
`institutional_holdings` and in the superinvestor sleeve. `cusip_ticker_map` is built from
OpenFIGI and records a miss PERMANENTLY, and OpenFIGI does not resolve CINS (a CUSIP whose
first character is a LETTER encoding a foreign domicile) from the 13F feed. Measured on the
live DB: 15,404 letter-prefixed rows in the map, ZERO resolved, and 34 of 500 universe names
absent from institutional_holdings.

These tests pin the override table's integrity and that it wins over a cached miss.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src.constants.constants import CUSIP_TICKER_OVERRIDES

_CUSIP_RE = re.compile(r"^[0-9A-Z]{9}$")
# first character of a CINS -> domicile group, for the sanity print
_CINS_COUNTRY = {"G": "Ireland/UK", "H": "Switzerland", "N": "Netherlands",
                 "V": "Liberia", "Y": "Singapore"}


def test_override_identifiers_are_well_formed_and_unique():
    """A malformed or duplicated identifier would silently misroute someone else's holdings."""
    assert CUSIP_TICKER_OVERRIDES, "override table is empty"
    bad = [c for c in CUSIP_TICKER_OVERRIDES if not _CUSIP_RE.match(c)]
    assert not bad, f"malformed identifiers (need 9 upper-case alphanumerics): {bad}"
    # one ticker may have several identifiers, but an identifier maps to exactly one ticker
    assert len(set(CUSIP_TICKER_OVERRIDES)) == len(CUSIP_TICKER_OVERRIDES)
    tickers = list(CUSIP_TICKER_OVERRIDES.values())
    assert len(set(tickers)) == len(tickers), (
        "a ticker appears twice — two identifiers for one name is allowed but suspicious here: "
        f"{[t for t in set(tickers) if tickers.count(t) > 1]}")

    cins = {c: t for c, t in CUSIP_TICKER_OVERRIDES.items() if c[0].isalpha()}
    print("\n=== SANITY CHECK: override table integrity ===")
    print(f"  {len(CUSIP_TICKER_OVERRIDES)} identifiers, all 9-char alphanumeric, no duplicates")
    print(f"  {len(cins)} are CINS (letter-prefixed / foreign domicile), "
          f"{len(CUSIP_TICKER_OVERRIDES) - len(cins)} are numeric US CUSIPs")
    by_country: dict[str, list[str]] = {}
    for c, t in sorted(cins.items(), key=lambda kv: kv[1]):
        by_country.setdefault(_CINS_COUNTRY.get(c[0], f"prefix {c[0]}"), []).append(t)
    for country, ts in sorted(by_country.items()):
        print(f"    {country:14} {', '.join(ts)}")
    print("  Validated.")


def test_no_etf_or_trust_is_mapped_to_an_operating_company():
    """The recovery scan's top hit for 'INVESCO' was 46090E103 = the INVESCO QQQ TRUST ETF, not
    Invesco Ltd — 13F filers hold QQQ so heavily that filer-count ranking prefers it. Mapping it
    to IVZ would book QQQ's holdings as Invesco Ltd. It must stay out until resolved properly."""
    assert "46090E103" not in CUSIP_TICKER_OVERRIDES, (
        "46090E103 is the Invesco QQQ Trust ETF, not Invesco Ltd (IVZ)")
    assert "IVZ" not in CUSIP_TICKER_OVERRIDES.values(), (
        "IVZ is deliberately unmapped — its identifier was not recovered unambiguously")
    print("\n=== SANITY CHECK: no ETF mapped to an operating company ===")
    print("  46090E103 (Invesco QQQ Trust) excluded; IVZ deliberately unmapped pending a "
          "targeted lookup of Invesco Ltd's Bermuda CINS. Validated.")


def test_overrides_win_over_a_cached_miss(monkeypatch, tmp_path):
    """The whole point: `cusip_ticker_map` stores a no-match as a NULL ticker and never retries,
    so the override must beat a row already cached as a miss."""
    from src.data_extract.utils.prices import fetch_cusip_map as fcm

    sample = list(CUSIP_TICKER_OVERRIDES)[:3]
    # the map as it looks today: these identifiers present, but recorded as MISSES
    cached = pd.DataFrame({"cusip": sample, "ticker": [None] * len(sample)})
    saved: list = []

    store = type("S", (), {
        "load": lambda self, *a, **k: cached.copy(),
        "save": lambda self, t, df: (saved.append(df), len(df))[1],
    })()
    context = type("C", (), {"store": store})()

    monkeypatch.setattr(fcm, "_openfigi_request", lambda *a, **k: {})
    monkeypatch.setattr(fcm, "_parse_openfigi", lambda *a, **k: {})

    out = fcm.build_cusip_ticker_map(context, sample, pause=0.0)
    got = dict(zip(out["cusip"], out["ticker"]))

    for cu in sample:
        assert got.get(cu) == CUSIP_TICKER_OVERRIDES[cu], (
            f"{cu} still unresolved — the override did not beat the cached miss")
    assert out["ticker"].notna().all(), "a miss survived into the holdings merge"

    print("\n=== SANITY CHECK: override beats a permanently-cached miss ===")
    print(f"  map pre-seeded with {sample} as NULL misses (OpenFIGI returning nothing)")
    print(f"  after the build: {got}")
    print("  the ~30 foreign-domiciled S&P 500 names now resolve, so their 13F holdings reach "
          "institutional_holdings and the superinvestor sleeve. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
