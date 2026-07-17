"""Tests for the management/ownership snapshot parsers
(src/data_extract/fetch_management.py).

The snapshot itself is not historical, so what matters is that the PARSING of
Yahoo's messy officer / roster / insider-purchase structures is correct and
robust: founder & CEO detection from titles, the family-ownership proxy from
the insider roster, and net-insider-buying extraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.structure.fetch_management import (
    _parse_officers, _parse_family, _parse_insider_net,
)


# ---- officer titles copied verbatim from the yfinance probe --------------- #
_META_OFFICERS = [
    {"name": "Mr. Mark Elliot Zuckerberg", "title": "Founder, Chairman & CEO",
     "age": 41, "yearBorn": 1984, "totalPay": 25125904},
    {"name": "Ms. Susan J. S. Li", "title": "Chief Financial Officer",
     "age": 39, "yearBorn": 1986, "totalPay": 3351064},
]
_AMD_OFFICERS = [
    {"name": "Dr. Lisa T. Su Ph.D.", "title": "Chair, President & CEO",
     "age": 55, "yearBorn": 1970, "totalPay": 4540876},
    {"name": "Ms. Jean X. Hu Ph.D.", "title": "Executive VP, CFO & Treasurer",
     "age": 62, "yearBorn": 1963, "totalPay": 1976642},
]


def test_officer_parsing_founder_ceo_and_aggregates():
    meta = _parse_officers(_META_OFFICERS)
    amd = _parse_officers(_AMD_OFFICERS)

    # founder-CEO detected only when the CEO's own title says "Founder"
    assert meta["founder_ceo"] == 1 and meta["founder_present"] == 1
    assert amd["founder_ceo"] == 0 and amd["founder_present"] == 0

    # CEO attributes come from the CEO row, not the CFO
    assert meta["ceo_age"] == 41 and meta["ceo_pay"] == 25125904
    assert amd["ceo_age"] == 55

    # aggregates
    assert meta["n_officers"] == 2
    assert abs(meta["avg_officer_age"] - 40.0) < 1e-9
    assert meta["total_officer_pay"] == 25125904 + 3351064
    assert _parse_officers([]) ["founder_present"] == 0  # empty-safe

    print("\n=== SANITY CHECK: officer / founder parsing ===")
    print(f"  META: founder-CEO={meta['founder_ceo']} (Zuckerberg 'Founder, Chairman & CEO'), "
          f"ceo_age={meta['ceo_age']}")
    print(f"  AMD : founder-CEO={amd['founder_ceo']} (Su is not a founder) -> correctly 0")
    print("  Founder flag keys off the CEO's title; aggregates match. Validated.")


def _wmt_roster():
    # WMT insider roster: five Waltons + a family trust, several >10% owners
    return pd.DataFrame({
        "Name": ["FURNER JOHN R", "WALTON ALICE L", "WALTON FAMILY HOLDINGS TRUST",
                 "WALTON JAMES CARR", "WALTON JIM C", "WALTON S ROBSON"],
        "Position": ["Chief Executive Officer",
                     "Beneficial Owner of more than 10% of a Class of Security",
                     "Beneficial Owner of more than 10% of a Class of Security",
                     "Beneficial Owner of more than 10% of a Class of Security",
                     "Beneficial Owner of more than 10% of a Class of Security",
                     "Beneficial Owner of more than 10% of a Class of Security"],
    })


def _amd_roster():
    return pd.DataFrame({
        "Name": ["SU LISA T", "HU JEAN X", "PAPERMASTER MARK D", "NORROD FORREST EUGENE"],
        "Position": ["Chief Executive Officer", "Chief Financial Officer",
                     "Chief Technology Officer", "Officer"],
    })


def test_family_ownership_proxy():
    wmt = _parse_family(_wmt_roster(), held_insiders=0.4485)
    amd = _parse_family(_amd_roster(), held_insiders=0.00397)

    # WMT: repeated surname + family trust + high insider stake -> family-owned
    assert wmt["family_owned"] == 1
    assert wmt["family_trust_present"] == 1
    assert wmt["max_surname_repeat"] >= 4        # 4 distinct Waltons (trust excluded)
    assert wmt["n_beneficial_owners"] == 5

    # AMD: distinct officers, tiny insider stake -> not family-owned
    assert amd["family_owned"] == 0
    assert amd["max_surname_repeat"] == 1

    # guard: repeated surname but no insider stake must NOT flag family-owned
    weak = _parse_family(_wmt_roster(), held_insiders=0.01)
    assert weak["family_owned"] == 0

    print("\n=== SANITY CHECK: family-ownership proxy ===")
    print(f"  WMT: surname_repeat={wmt['max_surname_repeat']} (Waltons), "
          f"trust={wmt['family_trust_present']}, insider=44.9% -> family_owned=1")
    print(f"  AMD: surname_repeat={amd['max_surname_repeat']}, insider=0.4% -> family_owned=0")
    print("  Requires BOTH a family cluster AND a >=10% insider stake. Validated.")


def test_insider_net_buying_extraction():
    purchases = pd.DataFrame({
        "Insider Purchases Last 6m": [
            "Purchases", "Sales", "Net Shares Purchased (Sold)",
            "Total Insider Shares Held", "% Net Shares Purchased (Sold)",
            "% Buy Shares", "% Sell Shares"],
        "Shares": [840531.0, 640700.0, 199831.0, 6473484.0, 0.032, 0.134, 0.102],
    })
    assert abs(_parse_insider_net(purchases) - 0.032) < 1e-9

    # missing / empty inputs are NaN-safe (META reports no net row)
    assert np.isnan(_parse_insider_net(None))
    assert np.isnan(_parse_insider_net(pd.DataFrame()))

    print("\n=== SANITY CHECK: net insider buying ===")
    print("  Extracted '% Net Shares Purchased (Sold)' = +3.2% for AMD; None/empty -> NaN.")
    print("  Robust to Yahoo's label-in-first-column layout. Validated.")
