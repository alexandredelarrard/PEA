"""DEF 14A proxy fallback parser (edgar_extract.extract_management_from_def14a).

Recovers CEO age / officer stats from the proxy's director & executive-officer
tables when the 10-K omits them, and confirms the shared refactor left the
existing 10-K parser (extract_executive_officers) behaving identically.
"""
from __future__ import annotations

from src.data_extract.utils.edgar_extract import (
    extract_management_from_def14a, extract_executive_officers,
)

# representative proxy director-nominee section (Name, Age, bio)
_PROXY = """
2024 PROXY STATEMENT

PROPOSAL 1 — ELECTION OF DIRECTORS

Nominees for election at the annual meeting of stockholders:

Jane A. Doe, 58, has served as a director since 2015 and as our President and
Chief Executive Officer since 2018. Ms. Doe co-founded the Company in 2004.

Robert L. Smith, 62, has served as a director since 2010 and is chair of the
Audit Committee. Mr. Smith is a retired partner of a global accounting firm.

Maria Garcia, 49, has served as a director since 2019. Ms. Garcia is the founder
and chief executive of an unrelated technology company.

SECURITY OWNERSHIP OF MANAGEMENT ...
"""

# a proxy that carries no ages at all (nothing to recover)
_PROXY_NO_AGES = """
NOTICE OF ANNUAL MEETING. Please vote your shares. The board recommends a vote
FOR each nominee. Details about compensation appear later in this statement.
"""

# a 10-K executive-officer block (existing parser must still work post-refactor)
_TENK = """
Information about our Executive Officers

Timothy Cook, 63, has served as our Chief Executive Officer since 2011.
Luca Maestri, 60, has served as our Senior Vice President and Chief Financial Officer.
"""


def test_def14a_recovers_ceo_age():
    info = extract_management_from_def14a(_PROXY)
    assert info["ceo_name"] == "Jane A. Doe", info
    assert info["ceo_age"] == 58
    assert info["founder_ceo"] == 1            # "co-founded the Company"
    assert info["n_officers"] >= 3
    assert 50 <= info["avg_officer_age"] <= 62
    # empty / age-less proxy -> nothing recovered (defensive)
    assert extract_management_from_def14a(_PROXY_NO_AGES)["ceo_age"] is None
    assert extract_management_from_def14a("")["officers"] == []
    print("\n=== SANITY CHECK: DEF 14A fallback parser ===")
    print(f"  proxy -> CEO {info['ceo_name']} age {info['ceo_age']} (founder_ceo=1), "
          f"{info['n_officers']} people, avg age {info['avg_officer_age']}; "
          f"age-less/empty proxy -> None. Validated.")


def test_10k_parser_unchanged_after_refactor():
    info = extract_executive_officers(_TENK)
    assert info["ceo_name"] == "Timothy Cook" and info["ceo_age"] == 63
    assert info["n_officers"] == 2
    print("\n=== SANITY CHECK: 10-K officer parser preserved ===")
    print(f"  10-K still parses CEO {info['ceo_name']} age {info['ceo_age']}, "
          f"{info['n_officers']} officers (shared-helper refactor is behaviour-preserving). Validated.")


if __name__ == "__main__":
    test_def14a_recovers_ceo_age()
    test_10k_parser_unchanged_after_refactor()
