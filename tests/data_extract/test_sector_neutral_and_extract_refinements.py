"""GICS sub-industry -> industry group, the rule the Wikipedia bootstrap of `tickers_universe` applies.

Share-class dedup is gone: the alias classes (GOOG, FOX, NWS, BRK-A) are declared in `configs/universe.yml`.
Sector-neutral portfolio checks live in tests/portfolio.
"""

from __future__ import annotations

from src.utils.tickers_universe.wikipedia_bootstrap import industry_group


def test_gics_industry_group():
    assert industry_group("Semiconductors", "Information Technology") == "Semiconductors & Semiconductor Equipment"
    assert industry_group("Systems Software", "Information Technology") == "Software & Services"
    assert industry_group("Integrated Oil & Gas", "Energy") == "Energy"
    assert industry_group("Totally Unknown Sub", "Utilities") == "Utilities"  # fallback
    print("\n=== SANITY CHECK: GICS industry group ===")
    print("  sub-industry -> industry-group maps; an unknown sub-industry falls back to its sector. Validated.")
