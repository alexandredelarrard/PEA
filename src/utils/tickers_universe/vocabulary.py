"""
vocabulary.py  (src/utils/tickers_universe/vocabulary.py)
---------------------------------------------------------
The GICS vocabulary, once: the 11 sector and 25 industry-group names `tickers_universe` stores.

GICS in `tickers_universe` is today's classification projected backwards over the whole
history (no date axis), the same look-ahead the static peer basket already carries.
"""

from __future__ import annotations

GICS_SECTORS: frozenset[str] = frozenset(
    {
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    }
)

GICS_INDUSTRY_GROUPS: frozenset[str] = frozenset(
    {
        "Automobiles & Components",
        "Banks",
        "Capital Goods",
        "Commercial & Professional Services",
        "Consumer Discretionary Distribution & Retail",
        "Consumer Durables & Apparel",
        "Consumer Services",
        "Consumer Staples Distribution & Retail",
        "Energy",
        "Equity Real Estate Investment Trusts (REITs)",
        "Financial Services",
        "Food, Beverage & Tobacco",
        "Health Care Equipment & Services",
        "Household & Personal Products",
        "Insurance",
        "Materials",
        "Media & Entertainment",
        "Pharmaceuticals, Biotechnology & Life Sciences",
        "Real Estate Management & Development",
        "Semiconductors & Semiconductor Equipment",
        "Software & Services",
        "Technology Hardware & Equipment",
        "Telecommunication Services",
        "Transportation",
        "Utilities",
    }
)
