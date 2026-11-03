"""
wikipedia_bootstrap.py  (src/utils/tickers_universe/wikipedia_bootstrap.py)
---------------------------------------------------------------------------
Name, CIK and GICS for a ticker entering `tickers_universe` with no stored row, read from
Wikipedia's S&P 500 table. Wikipedia gives the GICS sub-industry but not the industry group,
so the canonical GICS 2023 hierarchy below maps one to the other (case/punctuation-insensitive;
an unknown sub-industry falls back to the sector).

Delete this module when Sharadar categories replace GICS.
"""

from __future__ import annotations

import io
import re
from typing import cast

import pandas as pd
import requests

from src.constants.constants import BROWSER_HEADERS
from src.utils.string import pad_cik_series, yahoo_symbol

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_COLUMNS = {"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry", "CIK": "cik"}

# industry group -> its sub-industries (GICS 2023, S&P 500 coverage)
_GROUP_TO_SUBS: dict[str, list[str]] = {
    "Energy": [
        "Oil & Gas Drilling",
        "Oil & Gas Equipment & Services",
        "Integrated Oil & Gas",
        "Oil & Gas Exploration & Production",
        "Oil & Gas Refining & Marketing",
        "Oil & Gas Storage & Transportation",
        "Coal & Consumable Fuels",
    ],
    "Materials": [
        "Commodity Chemicals",
        "Diversified Chemicals",
        "Fertilizers & Agricultural Chemicals",
        "Industrial Gases",
        "Specialty Chemicals",
        "Construction Materials",
        "Metal, Glass & Plastic Containers",
        "Paper & Plastic Packaging Products & Materials",
        "Aluminum",
        "Diversified Metals & Mining",
        "Copper",
        "Gold",
        "Precious Metals & Minerals",
        "Silver",
        "Steel",
        "Forest Products",
        "Paper Products",
    ],
    "Capital Goods": [
        "Aerospace & Defense",
        "Building Products",
        "Construction & Engineering",
        "Electrical Components & Equipment",
        "Heavy Electrical Equipment",
        "Industrial Conglomerates",
        "Construction Machinery & Heavy Transportation Equipment",
        "Agricultural & Farm Machinery",
        "Industrial Machinery & Supplies & Components",
        "Trading Companies & Distributors",
    ],
    "Commercial & Professional Services": [
        "Commercial Printing",
        "Environmental & Facilities Services",
        "Office Services & Supplies",
        "Diversified Support Services",
        "Security & Alarm Services",
        "Human Resource & Employment Services",
        "Research & Consulting Services",
        "Data Processing & Outsourced Services",
    ],
    "Transportation": [
        "Air Freight & Logistics",
        "Passenger Airlines",
        "Airlines",
        "Marine Transportation",
        "Rail Transportation",
        "Cargo Ground Transportation",
        "Passenger Ground Transportation",
        "Airport Services",
        "Highways & Railtracks",
        "Marine Ports & Services",
    ],
    "Automobiles & Components": [
        "Automotive Parts & Equipment",
        "Tires & Rubber",
        "Automobile Manufacturers",
        "Motorcycle Manufacturers",
        "Automobiles",
    ],
    "Consumer Durables & Apparel": [
        "Consumer Electronics",
        "Home Furnishings",
        "Homebuilding",
        "Household Appliances",
        "Housewares & Specialties",
        "Leisure Products",
        "Apparel, Accessories & Luxury Goods",
        "Footwear",
        "Textiles",
    ],
    "Consumer Services": [
        "Casinos & Gaming",
        "Hotels, Resorts & Cruise Lines",
        "Leisure Facilities",
        "Restaurants",
        "Education Services",
        "Specialized Consumer Services",
    ],
    "Consumer Discretionary Distribution & Retail": [
        "Distributors",
        "Broadline Retail",
        "Internet & Direct Marketing Retail",
        "Apparel Retail",
        "Computer & Electronics Retail",
        "Home Improvement Retail",
        "Other Specialty Retail",
        "Automotive Retail",
        "Homefurnishing Retail",
    ],
    "Consumer Staples Distribution & Retail": ["Drug Retail", "Food Distributors", "Food Retail", "Consumer Staples Merchandise Retail"],
    "Food, Beverage & Tobacco": [
        "Brewers",
        "Distillers & Vintners",
        "Soft Drinks & Non-alcoholic Beverages",
        "Agricultural Products & Services",
        "Packaged Foods & Meats",
        "Tobacco",
    ],
    "Household & Personal Products": ["Household Products", "Personal Care Products", "Personal Products"],
    "Health Care Equipment & Services": [
        "Health Care Equipment",
        "Health Care Supplies",
        "Health Care Distributors",
        "Health Care Services",
        "Health Care Facilities",
        "Managed Health Care",
        "Health Care Technology",
    ],
    "Pharmaceuticals, Biotechnology & Life Sciences": ["Biotechnology", "Pharmaceuticals", "Life Sciences Tools & Services"],
    "Banks": ["Diversified Banks", "Regional Banks", "Commercial & Residential Mortgage Finance"],
    "Financial Services": [
        "Diversified Financial Services",
        "Multi-Sector Holdings",
        "Specialized Finance",
        "Consumer Finance",
        "Transaction & Payment Processing Services",
        "Asset Management & Custody Banks",
        "Investment Banking & Brokerage",
        "Diversified Capital Markets",
        "Financial Exchanges & Data",
    ],
    "Insurance": ["Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance", "Property & Casualty Insurance", "Reinsurance"],
    "Software & Services": ["IT Consulting & Other Services", "Internet Services & Infrastructure", "Application Software", "Systems Software"],
    "Technology Hardware & Equipment": [
        "Communications Equipment",
        "Technology Hardware, Storage & Peripherals",
        "Electronic Equipment & Instruments",
        "Electronic Components",
        "Electronic Manufacturing Services",
        "Technology Distributors",
    ],
    "Semiconductors & Semiconductor Equipment": ["Semiconductor Materials & Equipment", "Semiconductors"],
    "Telecommunication Services": ["Alternative Carriers", "Integrated Telecommunication Services", "Wireless Telecommunication Services"],
    "Media & Entertainment": [
        "Advertising",
        "Broadcasting",
        "Cable & Satellite",
        "Publishing",
        "Movies & Entertainment",
        "Interactive Home Entertainment",
        "Interactive Media & Services",
    ],
    "Real Estate Management & Development": [
        "Diversified Real Estate Activities",
        "Real Estate Operating Companies",
        "Real Estate Development",
        "Real Estate Services",
    ],
    "Equity Real Estate Investment Trusts (REITs)": [
        "Diversified REITs",
        "Industrial REITs",
        "Hotel & Resort REITs",
        "Office REITs",
        "Health Care REITs",
        "Residential REITs",
        "Retail REITs",
        "Specialized REITs",
        "Telecom Tower REITs",
        "Timber REITs",
        "Data Center REITs",
        "Single-Family Residential REITs",
        "Multi-Family Residential REITs",
        "Self-Storage REITs",
        "Other Specialized REITs",
    ],
    "Utilities": [
        "Electric Utilities",
        "Gas Utilities",
        "Multi-Utilities",
        "Water Utilities",
        "Independent Power Producers & Energy Traders",
        "Renewable Electricity",
    ],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_SUB_TO_GROUP = {_norm(sub): grp for grp, subs in _GROUP_TO_SUBS.items() for sub in subs}


def industry_group(sub_industry: str, sector: str | None = None) -> str:
    """GICS industry group of a sub-industry; the sector (or the raw sub-industry) when it is not in the table."""
    grp = _SUB_TO_GROUP.get(_norm(sub_industry))
    if grp is not None:
        return grp
    return sector if sector else str(sub_industry)


def parse_sp500_table(html: str) -> pd.DataFrame:
    """The first table of the Wikipedia page as `ticker, name, cik, sector, sub_industry, industry_group`,
    tickers in yfinance spelling and CIKs padded to 10 digits."""
    df = cast(pd.DataFrame, pd.read_html(io.StringIO(html))[0]).rename(columns=WIKIPEDIA_COLUMNS)
    df["ticker"] = df["ticker"].map(yahoo_symbol)
    df["cik"] = pad_cik_series(df["cik"])
    df["industry_group"] = [industry_group(sub, sector) for sub, sector in zip(df["sub_industry"], df["sector"], strict=True)]
    return cast(pd.DataFrame, df[["ticker", "name", "cik", "sector", "sub_industry", "industry_group"]])


def wikipedia_sp500() -> pd.DataFrame:
    """Fetch and parse Wikipedia's current S&P 500 constituents table."""
    response = requests.get(WIKIPEDIA_SP500_URL, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    return parse_sp500_table(response.text)
