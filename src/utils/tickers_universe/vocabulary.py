"""
vocabulary.py  (src/utils/tickers_universe/vocabulary.py)
---------------------------------------------------------
The GICS vocabulary, once: the 11 sector and 25 industry-group names `tickers_universe` stores,
and each group's sub-industries (the Wikipedia bootstrap maps a sub-industry to its group).

GICS in `tickers_universe` is today's classification projected backwards over the whole
history (no date axis), the same look-ahead the static peer basket already carries.
"""

from __future__ import annotations

# Names the sector-KPI scopes key on, or that are both a sector and a group; declared here only.
ENERGY = "Energy"
FINANCIALS = "Financials"
UTILITIES = "Utilities"
MATERIALS = "Materials"
BANKS = "Banks"
INSURANCE = "Insurance"
EQUITY_REITS = "Equity Real Estate Investment Trusts (REITs)"
PHARMA_BIOTECH_LIFE_SCIENCES = "Pharmaceuticals, Biotechnology & Life Sciences"

GICS_SECTORS: frozenset[str] = frozenset(
    {
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        ENERGY,
        FINANCIALS,
        "Health Care",
        "Industrials",
        "Information Technology",
        MATERIALS,
        "Real Estate",
        UTILITIES,
    }
)

# industry group -> its sub-industries (GICS 2023, S&P 500 coverage)
GICS_SUB_INDUSTRIES: dict[str, list[str]] = {
    ENERGY: [
        "Oil & Gas Drilling",
        "Oil & Gas Equipment & Services",
        "Integrated Oil & Gas",
        "Oil & Gas Exploration & Production",
        "Oil & Gas Refining & Marketing",
        "Oil & Gas Storage & Transportation",
        "Coal & Consumable Fuels",
    ],
    MATERIALS: [
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
    PHARMA_BIOTECH_LIFE_SCIENCES: ["Biotechnology", "Pharmaceuticals", "Life Sciences Tools & Services"],
    BANKS: ["Diversified Banks", "Regional Banks", "Commercial & Residential Mortgage Finance"],
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
    INSURANCE: ["Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance", "Property & Casualty Insurance", "Reinsurance"],
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
    EQUITY_REITS: [
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
    UTILITIES: [
        "Electric Utilities",
        "Gas Utilities",
        "Multi-Utilities",
        "Water Utilities",
        "Independent Power Producers & Energy Traders",
        "Renewable Electricity",
    ],
}

GICS_INDUSTRY_GROUPS: frozenset[str] = frozenset(GICS_SUB_INDUSTRIES)
