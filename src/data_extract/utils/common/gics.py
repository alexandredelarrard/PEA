"""
gics.py  (src/data_extract/utils/common/gics.py)
-----------------------------------------
Map a GICS **Sub-Industry** (the finest level, ~150, which Wikipedia gives) up to
its **Industry Group** (the 24-level, which Wikipedia does NOT give) so the
portfolio can be neutralized at the industry-group level top funds use.

The mapping is the canonical GICS 2023 hierarchy. Keys are matched case/space/
punctuation-insensitively (`_norm`); a Sub-Industry not in the table (GICS revisions
/ Wikipedia string drift) FALLS BACK to the GICS Sector, so neutralization always
has a valid group (coarser for the unmapped tail, never missing).
"""
from __future__ import annotations

import re

# industry group -> its sub-industries (GICS 2023, S&P 500 coverage)
_GROUP_TO_SUBS: dict[str, list[str]] = {
    "Energy": ["Oil & Gas Drilling", "Oil & Gas Equipment & Services",
               "Integrated Oil & Gas", "Oil & Gas Exploration & Production",
               "Oil & Gas Refining & Marketing", "Oil & Gas Storage & Transportation",
               "Coal & Consumable Fuels"],
    "Materials": ["Commodity Chemicals", "Diversified Chemicals", "Fertilizers & Agricultural Chemicals",
                  "Industrial Gases", "Specialty Chemicals", "Construction Materials",
                  "Metal, Glass & Plastic Containers", "Paper & Plastic Packaging Products & Materials",
                  "Aluminum", "Diversified Metals & Mining", "Copper", "Gold", "Precious Metals & Minerals",
                  "Silver", "Steel", "Forest Products", "Paper Products"],
    "Capital Goods": ["Aerospace & Defense", "Building Products", "Construction & Engineering",
                      "Electrical Components & Equipment", "Heavy Electrical Equipment",
                      "Industrial Conglomerates", "Construction Machinery & Heavy Transportation Equipment",
                      "Agricultural & Farm Machinery", "Industrial Machinery & Supplies & Components",
                      "Trading Companies & Distributors"],
    "Commercial & Professional Services": ["Commercial Printing", "Environmental & Facilities Services",
                                           "Office Services & Supplies", "Diversified Support Services",
                                           "Security & Alarm Services", "Human Resource & Employment Services",
                                           "Research & Consulting Services", "Data Processing & Outsourced Services"],
    "Transportation": ["Air Freight & Logistics", "Passenger Airlines", "Airlines", "Marine Transportation",
                       "Rail Transportation", "Cargo Ground Transportation", "Passenger Ground Transportation",
                       "Airport Services", "Highways & Railtracks", "Marine Ports & Services"],
    "Automobiles & Components": ["Automotive Parts & Equipment", "Tires & Rubber",
                                 "Automobile Manufacturers", "Motorcycle Manufacturers", "Automobiles"],
    "Consumer Durables & Apparel": ["Consumer Electronics", "Home Furnishings", "Homebuilding",
                                    "Household Appliances", "Housewares & Specialties", "Leisure Products",
                                    "Apparel, Accessories & Luxury Goods", "Footwear", "Textiles"],
    "Consumer Services": ["Casinos & Gaming", "Hotels, Resorts & Cruise Lines", "Leisure Facilities",
                          "Restaurants", "Education Services", "Specialized Consumer Services"],
    "Consumer Discretionary Distribution & Retail": ["Distributors", "Broadline Retail", "Internet & Direct Marketing Retail",
                                                     "Apparel Retail", "Computer & Electronics Retail", "Home Improvement Retail",
                                                     "Other Specialty Retail", "Automotive Retail", "Homefurnishing Retail"],
    "Consumer Staples Distribution & Retail": ["Drug Retail", "Food Distributors", "Food Retail",
                                               "Consumer Staples Merchandise Retail"],
    "Food, Beverage & Tobacco": ["Brewers", "Distillers & Vintners", "Soft Drinks & Non-alcoholic Beverages",
                                 "Agricultural Products & Services", "Packaged Foods & Meats", "Tobacco"],
    "Household & Personal Products": ["Household Products", "Personal Care Products", "Personal Products"],
    "Health Care Equipment & Services": ["Health Care Equipment", "Health Care Supplies",
                                         "Health Care Distributors", "Health Care Services",
                                         "Health Care Facilities", "Managed Health Care", "Health Care Technology"],
    "Pharmaceuticals, Biotechnology & Life Sciences": ["Biotechnology", "Pharmaceuticals",
                                                       "Life Sciences Tools & Services"],
    "Banks": ["Diversified Banks", "Regional Banks", "Commercial & Residential Mortgage Finance"],
    "Financial Services": ["Diversified Financial Services", "Multi-Sector Holdings", "Specialized Finance",
                           "Consumer Finance", "Transaction & Payment Processing Services",
                           "Asset Management & Custody Banks", "Investment Banking & Brokerage",
                           "Diversified Capital Markets", "Financial Exchanges & Data"],
    "Insurance": ["Insurance Brokers", "Life & Health Insurance", "Multi-line Insurance",
                  "Property & Casualty Insurance", "Reinsurance"],
    "Software & Services": ["IT Consulting & Other Services", "Internet Services & Infrastructure",
                            "Application Software", "Systems Software"],
    "Technology Hardware & Equipment": ["Communications Equipment", "Technology Hardware, Storage & Peripherals",
                                        "Electronic Equipment & Instruments", "Electronic Components",
                                        "Electronic Manufacturing Services", "Technology Distributors"],
    "Semiconductors & Semiconductor Equipment": ["Semiconductor Materials & Equipment", "Semiconductors"],
    "Telecommunication Services": ["Alternative Carriers", "Integrated Telecommunication Services",
                                   "Wireless Telecommunication Services"],
    "Media & Entertainment": ["Advertising", "Broadcasting", "Cable & Satellite", "Publishing",
                              "Movies & Entertainment", "Interactive Home Entertainment",
                              "Interactive Media & Services"],
    "Real Estate Management & Development": ["Diversified Real Estate Activities", "Real Estate Operating Companies",
                                            "Real Estate Development", "Real Estate Services"],
    "Equity Real Estate Investment Trusts (REITs)": [
        "Diversified REITs", "Industrial REITs", "Hotel & Resort REITs", "Office REITs",
        "Health Care REITs", "Residential REITs", "Retail REITs", "Specialized REITs",
        "Telecom Tower REITs", "Timber REITs", "Data Center REITs", "Single-Family Residential REITs",
        "Multi-Family Residential REITs", "Self-Storage REITs", "Other Specialized REITs"],
    "Utilities": ["Electric Utilities", "Gas Utilities", "Multi-Utilities", "Water Utilities",
                  "Independent Power Producers & Energy Traders", "Renewable Electricity"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_SUB_TO_GROUP = {_norm(sub): grp for grp, subs in _GROUP_TO_SUBS.items() for sub in subs}


def industry_group(sub_industry: str, sector: str | None = None) -> str:
    """GICS Industry Group for a Sub-Industry; falls back to `sector` (or the raw
    sub-industry) when the sub-industry isn't in the canonical table."""
    grp = _SUB_TO_GROUP.get(_norm(sub_industry))
    if grp is not None:
        return grp
    return sector if sector else str(sub_industry)
