"""Sharadar sector and security class, asset type of each line and the per-ticker aggregate that sizes positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import ListConfig

SECTORS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)
UNMAPPED = "pct_sector_unmapped"
#: Sector columns of the profile, one per Sharadar sector then the unmapped remainder.
SECTOR_COLUMNS = (*("pct_sector_" + name.lower().replace(" ", "_") for name in SECTORS), UNMAPPED)
#: Asset types of a line; each is a `pct_<asset>` mix column.
ASSETS = ("equity_us", "equity_foreign", "etf", "bond", "options", "other")
#: Issuer-name words of an exchange-traded product, used only for a security Sharadar does not list.
_ETF_NAME = r"\b(?:ETF|ETN|ISHARES|SPDR|VANGUARD|POWERSHARES|PROSHARES|INVESCO|INDEX FUND|TRUST SER)\b"


def sharadar_reference(df_tickers: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: `sector_col` (profile column of its sector, else unmapped) and `security_class` (us, foreign or other).

    A duplicated ticker keeps the row with a sector, then the latest `lastpricedate`, then the highest `permaticker`.
    """
    # One row per ticker.
    sector = df_tickers["sector"].fillna("").str.strip()
    ranked = df_tickers.assign(sector=sector, has_sector=sector.ne("")).sort_values(["has_sector", "lastpricedate", "permaticker"], ascending=False)
    df = ranked.drop_duplicates("ticker").set_index("ticker")

    # Profile column of the sector and class of the category.
    column = "pct_sector_" + df["sector"].str.lower().str.replace(" ", "_")
    category = df["category"].fillna("")
    security_class = np.select([category.str.startswith("Domestic"), category.str.startswith(("ADR", "Canadian"))], ["us", "foreign"], "other")
    return pd.DataFrame({"sector_col": column.where(df["sector"].isin(SECTORS), UNMAPPED), "security_class": security_class}, index=df.index)


def prefix_tickers(df_map: pd.DataFrame) -> pd.Series:
    """Ticker by 8-character CUSIP prefix (issuer plus issue; early filings omit the check digit); an ambiguous prefix is left out."""
    grouped = df_map.dropna(subset=["cusip", "ticker"]).groupby(df_map["cusip"].str[:8])["ticker"].agg(["nunique", "first"])
    return grouped.loc[grouped["nunique"] == 1, "first"]


def classify_securities(lines: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add each line's `asset` (one of ASSETS) and `sector_col`; a ticker Sharadar lacks is an ETF by issuer name, else other."""
    # Sharadar class and sector by ticker.
    df = lines.join(reference, on="ticker")
    kind, listed = df["kind"], df["security_class"].notna()
    common = kind.eq("common")

    # Asset type: debt and option rows by kind, equity by Sharadar class, unlisted equity by the ETF name rule.
    etf_name = df["issuer_name"].fillna("").str.upper().str.contains(_ETF_NAME, regex=True)
    conditions = [kind.eq("debt"), kind.isin(["call", "put"]), common & df["security_class"].eq("us"), common & df["security_class"].eq("foreign")]
    conditions.append(common & ~listed & etf_name)
    df["asset"] = np.select(conditions, ["bond", "options", "equity_us", "equity_foreign", "etf"], "other")
    df["sector_col"] = df["sector_col"].where(common & listed, UNMAPPED)
    return df.drop(columns="security_class")


def quarter_aggregate(lines: pd.DataFrame, breakpoints: ListConfig) -> pd.DataFrame:
    """Per position key: aggregate equity `value` and `weight` over all filers, and the `cap` bucket ranked by that value.

    Ranked by value (ties by key); a key is large while the value above it is under the first breakpoint, mid under the sum of both.
    """
    # Aggregate equity value per key, largest first.
    equity = lines[lines["kind"].eq("common")]
    value = equity.groupby("key")["value"].sum().sort_index().sort_values(ascending=False, kind="stable")
    weight = value / value.sum()

    # Bucket by the cumulative weight above each key.
    above = weight.cumsum() - weight
    large, mid = float(breakpoints[0]), float(breakpoints[0]) + float(breakpoints[1])
    cap = np.select([above < large, above < mid], ["large", "mid"], "small")
    return pd.DataFrame({"value": value, "weight": weight, "cap": cap})
