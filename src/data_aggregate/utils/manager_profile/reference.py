"""Sharadar sector and security class, asset type of each line and the per-ticker aggregate that sizes positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from src.utils.string import alternation

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


def _name_hits(text: pd.Series, markers: ListConfig) -> pd.Series:
    """Lines whose upper-case `text` holds any marker as whole words."""
    return text.str.contains(alternation(list(markers)), regex=True)


def _generic_titles(title: pd.Series, rule: DictConfig) -> pd.Series:
    """Titles made only of anchor and filler words with at least one anchor (COM, CL A, COMMON STOCK USD.01)."""
    word = "|".join([*rule.anchors, *rule.fillers])
    return title.str.fullmatch(rf"(?:{word})(?: (?:{word}))*") & title.str.contains(alternation(list(rule.anchors)), regex=True)


def classify_securities(lines: pd.DataFrame, reference: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Add each line's `asset` (one of ASSETS) and `sector_col`.

    A ticker Sharadar lists as a domestic, ADR or Canadian stock keeps its class and sector; any other common line (unlisted, or listed
    under a non-stock category) is read by `cfg.security_markers` on `title_of_class | issuer_name`: fund-like, then real-estate sector,
    then foreign, then a generic US title; else `other`.
    """
    # Sharadar class and sector by ticker.
    df = lines.join(reference, on="ticker")
    kind, listed = df["kind"], df["security_class"].isin(["us", "foreign"])
    common = kind.eq("common")

    # Name rules on the common lines Sharadar does not class as a stock, each later rule on the lines the earlier ones left.
    marks = cfg.security_markers
    unlisted = common & ~listed
    title = df["title_of_class"].fillna("").astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    text = title + " | " + df["issuer_name"].fillna("").astype(str).str.upper()
    fund = unlisted & _name_hits(text, marks.etf.markers)
    real_estate = unlisted & ~fund & _name_hits(text, marks.real_estate.markers)
    foreign = unlisted & ~fund & _name_hits(text, marks.foreign.markers)
    domestic = unlisted & ~fund & ~foreign & _generic_titles(title, marks.us_titles)

    # Asset type: debt and option rows by kind, equity by Sharadar class, unlisted equity by the name rules.
    conditions = [kind.eq("debt"), kind.isin(["call", "put"]), common & df["security_class"].eq("us"), common & df["security_class"].eq("foreign")]
    conditions += [fund, foreign, domestic]
    df["asset"] = np.select(conditions, ["bond", "options", "equity_us", "equity_foreign", "etf", "equity_foreign", "equity_us"], "other")
    df["sector_col"] = df["sector_col"].where(common & listed, UNMAPPED).mask(real_estate, "pct_sector_real_estate")
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
