"""Share lines that are really bond principal, found by the shape of the CUSIP and by the title, for the cube cleaning and the manager profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data_store.schema import EMPTY_MARKER


def bond_line_mask(lines: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """Lines whose position is bond principal filed as shares: a letter issue number on a numeric-issuer CUSIP, or a bond title with no ticker.

    `lines` carries `cusip`, `issuer_name`, `title_of_class` and `ticker` (a column the table lacks reads empty). `cfg` is
    `data_aggregate.bond_lines`. A CUSIP line is a bond unless its issuer or title names a fund, ADR or placeholder; a line with no ticker is
    also a bond when its title has a note, convertible, debenture, coupon or maturity marker and no preferred, warrant, unit or right word.
    """
    # Text of the line, upper case (empty where missing or the table lacks the column).
    text = lines.reindex(columns=["cusip", "issuer_name", "title_of_class"]).fillna("").astype(str).apply(lambda column: column.str.upper())
    cusip, issuer, title = text["cusip"], text["issuer_name"], text["title_of_class"]
    name = issuer + " " + title
    ticker = lines.get("ticker", pd.Series(None, index=lines.index, dtype=object))
    unlisted = ticker.isna() | ticker.isin(["", EMPTY_MARKER])

    # CUSIP rule: letter issue number, numeric issuer code, no fund, ADR or placeholder name.
    by_cusip = cusip.str.fullmatch(cfg["cusip_pattern"]) & ~cusip.str.startswith(tuple(cfg["cusip_excluded_prefixes"]))
    by_cusip &= ~name.str.contains("|".join(cfg["name_exclusions"]), regex=True)

    # Title rule, only where no ticker identifies the stock.
    by_title = title.str.contains("|".join(cfg["title_markers"]), regex=True) & ~(title + " " + issuer).str.contains(
        "|".join(cfg["title_exclusions"]), regex=True
    )
    return by_cusip | (unlisted & by_title)
