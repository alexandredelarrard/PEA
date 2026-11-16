"""Share lines that are really bond principal, found by the shape of the CUSIP and by the title, for the cube cleaning and the manager profile."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.data_store.schema import EMPTY_MARKER


def _upper(lines: pd.DataFrame, column: str) -> pd.Series:
    """The upper-case text of `column` (empty where it is missing or the table lacks the column)."""
    return lines.get(column, pd.Series("", index=lines.index)).fillna("").astype(str).str.upper()


def bond_line_mask(lines: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """Lines whose position is bond principal filed as shares: a letter issue number on a numeric-issuer CUSIP, or a bond title with no ticker.

    `lines` carries `cusip`, `issuer_name`, `title_of_class` and `ticker` (a column the table lacks reads empty). `cfg` is
    `build_cube.bond_lines`. A CUSIP line is a bond unless its issuer or title names a fund, ADR or placeholder; a line with no ticker is
    also a bond when its title has a note, convertible, debenture, coupon or maturity marker and no preferred, warrant, unit or right word.
    """
    # Text of the line, upper case.
    cusip, title = _upper(lines, "cusip"), _upper(lines, "title_of_class")
    name = _upper(lines, "issuer_name") + " " + title
    ticker = lines.get("ticker", pd.Series(None, index=lines.index, dtype=object))
    unlisted = ticker.isna() | ticker.isin(["", EMPTY_MARKER])

    # CUSIP rule: letter issue number, numeric issuer code, no fund, ADR or placeholder name.
    by_cusip = cusip.str.fullmatch(cfg["cusip_pattern"]) & ~cusip.str.startswith(tuple(cfg["cusip_excluded_prefixes"]))
    by_cusip &= ~name.str.contains("|".join(cfg["name_exclusions"]), regex=True)

    # Title rule, only where no ticker identifies the stock.
    by_title = title.str.contains("|".join(cfg["title_markers"]), regex=True) & ~(title + " " + _upper(lines, "issuer_name")).str.contains(
        "|".join(cfg["title_exclusions"]), regex=True
    )
    return by_cusip | (unlisted & by_title)
