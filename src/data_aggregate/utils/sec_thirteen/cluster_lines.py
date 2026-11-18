"""One report quarter of 13F books as company-level lines: the cleaned book per (manager, company)."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from src.utils.identity_tables import CusipTags

#: Columns of the company books: a manager's line per company and kind, its rank by value among all the manager's common
#: companies, and the manager's whole common book value.
BOOK_COLUMNS = ("cik", "kind", "ticker", "shares", "value", "rank", "book_total", "filing_date")
OPTION_KINDS = ("call", "put")


def company_books(classified: pd.DataFrame, tags: CusipTags, universe: Collection[str], deadline: pd.Timestamp) -> pd.DataFrame:
    """Each manager's lines of the quarter by company: `BOOK_COLUMNS`, one row per (cik, kind, ticker).

    `classified` is `classify_securities(clean_book(...))` of one quarter (it carries `cusip` and `asset`). Only lines filed on or before
    `deadline`, the quarter's filing deadline (`cluster_panel.filing_deadlines`), count. Every line takes the company ticker `tags.tag`
    gives its CUSIP on the period, so a company's share classes sum into one row (an untagged line stays under its own key); `rank` and
    `book_total` are taken over all the manager's common companies, then the rows keep the US-equity common companies of `universe` and
    the option lines (value only) on them.
    """
    # clean: filings made by the quarter's deadline
    members = set(universe)
    lines = classified[pd.to_datetime(classified["filing_date"]) <= deadline]
    if lines.empty:
        return pd.DataFrame(columns=list(BOOK_COLUMNS))

    # merge: one company per line; shares of one company sum (unknown only when no class reports them)
    keyed = lines.assign(ticker=tags.tag(lines).fillna(lines["key"]), us=lines["asset"].eq("equity_us"))
    keys = ["cik", "ticker"]
    common = keyed[keyed["kind"].eq("common")].groupby(keys, sort=False)
    book = common.agg(kind=("kind", "first"), value=("value", "sum"), filing_date=("filing_date", "first"), us=("us", "max"))
    book["shares"] = common["shares"].sum(min_count=1)

    # compute: rank and total over every common company of the manager, before the universe filter
    book["rank"] = book.groupby("cik")["value"].rank(ascending=False, method="first")
    book["book_total"] = book.groupby("cik")["value"].transform("sum")
    book = book.reset_index()
    equity = book[book["us"] & book["ticker"].isin(members)]

    # compute: the option lines of universe companies
    options = keyed[keyed["kind"].isin(OPTION_KINDS) & keyed["ticker"].isin(members)]
    option_book = options.groupby([*keys, "kind"], as_index=False, sort=False).agg(
        value=("value", "sum"), filing_date=("filing_date", "first"), shares=("shares", "sum")
    )
    return pd.concat([equity, option_book], ignore_index=True)[list(BOOK_COLUMNS)]
