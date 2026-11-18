"""One report quarter of 13F books as company-level lines: the raw read of the managers' lines and the cleaned book per (manager, company)."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import pandas as pd

from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.manager_profile.lines import LINE_COLUMNS
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.identity_tables import CusipTags

#: Columns of the company books: a manager's line per company and kind, its rank by value among all the manager's common
#: companies, and the manager's whole common book value.
BOOK_COLUMNS = ("cik", "kind", "ticker", "shares", "value", "rank", "book_total", "filing_date")
_OPTIONS = ("call", "put")
_CHUNK = 1_000


def read_quarter_lines(store: DataStore, ciks: Sequence[str], period: pd.Timestamp) -> pd.DataFrame:
    """Markers and lines of `ciks` for one report `period`, read on the primary key in CIK chunks (the table has no period index)."""
    # `iter_load` streams one scope; this read is one bounded load per CIK chunk, which it cannot express.
    stamp = period.strftime("%Y-%m-%d")
    frames = []
    for start in range(0, len(ciks), _CHUNK):
        where = {"cik": list(ciks[start : start + _CHUNK]), "period": [stamp]}
        frame = store.load(Tables.sec13f_hr, columns=list(LINE_COLUMNS), where=where, markers=True, optional=True)
        if frame is not None:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(LINE_COLUMNS))


def company_books(classified: pd.DataFrame, tags: CusipTags, universe: Collection[str]) -> pd.DataFrame:
    """Each manager's lines of the quarter by company: `BOOK_COLUMNS`, one row per (cik, kind, ticker).

    `classified` is `classify_securities(clean_book(...))` of one quarter (it carries `cusip`, `period` and `asset`). Only lines filed by
    the quarter's 45-day deadline count. Every line takes the company ticker `tags.tag` gives its CUSIP on the period, so a company's
    share classes sum into one row (an untagged line stays under its own key); `rank` and `book_total` are taken over all the manager's
    common companies, then the rows keep the US-equity common companies of `universe` and the option lines (value only) on them.
    """
    # clean: filings by the deadline of their own period
    members = set(universe)
    deadline = pd.to_datetime(classified["period"]) + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
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
    options = keyed[keyed["kind"].isin(_OPTIONS) & keyed["ticker"].isin(members)]
    option_book = options.groupby([*keys, "kind"], as_index=False, sort=False).agg(
        value=("value", "sum"), filing_date=("filing_date", "first"), shares=("shares", "sum")
    )
    return pd.concat([equity, option_book], ignore_index=True)[list(BOOK_COLUMNS)]
