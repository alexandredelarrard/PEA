"""Publish stage of the identity build: the one save of the three published tables (REQ-008).

`universe_cik_symbol` and `universe_cusip` are replaced through the identity commit only when they changed. The
`universe_tickers` rows are shared with the universe sync, so only the identity columns of the changed tickers are
updated (X19: `schema.sql` keeps `cik` and `universe_added_on` NOT NULL and Postgres checks them before ON CONFLICT,
so an upsert of the identity columns alone would fail); the sync's columns are never written here.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_extract.utils.common.identity_store import save_identity
from src.data_extract.utils.common.incremental import matches_stored
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.tickers_universe.sync import IDENTITY_COLUMNS

logger = logging.getLogger(__name__)


def cusip_company_ids(lines: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    """The CUSIP lines with their company's frozen `company_id` (X44), read off the lineage by universe ticker."""
    ids = lineage[lineage["canonical_ticker"].notna()].drop_duplicates("canonical_ticker").set_index("canonical_ticker")["company_id"]
    company = lines["canonical_company"].map(ids)
    return lines.assign(company_id=company.astype(object).where(company.notna(), None))


def changed_identity(rows: pd.DataFrame, stored: pd.DataFrame | None) -> pd.DataFrame:
    """The lifecycle rows whose identity columns differ from the stored `universe_tickers` row (dates compared as days)."""
    # both sides as text, dates as YYYY-MM-DD and nulls as ''
    previous = (stored if stored is not None else pd.DataFrame(columns=["ticker"])).reindex(columns=["ticker", *IDENTITY_COLUMNS])
    current, before = _comparable(rows.reindex(columns=["ticker", *IDENTITY_COLUMNS])), _comparable(previous)

    # the rows that are new or differ in any identity column
    joined = current.merge(before, on="ticker", how="left", suffixes=("", "_stored")).fillna("")
    same = pd.concat([joined[column].eq(joined[f"{column}_stored"]) for column in IDENTITY_COLUMNS], axis=1).all(axis=1)
    return rows[~same.to_numpy()]


def save_unless_unchanged(store: DataStore, table: Table, stored: pd.DataFrame | None, rows: pd.DataFrame) -> int:
    """Replace `table` through the identity commit unless `rows` equal the stored rows (none built: left as it is)."""
    if rows.empty:
        logger.info("%s: no rows built; table left unchanged", table)
        return 0
    if matches_stored(stored, rows, table):
        logger.info("%s: unchanged (%d row(s)); replace skipped", table, len(rows))
        return 0
    written = save_identity(store, table, rows)
    logger.info("%s: wrote %d row(s)", table, written)
    return written


def update_identity_columns(store: DataStore, changed: pd.DataFrame) -> int:
    """Write the identity columns of each changed universe ticker in place (`store.update`, never an upsert)."""
    written = 0
    for record in changed.to_dict("records"):
        values = {column: _value(record[column]) for column in IDENTITY_COLUMNS}
        written += store.update(Tables.universe_tickers, values, {"ticker": record["ticker"]})
    logger.info("%s: identity columns of %d ticker(s) updated", Tables.universe_tickers, written)
    return written


# Flat helpers
def _comparable(frame: pd.DataFrame) -> pd.DataFrame:
    """Every value as text, dates as YYYY-MM-DD and nulls as ''."""
    out = frame.copy()
    for column in ("first_price_date", "ended_on"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.astype("string").fillna("")


def _value(value: object) -> object:
    """A store value: None for a missing one."""
    return None if value is None or (not isinstance(value, str) and pd.isna(value)) else value
