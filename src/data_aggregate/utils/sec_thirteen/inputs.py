"""Single-pass immutable 13F accession packet loading."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


def load_filing_vintages(
    store: DataStore, until: object, ciks: Sequence[str] | None = None, *, table: Table = Tables.sec13f_hr
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Partition one marker-inclusive payload stream into manifests and whole books."""
    manifest_columns = (
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "accepted_at",
        "amendment_type",
        "amendment_number",
        "report_type",
        "restricted",
        "parsed_row_count",
        "parse_status",
    )
    numeric = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    position_columns = (
        "cik",
        "accession_number",
        "cusip",
        "period",
        "filing_date",
        "position_type",
        "class_evidence",
        *numeric,
        *("raw_" + col for col in numeric if "value" in col),
    )
    if not set(manifest_columns) <= set(store.columns(table)):
        logger.warning("%s has no completed accession schema; replay original sources before building PIT features.", table.name)
        return None
    where: dict[str, Any] = {"parse_status": ["complete", "position"]}
    if ciks is not None:
        where["cik"] = sorted({pad_cik(cik) for cik in ciks})
    manifests, positions = [], []
    for chunk in store.iter_load(
        table, columns=list(dict.fromkeys((*manifest_columns, *position_columns))), where=where, until=until, date_col="filing_date", markers=True
    ):
        manifests.append(chunk.loc[chunk["parse_status"].eq("complete") & chunk["cusip"].eq(table.empty_marker[1]), list(manifest_columns)])
        positions.append(chunk.loc[chunk["parse_status"].eq("position"), list(position_columns)])
    filings = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(columns=manifest_columns)
    if filings.empty:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    holdings = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(columns=position_columns)
    for column in ("period", "filing_date"):
        filings[column] = pd.to_datetime(filings[column])
        holdings[column] = pd.to_datetime(holdings[column])
    logger.info(
        "Immutable 13F: %s complete accessions / %s whole-book rows / %s managers; unverified legacy rows excluded.",
        len(filings),
        len(holdings),
        filings.cik.nunique(),
    )
    return filings, holdings
