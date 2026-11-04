"""Identity evidence capture: the Notes cover-page `dei:TradingSymbol` facts of every filer in the cached Notes zips, one
period at a time, into the `dei` partition of `symbol_tenure`. Offline: cached zips only (`zips-download --only notes`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.identity.symbol_tenure import aggregate_dei_symbols, captured_dei_periods, save_dei_period
from src.data_extract.utils.zips.fetch_financial_notes import CHUNK_ROWS, cached_notes_periods

logger = logging.getLogger(__name__)

_DEI_SUB_USECOLS = frozenset({"adsh", "cik", "name", "filed"})
_DEI_TXT_USECOLS = frozenset({"adsh", "tag", "coreg", "dimh", "value"})
_DEI_DIM_USECOLS = frozenset({"dimhash", "segments"})
_DEI_SYMBOL_TAG = "TradingSymbol"
#: A dimension on this axis names another entity's security on a combined cover (Entergy tagging `EAI`).
_LEGAL_ENTITY_SEGMENT = "LegalEntity="


def _cover_symbol_rows(chunk: pd.DataFrame) -> pd.Series:
    """`dei:TradingSymbol` facts of the primary filer; a co-registrant's symbol belongs to another CIK."""
    coreg = chunk.get("coreg", pd.Series("", index=chunk.index)).astype("string").fillna("").str.strip()
    return chunk["tag"].eq(_DEI_SYMBOL_TAG) & (coreg == "")


def _legal_entity_dims(chunk: pd.DataFrame) -> pd.Series:
    """`dim.tsv` rows whose segments include a LegalEntityAxis member."""
    return chunk["segments"].astype("string").str.contains(_LEGAL_ENTITY_SEGMENT, regex=False).fillna(False)


_DEI_READ = {
    "sub.tsv": ZipRead(usecols=_DEI_SUB_USECOLS),
    "txt.tsv": ZipRead(usecols=_DEI_TXT_USECOLS, keep=_cover_symbol_rows, chunksize=CHUNK_ROWS, skip_bad_lines=True),
    "dim.tsv": ZipRead(usecols=_DEI_DIM_USECOLS, keep=_legal_entity_dims, chunksize=CHUNK_ROWS, required=False),
}


def pending_dei_periods(context: Context, cache: Path, full: bool) -> list[str]:
    """The cached Notes periods whose cover-page symbols are not stored yet (`full`: every cached period)."""
    # the stored periods against the cached zips
    captured = captured_dei_periods(context)
    pending = [period for period in cached_notes_periods(cache) if full or period not in captured]
    logger.info("notes dei: %d cached period(s) to capture (%d already stored)", len(pending), len(captured))
    return pending


def capture_dei_period(context: Context, cache: Path, period: str) -> int:
    """Replace one period's `dei` rows with its zip's cover-page symbols; returns the rows written."""
    # read the period's cover-page symbol facts of every filer
    tables = read_zip_tables(cache / f"{period}_notes.zip", _DEI_READ, on_corrupt="delete", log=logger)
    facts = _primary_facts(tables)
    if facts is None or facts.empty:
        logger.info("notes dei %s: no cover-page symbol facts", period)
        return 0

    # one row per (symbol, CIK), saved over the period's stored rows
    rows = aggregate_dei_symbols(facts, period)
    written = save_dei_period(context, period, rows)
    logger.info("notes dei %s: %d accession(s) -> %d (symbol, CIK) row(s), %d written", period, facts["adsh"].nunique(), len(rows), written)
    return written


def _primary_facts(tables: dict[str, pd.DataFrame] | None) -> pd.DataFrame | None:
    """`[adsh, value, cik, name, filed]` symbol facts; a fact dimensioned on another legal entity is that entity's security."""
    if not tables:
        return None
    txt = tables["txt.tsv"]
    if txt.empty:
        return pd.DataFrame(columns=["adsh", "value", *sorted(_DEI_SUB_USECOLS - {"adsh"})])
    other_entity = txt["dimh"].isin(set(tables["dim.tsv"].get("dimhash", ())))
    return txt.loc[~other_entity, ["adsh", "value"]].merge(tables["sub.tsv"], on="adsh", how="inner")
