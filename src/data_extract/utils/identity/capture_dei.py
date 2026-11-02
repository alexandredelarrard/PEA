"""Notes cover-page symbol evidence: every filer's `dei:TradingSymbol` facts in the cached Notes zips, one period
at a time, into the `dei` partition of `symbol_tenure`. Offline: cached zips only (`notes-download` caches them).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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


def _read_dei_facts(path: Path) -> pd.DataFrame | None:
    """One notes zip -> `[adsh, value, cik, name, filed]` cover-page symbol facts of every filer; None when unreadable.

    A fact dimensioned on another legal entity is that entity's security (a subsidiary bond or preferred), so it is dropped.
    """
    specs = {
        "sub.tsv": ZipRead(usecols=_DEI_SUB_USECOLS),
        "txt.tsv": ZipRead(usecols=_DEI_TXT_USECOLS, keep=_cover_symbol_rows, chunksize=CHUNK_ROWS, skip_bad_lines=True),
        "dim.tsv": ZipRead(usecols=_DEI_DIM_USECOLS, keep=_legal_entity_dims, chunksize=CHUNK_ROWS, required=False),
    }
    tables = read_zip_tables(path, specs, on_corrupt="delete", log=logger)
    if not tables:
        return None
    if tables["txt.tsv"].empty:
        return pd.DataFrame(columns=["adsh", "value", *sorted(_DEI_SUB_USECOLS - {"adsh"})])
    txt = tables["txt.tsv"]
    other_entity = txt["dimh"].isin(set(tables["dim.tsv"].get("dimhash", ())))
    return txt.loc[~other_entity, ["adsh", "value"]].merge(tables["sub.tsv"], on="adsh", how="inner")


def capture_dei_symbols(context: Context, cache: Path, periods: list[str]) -> int:
    """Write each cached period's `dei` rows to `symbol_tenure`, replacing that period's rows; returns rows written."""
    written = 0
    for period in tqdm(periods, desc="SEC notes cover-page symbols"):
        facts = _read_dei_facts(cache / f"{period}_notes.zip")
        if facts is None or facts.empty:
            logger.info("notes dei %s: no cover-page symbol facts", period)
            continue
        rows = aggregate_dei_symbols(facts, period)
        n_saved = save_dei_period(context, period, rows)
        written += n_saved
        logger.info("notes dei %s: %d accession(s) -> %d (symbol, CIK) row(s), %d written", period, facts["adsh"].nunique(), len(rows), n_saved)
    return written


def capture_pending_dei(context: Context, cache: Path, *, full: bool) -> int:
    """Capture every cached zip whose period is not stored yet (`full`: every cached zip); returns the `dei` rows written."""
    captured = captured_dei_periods(context)
    pending = [period for period in cached_notes_periods(cache) if full or period not in captured]
    written = capture_dei_symbols(context, cache, pending)
    logger.info("notes dei: %d period(s) captured, %d row(s) written (%d already stored)", len(pending), written, len(captured))
    return written
