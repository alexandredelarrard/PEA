"""FTD identity evidence: the in-scope source lines of the cached FTD ZIPs, raw and per CUSIP, in `sec_fails_to_deliver_security`.

Scope is the lineage and master as stored before this run's rebuild: lineage symbols plus the CUSIP-6 of the
master's securities and the manual config. The periods read are those no stored line carries; every cached period
is re-read for the symbols and prefixes whose lineage or master rows changed recently and for a CUSIP-6 newly voted
for by a lineage symbol interval. Stamp columns stay NULL. Offline: cached ZIPs only (`ftd-download` caches them).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.identity.security_master import cusip_votes, lineage_scope_symbols, load_security_manual, squash
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd_zip
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

_LINEAGE_SCOPE_COLUMNS = ["entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]


def _ingest_scope(context: Context) -> tuple[pd.DataFrame | None, pd.DataFrame | None, frozenset[str], frozenset[str]]:
    """`(lineage, master, squashed lineage symbols, CUSIP-6 prefixes)`; prefixes come from the stored master and the manual config."""
    lineage = context.store.load(Tables.entity_lineage, columns=[*_LINEAGE_SCOPE_COLUMNS, "scope_changed_at"], optional=True)
    master = context.store.load(Tables.security_master, columns=["cusip", "scope_changed_at"], optional=True)
    if lineage is None:
        return None, master, frozenset(), frozenset()
    manual = load_security_manual(getattr(context, "config_dir", None))
    cusips = set() if master is None else set(master["cusip"].dropna().astype(str))
    cusips |= set(manual.boundaries["cusip"].dropna()) | set(manual.ratios["cusip"].dropna())
    return lineage, master, lineage_scope_symbols(lineage), frozenset(c[:6] for c in cusips if c)


def _recent_keys(rows: pd.DataFrame, key: str, as_of: pd.Timestamp) -> list[str]:
    """The values of `key` whose latest `scope_changed_at` is inside `resume.recently_changed`'s window on `as_of`."""
    stamps = pd.to_datetime(rows["scope_changed_at"], errors="coerce").groupby(rows[key].astype(str)).max()
    return recently_changed({str(k): v for k, v in stamps.items()}, as_of)


def _changed_scope(lineage: pd.DataFrame, master: pd.DataFrame | None, as_of: pd.Timestamp) -> tuple[frozenset[str], frozenset[str]]:
    """Changed symbols and CUSIP-6 prefixes derived from the initial identity snapshot."""
    entities = set(_recent_keys(lineage, "entity_id", as_of))
    symbols = lineage_scope_symbols(lineage.loc[lineage["entity_id"].astype(str).isin(entities)]) if entities else frozenset()
    prefixes = frozenset(cusip[:6] for cusip in _recent_keys(master, "cusip", as_of) if cusip) if master is not None else frozenset()
    return symbols, prefixes


def _in_scope(lines: pd.DataFrame, symbols: frozenset[str], prefixes: frozenset[str]) -> pd.DataFrame:
    keys = lines["source_symbol"].map(squash)
    return lines[keys.isin(symbols) | lines["cusip"].str[:6].isin(prefixes)]


def _read_cached(context: Context, cache: Path, periods: Sequence[str], symbols: frozenset[str], prefixes: frozenset[str]) -> dict[str, pd.DataFrame]:
    """`{period: in-scope lines}` of the cached, valid originals among `periods`."""
    cached = ftd_zip.cached_ftd_periods(cache)
    frames: dict[str, pd.DataFrame] = {}
    for period in tqdm([p for p in periods if p in cached], desc="FTD capture"):
        frame = ftd_zip.read_ftd_period(context, cache, period, download=False)
        if frame is not None:
            frames[period] = _in_scope(frame, symbols, prefixes)
    return frames


def capture_ftd_cusips(context: Context, cache: Path, years_history: int, *, full: bool, as_of: pd.Timestamp) -> int:
    """Store the in-scope raw lines of the pending cached periods; `full` replaces the table from every listed period.

    Returns the rows written; 0 before the first `entity_lineage` build (cold start: run it again with `full`).
    """
    lineage, master, symbols, prefixes = _ingest_scope(context)
    if lineage is None:
        logger.warning("FTD capture: no entity_lineage yet; nothing captured until the lineage is built")
        return 0
    changed_symbols, changed_prefixes = (frozenset(), frozenset()) if full else _changed_scope(lineage, master, as_of)
    pending = ftd_zip.pending_ftd_periods(context, cache, years_history, full=full, as_of=as_of)
    first = _read_cached(context, cache, pending, symbols, prefixes)
    kept = pd.concat(first.values(), ignore_index=True) if first else pd.DataFrame(columns=list(ftd_zip.SECURITY_COLUMNS[:9]))
    voted = cusip_votes(kept, lineage) if not kept.empty else pd.DataFrame(columns=["cusip"])
    added = frozenset(str(c)[:6] for c in voted["cusip"]) - prefixes
    rescan_prefixes = added | changed_prefixes
    reconciled = set(first)
    if rescan_prefixes or changed_symbols:
        logger.info(
            "FTD capture: re-reading every cached period for %d changed symbol(s) and %d CUSIP-6 prefix(es) (%s)",
            len(changed_symbols),
            len(rescan_prefixes),
            ", ".join(sorted(rescan_prefixes)),
        )
        rescanned = _read_cached(context, cache, sorted(ftd_zip.cached_ftd_periods(cache)), symbols, prefixes | rescan_prefixes)
        kept = pd.concat([kept, *rescanned.values()], ignore_index=True)
        reconciled |= set(rescanned)
    kept = ftd_zip.unique_ftd_keys(kept, "FTD capture")
    rows = kept.assign(security_id=None, ticker=None, lineage_role=None, security_class=None)[list(ftd_zip.SECURITY_COLUMNS)]
    written = _save_pending_fails(context, rows, sorted(reconciled))
    logger.info(
        "FTD capture: %d period(s) read, %d in-scope line(s) stored (%d symbol(s), %d CUSIP-6 prefix(es) in scope)",
        len(pending),
        written,
        len(symbols),
        len(prefixes | added),
    )
    return written


def _save_pending_fails(context: Context, rows: pd.DataFrame, periods: Sequence[str]) -> int:
    """New raw units and retained, explicitly unverified derived rows commit as one source pair."""
    previous = (
        context.store.load(
            Tables.sec_fails_to_deliver, columns=context.store.columns(Tables.sec_fails_to_deliver), where={"period": list(periods)}, optional=True
        )
        if periods
        else None
    )
    observed = previous if previous is not None else ftd_zip.empty_fails_grain()
    if "source_checked_at" not in observed:
        observed = observed.assign(source_checked_at=pd.Series(pd.NaT, index=observed.index, dtype="datetime64[ns]"))
    written = 0
    for period in periods:
        payload = rows.loc[rows["period"].eq(period)]
        expected = observed.loc[observed["period"].eq(period)]
        retained = expected.assign(source_checked_at=pd.Series(pd.NaT, index=expected.index, dtype="datetime64[ns]"))
        quiet = expected["source_checked_at"].notna() & expected["fails_quantity"].eq(0)
        retained.loc[quiet, ["fails_quantity", "fails_value"]] = float("nan")
        counts = ftd_zip.replace_fails_source(
            context.store,
            {Tables.sec_fails_to_deliver_security: payload, Tables.sec_fails_to_deliver: retained},
            {Tables.sec_fails_to_deliver_security: {"period": period}, Tables.sec_fails_to_deliver: {"period": period}},
            expected_source={Tables.sec_fails_to_deliver: expected},
        )
        written += counts[Tables.sec_fails_to_deliver_security]
    return written
