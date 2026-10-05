"""`identity-propagate`: carry an `entity_lineage` change into the stored SEC rows.

Per table, the tickers whose lineage stamp is at or after the table's manifest `last_run_date` (every
ticker when the table has no recorded run) are re-checked. Contraction: rows whose filer CIK is no
longer a CIK of the ticker's entity are deleted, one WARNING per table. Expansion: the bulk families
re-parse those tickers from their cached zips (EDGAR tables relist in their own fetchers). The raw FTD
lines of companies whose `security_master` rows changed are re-stamped from the stored rows and their
ticker rows rebuilt. A ticker whose facts were purged has its SEC and merged history rebuilt.
`sec_short_interest` is never touched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.run_manifest import get_entry, scope_changed_tickers
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_financial_notes import reparse_financial_notes
from src.data_extract.utils.fundamentals.fetch_financial_statements import reparse_financial_statements
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import load_fails_master, master_stamps, restamp_fails
from src.data_extract.utils.institutionals.fetch_insider_transactions import reparse_insider_transactions
from src.data_store.schema import Table, Tables
from src.utils.filer_tables import (
    PURGE_TABLES,
    PURGE_TABLES_BY_NAME,
    REMOVAL_COLUMNS,
    FilerTable,
    judged_cik_mask,
    own_filer_mask,
    removal_records,
)
from src.utils.string import normalise_ticker, pad_cik

#: Tickers read per scoped load, and keys per targeted delete.
_TICKER_CHUNK = 50
_KEY_CHUNK = 500
__all__ = ["PURGE_TABLES", "PURGE_TABLES_BY_NAME", "REMOVAL_COLUMNS", "PropagationResult", "propagate_identity"]


@dataclass(frozen=True)
class PropagationResult:
    """What one propagation removed (or, dry, would remove), re-parsed and rebuilt."""

    removals: pd.DataFrame
    reparsed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    rebuilt: tuple[str, ...] = ()


def _changed(context: Context, table: Table, stamps: Mapping[str, pd.Timestamp | None], *, every: bool = False) -> list[str]:
    """Tickers stamped at or after `table`'s last run; every ticker under `every` or when the table has no recorded run."""
    entry = get_entry(context, table)
    if every or not (entry or {}).get("last_run_date"):
        return sorted(stamps)
    return sorted(scope_changed_tickers(entry, stamps))


def _own_ciks(identity: Identity) -> dict[str, frozenset[str]]:
    """`{ticker: CIKs}` that `identity.ticker_for_cik(cik, None, "event")` maps to that ticker."""
    owned: dict[str, set[str]] = {}
    for entity, ciks in identity.event_ciks_by_entity.items():
        ticker = identity.ticker_by_entity.get(entity)
        if ticker is not None:
            owned.setdefault(ticker, set()).update(cik for cik in ciks if identity.entity_of(cik) == entity)
    return {ticker: frozenset(ciks) for ticker, ciks in owned.items()}


def _foreign_rows(context: Context, spec: FilerTable, tickers: Sequence[str], own_ciks: Mapping[str, frozenset[str]]) -> pd.DataFrame:
    """`tickers`' rows of `spec` whose filer CIK no longer belongs to the ticker's entity (a CIK with no digit is never judged)."""
    columns = ["ticker", spec.cik_col, spec.date_col, spec.key_col]
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), _TICKER_CHUNK):
        rows = context.store.load(spec.table, columns=columns, where={"ticker": list(tickers[start : start + _TICKER_CHUNK])}, optional=True)
        if rows is None or rows.empty:
            continue
        rows = rows[judged_cik_mask(rows[spec.cik_col])]
        foreign = rows[~own_filer_mask(rows["ticker"], rows[spec.cik_col].map(pad_cik), own_ciks)]
        if not foreign.empty:
            frames.append(foreign)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _warn(context: Context, table: str, records: list[dict], *, dry_run: bool) -> None:
    """One line per table: every (ticker, cik, window, rows) removed."""
    if not records:
        return
    detail = "; ".join(f"{r['ticker']} cik {r['cik'] or '-'} {r['first_filed']}..{r['last_filed']} {r['rows']} row(s)" for r in records)
    total = sum(r["rows"] for r in records)
    if dry_run:
        context.log.info("identity-propagate (dry run): would purge %d row(s) from '%s' -- %s", total, table, detail)
    else:
        context.log.warning("identity-propagate: purged %d row(s) from '%s' -- %s", total, table, detail)


def _purge(context: Context, spec: FilerTable, tickers: Sequence[str], own_ciks: Mapping[str, frozenset[str]], *, dry_run: bool) -> list[dict]:
    """Delete (or list) `tickers`' rows of `spec` filed by a CIK outside the ticker's entity."""
    if not tickers or not context.store.exists(spec.table):
        return []
    foreign = _foreign_rows(context, spec, tickers, own_ciks)
    records = removal_records(spec.table.name, foreign, spec)
    if not dry_run:
        for (ticker, cik), group in foreign.groupby(["ticker", spec.cik_col], sort=True):
            keys = sorted(group[spec.key_col].astype(str).unique())
            for start in range(0, len(keys), _KEY_CHUNK):
                context.store.delete(spec.table, where={"ticker": ticker, spec.cik_col: cik, spec.key_col: keys[start : start + _KEY_CHUNK]})
    _warn(context, spec.table.name, records, dry_run=dry_run)
    return records


def _reparse_bulk(context: Context, stamps: Mapping[str, pd.Timestamp | None]) -> dict[str, tuple[str, ...]]:
    """Re-parse each bulk family from its cache for the tickers whose scope changed since its last run."""
    families: tuple[tuple[Table, Callable[[Context, list[str]], int]], ...] = (
        (Tables.notes_num, reparse_financial_notes),
        (Tables.pension_facts, reparse_financial_statements),
        (Tables.insider_transactions, reparse_insider_transactions),
    )
    done: dict[str, tuple[str, ...]] = {}
    for table, reparse in families:
        tickers = _changed(context, table, stamps)
        if tickers:
            context.log.info("identity-propagate: re-parsing '%s' from cache for %d ticker(s): %s", table.name, len(tickers), ", ".join(tickers))
            reparse(context, tickers)
            done[table.name] = tuple(tickers)
    return done


def _refresh_fails(context: Context, tickers: Sequence[str], *, dry_run: bool, every: bool) -> list[dict]:
    """Re-stamp the stored FTD lines of companies whose master rows changed since the ticker table's last run; rebuild their rows."""
    if not context.store.exists(Tables.sec_fails_to_deliver_security):
        return []
    master = load_fails_master(context)
    if master is None or master.empty:
        return []
    changed = _changed(context, Tables.sec_fails_to_deliver, master_stamps(master), every=every)
    if not changed:
        return []
    context.log.info("identity-propagate: re-stamping FTD lines for %d company(ies): %s", len(changed), ", ".join(changed))
    records = restamp_fails(context, changed, tickers, master=master, dry_run=dry_run)
    _warn(context, Tables.sec_fails_to_deliver.name, records, dry_run=dry_run)
    return records


def _rebuild_history(context: Context, tickers: list[str]) -> None:
    """Delete and rebuild the SEC history of `tickers`, then their merged history in full."""
    context.log.warning("identity-propagate: rebuilding the SEC and merged history of %d purged ticker(s): %s", len(tickers), ", ".join(tickers))
    build_fundamentals_history(context, tickers, rebuild_history=True)
    build_merged_history(context, tickers, full=True, config_dir=str(context.config_dir))


def propagate_identity(
    context: Context, tickers: Sequence[str], *, dry_run: bool = False, every_ticker: bool = False, identity: Identity | None = None
) -> PropagationResult:
    """Purge, re-parse and rebuild for the lineage changes since each table's last run; `dry_run` only lists removals.

    `every_ticker` checks every ticker whatever its stamp (the validator's dry run; FTD then re-stamps every company).
    """
    resolver = identity or load_identity(context)
    universe = [normalise_ticker(ticker) for ticker in tickers if normalise_ticker(ticker) in resolver.roster_cik]
    cik_stamps = {ticker: resolver.filing_scope(ticker).scope_changed_at for ticker in universe}
    own_ciks = _own_ciks(resolver)
    records: list[dict] = []
    for spec in PURGE_TABLES:
        records += _purge(context, spec, _changed(context, spec.table, cik_stamps, every=every_ticker), own_ciks, dry_run=dry_run)
    reparsed = {} if dry_run else _reparse_bulk(context, cik_stamps)
    records += _refresh_fails(context, universe, dry_run=dry_run, every=every_ticker)
    removals = pd.DataFrame(records, columns=list(REMOVAL_COLUMNS))
    purged_facts = sorted(set(removals.loc[removals["table"] == Tables.fundamentals_facts.name, "ticker"]))
    if purged_facts and not dry_run:
        _rebuild_history(context, purged_facts)
    context.log.info(
        "identity-propagate%s: %d removal group(s), %d row(s); re-parsed %s; history rebuilt for %d ticker(s)",
        " (dry run)" if dry_run else "",
        len(removals),
        int(removals["rows"].sum()) if not removals.empty else 0,
        {table: len(names) for table, names in reparsed.items()},
        0 if dry_run else len(purged_facts),
    )
    return PropagationResult(removals=removals, reparsed=reparsed, rebuilt=() if dry_run else tuple(purged_facts))
