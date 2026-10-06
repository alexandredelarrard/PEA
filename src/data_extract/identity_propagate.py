"""`identity-propagate`: carry an `entity_lineage` change into the stored SEC rows.

The tickers whose lineage stamp (`entity_lineage.scope_changed_at`) falls inside
`resume.recently_changed`'s window on the run date are re-checked, on every table; each step is
idempotent, so a ticker seen on several runs inside the window costs reads only. Contraction: rows
whose filer CIK is no longer a CIK of the ticker's entity, and 8-K / 13D / 13G rows whose CIK has no
seam-widened window admitting their date, are deleted, one WARNING per table.
Expansion: the bulk families re-parse, from their cached zips, the changed tickers whose table holds their
rows but none from a scope CIK it can hold (EDGAR tables list a new CIK's filings in their own fetchers). The raw
FTD lines and RegSHO short-volume rows of companies whose `security_master` rows changed recently (for
short volume, also their lineage symbol rows) are re-stamped from the stored rows and their ticker rows
rebuilt. The stored insider rows of changed tickers (and every unstamped row) get their lineage stamp
rewritten in place, and co-registrant insider rows are purged. A ticker whose facts were purged has its
SEC and merged history rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity, load_identity
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_financial_notes import reparse_financial_notes
from src.data_extract.utils.fundamentals.fetch_financial_statements import reparse_financial_statements
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import load_fails_master, master_stamps, restamp_fails
from src.data_extract.utils.institutionals.fetch_insider_transactions import reparse_insider_transactions, restamp_insider_lineage
from src.data_extract.utils.institutionals.fetch_short_interest import change_stamps, restamp_short_volume
from src.data_extract.utils.institutionals.security_tape import restamp_targets
from src.data_store.schema import Table, Tables
from src.utils.filer_tables import (
    PURGE_TABLES,
    PURGE_TABLES_BY_NAME,
    REMOVAL_COLUMNS,
    FilerTable,
    ListedWindow,
    judged_cik_mask,
    own_filer_mask,
    removal_records,
    windowed_filer_mask,
)
from src.utils.string import normalise_ticker, pad_cik

#: Tickers read per scoped load, and keys per targeted delete.
_TICKER_CHUNK = 50
_KEY_CHUNK = 500
__all__ = ["PropagationResult", "propagate_identity"]


@dataclass(frozen=True)
class PropagationResult:
    """What one propagation removed (or, dry, would remove), re-parsed and rebuilt."""

    removals: pd.DataFrame
    reparsed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    rebuilt: tuple[str, ...] = ()


def _changed(stamps: Mapping[str, pd.Timestamp | None], as_of: pd.Timestamp, *, every: bool = False) -> list[str]:
    """Keys whose stamp is inside `resume.recently_changed`'s window on `as_of`; every key under `every`."""
    return sorted(stamps) if every else recently_changed(stamps, as_of)


def _lacking_scope_rows(context: Context, table: Table, tickers: Sequence[str], scope_ciks: Mapping[str, frozenset[str]]) -> list[str]:
    """`tickers` holding rows in `table` but none from one of the scope CIKs the table can hold (a bulk re-parse may
    add them); a ticker with no row at all is the table's own fetcher's work, not a lineage expansion."""
    cik_col = PURGE_TABLES_BY_NAME[table.name].cik_col
    lacking = []
    for ticker in tickers:
        stored = {pad_cik(cik) for cik in context.store.distinct(table, cik_col, where={"ticker": ticker})}
        if stored and scope_ciks.get(ticker, frozenset()) - stored:
            lacking.append(ticker)
    return lacking


def _own_ciks(identity: Identity) -> dict[str, frozenset[str]]:
    """`{ticker: CIKs}` that `identity.ticker_for_cik(cik, None, "event")` maps to that ticker."""
    owned: dict[str, set[str]] = {}
    for entity, ciks in identity.event_ciks_by_entity.items():
        ticker = identity.ticker_by_entity.get(entity)
        if ticker is not None:
            owned.setdefault(ticker, set()).update(cik for cik in ciks if identity.entity_of(cik) == entity)
    return {ticker: frozenset(ciks) for ticker, ciks in owned.items()}


def _own_windows(identity: Identity) -> dict[str, tuple[ListedWindow, ...]]:
    """`{ticker: (cik, listed_from, listed_to) per seam-widened window}` of every universe entity; a ticker whose
    event filings are undated (`Identity.undated_event_tickers`) is left out, so its rows are not date-limited."""
    return {
        ticker: tuple((window.cik, window.listed_from, window.listed_to) for window in identity.windows_by_entity.get(entity, ()))
        for entity, ticker in identity.ticker_by_entity.items()
        if ticker not in identity.undated_event_tickers
    }


def _foreign_rows(
    context: Context,
    spec: FilerTable,
    tickers: Sequence[str],
    own_ciks: Mapping[str, frozenset[str]],
    own_windows: Mapping[str, Sequence[ListedWindow]],
) -> pd.DataFrame:
    """`tickers`' rows of `spec` whose filer CIK no longer belongs to the ticker's entity, or for a dated table whose CIK
    has no window admitting the row's date (a CIK with no digit is never judged)."""
    columns = ["ticker", spec.cik_col, spec.date_col, spec.key_col]
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), _TICKER_CHUNK):
        rows = context.store.load(spec.table, columns=columns, where={"ticker": list(tickers[start : start + _TICKER_CHUNK])}, optional=True)
        if rows is None or rows.empty:
            continue
        rows = rows[judged_cik_mask(rows[spec.cik_col])]
        padded = rows[spec.cik_col].map(pad_cik)
        own = own_filer_mask(rows["ticker"], padded, own_ciks)
        if spec.dated:
            own &= windowed_filer_mask(rows["ticker"], padded, rows[spec.date_col], own_windows)
        foreign = rows[~own]
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


def _purge(
    context: Context,
    spec: FilerTable,
    tickers: Sequence[str],
    own_ciks: Mapping[str, frozenset[str]],
    own_windows: Mapping[str, Sequence[ListedWindow]],
    *,
    dry_run: bool,
) -> list[dict]:
    """Delete (or list) `tickers`' rows of `spec` filed by a CIK outside the ticker's entity, or outside its windows for a dated table."""
    if not tickers or not context.store.exists(spec.table):
        return []
    foreign = _foreign_rows(context, spec, tickers, own_ciks, own_windows)
    records = removal_records(spec.table.name, foreign, spec)
    if not dry_run:
        for (ticker, cik), group in foreign.groupby(["ticker", spec.cik_col], sort=True):
            keys = sorted(group[spec.key_col].astype(str).unique())
            for start in range(0, len(keys), _KEY_CHUNK):
                context.store.delete(spec.table, where={"ticker": ticker, spec.cik_col: cik, spec.key_col: keys[start : start + _KEY_CHUNK]})
    _warn(context, spec.table.name, records, dry_run=dry_run)
    return records


def _window_ciks(scopes: Mapping[str, FilingScope], table: Table) -> dict[str, frozenset[str]]:
    """`{ticker: window CIKs}` whose window is still open on `table`'s first data-set date."""
    start = pd.Timestamp(table.resume.source_start) if table.resume is not None and table.resume.source_start else None
    return {
        ticker: frozenset(w.cik for w in scope.windows if start is None or w.listed_to is None or pd.Timestamp(w.listed_to) > start)
        for ticker, scope in scopes.items()
    }


def _reparse_bulk(
    context: Context, changed: Sequence[str], scopes: Mapping[str, FilingScope], co_registrants: frozenset[str]
) -> dict[str, tuple[str, ...]]:
    """Re-parse each bulk family from its cache for the changed tickers whose table lacks rows of a CIK it can hold:
    a window CIK listed after the data set starts for the consolidating Notes and statement families, an event CIK
    other than a co-registrant for insider."""
    events = {ticker: frozenset(scope.event_ciks) - co_registrants for ticker, scope in scopes.items()}
    families: tuple[tuple[Table, Callable[[Context, list[str]], int], Mapping[str, frozenset[str]]], ...] = (
        (Tables.notes_num, reparse_financial_notes, _window_ciks(scopes, Tables.notes_num)),
        (Tables.pension_facts, reparse_financial_statements, _window_ciks(scopes, Tables.pension_facts)),
        (Tables.insider_transactions, reparse_insider_transactions, events),
    )
    done: dict[str, tuple[str, ...]] = {}
    for table, reparse, scope_ciks in families:
        tickers = _lacking_scope_rows(context, table, changed, scope_ciks) if context.store.exists(table) else list(changed)
        if tickers:
            context.log.info("identity-propagate: re-parsing '%s' from cache for %d ticker(s): %s", table.name, len(tickers), ", ".join(tickers))
            reparse(context, tickers)
            done[table.name] = tuple(tickers)
    return done


def _refresh_fails(context: Context, tickers: Sequence[str], as_of: pd.Timestamp, *, dry_run: bool, every: bool) -> list[dict]:
    """Re-stamp the stored FTD lines of companies whose master rows changed recently; rebuild their rows."""
    if not context.store.exists(Tables.sec_fails_to_deliver_security):
        return []
    master = load_fails_master(context)
    if master is None or master.empty:
        return []
    changed = restamp_targets(context, Tables.sec_fails_to_deliver_security, master_stamps(master), as_of, every=every)
    if not changed:
        return []
    context.log.info("identity-propagate: re-stamping FTD lines for %d company(ies): %s", len(changed), ", ".join(changed))
    records = restamp_fails(context, changed, tickers, master=master, dry_run=dry_run)
    _warn(context, Tables.sec_fails_to_deliver.name, records, dry_run=dry_run)
    return records


def _refresh_short_volume(
    context: Context, identity: Identity, tickers: Sequence[str], as_of: pd.Timestamp, *, dry_run: bool, every: bool
) -> list[dict]:
    """Re-stamp the stored short-volume rows of companies whose master or symbol rows changed recently."""
    if not context.store.exists(Tables.sec_short_volume_security):
        return []
    stamps = change_stamps(load_fails_master(context), identity)
    changed = restamp_targets(context, Tables.sec_short_volume_security, stamps, as_of, every=every)
    if not changed:
        return []
    context.log.info("identity-propagate: re-stamping short-volume rows for %d company(ies): %s", len(changed), ", ".join(changed))
    records = restamp_short_volume(context, changed, tickers, identity=identity, dry_run=dry_run)
    _warn(context, Tables.short_interest.name, records, dry_run=dry_run)
    return records


def _refresh_insider(context: Context, identity: Identity, changed: Sequence[str], *, dry_run: bool) -> list[dict]:
    """Re-stamp the stored insider rows of the changed tickers; purge co-registrant rows."""
    if not context.store.exists(Tables.insider_transactions):
        return []
    records = restamp_insider_lineage(context, changed, identity=identity, dry_run=dry_run)
    _warn(context, Tables.insider_transactions.name, records, dry_run=dry_run)
    return records


def _rebuild_history(context: Context, tickers: list[str]) -> None:
    """Delete and rebuild the SEC history of `tickers`, then their merged history in full."""
    context.log.warning("identity-propagate: rebuilding the SEC and merged history of %d purged ticker(s): %s", len(tickers), ", ".join(tickers))
    build_fundamentals_history(context, tickers, rebuild_history=True)
    build_merged_history(context, tickers, full=True, config_dir=str(context.config_dir))


def propagate_identity(
    context: Context,
    tickers: Sequence[str],
    *,
    dry_run: bool = False,
    every_ticker: bool = False,
    identity: Identity | None = None,
    as_of: pd.Timestamp | None = None,
) -> PropagationResult:
    """Purge, re-parse and rebuild for the lineage changes inside the re-check window on `as_of` (default
    today); `dry_run` only lists removals.

    `every_ticker` (`identity-propagate --every-ticker`) checks every ticker whatever its stamp; the tapes then re-stamp every company.
    """
    resolver = identity or load_identity(context)
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    universe = [normalise_ticker(ticker) for ticker in tickers if normalise_ticker(ticker) in resolver.roster_cik]
    scopes = {ticker: resolver.filing_scope(ticker) for ticker in universe}
    changed = _changed({ticker: scope.scope_changed_at for ticker, scope in scopes.items()}, run_date, every=every_ticker)
    if changed:
        context.log.info("identity-propagate: %d ticker lineage scope(s) to re-check: %s", len(changed), ", ".join(changed))
    own_ciks, own_windows = _own_ciks(resolver), _own_windows(resolver)
    records: list[dict] = []
    for spec in PURGE_TABLES:
        records += _purge(context, spec, changed, own_ciks, own_windows, dry_run=dry_run)
    reparsed = {} if dry_run else _reparse_bulk(context, changed, scopes, resolver.co_registrant_ciks)
    records += _refresh_insider(context, resolver, changed, dry_run=dry_run)
    records += _refresh_fails(context, universe, run_date, dry_run=dry_run, every=every_ticker)
    records += _refresh_short_volume(context, resolver, universe, run_date, dry_run=dry_run, every=every_ticker)
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
