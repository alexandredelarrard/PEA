"""`identity --only sync`: carry the published identity into the CIK-keyed stored rows, SQL only (REQ-009, decision Q1).

`run_sync` re-checks the tickers whose lineage stamp (`universe_cik_symbol.scope_changed_at`) is inside
`resume.recently_changed`'s window on the run date, every ticker on a first build or `--full` (PD-13). Per filer-CIK
table (`PURGE_TABLES`):
- Move: a row filed by a CIK that now belongs to another universe company whose scope holds it at the row's date (the
  table's own dated / windowed rule) is re-tickered there by delete + insert; a row already stored under the new key wins.
- Purge: every other row filed outside its company's scope; a row dated after the company's `ended_on` is left.
Then the insider lineage stamps of the changed and receiving tickers are rewritten (co-registrant rows purged). No zip
is read: a CIK new to a scope gets its archive rows from its fetcher's rescan (`resume.archive_worklist`), and a ticker
whose facts moved has its SEC history rebuilt by the nightly history build (`build_fundamentals_history`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity
from src.data_extract.utils.common.resume import recently_changed
from src.data_extract.utils.edgar_extract.insider_store import patch_insider_source
from src.data_store.schema import Tables
from src.utils.filer_tables import (
    MOVE_COLUMNS,
    REMOVAL_COLUMNS,
    FilerTable,
    ListedWindow,
    delete_filer_rows,
    foreign_filer_rows,
    move_filer_rows,
    moved_filer_rows,
    removal_records,
)
from src.utils.tickers_universe.universe import Bounds, before_end

#: `{ticker: CIKs of its company}` and `{ticker: its seam-widened windows}`.
OwnedScopes = tuple[dict[str, frozenset[str]], dict[str, tuple[ListedWindow, ...]]]
__all__ = ["SyncResult"]


@dataclass(frozen=True)
class SyncResult:
    """What one sync removed (`REMOVAL_COLUMNS`) and moved (`MOVE_COLUMNS`)."""

    removals: pd.DataFrame
    moves: pd.DataFrame


def recheck_tickers(scopes: Mapping[str, FilingScope], as_of: pd.Timestamp, every: bool) -> list[str]:
    """The tickers whose lineage scope stamp is inside the re-check window on `as_of`; every ticker under `every`."""
    # compute: the scope stamp of each ticker, then the recent ones
    stamps = {ticker: scope.scope_changed_at for ticker, scope in scopes.items()}
    return sorted(stamps) if every else recently_changed(stamps, as_of)


def owned_scopes(identity: Identity) -> OwnedScopes:
    """(`{ticker: CIKs its company owns}`, `{ticker: its seam-widened windows}`) of every universe company."""
    # compute: the event CIKs that map back to each company, and its windows
    owned: dict[str, set[str]] = {}
    for entity, ciks in identity.event_ciks_by_entity.items():
        ticker = identity.ticker_by_entity.get(entity)
        if ticker is not None:
            owned.setdefault(ticker, set()).update(cik for cik in ciks if identity.entity_of(cik) == entity)
    windows = {
        ticker: tuple((window.cik, window.listed_from, window.listed_to) for window in identity.windows_by_entity.get(entity, ()))
        for entity, ticker in identity.ticker_by_entity.items()
    }
    return {ticker: frozenset(ciks) for ticker, ciks in owned.items()}, windows


def sync_filer_rows(context: Context, spec: FilerTable, tickers: Sequence[str], owned: OwnedScopes, bounds: Bounds) -> tuple[list[dict], list[dict]]:
    """Move `tickers`' rows of `spec` that another universe company's scope now holds, then purge the other rows filed
    outside the ticker's scope (up to its end date); returns `(removal records, move records)`."""
    # load: the rows filed outside each ticker's scope
    if not tickers or not context.store.exists(spec.table):
        return [], []
    found = foreign_filer_rows(context.store, spec, tickers, *owned)

    # compute: the rows to move, and the rest up to each company's end
    moving = moved_filer_rows(spec, found, *owned)
    rest = found.drop(index=moving.index)
    foreign = rest[before_end(rest["ticker"], rest[spec.date_col], bounds)]

    # save: the moves first, then the purge (insider rows through their source patch: a shared accession keeps the other issuer)
    moves = move_filer_rows(context.store, spec, moving)
    if spec.table == Tables.insider_transactions:
        patch_insider_source(context.store, pd.DataFrame(), foreign[[*Tables.insider_transactions.pk, "ticker", "issuer_cik"]])
    else:
        delete_filer_rows(context.store, spec, foreign)
    records = removal_records(spec.table.name, foreign, spec)
    _log_table(context, spec.table.name, records, moves)
    return records, moves


def log_sync(context: Context, changed: Sequence[str], result: SyncResult) -> None:
    """One INFO summary of the run: the tickers re-checked, the rows moved and the rows removed."""
    # compute: the counts, then one line
    moved = int(result.moves["rows"].sum()) if not result.moves.empty else 0
    removed = int(result.removals["rows"].sum()) if not result.removals.empty else 0
    context.log.info(
        f"identity sync: {len(changed)} ticker scope(s) re-checked; {len(result.moves)} move group(s), {moved} row(s) re-tickered; "
        f"{len(result.removals)} removal group(s), {removed} row(s) purged; no zip read"
    )


def sync_result(removals: list[dict], moves: list[dict]) -> SyncResult:
    """The records of one sync as frames."""
    return SyncResult(pd.DataFrame(removals, columns=list(REMOVAL_COLUMNS)), pd.DataFrame(moves, columns=list(MOVE_COLUMNS)))


def _log_table(context: Context, table: str, records: list[dict], moves: list[dict]) -> None:
    """One INFO line per table and change: every (ticker, cik, window, rows) moved or removed (a performed change, PD-14)."""
    if moves:
        detail = "; ".join(
            f"{m['ticker']} cik {m['cik']} -> {m['new_ticker']} {m['first_filed']}..{m['last_filed']} {m['rows']} row(s)" for m in moves
        )
        kept = sum(m["kept"] for m in moves)
        context.log.info(f"identity sync: moved {sum(m['rows'] for m in moves)} row(s) in '{table}' ({kept} already under the new key) -- {detail}")
    if records:
        detail = "; ".join(f"{r['ticker']} cik {r['cik'] or '-'} {r['first_filed']}..{r['last_filed']} {r['rows']} row(s)" for r in records)
        context.log.info(f"identity sync: purged {sum(r['rows'] for r in records)} row(s) from '{table}' -- {detail}")
