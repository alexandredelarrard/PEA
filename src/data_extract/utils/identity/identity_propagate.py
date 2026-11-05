"""`identity --only propagate`: carry the published identity into the stored rows (REQ-014), one stage function each.

`run_propagate` re-checks the tickers whose lineage stamp (`universe_cik_symbol.scope_changed_at`) is inside
`resume.recently_changed`'s window on the run date, every ticker on a first build or `--full` (PD-13); a
`company_id`-only change moves no stamp, so it changes nothing. Each stage is idempotent: a ticker seen on several
runs inside the window costs reads only.
- Contraction (`purge_rows`): rows whose filer CIK is no longer a CIK of the ticker's company, 8-K / 13D / 13G rows
  whose CIK has no seam-widened window admitting their date, and consolidating rows (facts, notes, filing text,
  proxies) whose CIK holds no window of the ticker: the event-only CIKs (DD, DOW, JCI, PLD).
- Expansion (`reparse_notes`, `reparse_insider`): the bulk families re-parse, from their cached zips, the changed
  tickers whose table holds their rows but none from a scope CIK it can hold.
- Re-stamps: insider rows in place (co-registrant rows purged), FTD and short-volume lines of companies whose
  `universe_cusip` rows changed, and `sec13f_hr` rows whose CUSIP the dated lines now tag with another ticker.
- A ticker whose facts were purged has its SEC and merged history rebuilt. A dry run only reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity, load_source_identity
from src.data_extract.utils.common.resume import Bounds, before_end, recently_changed
from src.data_extract.utils.edgar_extract.insider_store import patch_insider_source
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.data_extract.utils.institutionals.fetch_short_interest import change_stamps, restamp_short_volume
from src.data_extract.utils.institutionals.security_tape import restamp_targets
from src.data_extract.utils.zips.fetch_fails_to_deliver import load_fails_master, master_stamps, prepare_master, restamp_fails
from src.data_extract.utils.zips.fetch_financial_notes import reparse_financial_notes
from src.data_extract.utils.zips.fetch_insider_transactions import reparse_insider_transactions, restamp_insider_lineage
from src.data_store.schema import Table, Tables
from src.utils.filer_tables import FilerTable, ListedWindow, delete_filer_rows, foreign_filer_rows, removal_records
from src.utils.identity_tables import load_cusip_lines, load_cusip_map, tag_cusips, universe_cusips
from src.utils.string import pad_cik_series

#: One row per (CUSIP, stored ticker, new ticker) the 13F re-stamp moves.
RESTAMP_COLUMNS = ("table", "cusip", "from_ticker", "to_ticker", "first_period", "last_period", "rows")
#: Keys per keyed 13F update.
_KEY_CHUNK = 500
#: `{ticker: CIKs of its company}` and `{ticker: its seam-widened windows}`.
OwnedScopes = tuple[dict[str, frozenset[str]], dict[str, tuple[ListedWindow, ...]]]
__all__ = ["PropagationResult", "RESTAMP_COLUMNS"]


@dataclass(frozen=True)
class PropagationResult:
    """What one propagation removed and re-stamped (dry: would), re-parsed and rebuilt."""

    removals: pd.DataFrame
    restamped: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(RESTAMP_COLUMNS)))
    reparsed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    rebuilt: tuple[str, ...] = ()


def recheck_tickers(scopes: Mapping[str, FilingScope], as_of: pd.Timestamp, every: bool) -> list[str]:
    """The tickers whose lineage scope stamp is inside the re-check window on `as_of`; every ticker under `every`."""
    # compute: the scope stamp of each ticker, then the recent ones
    stamps = {ticker: scope.scope_changed_at for ticker, scope in scopes.items()}
    return sorted(stamps) if every else recently_changed(stamps, as_of)


def rebuilt_since(scopes: Mapping[str, FilingScope], since: pd.Timestamp) -> list[str]:
    """The tickers whose lineage scope stamp is on or after `since` (this run's start): the scopes this run changed (X90)."""
    stamps = {ticker: scope.scope_changed_at for ticker, scope in scopes.items()}
    return sorted(ticker for ticker, stamp in stamps.items() if stamp is not None and not pd.isna(stamp) and pd.Timestamp(stamp) >= since)


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


def purge_rows(context: Context, spec: FilerTable, tickers: Sequence[str], owned: OwnedScopes, bounds: Bounds, dry_run: bool) -> list[dict]:
    """Delete (dry: list) `tickers`' rows of `spec` filed by a CIK outside the ticker's company, or outside its windows;
    a row dated after its company's `ended_on` is left as it is (REQ-014)."""
    # load: the rows filed outside each ticker's scope, up to its end date
    if not tickers or not context.store.exists(spec.table):
        return []
    found = foreign_filer_rows(context.store, spec, tickers, *owned)
    foreign = found[before_end(found["ticker"], found[spec.date_col], bounds)]
    records = removal_records(spec.table.name, foreign, spec)

    # save: insider rows through their source patch (a shared accession keeps the other issuer's payload), others by key
    if not dry_run and spec.table == Tables.insider_transactions:
        patch_insider_source(context.store, pd.DataFrame(), foreign[[*Tables.insider_transactions.pk, "ticker", "issuer_cik"]])
    elif not dry_run:
        delete_filer_rows(context.store, spec, foreign)
    _log_records(context, spec.table.name, records, dry_run)
    return records


def reparse_notes(context: Context, changed: Sequence[str], scopes: Mapping[str, FilingScope], identity: Identity) -> dict[str, tuple[str, ...]]:
    """Re-parse the cached Notes zips for the changed tickers whose rows lack a window CIK listed after the data set starts."""
    # load: the filer CIKs of each changed ticker's stored notes rows
    stored = context.store.summarize(Tables.notes_num, ["ticker", "cik"], {"rows": ("count", None)}, where={"ticker": list(changed)})
    exists = context.store.exists(Tables.notes_num)

    # compute: the tickers to re-parse (every changed one before the table exists)
    held = stored.assign(cik=pad_cik_series(stored["cik"]))
    tickers = _lacking(held, "cik", _window_ciks(scopes, Tables.notes_num), changed, exists)

    # save: re-parse them through the rebuilt identity
    if tickers:
        context.log.info(f"identity propagate: re-parsing '{Tables.notes_num.name}' from cache for {len(tickers)} ticker(s): {', '.join(tickers)}")
        reparse_financial_notes(context, tickers, identity=identity)
    return {Tables.notes_num.name: tuple(tickers)} if tickers else {}


def reparse_insider(
    context: Context, changed: Sequence[str], scopes: Mapping[str, FilingScope], identity: Identity, run_date: pd.Timestamp
) -> dict[str, tuple[str, ...]]:
    """Re-parse the cached Form 3/4/5 quarters for the changed tickers whose rows lack an event CIK (co-registrants aside)."""
    # load: the issuer CIKs of each changed ticker's stored insider rows
    stored = context.store.summarize(
        Tables.insider_transactions, ["ticker", "issuer_cik"], {"rows": ("count", None)}, where={"ticker": list(changed)}
    )
    exists = context.store.exists(Tables.insider_transactions)

    # compute: the tickers to re-parse (every changed one before the table exists)
    held = stored.assign(issuer_cik=pad_cik_series(stored["issuer_cik"]))
    events = {ticker: frozenset(scope.event_ciks) - identity.co_registrant_ciks for ticker, scope in scopes.items()}
    tickers = _lacking(held, "issuer_cik", events, changed, exists)

    # save: re-parse them through the rebuilt identity, up to the run date
    if tickers:
        context.log.info(
            f"identity propagate: re-parsing '{Tables.insider_transactions.name}' from cache for {len(tickers)} ticker(s): {', '.join(tickers)}"
        )
        reparse_insider_transactions(context, tickers, identity=identity, run_date=run_date)
    return {Tables.insider_transactions.name: tuple(tickers)} if tickers else {}


def refresh_insider(context: Context, identity: Identity, changed: Sequence[str], dry_run: bool) -> list[dict]:
    """Re-stamp the stored insider rows of the changed tickers in place; purge their co-registrant rows."""
    # save: the re-stamp writes only what changed
    if not context.store.exists(Tables.insider_transactions):
        return []
    records = restamp_insider_lineage(context, changed, identity=identity, dry_run=dry_run)
    _log_records(context, Tables.insider_transactions.name, records, dry_run)
    return records


def refresh_fails(context: Context, tickers: Sequence[str], as_of: pd.Timestamp, dry_run: bool, every: bool) -> list[dict]:
    """Re-stamp the stored FTD lines of companies whose `universe_cusip` rows changed recently; rebuild their rows."""
    # load: the published cohort and its CUSIP lines
    if not context.store.exists(Tables.sec_fails_to_deliver_security):
        return []
    cohort = load_source_identity(context.store)
    master = prepare_master(cohort[Tables.universe_cusip])

    # save: the companies to re-stamp, then their lines
    changed = [] if master.empty else restamp_targets(context, Tables.sec_fails_to_deliver_security, master_stamps(master), as_of, every=every)
    if not changed:
        return []
    context.log.info(f"identity propagate: re-stamping FTD lines for {len(changed)} company(ies): {', '.join(changed)}")
    records = restamp_fails(context, changed, tickers, identity=cohort, dry_run=dry_run)
    _log_records(context, Tables.sec_fails_to_deliver.name, records, dry_run)
    return records


def refresh_short_volume(context: Context, identity: Identity, tickers: Sequence[str], as_of: pd.Timestamp, dry_run: bool, every: bool) -> list[dict]:
    """Re-stamp the stored short-volume rows of companies whose CUSIP or symbol rows changed recently."""
    # save: the companies to re-stamp, then their rows
    if not context.store.exists(Tables.sec_short_volume_security):
        return []
    changed = restamp_targets(context, Tables.sec_short_volume_security, change_stamps(load_fails_master(context), identity), as_of, every=every)
    if not changed:
        return []
    context.log.info(f"identity propagate: re-stamping short-volume rows for {len(changed)} company(ies): {', '.join(changed)}")
    records = restamp_short_volume(context, changed, tickers, identity=identity, dry_run=dry_run)
    _log_records(context, Tables.short_interest.name, records, dry_run)
    return records


def thirteenf_moves(context: Context, tickers: Sequence[str], as_of: pd.Timestamp, every: bool) -> pd.DataFrame:
    """The stored `sec13f_hr` (cusip, period, ticker) groups that `tag_cusips` now tags with another ticker, for the
    companies whose `universe_cusip` rows changed recently (`every`: every ticker): a CUSIP moved to or from a company."""
    # load: the CUSIP lines, the market-wide map, the companies to re-check and their stored 13F groups
    lines, cmap, master = load_cusip_lines(context.store), load_cusip_map(context.store), load_fails_master(context)
    stamps = {} if master is None else master_stamps(master)
    companies = sorted(set(tickers) if every else set(recently_changed(stamps, as_of)) & set(tickers))
    scope = [{"cusip": sorted(universe_cusips(lines, companies))}, {"ticker": companies}]
    groups = context.store.summarize(Tables.sec13f_hr, ["cusip", "period", "ticker"], {"rows": ("count", None)}, where=scope)

    # compute: each group's ticker under the published lines (unmapped: the sentinel), and the groups that move
    marker = str(Tables.sec13f_hr.empty_marker[1]) if Tables.sec13f_hr.empty_marker else ""
    groups = groups[groups["cusip"].astype(str).ne(marker)].reset_index(drop=True)
    fresh = tag_cusips(groups, lines, cmap).astype(object).where(lambda tags: tags.notna(), marker)
    moves = groups.assign(new_ticker=fresh.to_numpy(dtype=object))
    return moves[moves["ticker"].astype(str).ne(moves["new_ticker"].astype(str))].reset_index(drop=True)


def retag_13f(context: Context, moves: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    """Move (dry: report) the `sec13f_hr` rows of `moves` to their new ticker by keyed updates; one report row per
    (CUSIP, stored ticker, new ticker)."""
    # load: the stored keys of the moving CUSIPs (none on a dry run)
    keys_wanted = not dry_run and not moves.empty
    stored = (
        context.store.load(Tables.sec13f_hr, columns=list(Tables.sec13f_hr.pk), where={"cusip": sorted(set(moves["cusip"]))}) if keys_wanted else None
    )

    # compute: the report, and the rows each new ticker takes
    report = _move_report(moves)
    moving = _moving_keys(stored, moves, list(Tables.sec13f_hr.pk)) if stored is not None else pd.DataFrame()

    # save: one keyed update per new ticker and key chunk (the ticker is part of the key, so no row can collide)
    for new_ticker, group in moving.groupby("new_ticker") if not moving.empty else ():
        for start in range(0, len(group), _KEY_CHUNK):
            context.store.update(Tables.sec13f_hr, {"ticker": new_ticker}, where=group[list(Tables.sec13f_hr.pk)].iloc[start : start + _KEY_CHUNK])
    _log_moves(context, report, dry_run)
    return report


def rebuild_history(context: Context, removals: pd.DataFrame, dry_run: bool) -> tuple[str, ...]:
    """Delete and rebuild the SEC history, then the merged history, of every ticker whose facts were purged."""
    # compute: the tickers whose facts were purged
    tickers = sorted(set(removals.loc[removals["table"] == Tables.fundamentals_facts.name, "ticker"]))
    if dry_run or not tickers:
        return ()

    # save: both histories in full
    context.log.info(f"identity propagate: rebuilding the SEC and merged history of {len(tickers)} purged ticker(s): {', '.join(tickers)}")
    build_fundamentals_history(context, tickers, rebuild_history=True)
    build_merged_history(context, tickers, full=True, config_dir=str(context.config_dir))
    return tuple(tickers)


def log_propagation(context: Context, changed: Sequence[str], result: PropagationResult, dry_run: bool) -> None:
    """One INFO summary of the run: the tickers re-checked, the removals, the 13F moves, the re-parses and the rebuilds."""
    # compute: the counts, then one line
    removed = int(result.removals["rows"].sum()) if not result.removals.empty else 0
    moved = int(result.restamped["rows"].sum()) if not result.restamped.empty else 0
    context.log.info(
        f"identity propagate{' (dry run)' if dry_run else ''}: {len(changed)} ticker scope(s) re-checked; "
        f"{len(result.removals)} removal group(s), {removed} row(s); {len(result.restamped)} 13F move(s), {moved} row(s); "
        f"re-parsed { ({table: len(names) for table, names in result.reparsed.items()}) }; history rebuilt for {len(result.rebuilt)} ticker(s)"
    )


def _lacking(stored: pd.DataFrame, cik_col: str, scope_ciks: Mapping[str, frozenset[str]], tickers: Sequence[str], exists: bool) -> list[str]:
    """`tickers` holding stored rows (padded `cik_col`) but none from one of their scope CIKs; every ticker when the table
    does not exist. A ticker with no row at all is the table's own fetcher's work, not a lineage expansion."""
    if not exists:
        return list(tickers)
    held = stored.groupby("ticker")[cik_col].agg(lambda ciks: frozenset(ciks.astype(str))).to_dict()
    return [ticker for ticker in tickers if ticker in held and scope_ciks.get(ticker, frozenset()) - held[ticker]]


def _window_ciks(scopes: Mapping[str, FilingScope], table: Table) -> dict[str, frozenset[str]]:
    """`{ticker: window CIKs}` whose window is still open on `table`'s first data-set date."""
    start = pd.Timestamp(table.resume.source_start) if table.resume is not None and table.resume.source_start else None
    return {
        ticker: frozenset(w.cik for w in scope.windows if start is None or w.listed_to is None or pd.Timestamp(w.listed_to) > start)
        for ticker, scope in scopes.items()
    }


def _moving_keys(stored: pd.DataFrame, moves: pd.DataFrame, pk: list[str]) -> pd.DataFrame:
    """The stored keys of the moving groups, with their `new_ticker` (periods compared as dates)."""
    left = stored.assign(day=pd.to_datetime(stored["period"]).dt.normalize())
    right = moves.assign(day=pd.to_datetime(moves["period"]).dt.normalize())[["cusip", "day", "ticker", "new_ticker"]]
    return left.merge(right, on=["cusip", "day", "ticker"], how="inner")[[*pk, "new_ticker"]]


def _move_report(moves: pd.DataFrame) -> pd.DataFrame:
    """One `RESTAMP_COLUMNS` row per (CUSIP, stored ticker, new ticker) of `moves`."""
    if moves.empty:
        return pd.DataFrame(columns=list(RESTAMP_COLUMNS))
    days = moves.assign(day=pd.to_datetime(moves["period"]))
    grouped = days.groupby(["cusip", "ticker", "new_ticker"], as_index=False).agg(
        first_period=("day", "min"), last_period=("day", "max"), rows=("rows", "sum")
    )
    report = grouped.rename(columns={"ticker": "from_ticker", "new_ticker": "to_ticker"}).assign(table=Tables.sec13f_hr.name)
    report["first_period"], report["last_period"] = report["first_period"].dt.date.astype(str), report["last_period"].dt.date.astype(str)
    report["rows"] = report["rows"].astype(int)
    return report[list(RESTAMP_COLUMNS)]


def _log_records(context: Context, table: str, records: list[dict], dry_run: bool) -> None:
    """One INFO line per table: every (ticker, cik, window, rows) removed (a performed change, not an item: PD-14)."""
    if not records:
        return
    detail = "; ".join(f"{r['ticker']} cik {r['cik'] or '-'} {r['first_filed']}..{r['last_filed']} {r['rows']} row(s)" for r in records)
    verb = "would purge" if dry_run else "purged"
    context.log.info(
        f"identity propagate{' (dry run)' if dry_run else ''}: {verb} {sum(r['rows'] for r in records)} row(s) from '{table}' -- {detail}"
    )


def _log_moves(context: Context, report: pd.DataFrame, dry_run: bool) -> None:
    """One INFO line for the 13F re-stamp: every (CUSIP, from, to, rows) moved."""
    if report.empty:
        return
    detail = "; ".join(f"{r.cusip} {r.from_ticker}->{r.to_ticker} {r.rows} row(s)" for r in report.itertuples(index=False))
    verb = "would move" if dry_run else "moved"
    context.log.info(
        f"identity propagate{' (dry run)' if dry_run else ''}: {verb} {int(report['rows'].sum())} '{Tables.sec13f_hr.name}' row(s) -- {detail}"
    )
