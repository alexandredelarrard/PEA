"""Data-extraction CLI: one command per data source, so the Airflow extraction DAG can schedule each independently.

    python -m src data_extract <command> [-c ./configs] [-t AAPL,MSFT]

Every command builds a fresh Context, resolves the ticker universe (or a --tickers subset) and runs its
fetcher. Fetchers resume from the DB, so a nightly rerun pulls only new data. `seed-universe` must run first.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd

from src.constants.command_line_interface import (
    ACCESSION_ARGS,
    ACCESSION_KWARGS,
    APPROVE_REKEY_ARGS,
    APPROVE_REKEY_KWARGS,
    ARCHIVE_ARGS,
    ARCHIVE_KWARGS,
    AS_OF_ARGS,
    AS_OF_KWARGS,
    CIK_ARGS,
    CIK_KWARGS,
    CONFIG_ARGS,
    CONFIG_KWARGS,
    COUNT_ARGS,
    COUNT_KWARGS,
    DELETE_ARGS,
    DELETE_KWARGS,
    DRY_RUN_ARGS,
    DRY_RUN_KWARGS,
    FILED_THROUGH_ARGS,
    FILED_THROUGH_KWARGS,
    FILING_WINDOW_ARGS,
    FILING_WINDOW_KWARGS,
    FULL_ARGS,
    FULL_KWARGS,
    MAX_FILINGS_ARGS,
    MAX_FILINGS_KWARGS,
    NO_CAP_ARGS,
    NO_CAP_KWARGS,
    PERIOD_END_ARGS,
    PERIOD_END_KWARGS,
    PERIOD_START_ARGS,
    PERIOD_START_KWARGS,
    PROPOSE_ARGS,
    PROPOSE_KWARGS,
    REBUILD_ARGS,
    REBUILD_KWARGS,
    REPAIR_AVAILABILITY_ARGS,
    REPAIR_AVAILABILITY_KWARGS,
    REPORT_PATH_ARGS,
    REPORT_PATH_KWARGS,
    TABLE_ARGS,
    TABLE_KWARGS,
    TICKERS_ARGS,
    TICKERS_KWARGS,
    TIMINGS_ARGS,
    TIMINGS_KWARGS,
    VERIFY_HISTORY_ARGS,
    VERIFY_HISTORY_KWARGS,
    VINTAGES_ARGS,
    VINTAGES_KWARGS,
    YEARS_ARGS,
    YEARS_KWARGS,
)
from src.context import Context, get_config_context
from src.data_extract.transformers.step_extract_edgar import EdgarOptions, StepExtractEdgar
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.transformers.step_extract_llm import LlmOptions, StepExtractLlm
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from src.data_extract.transformers.step_extract_zips import StepExtractZips, ZipsOptions
from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import (
    extract_earnings_calls as _extract_earnings_calls,
)
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun, plan_fetch
from src.data_extract.utils.edgar_extract.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.edgar_extract.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.edgar_extract.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.edgar_extract.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.edgar_extract.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_insider_edgar import INSIDER_EDGAR_FETCH
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec, fetched_filing_dates, fundamentals_fetch
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_actions,
    fetch_sharadar_sp500,
    fetch_sharadar_tickers,
)
from src.data_extract.utils.fundamentals_sharadar.gap_check import (
    DEFAULT_REPORT_PATH as GAP_REPORT_PATH,
)
from src.data_extract.utils.fundamentals_sharadar.gap_check import (
    run_gap_check,
)
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_13f_backfill import fetch_13f_backfill
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers, recover_13f_vintages
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.institutionals.fetch_superinvestors import rebuild_roster, upsert_roster_snapshot
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import fetch_prices_and_actions
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_store.schema import Table, Tables, freshness_tables, marker_tables, resolve
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.freshness import PREDICTION_INPUTS, fresh_share, last_dates_by_key, max_age_days, table_freshness
from src.utils.string import normalise_ticker, run_date
from src.utils.universe import load_universe_tickers, unverified_ciks


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA EXTRACTION — one command per source (scheduled by the Airflow extraction DAG)."""


def _tickers(context: Context, tickers: str | None) -> list[str]:
    """The --tickers subset if given, else the full sp500_tickers universe."""
    if tickers:
        return list(dict.fromkeys(ticker for raw in tickers.split(",") if (ticker := normalise_ticker(raw))))
    return load_universe_tickers(context)


def _ticker_scope(context: Context, tickers: str | None) -> tuple[list[str], list[str]]:
    """CLI selection and the full analysis universe, read once for scoped-write guards."""
    universe = load_universe_tickers(context)
    return (_tickers(context, tickers) if tickers else universe), universe


def _market_as_of(as_of: datetime | None) -> pd.Timestamp | None:
    """The `--as-of` date for market-data fetchers, or None so they end at the last session completed now."""
    return None if as_of is None else run_date(as_of)


def _table_status(context: Context, table: Table, universe: list[str], as_of: pd.Timestamp, min_share: float) -> dict[str, object]:
    """One table's freshness: the table-wide age, and for a ticker table the per-key share of the universe.

    RED when the age exceeds the cadence, or when a prediction input's share is below `min_share`."""
    age_days = table_freshness(context.store, table, as_of)
    max_age = max_age_days(table)
    ok = age_days is not None and age_days <= max_age
    share = None
    if table.ticker_col is not None:
        last_by_key = last_dates_by_key(context.store, table)
        if table in PREDICTION_INPUTS or not set(universe).isdisjoint(last_by_key):
            share = fresh_share(last_by_key, universe, as_of, max_age)
    if table in PREDICTION_INPUTS:
        ok = ok and share is not None and share >= min_share
    return {
        "date_column": table.freshness_col,
        "cadence": table.freshness,
        "max_date": None if age_days is None else (as_of.date() - timedelta(days=age_days)).isoformat(),
        "age_days": age_days,
        "max_age_days": max_age,
        "fresh_share": None if share is None else round(share, 4),
        "ok": ok,
    }


def _extraction_status_report(context: Context, *, as_of: pd.Timestamp | None = None) -> dict[str, object]:
    """Freshness report over exactly the tables and cadences declared by the schema registry."""
    as_of = (as_of if as_of is not None else pd.Timestamp.today()).normalize()
    universe = load_universe_tickers(context)
    min_share = float(context.config.data_extract.prediction_fresh_share)
    statuses = {table.name: _table_status(context, table, universe, as_of, min_share) for table in freshness_tables()}
    behind = [name for name, status in statuses.items() if not status["ok"]]
    for name in behind:
        status = statuses[name]
        context.log.warning(
            "Extraction freshness RED: %s max_date=%s age_days=%s (max %s) fresh_share=%s",
            name,
            status["max_date"],
            status["age_days"],
            status["max_age_days"],
            status["fresh_share"],
        )
    context.log.info("Extraction freshness report ok=%s behind=%s", not behind, behind)
    return {"as_of": as_of.date().isoformat(), "ok": not behind, "behind": behind, "tables": statuses}


@cli.command(name="extraction-status", help="Report every schema-declared extraction table's freshness (JSON); RED tables warn, the exit code is 0.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def extraction_status(config_path: str, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    report = _extraction_status_report(context, as_of=run_date(as_of))
    click.echo(json.dumps(report, sort_keys=True))


# --- Universe seed (must run first; everything else resolves the universe from it) ---
@cli.command(help="Seed the sp500_tickers universe (idempotent; scrapes only if empty or --full).", help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def seed_universe(config_path: str, full: bool, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    if full or context.store.row_count(Tables.sp500_tickers) == 0:
        context.log.info(f"Seeding {Tables.sp500_tickers} via the S&P 500 scraper (full={full})")
        get_sp500_tickers(context, as_of=run_date(as_of))
    context.log.info("Universe ready: %d tickers.", len(load_universe_tickers(context)))
    # A wrong CIK is invisible downstream (prices key on the ticker), so it is reported here; warned, not raised,
    # because a genuine recent spin-off has no filing rows either.
    suspect = [r for r in unverified_ciks(context) if r["shape"] == "SUSPECT CIK"]
    for row in suspect:
        context.log.warning(
            "%s: CIK %s appears in NO filing table (%s) yet the ticker HAS price history — verify the company ID against EDGAR",
            row["ticker"],
            row["cik"],
            ", ".join(row["filing_tables_checked"]),
        )
    if suspect:
        context.log.warning("%d universe CIK(s) unconfirmed by any filing table.", len(suspect))


# --- Prices / market / macro ---
@cli.command(
    help="Daily OHLCV, dividend and split ex-dates from one yfinance download (prices, prices_dividends, prices_splits). HEAVY.", help_priority=2
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def price_history(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """`--full` re-pulls the whole window for every requested ticker; new tickers, holes and splits need no flag."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    years_history = int(context.config.data_extract.years_history)
    fetch_prices_and_actions(context, tickers=_tickers(context, tickers), years_history=years_history, full=full, as_of=_market_as_of(as_of))


@cli.command(
    help="FINRA RegSHO daily short volume: raw rows per symbol (sec_short_volume_security) stamped from security_master, "
    "summed per ticker by class into sec_short_interest. Day files: business days from the earlier of the latest stored "
    "day and today minus the 7-day overlap, to today. --full re-fetches from 2018-08-01 and aborts before writing on "
    "a fetch error."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def short_interest(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    names, _ = _ticker_scope(context, tickers)
    fetch_short_interest(context, tickers=names, full=full, as_of=_market_as_of(as_of))


@cli.command(help="ALL macro / market series -> prices_macro (yfinance + FRED). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def macro(config_path: str) -> None:
    """Every non-equity series: market/commodity/FX closes, FRED levels, and the derived spreads + 10Y total-return index."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_macro(context, years_history=context.config.data_extract.macro_years_history)


@cli.command(help="13F institutional holdings (EDGAR by filing date + OpenFIGI cusip map). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FILING_WINDOW_ARGS, **FILING_WINDOW_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def thirteen_f(config_path: str, tickers: str | None, filing_window: str | None, as_of: datetime | None) -> None:
    """`tickers` is the universe the CUSIP map is resolved against, so it is always passed (never None).
    A `-t` run reads no filing past the stored frontier, so it leaves the next window unchanged."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    window = None
    if filing_window:
        parts = filing_window.split(":")
        if len(parts) != 2 or not all(parts):
            raise click.BadParameter("--filing-window must be FROM:TO, e.g. 2024-01-01:2024-03-01")
        window = (parts[0], parts[1])
    names, universe = _ticker_scope(context, tickers)
    fetch_13f(
        context,
        tickers=names,
        years_history=int(config.data_extract.years_history),
        filing_window=window,
        as_of=run_date(as_of),
        all_tickers=universe,
    )


@cli.command(
    name="thirteen-f-backfill",
    help="13F history of NEW universe tickers: SEC 13F data sets (cached ZIPs), then one EDGAR walk over the gap after them.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def thirteen_f_backfill(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """Tickers added inside `sec13f_hr`'s overlap: discover from 2013 Q2, skipping verified completed accessions.
    `-t X` narrows the new tickers (an established X does nothing); `-t X -F` re-reads every data set for X."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    names, universe = _ticker_scope(context, tickers)
    fetch_13f_backfill(context, tickers=names if tickers else None, as_of=run_date(as_of), full=full, all_tickers=universe)


@cli.command(name="thirteen-f-managers", help="FULL 13F portfolios of the superinvestor roster (all securities, CUSIP grain).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*VINTAGES_ARGS, **VINTAGES_KWARGS)
@click.option(*CIK_ARGS, **CIK_KWARGS)
@click.option(*ACCESSION_ARGS, **ACCESSION_KWARGS)
@click.option(*PERIOD_START_ARGS, **PERIOD_START_KWARGS)
@click.option(*PERIOD_END_ARGS, **PERIOD_END_KWARGS)
@click.option(*FILED_THROUGH_ARGS, **FILED_THROUGH_KWARGS)
@click.option(*MAX_FILINGS_ARGS, **MAX_FILINGS_KWARGS)
@click.option(*ARCHIVE_ARGS, **ARCHIVE_KWARGS)
@click.option(*DRY_RUN_ARGS, **DRY_RUN_KWARGS)
def thirteen_f_managers(
    config_path: str,
    years: int | None,
    vintages: bool,
    ciks: tuple[str, ...],
    accessions: tuple[str, ...],
    period_start: str | None,
    period_end: str | None,
    filed_through: str | None,
    max_filings: int | None,
    archives: tuple[Path, ...],
    dry_run: bool,
) -> None:
    """Scope is the union of every CIK ever in `superinvestor_roster`, so a departed manager keeps its history.
    An empty roster raises: run `superinvestors -F` first on a cold database."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    if vintages:
        if not (period_start and period_end and filed_through and max_filings):
            raise click.UsageError("--vintages requires --period-start, --period-end, --filed-through and --max-filings")
        recover_13f_vintages(
            context,
            ciks=list(ciks),
            start_period=period_start,
            end_period=period_end,
            filed_through=filed_through,
            accessions=list(accessions),
            max_filings=max_filings,
            dry_run=dry_run,
            archives=list(archives) or None,
        )
    else:
        if ciks or accessions or archives or dry_run or period_start or period_end or filed_through or max_filings is not None:
            raise click.UsageError("Version recovery options require --vintages")
        fetch_13f_managers(context, years_history=years or config.data_extract.years_history)


@cli.command(
    help="Superinvestor roster (Dataroma) -> today's `superinvestor_roster` snapshot, written only when it changed. Light. "
    "-F rebuilds first: upsert the committed quarterly Wayback history (configs/superinvestors/) plus every stored live "
    "snapshot, all re-resolved and 13F-gated, then delete stale keys; idempotent. Rerun -F after any overrides.json change."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def superinvestors(config_path: str, full: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    if full:
        rebuild_roster(context)
    upsert_roster_snapshot(context)


# --- Fundamentals: SEC ---
# Facts (network walk) and history (pure replay) are separate commands so a history-layer bug never costs a re-download.
# `-F/--full` re-reads every listed filing, stored ones included (see `run_edgar_fetch`).


@cli.command(
    name="fundamentals-facts",
    help="SEC per-filing XBRL -> fundamentals_facts, resolved from each filer's own calculation linkbase. As-filed only; append-only.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def fundamentals_facts(config_path: str, tickers: str | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fundamentals_sec(
        context,
        tickers=_tickers(context, tickers),
        full=full,
        years_history=int(config.data_extract.years_history),
        as_of=run_date(as_of),
        no_cap=no_cap,
    )


@cli.command(
    name="fundamentals-employees",
    help="SEC 10-K prose -> fundamentals_employees: total, full-time and part-time counts with basis, status, filer CIK, accession "
    "and source quotes, one row per filing date over the registrant lineage. Dates that already have a row (NULL-status rows included) "
    "are skipped unless --full. A failed LLM call or ticker is logged and retried next run; the task exits 0.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fundamentals_employees(config_path: str, tickers: str | None, full: bool) -> None:
    _llm_step(config_path, tickers, full).run_employees()


@cli.command(
    name="fundamentals-history-sec",
    help="fundamentals_facts -> fundamentals_history_sec + _reason_codes, on the "
    "publication-event grain. No network. Append-only: refuses to overwrite "
    "an already-published row unless --full deletes the requested tickers' history and reason codes first. Stored facts are unchanged; use `fundamentals --rebuild` for a facts-resolution bug.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*VERIFY_HISTORY_ARGS, **VERIFY_HISTORY_KWARGS)
def fundamentals_history_sec(config_path: str, tickers: str | None, full: bool, verify_history: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    build_fundamentals_history(context, tickers=_tickers(context, tickers), rebuild_history=full, verify_history=verify_history)


def _delete_fundamentals_layers(context: Context, names: list[str]) -> None:
    """Delete each ticker's facts, history and reason-code rows, so the next fetch refetches every filing."""
    for ticker in names:
        for table in (
            Tables.fundamentals_facts,
            Tables.fundamentals_history_sec,
            Tables.fundamentals_reason_codes,
        ):
            context.store.delete(table, {"ticker": ticker})
    context.log.warning("fundamentals: --rebuild deleted the facts/history tables for %d ticker(s); every XBRL filing will be refetched", len(names))


@cli.command(help="Both fundamentals layers in order: facts (network) then history (replay).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*REBUILD_ARGS, **REBUILD_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*VERIFY_HISTORY_ARGS, **VERIFY_HISTORY_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def fundamentals(
    config_path: str, tickers: str | None, rebuild: bool, full: bool, verify_history: bool, as_of: datetime | None, no_cap: bool
) -> None:
    """The fetch hands the filing dates it read to the history triage, so a back-dated read replays that ticker in full."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers)
    if rebuild:
        _delete_fundamentals_layers(context, names)
    summary = fetch_fundamentals_sec(
        context, tickers=names, full=full or rebuild, years_history=int(config.data_extract.years_history), as_of=run_date(as_of), no_cap=no_cap
    )
    build_fundamentals_history(
        context,
        tickers=names,
        rebuild_history=rebuild,
        verify_history=verify_history,
        fetched=fetched_filing_dates(summary),
        full_fetch=full or rebuild,
    )


# --- Fundamentals: Sharadar (SF1) ---
# Single-table commands are for a manual targeted refresh; `-F/--full` re-pulls the whole window and makes the merge DELETE first.
# History depth is `data_extract.sharadar_years_history`, not `years_history`.
@cli.command(
    name="fundamentals-sharadar",
    help="The whole Sharadar producer in dependency order: tickers -> SF1 fundamentals -> actions -> sp500 -> the MERGED fundamentals_history.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def fundamentals_sharadar(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """Delegates to `StepExtractFundamentalsSharadar` so the dependency order, ending with the merge, lives in one place."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    step = StepExtractFundamentalsSharadar(context=context, config=config)
    step.run(tickers=_tickers(context, tickers), full=full, config_dir=config_path, as_of=run_date(as_of))


@cli.command(
    name="sharadar-tickers",
    help="Sharadar entity dimension (permaticker, currency, category) -> sharadar_tickers. Full refresh. Prerequisite of fundamentals-sharadar.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def sharadar_tickers(config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_sharadar_tickers(context)


@cli.command(name="sharadar-actions", help="Sharadar corporate actions (dividends, splits, spinoffs, acquisitions, relations) -> sharadar_actions.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def sharadar_actions(config_path: str, full: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_sharadar_actions(context, full=full, years_history=int(config.data_extract.sharadar_years_history), as_of=run_date(as_of))


@cli.command(
    name="sharadar-sp500",
    help="S&P 500 membership events (added / removed / historical, from 1992) -> sharadar_sp500. Ingested only; universe.py is NOT re-pointed (D27).",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def sharadar_sp500(config_path: str, full: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_sharadar_sp500(context, full=full)


# --- Fundamentals: the merged table, its gap check, and the remaining SEC sources ---
@cli.command(
    name="fundamentals-history-merged",
    help="fundamentals_sharadar + fundamentals_history_sec -> fundamentals_history, "
    "the MERGED table every consumer reads. Build only, no network. Both "
    "inputs are read-only, so the rollback is a drop-and-rebuild.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fundamentals_history_merged(config_path: str, tickers: str | None, full: bool) -> None:
    """Field-block precedence: each column has one declared source for all history and never switches mid-series.

    `--full` deletes these tickers' rows before rebuilding; the default upsert cannot remove a row that no longer exists.
    """
    _, context = get_config_context(config_path, use_cache=False, save=False)
    build_merged_history(context, tickers=_tickers(context, tickers), full=full, config_dir=config_path)


@cli.command(
    name="sharadar-gap-check",
    help="READ-ONLY: where Sharadar and the SEC layer disagree on their SHARED "
    "dates, and which gaps are SYSTEMATIC enough to be a basis conflict. "
    "Writes a markdown report and, with --propose, INERT override candidates.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*REPORT_PATH_ARGS, **REPORT_PATH_KWARGS)
@click.option(*PROPOSE_ARGS, **PROPOSE_KWARGS)
def sharadar_gap_check(config_path: str, tickers: str | None, report_path: str | None, propose: bool) -> None:
    """Read-only Sharadar-vs-SEC disagreement report; reason codes stay with the SEC table and do not gate `fundamentals_history`.

    `--tickers` defaults to every stored ticker, not the sp500 universe. Never imports `src/validate/`.
    """
    _, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers) if tickers else None
    run_gap_check(context, tickers=names, report_path=report_path or GAP_REPORT_PATH, propose_overrides=propose, config_dir=config_path)


@cli.command(help="Earnings surprises -> historical forward P/E (the prices step's `run_earnings_surprises`).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def earnings_surprises(config_path: str, tickers: str | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    StepExtractPrices(context=context, config=config).run_earnings_surprises(_tickers(context, tickers))


@cli.command(
    name="insider-edgar",
    help="The EDGAR half of the insider table: every indexed Forms 3/4/5 filing of the ticker's filing scope after the last "
    "stored zip quarter not yet stored from EDGAR. An EDGAR walk, so the DAG runs it in the sec_api pool, after "
    "`zips-ingest --only insider` (the zip half). Both halves by hand: `zips-ingest --only insider [-F]`, then `insider-edgar [-F]`.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def insider_edgar(config_path: str, tickers: str | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _edgar_step(config_path, tickers, None, full, as_of, no_cap).run_insider()


_ZIP_DOWNLOADS = ("insider", "notes", "statements", "ftd", "sec-tickers")
_ZIP_INGESTS = ("insider", "notes", "statements", "ftd")


@cli.command(
    name="zips-download",
    help="Cache the SEC bulk zips (Form 3/4/5, Notes, Financial Statements, FTD) and snapshot SEC's current tickers "
    f"(the one table written). --only runs the named source(s) ({', '.join(_ZIP_DOWNLOADS)}). SEC-bulk.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option("--only", "only", type=click.Choice(_ZIP_DOWNLOADS), multiple=True, help="Run only this source; repeatable.")
def zips_download(config_path: str, full: bool, as_of: datetime | None, only: tuple[str, ...]) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    step = StepExtractZips(context, config, ZipsOptions(full=full, as_of=run_date(as_of)))
    sources = {
        "insider": step.run_download_insider,
        "notes": step.run_download_financial_notes,
        "statements": step.run_download_financial_statements,
        "ftd": step.run_download_fails_to_deliver,
        "sec-tickers": step.run_download_sec_tickers,
    }
    for name in only or _ZIP_DOWNLOADS:
        sources[name]()


@cli.command(
    name="zips-ingest",
    help="Parse the cached SEC bulk zips into insider_transactions (zip quarters), notes_num/notes_text, pension_facts and the "
    f"two FTD tables, downloading a pending period not cached yet. --only runs the named source(s) ({', '.join(_ZIP_INGESTS)}); "
    "-F re-parses every cached period; --repair-availability only repairs the stored Notes clocks. SEC-bulk.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*REPAIR_AVAILABILITY_ARGS, **REPAIR_AVAILABILITY_KWARGS)
@click.option("--only", "only", type=click.Choice(_ZIP_INGESTS), multiple=True, help="Run only this source; repeatable.")
def zips_ingest(config_path: str, tickers: str | None, full: bool, as_of: datetime | None, repair_availability: bool, only: tuple[str, ...]) -> None:
    if repair_availability and set(only) != {"notes"}:
        raise click.UsageError("--repair-availability needs `--only notes` and no other source.")
    config, context = get_config_context(config_path, use_cache=False, save=False)
    options = ZipsOptions(
        tickers=_tickers(context, tickers) if tickers else None, full=full, as_of=run_date(as_of), repair_availability=repair_availability
    )
    step = StepExtractZips(context, config, options)
    sources = {
        "insider": step.run_ingest_insider,
        "notes": step.run_ingest_financial_notes,
        "statements": step.run_ingest_financial_statements,
        "ftd": step.run_ingest_fails_to_deliver,
    }
    for name in only or _ZIP_INGESTS:
        sources[name]()


_IDENTITY_STAGES = ("capture-dei", "capture-ftd", "tenure", "lineage", "master", "propagate")


@cli.command(
    name="identity",
    help="Identity layer, OFFLINE (cached zips + DB): capture the Notes cover-page symbols and the FTD lines, then build "
    "symbol_tenure -> entity_lineage -> security_master and propagate the lineage changes. --only runs one stage "
    f"({', '.join(_IDENTITY_STAGES)}); --full re-captures every cached zip and re-checks every ticker. Cold start: "
    "`identity`, then `identity --only capture-ftd -F`, then `identity --only master`.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*APPROVE_REKEY_ARGS, **APPROVE_REKEY_KWARGS)
@click.option(*DRY_RUN_ARGS, **DRY_RUN_KWARGS)
@click.option("--only", "only", type=click.Choice(_IDENTITY_STAGES), default=None, help="Run this one stage instead of all six.")
def identity(
    config_path: str, tickers: str | None, full: bool, as_of: datetime | None, approved_rekeys: tuple[str, ...], dry_run: bool, only: str | None
) -> None:
    if dry_run and only != "propagate":
        raise click.UsageError("--dry-run applies to `--only propagate` only.")
    if approved_rekeys and only not in (None, "lineage"):
        raise click.UsageError("--approve-rekey applies to the full run or `--only lineage`.")
    config, context = get_config_context(config_path, use_cache=False, save=False)
    options = IdentityOptions(
        tickers=_tickers(context, tickers) if tickers else None,
        full=full,
        dry_run=dry_run,
        as_of=run_date(as_of),
        approved_rekeys=_approved_rekeys(approved_rekeys),
    )
    step = StepExtractIdentity(context, config, options)
    stages = {
        "capture-dei": step.run_capture_dei_symbols,
        "capture-ftd": step.run_capture_ftd_cusips,
        "tenure": step.run_symbol_tenure,
        "lineage": step.run_entity_lineage,
        "master": step.run_security_master,
        "propagate": step.run_propagate,
    }
    if only is None:
        step.run()
    elif dry_run:
        removals = step.run_propagate().removals
        click.echo(removals.to_string(index=False) if not removals.empty else "identity --only propagate: no pending removals")
    else:
        stages[only]()


def _approved_rekeys(values: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """`--approve-rekey OLD:NEW` values as `(old, new)` entity-id pairs."""
    parsed: set[tuple[str, str]] = set()
    for value in values:
        old, separator, new = value.partition(":")
        if not separator or not old or not new:
            raise click.BadParameter("expected OLD_ENTITY_ID:NEW_ENTITY_ID", param_hint="--approve-rekey")
        parsed.add((old, new))
    return frozenset(parsed)


# --- Structure (governance) ---
@cli.command(help="DEF 14A governance / executive pay (LLM-parsed). SEC-api + LLM.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def def14a(config_path: str, tickers: str | None) -> None:
    """Every run lists each ticker's whole `years_history`; `--full` also re-sends the saved proxies without
    evidence, markers included (a proxy with stored evidence is never re-sent)."""
    _llm_step(config_path, tickers).run_def14a()


@cli.command(help="8-K events: item codes + has_earnings/has_press_release (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def sec_8k_items(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _edgar_step(config_path, tickers, years, full, as_of, no_cap).run_8k()


@cli.command(help="Shareholder vote tallies from the STORED 8-K Item 5.07 narratives (LLM). No download — reads sec_8k.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def sec_8k_votes(config_path: str, tickers: str | None) -> None:
    _llm_step(config_path, tickers).run_8k_votes()


@cli.command(help="SC 13D activist filings + amendments: reporting persons, CUSIP, ownership (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def sec_13d(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _edgar_step(config_path, tickers, years, full, as_of, no_cap).run_13d()


@cli.command(help="SC 13G passive 5%+ beneficial ownership + amendments (edgartools). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def sec_13g(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    """Passive counterpart of `sec-13d`. A schedule a universe company only FILED is stored as an empty-filing marker under it."""
    _edgar_step(config_path, tickers, years, full, as_of, no_cap).run_13g()


@cli.command(help="Filing text: 10-K Item 1A (Risk Factors) + Item 7 (MD&A) & 10-Q Item 2 (MD&A). SEC-api.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def filing_text(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _edgar_step(config_path, tickers, years, full, as_of, no_cap).run_filing_text()


@cli.command(help="DEF 14A structured: pay-vs-performance, audit fees, comp/ownership/vote tables (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
@click.option(*NO_CAP_ARGS, **NO_CAP_KWARGS)
def def14a_edgar(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _edgar_step(config_path, tickers, years, full, as_of, no_cap).run_def14a_ecd()


def _edgar_step(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> StepExtractEdgar:
    """The EDGAR step over the --tickers subset (or the universe), the -y window (or the configured one) and the run options."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    return StepExtractEdgar(
        context, config, EdgarOptions(tickers=_tickers(context, tickers), years=years, full=full, as_of=run_date(as_of), no_cap=no_cap)
    )


def _llm_step(config_path: str, tickers: str | None, full: bool = False) -> StepExtractLlm:
    """The LLM step over the --tickers subset (or the universe); `full` re-decides every employee filing date."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    return StepExtractLlm(context, config, LlmOptions(tickers=_tickers(context, tickers), full=full))


# --- EDGAR index, work-list dry run and empty-filing markers ---
@cli.command(
    name="edgar-index",
    help="Download the local EDGAR filing index (closed quarters once, the current quarter again) and print rows per quarter. "
    "--full downloads every quarter again.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def edgar_index_command(config_path: str, full: bool, as_of: datetime | None) -> None:
    """One EDGAR walk: run it alone. The first build fetches about 128 quarters."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    context.ensure_edgar_identity()
    edgar_index.refresh(context, run_date(as_of), int(config.data_extract.years_history), rebuild=full)
    for quarter, n_rows in edgar_index.quarter_counts(context).items():
        click.echo(f"{quarter}\t{n_rows}")


def _document_fetches(context: Context, cik_map: pd.DataFrame) -> dict[str, EdgarFetch]:
    """Every EDGAR document fetch, keyed by its done table's name."""
    fetches = (
        SEC_8K_FETCH,
        SEC_13D_FETCH,
        SEC_13G_FETCH,
        FILING_TEXT_FETCH,
        DEF14A_EDGAR_FETCH,
        fundamentals_fetch(context, cik_map),
        INSIDER_EDGAR_FETCH,
    )
    return {fetch.done.name: fetch for fetch in fetches}


@cli.command(name="resume-plan", help="READ-ONLY: each EDGAR document table's work list for the run date, by key class, from the cached index.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*TABLE_ARGS, **TABLE_KWARGS)
@click.option(*TIMINGS_ARGS, **TIMINGS_KWARGS)
@click.option(*AS_OF_ARGS, **AS_OF_KWARGS)
def resume_plan(config_path: str, tickers: str | None, table_name: str | None, timings: bool, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    run = EdgarRun.load(context, _tickers(context, tickers), as_of=as_of)
    fetches = _document_fetches(context, run.cik_map)
    if table_name is not None and table_name not in fetches:
        raise click.BadParameter(f"not an EDGAR document table: {table_name}; one of {sorted(fetches)}", param_hint="--table")
    for name, fetch in fetches.items():
        if table_name is not None and name != table_name:
            continue
        work = plan_fetch(context, fetch, run)
        report: dict[str, object] = {
            "table": name,
            "documents": work.size,
            "uncapped": work.uncapped,
            "keys": len(work.units),
            "by_class": work.counts,
        }
        if timings:
            report["timings_s"] = {k: round(v, 2) for k, v in work.timings.items()}
        click.echo(json.dumps(report, sort_keys=True))


@cli.command(help="Count (or, for a rollback only, delete) empty-filing marker rows by their declared sentinel.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TABLE_ARGS, **TABLE_KWARGS)
@click.option(*COUNT_ARGS, **COUNT_KWARGS)
@click.option(*DELETE_ARGS, **DELETE_KWARGS)
def markers(config_path: str, table_name: str | None, count_only: bool, delete: bool) -> None:
    if count_only and delete:
        raise click.UsageError("--count and --delete are exclusive")
    _, context = get_config_context(config_path, use_cache=False, save=False)
    tables: list[Table] = [resolve(table_name)] if table_name else list(marker_tables())
    for table in tables:
        if table.empty_marker is None:
            raise click.BadParameter(f"{table.name} declares no empty-filing marker", param_hint="--table")
        column, sentinel = table.empty_marker
        df = context.store.load(table, columns=[column], where={column: sentinel}, markers=True, optional=True)
        count = 0 if df is None else len(df)
        deleted = context.store.delete(table, {column: sentinel}) if delete and count else 0
        click.echo(json.dumps({"table": table.name, "markers": count, "deleted": deleted}, sort_keys=True))


# --- Behavioral (retail attention) ---
@cli.command(
    help="Earnings-call transcripts (HuggingFace defeatbeta) -> earnings_call_sections, one row per "
    "paragraph. Every run compares complete scoped metadata for current and historical symbols; payloads are read only for missing or reissued calls. "
    "-F also removes source-absent calls for tickers still present in the source."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def extract_earnings_calls(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    _extract_earnings_calls(context, config.earnings_calls, full=full, tickers=_tickers(context, tickers))
