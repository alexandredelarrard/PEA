"""Data-extraction CLI: one command per data source, so the Airflow extraction DAG can schedule each independently.

    python -m src data_extract <command> [-c ./configs] [-t AAPL,MSFT]

Every command builds a fresh Context, resolves the ticker universe (or a --tickers subset) and runs its
fetcher. Fetchers resume from the DB, so a nightly rerun pulls only new data. `seed-universe` must run first.
"""

import json
from typing import Any, cast

import click
import pandas as pd

from src.constants.command_line_interface import (
    CONFIG_ARGS,
    FULL_ARGS,
    TICKERS_ARGS,
    YEARS_ARGS,
)
from src.constants.command_line_interface import (
    CONFIG_KWARGS as _CONFIG_KWARGS,
)
from src.constants.command_line_interface import (
    FULL_KWARGS as _FULL_KWARGS,
)
from src.constants.command_line_interface import (
    TICKERS_KWARGS as _TICKERS_KWARGS,
)
from src.constants.command_line_interface import (
    YEARS_KWARGS as _YEARS_KWARGS,
)
from src.constants.constants import DATA_FRESHNESS_MAX_AGE_DAYS
from src.context import Context, get_config_context
from src.data_extract.identity_propagate import propagate_identity
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import (
    extract_earnings_calls as _extract_earnings_calls,
)
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.edgar_driver import run_edgar_fetch
from src.data_extract.utils.common.entity_lineage import build_entity_lineage
from src.data_extract.utils.common.sec_tickers import download_sec_company_tickers
from src.data_extract.utils.common.security_master import build_security_master
from src.data_extract.utils.common.symbol_tenure import build_symbol_tenure, scan_form345_cache
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_financial_notes import download_financial_notes, fetch_financial_notes
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec
from src.data_extract.utils.fundamentals.fundamentals_employees import fetch_fundamentals_employees
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
from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.institutionals.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import download_fails_to_deliver, fetch_fails_to_deliver
from src.data_extract.utils.institutionals.fetch_insider_edgar import fetch_insider_edgar
from src.data_extract.utils.institutionals.fetch_insider_transactions import download_insider_transactions, fetch_insider_transactions
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.institutionals.fetch_superinvestors import seed_roster_history, upsert_roster_snapshot
from src.data_extract.utils.prices.fetch_dividends import fetch_dividends
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import fetch_price_history
from src.data_extract.utils.prices.fetch_splits import fetch_splits
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_extract.utils.structure.def14a import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.structure.votes import fetch_8k_votes_llm
from src.data_store.schema import Tables, freshness_tables
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers, unverified_ciks

CONFIG_KWARGS = cast(dict[str, Any], _CONFIG_KWARGS)
TICKERS_KWARGS = cast(dict[str, Any], _TICKERS_KWARGS)
FULL_KWARGS = cast(dict[str, Any], _FULL_KWARGS)
YEARS_KWARGS = cast(dict[str, Any], _YEARS_KWARGS)


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA EXTRACTION — one command per source (scheduled by the Airflow extraction DAG)."""


def _tickers(context: Context, tickers: str | None) -> list[str]:
    """The --tickers subset if given, else the full sp500_tickers universe."""
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return load_universe_tickers(context)


def _extraction_status_report(context: Context, *, as_of: pd.Timestamp | None = None) -> dict[str, object]:
    """Measure exactly the tables and cadences declared by the schema registry."""
    as_of = (as_of if as_of is not None else pd.Timestamp.today()).normalize()
    statuses: dict[str, dict[str, object]] = {}
    for table in freshness_tables():
        cadence = cast(str, table.freshness)
        maximum = context.store.max_date(table, table.freshness_col)
        age_days = None if maximum is None else int((as_of - maximum).days)
        max_age_days = DATA_FRESHNESS_MAX_AGE_DAYS[cadence]
        statuses[table.name] = {
            "date_column": table.freshness_col,
            "cadence": cadence,
            "max_date": None if maximum is None else maximum.date().isoformat(),
            "age_days": age_days,
            "max_age_days": max_age_days,
            "ok": age_days is not None and age_days <= max_age_days,
        }
    behind = [name for name, status in statuses.items() if not status["ok"]]
    context.log.info("Extraction freshness gate ok=%s behind=%s", not behind, behind)
    return {"as_of": as_of.date().isoformat(), "ok": not behind, "behind": behind, "tables": statuses}


@cli.command(name="extraction-status", help="Fail unless every schema-declared extraction table is fresh enough for aggregation.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def extraction_status(config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    report = _extraction_status_report(context)
    click.echo(json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise click.ClickException("stale or incomplete extraction tables: " + ", ".join(cast(list[str], report["behind"])))


# --- Universe seed (must run first; everything else resolves the universe from it) ---
@cli.command(help="Seed the sp500_tickers universe (idempotent; scrapes only if empty or --refresh).", help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("--refresh", is_flag=True, default=False, help="Re-scrape the S&P 500 even if populated.")
def seed_universe(config_path: str, refresh: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    if refresh or context.store.row_count(Tables.sp500_tickers) == 0:
        context.log.info(f"Seeding {Tables.sp500_tickers} via the S&P 500 scraper (refresh={refresh})")
        get_sp500_tickers(context)
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
@cli.command(help="Daily price history, OHLCV (yfinance). HEAVY.", help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def price_history(config_path: str, tickers: str | None, full: bool) -> None:
    """`--full` re-pulls the whole window: split adjustment is retroactive and an incremental upsert never revisits restated bars."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_price_history(context, tickers=_tickers(context, tickers), years_history=context.config.data_extract.years_history, full=full)


@cli.command(help="Cash-dividend ex-dates (yfinance). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def dividends(config_path: str, tickers: str | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_dividends(context, tickers=_tickers(context, tickers), years_history=context.config.data_extract.years_history)


@cli.command(help="Share-split ex-dates (yfinance) -> prices_splits. HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def splits(config_path: str, tickers: str | None, full: bool) -> None:
    """Fills holes in `sharadar_actions`; use `--full` on a cold table, since resuming from an empty frontier misses all history."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_splits(context, tickers=_tickers(context, tickers), years_history=context.config.data_extract.years_history, full=full)


@cli.command(help="FINRA RegSHO short interest / short volume.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def short_interest(config_path: str, tickers: str | None, full: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_short_interest(context, tickers=_tickers(context, tickers), years_history=context.config.data_extract.years_history, full=full)


@cli.command(help="SEC fails-to-deliver (settlement fails). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fails_to_deliver(config_path: str, tickers: str | None, full: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fails_to_deliver(context, tickers=_tickers(context, tickers), full=full)


@cli.command(help="ALL macro / market series -> prices_macro (yfinance + FRED). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def macro(config_path: str) -> None:
    """Every non-equity series: market/commodity/FX closes, FRED levels, and the derived spreads + 10Y total-return index."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_macro(context, years_history=context.config.data_extract.macro_years_history)


@cli.command(help="13F institutional holdings (EDGAR by filing date + OpenFIGI cusip map). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--filing-window",
    default=None,
    metavar="FROM:TO",
    help="BACKFILL a lost filing season, e.g. 2024-01-01:2024-03-01. Replaces the "
    "watermark and advances nothing -- a gap BEHIND max(filing_date) is "
    "unreachable by the normal 7-day lookback. ONE EDGAR WALK AT A TIME.",
)
def thirteen_f(config_path: str, tickers: str | None, filing_window: str | None) -> None:
    """`tickers` is the universe the CUSIP map is resolved against, so it is always passed (never None)."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    window = None
    if filing_window:
        parts = filing_window.split(":")
        if len(parts) != 2 or not all(parts):
            raise click.BadParameter("--filing-window must be FROM:TO, e.g. 2024-01-01:2024-03-01")
        window = (parts[0], parts[1])
    fetch_13f(context, tickers=_tickers(context, tickers), filing_window=window)


@cli.command(name="thirteen-f-managers", help="FULL 13F portfolios of the superinvestor roster (all securities, CUSIP grain).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def thirteen_f_managers(config_path: str, years: int | None) -> None:
    """Scope is the union of every CIK ever in `superinvestor_roster`, so a departed manager keeps its history.
    An empty roster raises: run `superinvestors --seed` first on a cold database."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_13f_managers(context, years_history=years or config.data_extract.years_history)


@cli.command(help="Superinvestor roster (Dataroma) -> today's `superinvestor_roster` snapshot. Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(
    "--seed",
    is_flag=True,
    default=False,
    help="ONE-OFF: also replay the 13 committed web.archive.org captures (2013-2026) so the roster has a history to be point-in-time about.",
)
def superinvestors(config_path: str, seed: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    if seed:
        seed_roster_history(context)
    upsert_roster_snapshot(context)


# --- Fundamentals: SEC ---
# Facts (network walk) and history (pure replay) are separate commands so a history-layer bug never costs a re-download.
# `-F/--full` bypasses the manifest's ticker-count incremental window, which a chunked backfill defeats (see `run_edgar_fetch`).


@cli.command(
    name="fundamentals-facts",
    help="SEC per-filing XBRL -> fundamentals_facts, resolved from each filer's own calculation linkbase. As-filed only; append-only.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fundamentals_facts(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fundamentals_sec(context, tickers=_tickers(context, tickers), full=full, years_history=int(config.data_extract.years_history))


@cli.command(
    name="fundamentals-employees",
    help="SEC 10-K prose -> fundamentals_employees, skipping filing dates that already have a row (NULL decisions included), over the registrant lineage.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fundamentals_employees(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fundamentals_employees(
        context,
        tickers=_tickers(context, tickers),
        full=full,
        years_history=int(config.data_extract.years_history),
    )


@cli.command(
    name="fundamentals-history-sec",
    help="fundamentals_facts -> fundamentals_history_sec + _reason_codes, on the "
    "publication-event grain. No network. Append-only: refuses to overwrite "
    "an already-published row unless --rebuild-history is passed.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--rebuild-history",
    is_flag=True,
    help="Delete these tickers' fundamentals_history_sec / _reason_codes rows and "
    "rebuild from the facts ALREADY STORED. For a bug in the history layer; "
    "costs no network. (Use `fundamentals --rebuild` for a resolution bug.)",
)
def fundamentals_history_sec(config_path: str, tickers: str | None, rebuild_history: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    build_fundamentals_history(context, tickers=_tickers(context, tickers), rebuild_history=rebuild_history)


@cli.command(help="Both fundamentals layers in order: facts (network) then history (replay).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Delete these tickers' rows from BOTH layers and refetch every filing. "
    "For a bug in the RESOLUTION layer, where the stored facts are themselves "
    "wrong. A deleted ticker looks exactly like a never-fetched one to the "
    "fetcher's accession-set resume, so there is no third state to reason "
    "about. There is no build_version column: the rebuild IS the version.",
)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def fundamentals(config_path: str, tickers: str | None, rebuild: bool, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers)
    if rebuild:
        for ticker in names:
            for table in (
                Tables.fundamentals_facts,
                Tables.fundamentals_history_sec,
                Tables.fundamentals_reason_codes,
            ):
                context.store.delete(table, {"ticker": ticker})
        context.log.warning(
            "fundamentals: --rebuild deleted the facts/history tables for %d ticker(s); every XBRL filing will be refetched", len(names)
        )
    fetch_fundamentals_sec(context, tickers=names, full=full or rebuild, years_history=int(config.data_extract.years_history))
    build_fundamentals_history(context, tickers=names, rebuild_history=rebuild)


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
def fundamentals_sharadar(config_path: str, tickers: str | None, full: bool) -> None:
    """Delegates to `StepExtractFundamentalsSharadar` so the dependency order, ending with the merge, lives in one place."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    StepExtractFundamentalsSharadar(context=context, config=config).run(tickers=_tickers(context, tickers), full=full, config_dir=config_path)


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
def sharadar_actions(config_path: str, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_sharadar_actions(context, full=full, years_history=int(config.data_extract.sharadar_years_history))


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
@click.option("--out", "report_path", default=GAP_REPORT_PATH, show_default=True, help="Where to write the findings markdown.")
@click.option(
    "--propose",
    is_flag=True,
    help="Merge candidate entries into sharadar_source_overrides.json with "
    "`approved: null`. They change NOTHING until a human adjudicates them, "
    "and an entry that already exists is never touched.",
)
def sharadar_gap_check(config_path: str, tickers: str | None, report_path: str, propose: bool) -> None:
    """Read-only Sharadar-vs-SEC disagreement report; reason codes stay with the SEC table and do not gate `fundamentals_history`.

    `--tickers` defaults to every stored ticker, not the sp500 universe. Never imports `src/validate/`.
    """
    _, context = get_config_context(config_path, use_cache=False, save=False)
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    run_gap_check(context, tickers=names, report_path=report_path, propose_overrides=propose, config_dir=config_path)


@cli.command(help="Earnings surprises -> historical forward P/E.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def earnings_surprises(config_path: str, tickers: str | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_earnings_surprises(context, tickers=_tickers(context, tickers))


@cli.command(help="SEC Financial Statement Data Sets -> pension_facts (num/sub XBRL). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help="Re-read every cached period even when already ingested. For a PARSE change -- a new column, or a registrant-resolution change -- not a data change. Nothing is re-downloaded.",
)
def financial_statements(config_path: str, tickers: str | None, reparse: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_financial_statements(context, tickers=_tickers(context, tickers), reparse=reparse)


@cli.command(help="SEC insider transactions (Forms 3/4/5): quarterly bulk history plus the daily EDGAR tail.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help="Re-read every cached quarter even when already ingested. For a PARSE change (a new column), not a data change -- nothing is re-downloaded.",
)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option(
    "--bulk-only",
    is_flag=True,
    default=False,
    help="Parse the quarterly zips only; the daily EDGAR tail is left to `insider-edgar` (the DAG runs it in the sec_api pool).",
)
def insider_transactions(
    config_path: str,
    tickers: str | None,
    reparse: bool,
    full: bool,
    bulk_only: bool,
) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers)
    fetch_insider_transactions(context, tickers=names, reparse=reparse)
    if not bulk_only:
        fetch_insider_edgar(context, tickers=names, years_history=int(config.data_extract.years_history), full=full)


@cli.command(name="insider-edgar", help="SEC insider transactions: the daily EDGAR tail (Forms 3/4/5) after the latest bulk quarter. SEC-API.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def insider_edgar(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_insider_edgar(context, tickers=_tickers(context, tickers), years_history=int(config.data_extract.years_history), full=full)


@cli.command(
    name="insider-download",
    help="Cache the SEC Form 3/4/5 quarterly zips (every quarter since 2006) that identity-tables and insider-transactions read. SEC-bulk.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def insider_download(config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    download_insider_transactions(context)


@cli.command(
    name="ftd-download",
    help="Cache the SEC fails-to-deliver zips and store their in-scope lines raw, per CUSIP "
    "(sec_fails_to_deliver_security), for the security master. SEC-bulk.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def ftd_download(config_path: str, full: bool) -> None:
    """`--full` re-reads every cached zip and replaces the raw table; the default reads only periods not yet stored."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    download_fails_to_deliver(context, years_history=int(config.data_extract.years_history), full=full)


@cli.command(name="sec-tickers", help="Snapshot SEC's current tickers with exchanges (one GET) into sec_company_tickers. SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def sec_tickers(config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    download_sec_company_tickers(context)


@cli.command(
    name="identity-tables",
    help="symbol_tenure + entity_lineage + security_master: WHICH COMPANY a ticker was, and when, and "
    "which security each FTD line is. OFFLINE reference build -- derived from the cached Form 345 zips "
    "and the DB, no network. Run after `insider-download`, `notes-download`, `ftd-download` and `sec-tickers`.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(
    "--approve-rekey",
    "approved_rekeys",
    multiple=True,
    metavar="OLD_ENTITY_ID:NEW_ENTITY_ID",
    help="Apply one exact older-CIK entity-id change that the build excluded and backlogged (WARNING log).",
)
def identity_tables(config_path: str, approved_rekeys: tuple[str, ...]) -> None:
    """Builds `symbol_tenure`, `entity_lineage` then `security_master` together: lineage candidates are read off
    tenure and the master's issuers off the lineage, so a stale input silently narrows the next. Market-wide,
    so the manifest records `ticker_count=0`.
    """
    _, context = get_config_context(config_path, use_cache=False, save=False)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)
    parsed_rekeys: set[tuple[str, str]] = set()
    for value in approved_rekeys:
        old, separator, new = value.partition(":")
        if not separator or not old or not new:
            raise click.BadParameter(
                "expected OLD_ENTITY_ID:NEW_ENTITY_ID",
                param_hint="--approve-rekey",
            )
        parsed_rekeys.add((old, new))
    scan = scan_form345_cache(cache)
    tenure = build_symbol_tenure(context, scan, config_path)
    lineage = build_entity_lineage(
        context,
        tenure,
        scan.owner_pairs,
        config_path,
        approved_rekeys=frozenset(parsed_rekeys),
        redundant_symbols=frozenset(context.config.data_extract.redundant_ticks),
    )
    build_security_master(context, lineage, config_path)


@cli.command(
    name="identity-propagate",
    help="Carry an entity_lineage change into the stored SEC rows: purge rows whose filer CIK left the "
    "ticker's entity (one WARNING per table), re-parse notes, pension and insider bulk from cache, "
    "re-resolve FTD for changed symbols, rebuild purged tickers' history. OFFLINE. Run after "
    "`identity-tables`; EDGAR tables relist in their own fetchers. sec_short_interest is never touched.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--dry-run", is_flag=True, help="List the pending removals (table, ticker, cik, window, rows); delete, re-parse and rebuild nothing.")
def identity_propagate(config_path: str, tickers: str | None, dry_run: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    result = propagate_identity(context, _tickers(context, tickers), dry_run=dry_run)
    if dry_run:
        click.echo(result.removals.to_string(index=False) if not result.removals.empty else "identity-propagate: no pending removals")


@cli.command(help="SEC Financial Statement & NOTES sets -> notes_num / notes_text. VERY HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help="Re-read every cached period even when already ingested. For a PARSE change -- a new column, or a registrant-resolution change -- not a data change. Nothing is re-downloaded.",
)
@click.option(
    "--repair-availability",
    is_flag=True,
    default=False,
    help="Rewrite notes available_at metadata using the historical estimate or cached download date; do not reparse ZIPs.",
)
def financial_notes(config_path: str, tickers: str | None, reparse: bool, repair_availability: bool) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_financial_notes(context, tickers=_tickers(context, tickers), reparse=reparse, repair_availability=repair_availability)


@cli.command(
    name="notes-download",
    help="Cache the SEC Notes zips and capture every filer's cover-page symbols (dei:TradingSymbol) into symbol_tenure. SEC-bulk.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def notes_download(config_path: str, full: bool) -> None:
    """`--full` re-captures every cached zip (one-off backfill); the default captures only periods not yet stored."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    download_financial_notes(context, full=full)


# --- Structure (governance) ---
@cli.command(help="DEF 14A governance / executive pay (LLM-parsed). SEC-api + LLM.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def def14a(config_path: str, tickers: str | None, full: bool) -> None:
    """`--full` is needed after a registrant chain grows: the incremental gates are run-wide (global `last_run_date`,
    any-rows ticker check), so a same-day rerun would skip the newly exposed history."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_def14a_llm(context, config, tickers=_tickers(context, tickers), full=full)


@cli.command(help="8-K events: item codes + has_earnings/has_press_release (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def sec_8k_items(config_path: str, tickers: str | None, years: int | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(
        context, tickers=_tickers(context, tickers), years_history=years or config.data_extract.years_history, fetch=SEC_8K_FETCH, full=full
    )


@cli.command(help="Shareholder vote tallies from the STORED 8-K Item 5.07 narratives (LLM). No download — reads sec_8k.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def sec_8k_votes(config_path: str, tickers: str | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_8k_votes_llm(context, config, tickers=_tickers(context, tickers))


@cli.command(help="SC 13D activist filings + amendments: reporting persons, CUSIP, ownership (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def sec_13d(config_path: str, tickers: str | None, years: int | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(
        context, tickers=_tickers(context, tickers), years_history=years or config.data_extract.years_history, fetch=SEC_13D_FETCH, full=full
    )


@cli.command(help="SC 13G passive 5%+ beneficial ownership + amendments (edgartools). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def sec_13g(config_path: str, tickers: str | None, years: int | None, full: bool) -> None:
    """Passive counterpart of `sec-13d`. Chunk a from-scratch backfill with `-t` + `-F`: the manifest's incremental
    test is "did the universe change size", which a chunked walk defeats."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(
        context, tickers=_tickers(context, tickers), years_history=years or config.data_extract.years_history, fetch=SEC_13G_FETCH, full=full
    )


@cli.command(help="Filing text: 10-K Item 1A (Risk Factors) + Item 7 (MD&A) & 10-Q Item 2 (MD&A). SEC-api.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def filing_text(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(context, tickers=_tickers(context, tickers), years_history=years or config.data_extract.years_history, fetch=FILING_TEXT_FETCH)


@cli.command(help="DEF 14A structured: pay-vs-performance, audit fees, comp/ownership/vote tables (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def def14a_edgar(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(context, tickers=_tickers(context, tickers), years_history=years or config.data_extract.years_history, fetch=DEF14A_EDGAR_FETCH)


# --- Behavioral (retail attention) ---
@cli.command(
    help="Earnings-call transcripts (HuggingFace defeatbeta) -> earnings_call_sections, one row per "
    "paragraph. Incremental: an unchanged source file is a no-op; -F compares every call."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def extract_earnings_calls(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    _extract_earnings_calls(context, config.earnings_calls, full=full, tickers=_tickers(context, tickers))
