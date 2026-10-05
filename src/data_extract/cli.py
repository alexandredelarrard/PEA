"""Data-extraction CLI: one command per data source, so the Airflow extraction DAG can schedule each independently.

    python -m src data_extract <command> [-c ./configs] [-t AAPL,MSFT]

Every command builds a fresh Context, resolves the ticker universe (or a --tickers subset) and runs its
fetcher. Fetchers resume from the DB, so a nightly rerun pulls only new data. `seed-universe` must run first.
"""

import json
from datetime import datetime
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
from src.context import Context, get_config_context
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import (
    extract_earnings_calls as _extract_earnings_calls,
)
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.edgar_driver import EdgarFetch, load_edgar_scope, plan_fetch, run_edgar_fetch
from src.data_extract.utils.common.entity_lineage import build_entity_lineage
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.common.symbol_tenure import build_symbol_tenure, scan_form345_cache
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec, fundamentals_fetch
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
from src.data_extract.utils.institutionals.fetch_13f_backfill import fetch_13f_backfill
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.institutionals.fetch_insider_edgar import fetch_insider_edgar, insider_fetch
from src.data_extract.utils.institutionals.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.institutionals.fetch_superinvestors import seed_roster_history, upsert_roster_snapshot
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import fetch_prices_and_actions
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_extract.utils.structure.def14a import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.structure.votes import fetch_8k_votes_llm
from src.data_store.schema import Table, Tables, freshness_tables, marker_tables, resolve
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.freshness import PREDICTION_INPUTS, fresh_share, last_dates_by_key, max_age_days, table_freshness
from src.utils.universe import load_universe_tickers, unverified_ciks

CONFIG_KWARGS = cast(dict[str, Any], _CONFIG_KWARGS)
TICKERS_KWARGS = cast(dict[str, Any], _TICKERS_KWARGS)
FULL_KWARGS = cast(dict[str, Any], _FULL_KWARGS)
YEARS_KWARGS = cast(dict[str, Any], _YEARS_KWARGS)
# Hidden run date: what a command treats as "today", so a run can be replayed on an injected date.
AS_OF_OPTION = click.option(
    "--as-of", "as_of", type=click.DateTime(formats=["%Y-%m-%d"]), default=None, hidden=True, help="Run date YYYY-MM-DD (default: today)."
)
# The one-time EDGAR backlog run lifts `data_extract.max_documents_per_run`; nightly runs keep it.
NO_CAP_OPTION = click.option("--no-cap", "no_cap", is_flag=True, default=False, help="Lift the per-run document cap (one-time backlog runs only).")


@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA EXTRACTION — one command per source (scheduled by the Airflow extraction DAG)."""


def _tickers(context: Context, tickers: str | None) -> list[str]:
    """The --tickers subset if given, else the full sp500_tickers universe."""
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return load_universe_tickers(context)


def _run_date(as_of: datetime | None) -> pd.Timestamp:
    """The `--as-of` date at midnight, or today."""
    return cast(pd.Timestamp, pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today())).normalize()


def _market_as_of(as_of: datetime | None) -> pd.Timestamp | None:
    """The `--as-of` date for market-data fetchers, or None so they end at the last session completed now."""
    return None if as_of is None else _run_date(as_of)


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
        "max_date": None if age_days is None else (as_of - pd.Timedelta(days=age_days)).date().isoformat(),
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
@AS_OF_OPTION
def extraction_status(config_path: str, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    report = _extraction_status_report(context, as_of=_run_date(as_of))
    click.echo(json.dumps(report, sort_keys=True))


# --- Universe seed (must run first; everything else resolves the universe from it) ---
@cli.command(help="Seed the sp500_tickers universe (idempotent; scrapes only if empty or --refresh).", help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("--refresh", is_flag=True, default=False, help="Re-scrape the S&P 500 even if populated.")
@AS_OF_OPTION
def seed_universe(config_path: str, refresh: bool, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    if refresh or context.store.row_count(Tables.sp500_tickers) == 0:
        context.log.info(f"Seeding {Tables.sp500_tickers} via the S&P 500 scraper (refresh={refresh})")
        get_sp500_tickers(context, as_of=_run_date(as_of))
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
@AS_OF_OPTION
def price_history(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """`--full` re-pulls the whole window for every requested ticker; new tickers, holes and splits need no flag."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    years_history = int(context.config.data_extract.years_history)
    fetch_prices_and_actions(context, tickers=_tickers(context, tickers), years_history=years_history, full=full, as_of=_market_as_of(as_of))


@cli.command(help="FINRA RegSHO short interest / short volume.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@click.option("--repair-gaps", "repair", is_flag=True, default=False, help="Also re-read every day a key misses inside its stored span (one-time).")
@AS_OF_OPTION
def short_interest(config_path: str, tickers: str | None, full: bool, repair: bool, as_of: datetime | None) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=False)
    years_history = int(context.config.data_extract.years_history)
    universe = _tickers(context, tickers)
    fetch_short_interest(context, tickers=universe, years_history=years_history, full=full, as_of=_market_as_of(as_of), repair=repair)


@cli.command(help="SEC fails-to-deliver (settlement fails). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
def fails_to_deliver(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fails_to_deliver(
        context, tickers=_tickers(context, tickers), years_history=int(config.data_extract.years_history), full=full, as_of=_run_date(as_of)
    )


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
@AS_OF_OPTION
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
    fetch_13f(
        context,
        tickers=_tickers(context, tickers),
        years_history=int(config.data_extract.years_history),
        filing_window=window,
        as_of=_run_date(as_of),
    )


@cli.command(
    name="thirteen-f-backfill",
    help="13F history of NEW universe tickers: SEC 13F data sets (cached ZIPs), then one EDGAR walk over the gap after them.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
def thirteen_f_backfill(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """Tickers added inside `sec13f_hr`'s overlap, each from its own earliest stored period back to 2013 Q2.
    `-t X` narrows the new tickers (an established X does nothing); `-t X -F` re-reads every data set for X."""
    _, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_13f_backfill(context, tickers=_tickers(context, tickers) if tickers else None, as_of=_run_date(as_of), full=full)


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
# `-F/--full` re-reads every listed filing, stored ones included (see `run_edgar_fetch`).


@cli.command(
    name="fundamentals-facts",
    help="SEC per-filing XBRL -> fundamentals_facts, resolved from each filer's own calculation linkbase. As-filed only; append-only.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def fundamentals_facts(config_path: str, tickers: str | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_fundamentals_sec(
        context,
        tickers=_tickers(context, tickers),
        full=full,
        years_history=int(config.data_extract.years_history),
        as_of=_run_date(as_of),
        no_cap=no_cap,
    )


@cli.command(
    name="fundamentals-employees",
    help="SEC 10-K prose -> fundamentals_employees, skipping filing dates that already have a row (NULL decisions included), over the registrant lineage. "
    "A ticker that fails is logged and retried next run; the task exits 0.",
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
@AS_OF_OPTION
@NO_CAP_OPTION
def fundamentals(config_path: str, tickers: str | None, rebuild: bool, full: bool, as_of: datetime | None, no_cap: bool) -> None:
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
    fetch_fundamentals_sec(
        context, tickers=names, full=full or rebuild, years_history=int(config.data_extract.years_history), as_of=_run_date(as_of), no_cap=no_cap
    )
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
@AS_OF_OPTION
def fundamentals_sharadar(config_path: str, tickers: str | None, full: bool, as_of: datetime | None) -> None:
    """Delegates to `StepExtractFundamentalsSharadar` so the dependency order, ending with the merge, lives in one place."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    step = StepExtractFundamentalsSharadar(context=context, config=config)
    step.run(tickers=_tickers(context, tickers), full=full, config_dir=config_path, as_of=_run_date(as_of))


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
@AS_OF_OPTION
def sharadar_actions(config_path: str, full: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_sharadar_actions(context, full=full, years_history=int(config.data_extract.sharadar_years_history), as_of=_run_date(as_of))


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
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_earnings_surprises(context, tickers=_tickers(context, tickers), years_history=int(config.data_extract.years_history))


@cli.command(help="SEC Financial Statement Data Sets -> pension_facts (num/sub XBRL). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help="Re-read every cached period even when already ingested. For a PARSE change -- a new column, or a registrant-resolution change -- not a data change. Nothing is re-downloaded.",
)
@AS_OF_OPTION
def financial_statements(config_path: str, tickers: str | None, reparse: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_financial_statements(
        context, tickers=_tickers(context, tickers), years_history=int(config.data_extract.years_history), reparse=reparse, as_of=_run_date(as_of)
    )


@cli.command(
    help="SEC insider transactions (Forms 3/4/5) into one table: the quarterly zips no stored row carries add the filings "
    "EDGAR lacks (a new ticker also re-parses the cached zips), then EDGAR reads every indexed filing after the last stored "
    "zip quarter that the ticker has not stored from EDGAR. Stored rows are re-screened and rejects deleted only on a "
    "full-universe run (no -t). -F re-parses every cached zip (only the -t tickers' rows under -t), then re-reads every "
    "EDGAR filing after the last zip quarter, including those already stored."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(
    "--reparse",
    is_flag=True,
    default=False,
    help="Re-read every cached quarter even when already ingested (implied by -F). For a PARSE change (a new column), not a data change -- nothing is re-downloaded.",
)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def insider_transactions(config_path: str, tickers: str | None, reparse: bool, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers)
    fetch_insider_transactions(
        context, tickers=names, years_history=int(config.data_extract.years_history), reparse=reparse or full, as_of=_run_date(as_of)
    )
    fetch_insider_edgar(
        context, tickers=names, years_history=int(config.data_extract.years_history), full=full, as_of=_run_date(as_of), no_cap=no_cap
    )


@cli.command(
    name="identity-tables",
    help="symbol_tenure + entity_lineage: WHICH COMPANY a ticker was, and when. "
    "OFFLINE reference build -- derived from the cached Form 345 zips and the "
    "DB, no network. Run after `insider-transactions` has populated the cache.",
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(
    "--approve-rekey",
    "approved_rekeys",
    multiple=True,
    metavar="OLD_ENTITY_ID:NEW_ENTITY_ID",
    help="Acknowledge one exact older-CIK entity-id change reported by the rekey impact file.",
)
def identity_tables(config_path: str, approved_rekeys: tuple[str, ...]) -> None:
    """Builds `symbol_tenure` then `entity_lineage` together: lineage candidates are read off tenure, so a stale
    tenure silently narrows lineage. Market-wide: both tables are rebuilt from the cache and the DB.
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
    build_entity_lineage(
        context,
        tenure,
        scan.owner_pairs,
        config_path,
        approved_rekeys=frozenset(parsed_rekeys),
    )


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
@AS_OF_OPTION
def financial_notes(config_path: str, tickers: str | None, reparse: bool, repair_availability: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_financial_notes(
        context,
        tickers=_tickers(context, tickers),
        years_history=int(config.data_extract.years_history),
        reparse=reparse,
        repair_availability=repair_availability,
        as_of=_run_date(as_of),
    )


# --- Structure (governance) ---
@cli.command(help="DEF 14A governance / executive pay (LLM-parsed). SEC-api + LLM.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def def14a(config_path: str, tickers: str | None, full: bool) -> None:
    """Every run lists each ticker's whole `years_history`; `--full` also re-sends the saved proxies without
    evidence, markers included (a proxy with stored evidence is never re-sent)."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    fetch_def14a_llm(context, config, tickers=_tickers(context, tickers), full=full)


@cli.command(help="8-K events: item codes + has_earnings/has_press_release (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def sec_8k_items(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _run_document_command(config_path, tickers, years, SEC_8K_FETCH, full=full, as_of=as_of, no_cap=no_cap)


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
@AS_OF_OPTION
@NO_CAP_OPTION
def sec_13d(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _run_document_command(config_path, tickers, years, SEC_13D_FETCH, full=full, as_of=as_of, no_cap=no_cap)


@cli.command(help="SC 13G passive 5%+ beneficial ownership + amendments (edgartools). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def sec_13g(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    """Passive counterpart of `sec-13d`. A schedule a universe company only FILED is stored as an empty-filing marker under it."""
    _run_document_command(config_path, tickers, years, SEC_13G_FETCH, full=full, as_of=as_of, no_cap=no_cap)


@cli.command(help="Filing text: 10-K Item 1A (Risk Factors) + Item 7 (MD&A) & 10-Q Item 2 (MD&A). SEC-api.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def filing_text(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _run_document_command(config_path, tickers, years, FILING_TEXT_FETCH, full=full, as_of=as_of, no_cap=no_cap)


@cli.command(help="DEF 14A structured: pay-vs-performance, audit fees, comp/ownership/vote tables (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
@AS_OF_OPTION
@NO_CAP_OPTION
def def14a_edgar(config_path: str, tickers: str | None, years: int | None, full: bool, as_of: datetime | None, no_cap: bool) -> None:
    _run_document_command(config_path, tickers, years, DEF14A_EDGAR_FETCH, full=full, as_of=as_of, no_cap=no_cap)


def _run_document_command(
    config_path: str, tickers: str | None, years: int | None, fetch: EdgarFetch, *, full: bool, as_of: datetime | None, no_cap: bool
) -> None:
    """One declared EDGAR document fetch over the --tickers subset (or the universe)."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    run_edgar_fetch(
        context,
        tickers=_tickers(context, tickers),
        years_history=years or config.data_extract.years_history,
        fetch=fetch,
        full=full,
        as_of=_run_date(as_of),
        no_cap=no_cap,
    )


# --- EDGAR index, work-list dry run and empty-filing markers ---
@cli.command(
    name="edgar-index", help="Download the local EDGAR filing index (closed quarters once, the current quarter again) and print rows per quarter."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("--build", is_flag=True, default=False, help="Download every quarter of the window again, closed ones included.")
@AS_OF_OPTION
def edgar_index_command(config_path: str, build: bool, as_of: datetime | None) -> None:
    """One EDGAR walk: run it alone. The first build fetches about 128 quarters."""
    config, context = get_config_context(config_path, use_cache=False, save=False)
    context.ensure_edgar_identity()
    edgar_index.refresh(context, _run_date(as_of), int(config.data_extract.years_history), rebuild=build)
    for quarter, n_rows in edgar_index.quarter_counts(context).items():
        click.echo(f"{quarter}\t{n_rows}")


def _document_fetches(context: Context, tickers: list[str]) -> dict[str, EdgarFetch]:
    """Every EDGAR document fetch, keyed by its done table's name."""
    fetches = (
        SEC_8K_FETCH,
        SEC_13D_FETCH,
        SEC_13G_FETCH,
        FILING_TEXT_FETCH,
        DEF14A_EDGAR_FETCH,
        fundamentals_fetch(context, load_cik_mapping(context, tickers)),
        insider_fetch(tickers),
    )
    return {fetch.done.name: fetch for fetch in fetches}


@cli.command(name="resume-plan", help="READ-ONLY: each EDGAR document table's work list for the run date, by key class, from the cached index.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--table", "table_name", default=None, help="One done table (e.g. sec_8k); default: every document table.")
@click.option("--timings", is_flag=True, default=False, help="Also print the plan-read timings.")
@AS_OF_OPTION
def resume_plan(config_path: str, tickers: str | None, table_name: str | None, timings: bool, as_of: datetime | None) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    names = _tickers(context, tickers)
    cik_map = load_cik_mapping(context, names)
    fetches = _document_fetches(context, names)
    if table_name is not None and table_name not in fetches:
        raise click.BadParameter(f"not an EDGAR document table: {table_name}; one of {sorted(fetches)}", param_hint="--table")
    for name, fetch in fetches.items():
        if table_name is not None and name != table_name:
            continue
        scope = load_edgar_scope(context, identity_aware=fetch.identity_aware)
        work = plan_fetch(context, fetch, cik_map, scope, _run_date(as_of), int(config.data_extract.years_history))
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
@click.option("--table", "table_name", default=None, help="One marker table; default: every table that declares a marker.")
@click.option("--count", "count_only", is_flag=True, default=False, help="Count only (the default without --delete).")
@click.option("--delete", "delete", is_flag=True, default=False, help="Delete the marker rows (rollback only).")
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
    "paragraph. Incremental from the stored frontier minus the contract overlap; -F compares every call and removes calls gone from the source."
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*FULL_ARGS, **FULL_KWARGS)
def extract_earnings_calls(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = get_config_context(config_path, use_cache=False, save=False)
    _extract_earnings_calls(context, config.earnings_calls, full=full, tickers=_tickers(context, tickers))
