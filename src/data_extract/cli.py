"""
cli.py  (src/data_extract/cli.py)
---------------------------------
DATA-EXTRACTION command-line interface — ONE command per data SOURCE (fetcher), so the Airflow
extraction DAG can schedule each independently and tune parallelism by load (fan out the light
sources; throttle the heavy / rate-limited ones via pools). Invoked as:

    python -m src data_extract <command> [-c ./configs] [-t AAPL,MSFT]

Every command builds a fresh Context, resolves the ticker universe (or a --tickers subset) and runs
its fetcher. Fetchers are incremental (resume from the DB), so re-running nightly only pulls new data.
"""
import click

from src.data_store.schema import Tables
from src.constants.command_line_interface import (
    CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS,
)
from src.context import Context, get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers

# --- prices / market / macro ------------------------------------------------ #
from src.data_extract.utils.prices.fetch_prices import fetch_price_history
from src.data_extract.utils.prices.fetch_dividends import fetch_dividends
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_extract.utils.prices.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.prices.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.prices.fetch_13f import fetch_13f
from src.data_extract.utils.prices.fetch_superinvestors import build_superinvestors_json
from src.data_extract.utils.prices.fetch_macro import fetch_macro
# --- fundamentals ----------------------------------------------------------- #
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_actions, fetch_sharadar_sp500, fetch_sharadar_tickers,
)
from src.data_extract.utils.fundamentals_sharadar.diagnostics import (
    DEFAULT_REPORT_PATH, run_diagnostics,
)
from src.data_extract.utils.fundamentals_sharadar.gap_check import (
    DEFAULT_REPORT_PATH as GAP_REPORT_PATH, run_gap_check,
)
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
# --- structure -------------------------------------------------------------- #
from src.data_extract.utils.structure.fetch_def14a_edgar import fetch_def14a_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.structure.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.structure.fetch_filing_text import fetch_filing_text
# --- behavioral ------------------------------------------------------------- #
from src.data_extract.utils.behavioral.fetch_wiki_pageviews import fetch_wiki_pageviews
from src.data_extract.utils.behavioral.fetch_google_trends import fetch_google_trends
from src.data_extract.utils.behavioral.fetch_earnings_calls import (
    download_earnings_calls as _download_earnings_calls,
    ingest_all_earnings_calls as _ingest_earnings_calls,
)

@click.group(cls=SpecialHelpOrder)
def cli() -> None:
    """DATA EXTRACTION — one command per source (scheduled by the Airflow extraction DAG)."""


def _ctx(config_path: str) -> tuple[object, Context]:
    return get_config_context(config_path, use_cache=False, save=False)


def _tickers(context: Context, tickers: str | None) -> list[str]:
    """The --tickers subset if given, else the full sp500_tickers universe."""
    if tickers:
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return load_universe_tickers(context)


# --------------------------------------------------------------------------- #
# Universe seed — MUST run first; everything else resolves the universe from it #
# --------------------------------------------------------------------------- #
@cli.command(help="Seed the sp500_tickers universe (idempotent; scrapes only if empty or --refresh).",
             help_priority=1)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option("--refresh", is_flag=True, default=False, help="Re-scrape the S&P 500 even if populated.")
def seed_universe(config_path: str, refresh: bool) -> None:
    _, context = _ctx(config_path)
    if refresh or context.store.row_count(Tables.sp500_tickers) == 0:
        context.log.info(F"Seeding {Tables.sp500_tickers} via the S&P 500 scraper (refresh={refresh})")
        get_sp500_tickers(context)
    context.log.info("Universe ready: %d tickers.", len(load_universe_tickers(context)))


# --------------------------------------------------------------------------- #
# Prices / market / macro                                                       #
# --------------------------------------------------------------------------- #
@cli.command(help="Daily price history, OHLCV (yfinance). HEAVY.", help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def price_history(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_price_history(context, tickers=_tickers(context, tickers),
                        years_history=context.config.data_extract.years_history)


@cli.command(help="Cash-dividend ex-dates (yfinance). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def dividends(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_dividends(context, tickers=_tickers(context, tickers),
                    years_history=context.config.data_extract.years_history)


@cli.command(help="FINRA RegSHO short interest / short volume.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def short_interest(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_short_interest(context, tickers=_tickers(context, tickers))


@cli.command(help="SEC fails-to-deliver (settlement fails). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def fails_to_deliver(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_fails_to_deliver(context, tickers=_tickers(context, tickers))


@cli.command(help="ALL macro / market series -> prices_macro (yfinance + FRED). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def macro(config_path: str) -> None:
    """One command for every non-equity series: the market/commodity/FX closes (SPY, ^VIX,
    oil, gold, energy, FX), the FRED levels (yields, cash, credit spread, breakeven) and the
    derived spreads + 10Y total-return index. Replaced three commands -- `market-prices`
    (which wrote its tickers into `prices`), `macro` and `macro-assets`."""
    _, context = _ctx(config_path)
    fetch_macro(context, years_history=context.config.data_extract.macro_years_history)


@cli.command(help="13F institutional holdings (EDGAR by filing date + OpenFIGI cusip map). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def thirteen_f(config_path: str) -> None:
    _, context = _ctx(config_path)
    fetch_13f(context)


@cli.command(help="Superinvestor roster (Dataroma) -> ranked CIK subset JSON. Needs 13F. Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def superinvestors(config_path: str) -> None:
    _, context = _ctx(config_path)
    build_superinvestors_json(context)


# --------------------------------------------------------------------------- #
# Fundamentals                                                                  #
# --------------------------------------------------------------------------- #
# The two layers are separate commands AND joined by one, because their costs differ by three
# orders of magnitude: the facts walk is network-bound (~2h for 52 tickers), the history build
# is a pure in-memory replay of what that walk already stored. A bug in the history layer must
# not cost a re-download, which is exactly what the two rebuild flags encode (decision 27).
#: `-F/--full` on the fundamentals commands. The manifest's incremental window keys on the
#: TICKER COUNT, so a chunked from-scratch backfill reads as a repeat of the previous chunk and
#: fetches nothing; this bypasses it. See `run_edgar_fetch`.
_FULL_ARGS = ("-F", "--full")
_FULL_KWARGS = dict(is_flag=True, default=False,
                    help="Ignore the run manifest and take the whole years-history window. "
                         "Needed for a chunked from-scratch backfill.")


@cli.command(name="fundamentals-facts",
             help="SEC per-filing XBRL -> fundamentals_facts (+ headcount from the same "
                  "10-K), resolved from each filer's own calculation linkbase. As-filed "
                  "only; append-only.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def fundamentals_facts(config_path: str, tickers: str | None, full: bool) -> None:
    config, context = _ctx(config_path)
    fetch_fundamentals_sec(context, tickers=_tickers(context, tickers), full=full,
                           years_history=int(config.data_extract.years_history))


@cli.command(name="fundamentals-history-sec",
             help="fundamentals_facts -> fundamentals_history_sec + _reason_codes, on the "
                  "publication-event grain. No network. Append-only: refuses to overwrite "
                  "an already-published row unless --rebuild-history is passed.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--rebuild-history", is_flag=True,
              help="Delete these tickers' fundamentals_history_sec / _reason_codes rows and "
                   "rebuild from the facts ALREADY STORED. For a bug in the history layer; "
                   "costs no network. (Use `fundamentals --rebuild` for a resolution bug.)")
def fundamentals_history_sec(config_path: str, tickers: str | None,
                             rebuild_history: bool) -> None:
    _, context = _ctx(config_path)
    build_fundamentals_history(context, tickers=_tickers(context, tickers),
                               rebuild_history=rebuild_history)


@cli.command(help="Both fundamentals layers in order: facts (network) then history (replay).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--rebuild", is_flag=True,
              help="Delete these tickers' rows from BOTH layers and refetch every filing. "
                   "For a bug in the RESOLUTION layer, where the stored facts are themselves "
                   "wrong. A deleted ticker looks exactly like a never-fetched one to the "
                   "fetcher's accession-set resume, so there is no third state to reason "
                   "about. There is no build_version column: the rebuild IS the version.")
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def fundamentals(config_path: str, tickers: str | None, rebuild: bool, full: bool) -> None:
    config, context = _ctx(config_path)
    names = _tickers(context, tickers)
    if rebuild:
        for ticker in names:
            for table in (Tables.fundamentals_facts, Tables.fundamentals_history_sec,
                          Tables.fundamentals_reason_codes, Tables.fundamentals_employees):
                context.store.delete(table, {"ticker": ticker})
        context.log.warning("fundamentals: --rebuild deleted all four tables for %d "
                            "ticker(s); every filing will be refetched", len(names))
    fetch_fundamentals_sec(context, tickers=names, full=full or rebuild,
                           years_history=int(config.data_extract.years_history))
    build_fundamentals_history(context, tickers=names, rebuild_history=rebuild)


# --------------------------------------------------------------------------- #
# Fundamentals -- Sharadar (SF1), the OTHER producer                            #
# --------------------------------------------------------------------------- #
# The joined command runs the whole producer; the three single-table commands exist for a
# TARGETED refresh when only one dimension is stale, which is a manual operation -- nothing in
# `src/dags/` schedules them. `-F/--full` re-pulls the whole configured window instead of
# resuming from the stored max date, and makes the merge DELETE before it rebuilds.
#
# History depth is `data_extract.sharadar_years_history`, NOT `years_history`: the SEC walk
# is limited by patience, Sharadar by subscription tier (D3).
@cli.command(name="fundamentals-sharadar",
             help="The whole Sharadar producer in dependency order: tickers -> SF1 "
                  "fundamentals -> actions -> sp500 -> the MERGED fundamentals_history.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def fundamentals_sharadar(config_path: str, tickers: str | None, full: bool) -> None:
    """Delegates to `StepExtractFundamentalsSharadar` rather than restating the order.

    The two used to be written out separately and had already diverged: this command stopped
    before the merge, so it refreshed the vendor tables and left `fundamentals_history` a run
    behind them -- exactly the staleness the step's own comment warns about.
    """
    config, context = _ctx(config_path)
    StepExtractFundamentalsSharadar(context=context, config=config).run(
        tickers=_tickers(context, tickers), full=full, config_dir=config_path)


@cli.command(name="sharadar-tickers",
             help="Sharadar entity dimension (permaticker, currency, category) -> "
                  "sharadar_tickers. Full refresh. Prerequisite of fundamentals-sharadar.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def sharadar_tickers(config_path: str) -> None:
    _, context = _ctx(config_path)
    fetch_sharadar_tickers(context)


@cli.command(name="sharadar-actions",
             help="Sharadar corporate actions (dividends, splits, spinoffs, acquisitions, "
                  "relations) -> sharadar_actions.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def sharadar_actions(config_path: str, full: bool) -> None:
    config, context = _ctx(config_path)
    fetch_sharadar_actions(context, full=full,
                           years_history=int(config.data_extract.sharadar_years_history))


@cli.command(name="sharadar-sp500",
             help="S&P 500 membership events (added / removed / historical, from 1992) -> "
                  "sharadar_sp500. Ingested only; universe.py is NOT re-pointed (D27).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def sharadar_sp500(config_path: str, full: bool) -> None:
    _, context = _ctx(config_path)
    fetch_sharadar_sp500(context, full=full)


# --------------------------------------------------------------------------- #
# Fundamentals -- the MERGED table, and the instrument that governs it           #
# --------------------------------------------------------------------------- #
@cli.command(name="fundamentals-history-merged",
             help="fundamentals_sharadar + fundamentals_history_sec -> fundamentals_history, "
                  "the MERGED table every consumer reads. Build only, no network. Both "
                  "inputs are read-only, so the rollback is a drop-and-rebuild.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*_FULL_ARGS, **_FULL_KWARGS)
def fundamentals_history_merged(config_path: str, tickers: str | None, full: bool) -> None:
    """Field-block precedence (D14): Sharadar owns a declared column block for ALL history,
    the SEC table owns 15, and no column ever switches source mid-series.

    `--full` DELETES these tickers' rows before rebuilding. The default upsert refreshes every
    row it rebuilds but cannot REMOVE one that no longer exists -- a row the same-date collapse
    now drops would otherwise survive as a fossil under an unchanged key.
    """
    _, context = _ctx(config_path)
    build_merged_history(context, tickers=_tickers(context, tickers), full=full,
                         config_dir=config_path)


@cli.command(name="sharadar-gap-check",
             help="READ-ONLY: where Sharadar and the SEC layer disagree on their SHARED "
                  "dates, and which gaps are SYSTEMATIC enough to be a basis conflict. "
                  "Writes a markdown report and, with --propose, INERT override candidates.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--out", "report_path", default=GAP_REPORT_PATH, show_default=True,
              help="Where to write the findings markdown.")
@click.option("--propose", is_flag=True,
              help="Merge candidate entries into sharadar_source_overrides.json with "
                   "`approved: null`. They change NOTHING until a human adjudicates them, "
                   "and an entry that already exists is never touched.")
def sharadar_gap_check(config_path: str, tickers: str | None, report_path: str,
                       propose: bool) -> None:
    """The merged table's instrument, and it replaces one: reason codes stay with the SEC
    table (D24), so `unexplained_null` no longer gates `fundamentals_history`.

    `--tickers` defaults to EVERY stored ticker rather than the sp500 universe -- the two
    sources overlap on a subset, and asking for the rest would report a gap on tickers one of
    them never had. Nothing here imports `src/validate/` (D25).
    """
    _, context = _ctx(config_path)
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    run_gap_check(context, tickers=names, report_path=report_path,
                  propose_overrides=propose, config_dir=config_path)


@cli.command(name="sharadar-diagnostics",
             help="READ-ONLY acceptance gates on fundamentals_sharadar -> a markdown report "
                  "of what this run measured. Writes no production data and is NOT the SEC "
                  "check scheme (D25).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("--out", "report_path", default=DEFAULT_REPORT_PATH, show_default=True,
              help="Where to write the findings markdown.")
def sharadar_diagnostics(config_path: str, tickers: str | None, report_path: str) -> None:
    """Completeness, implausible quarters and per-field zero-fill prevalence (D28), measured
    from POSTGRES rather than from the API (D29).

    `--tickers` is optional and defaults to EVERY ticker already stored, not to the sp500
    universe: the entitlement covers a subset, and asking for the rest would report a gate
    failure on tickers that were never fetched. Nothing here registers a check, writes a
    `fundamentals_check` row or imports `src/validate/`.
    """
    _, context = _ctx(config_path)
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    run_diagnostics(context, tickers=names, report_path=report_path)


@cli.command(help="Earnings surprises -> historical forward P/E.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def earnings_surprises(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_earnings_surprises(context, tickers=_tickers(context, tickers))


@cli.command(help="SEC Financial Statement Data Sets -> pension_facts (num/sub XBRL). SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def financial_statements(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_financial_statements(context, tickers=_tickers(context, tickers))


@cli.command(help="SEC insider transactions (Forms 3/4/5) -> insider_transactions. SEC-bulk.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def insider_transactions(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_insider_transactions(context, tickers=_tickers(context, tickers))


@cli.command(help="SEC Financial Statement & NOTES sets -> notes_num / notes_text. VERY HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def financial_notes(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_financial_notes(context, tickers=_tickers(context, tickers))


YEARS_ARGS = ("-y", "--years")
YEARS_KWARGS = dict(
    type=int, default=None,
    help="Override data_extract.years_history for THIS run. A rebuild-from-scratch needs "
         "this to reach as far back as the incrementally-grown table it replaces.")


# --------------------------------------------------------------------------- #
# Structure (governance)                                                        #
# --------------------------------------------------------------------------- #
# NOTE: there is no `employees` command any more. Headcount is parsed out of the same 10-K
# the fundamentals walk already opens (`fundamentals_employees.py`) and lands in
# `fundamentals_employees`, so `fundamentals-facts` above covers it.
@cli.command(help="DEF 14A governance / executive pay (LLM-parsed). SEC-api + LLM.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def def14a(config_path: str, tickers: str | None) -> None:
    config, context = _ctx(config_path)
    fetch_def14a_llm(context, tickers=_tickers(context, tickers),
                     model=config.data_extract.llm_model)


@cli.command(help="8-K events: item codes + has_earnings/has_press_release (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def sec_8k_items(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = _ctx(config_path)
    fetch_8k_edgar(context, tickers=_tickers(context, tickers),
                   years_history=years or config.data_extract.years_history)


@cli.command(help="SC 13D activist filings + amendments: reporting persons, CUSIP, ownership (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def sec_13d(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = _ctx(config_path)
    fetch_13d_edgar(context, tickers=_tickers(context, tickers),
                    years_history=years or config.data_extract.years_history)


@cli.command(help="Filing text: 10-K Item 1A (Risk Factors) + Item 7 (MD&A) & 10-Q Item 2 (MD&A). SEC-api.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def filing_text(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = _ctx(config_path)
    fetch_filing_text(context, tickers=_tickers(context, tickers),
                      years_history=years or config.data_extract.years_history)


@cli.command(help="DEF 14A structured: pay-vs-performance, audit fees, comp/ownership/vote tables (edgartools).")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option(*YEARS_ARGS, **YEARS_KWARGS)
def def14a_edgar(config_path: str, tickers: str | None, years: int | None) -> None:
    config, context = _ctx(config_path)
    fetch_def14a_edgar(context, tickers=_tickers(context, tickers),
                       years_history=years or config.data_extract.years_history)


# --------------------------------------------------------------------------- #
# Behavioral (retail attention)                                                 #
# --------------------------------------------------------------------------- #
@cli.command(help="Wikipedia pageviews (retail attention). Scrape.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def wiki_pageviews(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_wiki_pageviews(context, tickers=_tickers(context, tickers))


@cli.command(help="Google Trends search interest (rate-limited, SLOW). Scrape.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def google_trends(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_google_trends(context, tickers=_tickers(context, tickers))


@cli.command(help="DOWNLOAD earnings-call transcripts to disk: HuggingFace backbone parquet + "
                  "Motley Fool quote-page discovery + MF HTML (no DB). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def download_earnings_calls(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    _download_earnings_calls(context, tickers=_tickers(context, tickers))


@cli.command(help="INGEST cached earnings-call transcripts (HF parquet + MF HTML) -> "
                  "earnings_call_sections. Incremental: skips (ticker,quarter) already ingested; "
                  "--force re-parses everything. Runs after download-earnings-calls.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
@click.option("-F", "--force", is_flag=True, default=False,
              help="Re-ingest all cached transcripts (ignore what's already in the DB).")
def ingest_earnings_calls(config_path: str, tickers: str | None, force: bool) -> None:
    _, context = _ctx(config_path)
    _ingest_earnings_calls(context, tickers=_tickers(context, tickers), force=force)


