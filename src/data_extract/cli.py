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
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes
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
# NOTE: the combined `fundamentals` command (facts THEN history) is still absent while the
# stack is rebuilt (reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md).
# Phase 3 landed the facts layer and exposes it on its own below; Phase 5 adds the history
# build and re-joins the two under one command.
@cli.command(name="fundamentals-facts",
             help="SEC per-filing XBRL -> fundamentals_facts, resolved from each filer's "
                  "own calculation linkbase. As-filed only; append-only.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def fundamentals_facts(config_path: str, tickers: str | None) -> None:
    config, context = _ctx(config_path)
    fetch_fundamentals_sec(context, tickers=_tickers(context, tickers),
                           years_history=int(config.data_extract.years_history))


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
# NOTE: there is no `employees` command any more. Employee counts are extracted
# from the same 10-K as the fundamentals (`fundamentals_employees.py`), so the
# `fundamentals` command above now covers them.
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


