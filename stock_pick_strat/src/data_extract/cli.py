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
import json

import click

from src.constants.command_line_interface import (
    CONFIG_ARGS, CONFIG_KWARGS, TICKERS_ARGS, TICKERS_KWARGS,
)
from src.context import Context, get_config_context
from src.utils.cli_helper import SpecialHelpOrder
from src.utils.universe import load_universe_tickers
from src.data_extract.step_check_freshness import StepCheckFreshness

# --- prices / market / macro ------------------------------------------------ #
from src.data_extract.utils.prices.fetch_prices import (
    fetch_market_prices, fetch_price_history, get_sp500_tickers,
)
from src.data_extract.utils.prices.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.prices.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.prices.fetch_13f import fetch_13f
from src.data_extract.utils.prices.fetch_superinvestors import build_superinvestors_json
from src.data_extract.utils.prices.fetch_macro_assets import fetch_macro_assets
from src.data_extract.utils.fundamentals.fetch_macro import fetch_macro
# --- fundamentals ----------------------------------------------------------- #
from src.data_extract.utils.fundamentals.fetch_fundamentals import fetch_fundamentals
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
# --- structure -------------------------------------------------------------- #
from src.data_extract.utils.structure.fetch_employees_edgar import fetch_employees_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
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
    if refresh or context.store.row_count("sp500_tickers") == 0:
        context.log.info("Seeding sp500_tickers via the S&P 500 scraper (refresh=%s)", refresh)
        get_sp500_tickers(context)
    context.log.info("Universe ready: %d tickers.", len(load_universe_tickers(context)))


# --------------------------------------------------------------------------- #
# Prices / market / macro                                                       #
# --------------------------------------------------------------------------- #
@cli.command(help="Daily price history + dividends (yfinance). HEAVY.", help_priority=2)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def price_history(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_price_history(context, tickers=_tickers(context, tickers))


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


@cli.command(help="Benchmark + commodity/FX OHLCV (SPY, ^VIX, oil, gold, FX). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def market_prices(config_path: str) -> None:
    _, context = _ctx(config_path)
    fetch_market_prices(context)


@cli.command(help="FRED macro series (yields, VIX, credit spread, breakevens). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def macro(config_path: str) -> None:
    _, context = _ctx(config_path)
    fetch_macro(context)


@cli.command(help="Long-history multi-asset allocation series (FRED + yfinance, since ~1995). Light.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def macro_assets(config_path: str) -> None:
    _, context = _ctx(config_path)
    fetch_macro_assets(context)


@cli.command(help="13F institutional holdings (SEC bulk + OpenFIGI cusip map). HEAVY.")
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
@cli.command(help="SEC companyfacts fundamentals (balance sheet / income / cash flow). HEAVY.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def fundamentals(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_fundamentals(context, tickers=_tickers(context, tickers))


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


# --------------------------------------------------------------------------- #
# Structure (workforce / governance)                                            #
# --------------------------------------------------------------------------- #
@cli.command(help="Employee counts from SEC 10-K text (EDGAR). SEC-api.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def employees(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    fetch_employees_edgar(context, tickers=_tickers(context, tickers))


@cli.command(help="DEF 14A governance / executive pay (LLM-parsed). SEC-api + LLM.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def def14a(config_path: str, tickers: str | None) -> None:
    config, context = _ctx(config_path)
    fetch_def14a_llm(context, tickers=_tickers(context, tickers),
                     model=config.data_extract.llm_model)


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
                  "earnings_call_sections. Runs after download-earnings-calls.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*TICKERS_ARGS, **TICKERS_KWARGS)
def ingest_earnings_calls(config_path: str, tickers: str | None) -> None:
    _, context = _ctx(config_path)
    _ingest_earnings_calls(context, tickers=_tickers(context, tickers))


# --------------------------------------------------------------------------- #
# Data-freshness / gap gate — runs LAST, before triggering aggregation         #
# --------------------------------------------------------------------------- #
@cli.command(help="Check every source is up to date for its cadence (daily..yearly). Prints the "
                  "report as a JSON last line (the DAG pushes it to XCom); exits non-zero if any "
                  "source is stale / gapped.")
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
def check_freshness(config_path: str) -> None:
    config, context = _ctx(config_path)
    report = StepCheckFreshness(context=context, config=config).run()
    # emit the report as the FINAL stdout line so the DAG can capture it into XCom
    click.echo(json.dumps(report, separators=(",", ":")))
    if not report["ok"]:
        raise SystemExit(2)                       # non-zero -> "not as expected" (RED)
