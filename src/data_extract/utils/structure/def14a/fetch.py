"""Daily five-domain DEF 14A extraction from the stored `sec_def14a_text` payloads, with one durable section retry.

Reads only `sec_def14a_text` and the five DEF 14A tables; no SEC read. A proxy the walk's subject guard rejected is a
hidden marker, so it is never reserved; a walked proxy missing a section (or holding only the `_none` row) has that
domain empty, and an open empty domain consumes its attempt. The prompt gets the first
`data_extract.llm_text_budget.def14a` characters of a payload; postprocessing gets the stored uncut payload.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.resume import document_floor, planner_bounds
from src.data_extract.utils.edgar_extract.edgar_driver import universe_rows
from src.data_extract.utils.structure.def14a.gender import finalise_gender
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_extract.utils.structure.def14a.work import call_sections, section_requests, ticker_works
from src.data_store.schema import Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.utils.string import run_date
from src.utils.tickers_universe.universe import before_end

#: The `sec_def14a_text` columns the worklist reads; `payload` is read per ticker, for its proxies worth a payload.
_TEXT_COLUMNS = ["ticker", "accession_number", "cik", "filing_date"]


def section_extractors(context: Context, config: DictConfig) -> dict[str, LLMExtractor]:
    """One extractor per section, bound to its own action, prompt and schema; two concurrent calls each, which also
    bounds the answers in flight."""
    workers = 2
    return {section.key: LLMExtractor(context, config, action=section.action, threads=workers) for section in SECTIONS}


def _walked_proxies(df_text: pd.DataFrame | None, names: Mapping[str, str]) -> pd.DataFrame:
    """One row per walked proxy, in filing order: the filings frame `load_work` reads, with the universe company name."""
    df = (pd.DataFrame(columns=_TEXT_COLUMNS) if df_text is None else df_text).drop_duplicates(["ticker", "accession_number"])
    df = df.assign(filing_date=pd.to_datetime(df["filing_date"]).dt.normalize(), ticker=df["ticker"].astype(str))
    df = df.assign(company_name=df["ticker"].map(names).fillna(""))
    return df.sort_values(["ticker", "filing_date", "accession_number"], kind="stable").reset_index(drop=True)


def fetch_def14a_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    extractors: Mapping[str, LLMExtractor],
) -> None:
    """Extract every section of a walked proxy filed by its ticker's end (plus the grace) that has a key gap and fewer
    than two attempts (`retry=1`, then one more call and `retry=2`), from the stored tables alone. `extractors`
    (`section_extractors`) holds one extractor per section; their worker count bounds the answers in flight."""
    # load: the five output tables (created from their committed DDL on a first run), the walked proxies in the window
    # (no payloads), the ticker names and end bounds
    for section in SECTIONS:
        context.store.create_declared(section.table)
    years = int(config.data_extract.years_history)
    floor = document_floor(Tables.def14a_llm, run_date(None), years)
    df_text = context.store.load(Tables.sec_def14a_text, columns=_TEXT_COLUMNS, where={"ticker": tickers}, since=floor, optional=True)
    df_names = universe_rows(context, tickers)
    bounds = planner_bounds(context)
    budget = int(config.data_extract.llm_text_budget.def14a)

    # clean: the filings frame of the walked proxies filed on or before their ticker's end (plus the grace)
    df_proxies = _walked_proxies(df_text, dict(zip(df_names["ticker"], df_names["name"], strict=True)))
    df_proxies = df_proxies[before_end(df_proxies["ticker"], df_proxies["filing_date"], bounds)].reset_index(drop=True)
    batch_size = max((extractor.threads for extractor in extractors.values()), default=1)

    # compute: each ticker's open sections, saved per section
    updated = 0
    for ticker, df_filings in tqdm(df_proxies.groupby("ticker", sort=False), desc="DEF 14A LLM"):
        works = ticker_works(context, str(ticker), df_filings, floor)
        updated += call_sections(context, str(ticker), section_requests(context, works, budget), extractors, batch_size)

    # save: the cross-ticker gender reconciliation after saved updates
    if updated:
        finalise_gender(context)
    context.log.info("DEF 14A: %d section answer(s) extracted; sections at retry=2 skipped", updated)
