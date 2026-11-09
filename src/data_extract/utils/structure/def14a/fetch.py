"""Daily five-domain DEF 14A extraction from the stored `sec_def14a_text` payloads, with one durable section retry.

Reads only `sec_def14a_text` and the five DEF 14A tables; no SEC read. A proxy the walk's subject guard rejected is a
hidden marker, so it is never reserved; a walked proxy missing a section (or holding only the `_none` row) has that
domain empty, and an open empty domain consumes its attempt. The prompt gets the first
`data_extract.llm_text_budget.def14a` characters of a payload; postprocessing gets the stored uncut payload.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.resume import document_floor, planner_bounds
from src.data_extract.utils.edgar_extract.edgar_driver import universe_rows
from src.data_extract.utils.structure.def14a.gender import (
    basis_distribution,
    consensus,
    log_consensus,
    recompute_parent_gender,
)
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section
from src.data_extract.utils.structure.def14a.work import (
    FilingWork,
    load_archives,
    load_work,
    matched_batches,
    open_sections,
    requests_for,
    reserve,
    save_archive,
    save_section,
)
from src.data_store.schema import Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmTask
from src.utils.string import run_date
from src.utils.tickers_universe.universe import before_end

logger = logging.getLogger(__name__)

#: The `sec_def14a_text` columns the worklist reads; `payload` is read per ticker, for its proxies worth a payload.
_TEXT_COLUMNS = ["ticker", "accession_number", "cik", "filing_date", "period"]


def _finalise_gender(context: Context) -> None:
    """Cross-ticker gender consensus over `def14a_directors`, run once after the per-ticker loop.

    A director recurs across companies, so the consensus needs every ticker's rows. Rewrites
    `def14a_directors` gender columns and the parent's gender ratios only when something changed.
    """
    # load: every ticker's directors
    directors = context.store.load(
        Tables.def14a_directors, columns=["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"], optional=True
    )
    if directors is None or directors.empty:
        return

    # compute: the consensus, logged against the basis distribution before it
    before = basis_distribution(directors)
    resolved, stats = consensus(directors)
    after = basis_distribution(resolved)
    log_consensus(context.log, stats, before, after)
    if not (stats["filled"] or stats["overturned"]):
        return

    # save: the changed gender columns
    context.store.save(
        Tables.def14a_directors,
        resolved[["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"]],
        pk=["ticker", "accession_number", "name"],
    )

    # The filing's stated `n_women_directors` still wins; this ratio only improves the fallback.
    parent = recompute_parent_gender(resolved)
    if not parent.empty:
        context.store.save(Tables.def14a_llm, parent, pk=["ticker", "accession_number"])
        context.log.info("gender consensus: refreshed pct_female_directors / pct_gender_stated on %d filings", len(parent))


def section_extractors(context: Context, config: DictConfig) -> dict[str, LLMExtractor]:
    """One extractor per section, bound to its own action, prompt and schema; two concurrent calls each, which also
    bounds the answers in flight."""
    workers = 2
    return {section.key: LLMExtractor(context, config, action=section.action, threads=workers) for section in SECTIONS}


def _walked_proxies(df_text: pd.DataFrame | None, names: Mapping[str, str]) -> pd.DataFrame:
    """One row per walked proxy, in filing order: the filings frame `load_work` reads, with the universe company name;
    a NULL period reads back as '' (Postgres COPY stores the walk's '' as NULL)."""
    df = (pd.DataFrame(columns=_TEXT_COLUMNS) if df_text is None else df_text).drop_duplicates(["ticker", "accession_number"])
    df = df.assign(filing_date=pd.to_datetime(df["filing_date"]).dt.normalize(), ticker=df["ticker"].astype(str))
    df = df.assign(period=df["period"].fillna(""), company_name=df["ticker"].map(names).fillna(""))
    return df.sort_values(["ticker", "filing_date", "accession_number"], kind="stable").reset_index(drop=True)


def _ticker_works(context: Context, ticker: str, df_filings: pd.DataFrame, floor: pd.Timestamp) -> list[FilingWork]:
    """The ticker's walked proxies with their saved facts and archives, and the uncut payloads of those with an open
    section."""
    # load: the saved facts and archives, then the payloads of the proxies that need them
    works = load_work(context, ticker, df_filings, floor)
    load_archives(context, works)
    wanted = [work.stamp["accession_number"] for work in works if open_sections(work)]
    if not wanted:
        return works
    df_payloads = context.store.load(
        Tables.sec_def14a_text, columns=["accession_number", "section", "payload"], where={"ticker": ticker, "accession_number": wanted}
    )

    # merge: each section's payload onto its proxy (the `_none` row carries no domain)
    payloads = {str(acc): dict(zip(df.section, df.payload.fillna(""), strict=True)) for acc, df in df_payloads.groupby("accession_number")}
    for work in works:
        work.payloads = payloads.get(str(work.stamp["accession_number"]), {})
    return works


def _requests(context: Context, works: list[FilingWork], budget: int) -> list[tuple[Section, LlmTask]]:
    """The prompt task of every open section with a payload, cut to `budget`; an open section the proxy lacks consumes
    its attempt."""
    requests = []
    for work in works:
        requests.extend(requests_for(work, budget))
        for section in (item for item in open_sections(work) if not work.payloads.get(item.key, "").strip()):
            reserve(context, work, section, work.attempts[section.key] + 1)
    return requests


def _call_sections(
    context: Context, ticker: str, requests: list[tuple[Section, LlmTask]], extractors: Mapping[str, LLMExtractor], batch_size: int
) -> int:
    """Reserve each owned attempt, call its section's prompt/schema, archive the answer, then postprocess the stored
    uncut payload and save that section only. Returns the sections answered."""
    answered = 0
    for section, tasks in matched_batches(requests, batch_size):
        for task in tasks:
            work = cast(FilingWork, task.meta["work"])
            reserve(context, work, section, min(2, work.attempts[section.key] + 1))
        results = extractors[section.key].run(tasks)
        failed = {cast(FilingWork, r.task.meta["work"]).stamp["accession_number"]: r.error for r in results if not r.ok}
        if failed:
            context.log.warning("%s %s: reserved attempts failed: %s", ticker, section.key, failed)
        for result in (r for r in results if r.ok):
            work = cast(FilingWork, result.task.meta["work"])
            assert result.parsed is not None
            work.raw = append_answer(work.raw, section, result.parsed)
            save_archive(context, work)
            processed = postprocess_section(section, result.parsed, work.stamp, work.saved, work.parent, work.payloads.get(section.key, ""))
            save_section(context, work, processed)
            work.raw = refresh_archive(work.raw, section, processed.legacy, processed.corrections)
            save_archive(context, work)
            answered += 1
    return answered


def fetch_def14a_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    extractors: Mapping[str, LLMExtractor],
) -> None:
    """Extract every section of a walked proxy filed by its ticker's end (plus the grace) that has a key gap and fewer
    than two attempts (`retry=1`, then one more call and `retry=2`), from the stored tables alone. `extractors`
    (`section_extractors`) holds one extractor per section; their worker count bounds the answers in flight."""
    # load: the walked proxies in the window (no payloads), the ticker names and end bounds
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
        works = _ticker_works(context, str(ticker), df_filings, floor)
        updated += _call_sections(context, str(ticker), _requests(context, works, budget), extractors, batch_size)

    # save: the cross-ticker gender reconciliation after saved updates
    if updated:
        _finalise_gender(context)
    context.log.info("DEF 14A: %d section answer(s) extracted; sections at retry=2 skipped", updated)
