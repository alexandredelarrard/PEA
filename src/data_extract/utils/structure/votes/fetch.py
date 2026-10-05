"""Parse the Form 8-K Item 5.07 narratives already stored in `sec_8k.item_text` into `sec_8k_votes`.

One row per proposal; a director election collapses to one row with per-role-category vote columns.
No new download. A filing that was read and yields no vote row (the guard refused its text, or the
answer has no surviving proposal) is stored as one empty-filing marker and never queued again; a
failed LLM call writes nothing and is queued on the next run. A vote table has no independent total, so there is no validation gate: the per-nominee
sum is a flag (`nominee_sum_matches`), not a filter. Three hard rules instead:
1. Fabrication guard: no task unless the text passes `rejection_reason`, and nominees not grounded in the source are dropped.
2. Never "latest wins" on an amendment: each accession's rows are stored under it, nothing is deduped
   across accessions, and the reader unions on `(ticker, period_of_report)`.
3. A per-line tally (e.g. a say-on-pay frequency vote) is never relabelled For/Against: the buckets go to
   `nominee_votes_json` and `votes_for` / `votes_against` are nulled in `flatten`.
Role categories join each nominee to the nearest prior proxy on the `lastname|firstinitial` key; the
per-filing unmatched count is stored. Zero rows is a normal outcome (e.g. a 5.07(d) board-response filing).
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.schemas.vote_schema import Item507Extract
from src.data_extract.utils.structure.votes.flatten import _marker_frame, _prepare_frame, _proposal_rows
from src.data_extract.utils.structure.votes.guard import rejection_reason
from src.data_extract.utils.structure.votes.roles import _role_map, _role_source
from src.data_store.schema import Table, Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

logger = logging.getLogger(__name__)

#: The 8-K item code this module reads.
_ITEM = "5.07"

#: Columns read out of `sec_8k` (a narrow projection: `item_text` is its widest column).
_SOURCE_COLS = ("ticker", "cik", "accession_number", "form", "filing_date", "period_of_report", "is_amendment", "item_text")


def _result_frames(result: LlmResult, tally: dict) -> dict[Table, pd.DataFrame]:
    """One answer -> the `sec_8k_votes` rows that survive the fabrication guard, else its marker.

    Runs on the main thread after the pool joins; `roles` / `titles` were resolved at task-build time.
    """
    meta = result.task.meta
    extract = result.parsed
    assert isinstance(extract, Item507Extract)
    rows, rejected = _proposal_rows(
        str(meta["ticker"]),
        cast(pd.Series, meta["filing"]),
        extract,
        str(meta["text"]),
        cast(dict[str, str], meta["roles"]),
        cast(dict[str, str], meta["titles"]),
    )
    tally["rejected"] += rejected
    if not rows:
        # Not a failure: a 5.07(d) board-response filing correctly yields zero rows.
        tally["skips"]["no proposals survived the guard"] = tally["skips"].get("no proposals survived the guard", 0) + 1
        return {Tables.sec_8k_votes: _marker_frame(str(meta["ticker"]), cast(pd.Series, meta["filing"]))}
    tally["rows"].extend(rows)
    return {Tables.sec_8k_votes: _prepare_frame(rows)}


def fetch_8k_votes_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    model: str | None = None,
    max_chars: int | None = None,
    cache: bool | None = None,
    workers: int | None = None,
) -> None:
    """Build/refresh `sec_8k_votes` from the stored Item 5.07 narratives, ticker by ticker.

    Reads a projection of `sec_8k`, skips accessions already stored (markers included), and upserts
    each ticker's rows and markers before starting the next. Skips gracefully when OPENAI_API_KEY is absent.

    `model` / `max_chars` / `cache` default to `config.gpt` (`max_chars.sec8k_votes`), and `workers`
    (concurrent LLM calls) to `config.gpt.threads`; an explicit keyword pins one without touching config.
    """
    config = with_gpt_overrides(config, "sec8k_votes", model=model, max_chars=max_chars, cache=cache)
    if not context.store.exists(Tables.sec_8k):
        context.log.warning("sec_8k does not exist yet — run the 8-K fetcher first; Item 5.07 votes skipped")
        return

    df_source = context.store.load(Tables.sec_8k, columns=list(_SOURCE_COLS), where={"item": _ITEM, "ticker": list(tickers)})
    if df_source is None or df_source.empty:
        context.log.info("no stored Item 5.07 narratives for the %d requested ticker(s)", len(tickers))
        return

    seen = stored_values(context, Tables.sec_8k_votes, "accession_number")
    df_todo = df_source[~df_source["accession_number"].isin(seen)]
    context.log.info(
        "Item 5.07: %d stored filing(s) for %d ticker(s), %d already parsed, %d to read",
        len(df_source),
        df_source["ticker"].nunique(),
        len(df_source) - len(df_todo),
        len(df_todo),
    )
    if df_todo.empty:
        return

    try:
        extractor = LLMExtractor(context, config, action="sec8k_votes", threads=workers)
    except OSError as e:
        context.log.warning("Item 5.07 vote extraction skipped: %s", e)
        return

    total_rows, total_rejected = 0, 0
    skips: dict[str, int] = {}
    for ticker, df_filings in tqdm(df_todo.groupby("ticker"), desc="8-K votes"):
        # Read once per ticker on this thread, so a worker never needs the database to categorise a nominee.
        role_source = _role_source(context, str(ticker))

        tasks: list[LlmTask] = []
        refused: list[pd.DataFrame] = []
        for _, f in df_filings.iterrows():
            text = f.get("item_text")
            reason = rejection_reason(text)
            if reason is not None:
                # Rejected before a task exists, so a truncated or tally-free narrative costs nothing; it is marked as read.
                skips[reason] = skips.get(reason, 0) + 1
                refused.append(_marker_frame(str(ticker), f))
                continue
            roles, titles = _role_map(role_source, f.get("period_of_report"))
            tasks.append(
                LlmTask(
                    seq=len(tasks),
                    payload=cast(str, text),
                    schema=Item507Extract,
                    table=Tables.sec_8k_votes,
                    meta={"ticker": str(ticker), "filing": f, "text": text, "roles": roles, "titles": titles},
                )
            )

        if refused:
            context.store.save(Tables.sec_8k_votes, pd.concat(refused, ignore_index=True))
        tally: dict = {"rejected": 0, "rows": [], "skips": {}}
        results = extractor.run_extraction(
            tasks,
            flatten=lambda r, tally=tally: _result_frames(r, tally),
            group_key=lambda t: str(t.meta["ticker"]),
        )

        total_rejected += tally["rejected"]
        for reason, n in tally["skips"].items():
            skips[reason] = skips.get(reason, 0) + n
        for failed in (r for r in results if not r.ok):
            logger.warning("%s: Item 5.07 LLM extraction failed (%s)", ticker, failed.error)
            skips["extraction failed"] = skips.get("extraction failed", 0) + 1

        ticker_rows = tally["rows"]
        if ticker_rows:
            total_rows += len(ticker_rows)
            unmatched = sum(r.get("n_nominees_unmatched") or 0 for r in ticker_rows)
            nominees = sum(r.get("n_nominees") or 0 for r in ticker_rows)
            context.log.info(
                "%s: +%d vote row(s) from %d filing(s); %d/%d nominees unmatched",
                ticker,
                len(ticker_rows),
                len(df_filings),
                int(unmatched),
                int(nominees),
            )

    context.log.info("Item 5.07: %d row(s) written, %d row(s) rejected by the fabrication guard", total_rows, total_rejected)
    for reason, n in sorted(skips.items(), key=lambda kv: -kv[1]):
        # Each is a filing that produced no rows; the reason separates an accepted loss from a correct empty answer.
        context.log.info("Item 5.07: %d filing(s) skipped — %s", n, reason)
