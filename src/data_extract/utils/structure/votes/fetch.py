"""Extract stored Item 5.07 narratives, preserving accession decisions and per-ticker durable writes."""

from __future__ import annotations

import logging
from collections import Counter
from functools import partial
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.frame_sanitize import strip_nul
from src.data_extract.utils.common.resume import DONE_PER_KEY, reference_values, stored_accessions
from src.data_extract.utils.edgar_extract.empty_markers import marker_frame
from src.data_extract.utils.edgar_extract.parallel_fetch import run_per_ticker
from src.data_extract.utils.structure.votes.flatten import _NUMERIC_COLS, _clean_proposal_number, _director_columns, _proposal_rows
from src.data_extract.utils.structure.votes.guard import _grounded_nominees, _source_counts, mentions_preliminary, rejection_reason
from src.data_extract.utils.structure.votes.roles import _role_map, _role_source
from src.data_extract.utils.structure.votes.vote_schema import PROPOSAL_TYPES, Item507Extract
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmTask
from src.utils.data_policy import overlap_column, source_read_window, window_policy
from src.utils.string import clean_text

logger = logging.getLogger(__name__)
VOTES_ACTION = "sec8k_votes"
_SOURCE_COLS = ("ticker", "cik", "accession_number", "form", "filing_date", "period_of_report", "is_amendment", "item_text")


def _load_narratives(context: Context, tickers: list[str]) -> pd.DataFrame | None:
    """Load requested voting narratives inside the source/output/run intersection."""
    policy = window_policy(context.config, Tables.sec_8k_votes, context.run.as_of)
    source = source_read_window(context.config, Tables.sec_8k, context.run.as_of)
    window = policy.requested.intersection(source) if policy is not None and source is not None else None
    if window is None:
        return None
    df_source = context.store.load(
        Tables.sec_8k,
        columns=list(_SOURCE_COLS),
        where={"item": "5.07", "ticker": tickers},
        since=window.start,
        until=window.end,
        optional=True,
    )
    if df_source is None or df_source.empty:
        context.log.info("no stored Item 5.07 narratives for the %d requested ticker(s)", len(tickers))
        return None
    return df_source


def _plan_todo(context: Context, df_source: pd.DataFrame) -> pd.DataFrame:
    """Exclude saved decisions, markers included, by each ticker's configured completion reference."""
    column = overlap_column(context.config, Tables.sec_8k_votes)
    seen = stored_accessions(
        context,
        Tables.sec_8k_votes,
        df_source.ticker.unique().tolist(),
        DONE_PER_KEY,
        where={column: df_source[column].dropna().unique().tolist()},
    )
    references = reference_values(df_source[column], Tables.sec_8k_votes, column)
    completed = {(ticker, value) for ticker, values in seen.items() for value in values}
    df_todo = df_source[~pd.MultiIndex.from_arrays([df_source.ticker.astype(str), references]).isin(completed)]
    context.log.info("Item 5.07: %d stored filing(s), %d already parsed, %d to read", len(df_source), len(df_source) - len(df_todo), len(df_todo))
    return df_todo


def _extract_ticker(ticker: str, context: Context, df_source: pd.DataFrame, extractor: LLMExtractor) -> tuple[int, int, dict[str, int]]:
    """Load one proxy roster set, call that ticker's filings, ground answers and save decisions."""
    # Load proxy evidence once before preparing this ticker's tasks.
    role_source = _role_source(context, ticker)
    df_filings = df_source[df_source.ticker.eq(ticker)]
    budget = int(context.config.data_extract.llm_text_budget.votes)
    tasks: list[LlmTask] = []
    frames: list[pd.DataFrame] = []
    skips: Counter[str] = Counter()
    for _, filing in df_filings.iterrows():
        text = cast(str, filing.get("item_text"))
        stamp = {"ticker": ticker, **{column: filing.get(column) for column in ("accession_number", "filing_date", "cik", "form")}}
        stamp["form"] = clean_text(stamp["form"])
        reason = rejection_reason(text)
        if reason is not None:
            skips[reason] += 1
            frames.append(marker_frame(Tables.sec_8k_votes, stamp))
            continue
        tasks.append(
            LlmTask(
                seq=len(tasks),
                payload=text[:budget],
                schema=Item507Extract,
                table=Tables.sec_8k_votes,
                meta={"ticker": ticker, "filing": filing, "text": text, "stamp": stamp},
            )
        )

    # Persist refused inputs before calling the provider; failed calls receive no decision.
    if frames:
        context.store.save(Tables.sec_8k_votes, pd.concat(frames, ignore_index=True))
    results = extractor.run(tasks)
    frames = []
    all_rows: list[dict[str, Any]] = []
    rejected = 0
    for result in results:
        if not result.ok:
            logger.warning("%s: Item 5.07 LLM extraction failed (%s)", ticker, result.error)
            skips["extraction failed"] += 1
            continue
        extract = result.parsed
        if not isinstance(extract, Item507Extract):
            raise TypeError("Item 5.07 answer does not match its requested schema")
        meta = result.task.meta
        filing = cast(pd.Series, meta["filing"])
        text = str(meta["text"])
        counts = _source_counts(text)
        roles, titles = _role_map(role_source, filing.get("period_of_report"))
        prepared: list[dict[str, Any]] = []
        for proposal in extract.proposals:
            nominees, n_bad = _grounded_nominees(proposal, text, counts)
            ptype = (proposal.proposal_type or "").strip().lower()
            prepared.append(
                {
                    **proposal.model_dump(exclude={"nominees"}),
                    "proposal_type": ptype if ptype in PROPOSAL_TYPES else None,
                    "proposal_number": _clean_proposal_number(proposal.proposal_number),
                    "nominees": nominees,
                    "rejected": n_bad,
                    "director": _director_columns(nominees, roles, titles) if ptype == "director_election" else {},
                }
            )
        rows, n_bad = _proposal_rows(ticker, filing, extract, prepared, counts, mentions_preliminary(text))
        all_rows.extend(rows)
        rejected += n_bad
        if not rows:
            skips["no proposals survived the guard"] += 1
            frames.append(marker_frame(Tables.sec_8k_votes, cast(dict[str, object], meta["stamp"])))

    # Preserve numeric, NUL, date and primary-key cleaning order before the ticker save.
    if all_rows:
        df_rows = pd.DataFrame(all_rows)
        df_rows = df_rows.assign(**{column: pd.to_numeric(df_rows[column], errors="coerce") for column in _NUMERIC_COLS})
        df_rows = strip_nul(df_rows)
        df_rows = df_rows.assign(
            **{column: pd.to_datetime(df_rows[column], errors="coerce") for column in ("filing_date", "period_of_report", "meeting_date")}
        )
        frames.append(df_rows.drop_duplicates(subset=list(Tables.sec_8k_votes.pk), keep="last"))
    if frames:
        context.store.save(Tables.sec_8k_votes, pd.concat(frames, ignore_index=True))
    logger.info("%s: %d vote rows, %d guard rejections", ticker, len(all_rows), rejected)
    return len(all_rows), rejected, dict(skips)


def fetch_8k_votes_llm(context: Context, tickers: list[str], extractor: LLMExtractor) -> None:
    """Plan narratives once and run serial ticker workers around the shared concurrent extractor."""
    # Load and plan source keys without replaying completed decisions in any run mode.
    df_source = _load_narratives(context, tickers)
    if df_source is None:
        return
    df_todo = _plan_todo(context, df_source)
    if df_todo.empty:
        return

    # The extractor's queues are shared; only its within-ticker filing calls run concurrently.
    results = run_per_ticker(
        df_todo[["ticker"]].drop_duplicates().sort_values("ticker"),
        partial(_extract_ticker, context=context, df_source=df_todo, extractor=extractor),
        "8-K votes",
        max_workers=1,
        key_cols=("ticker",),
    )
    completed = [result for result in results if result is not None]
    skips: Counter[str] = Counter()
    for _, _, reasons in completed:
        skips.update(reasons)
    context.log.info("Item 5.07: %d rows written, %d guard rejections", sum(row[0] for row in completed), sum(row[1] for row in completed))
    for reason, count in skips.most_common():
        context.log.info("Item 5.07: %d filings skipped — %s", count, reason)
