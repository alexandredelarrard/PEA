"""
fetch.py  (src/data_extract/utils/structure/votes/fetch.py)
--------------------------------------------------------------------------------
Parse the Form 8-K **Item 5.07** narratives ALREADY STORED in `sec_8k.item_text` into
`sec_8k_votes` — one row per proposal, with a director election collapsed to one row
carrying per-role-category vote columns.

No new download. `fetch_8k_edgar` already tags item 5.07 as high-signal and stores the
narrative: 6,657 rows over 405 tickers, 2010-03-01 -> 2026-08-19, 99.2% with `item_text`
longer than 200 chars, and nothing has ever parsed them. `item_text` carries zero HTML
tags but IS edgartools' rendered view, so column alignment survives (84.6% carry a
`U+2500` rule under the header, the rest are whitespace-aligned, one filing row per
text line) — which is why the model can read the tables out of plain text.

**There is no validation gate for a vote table, and this module does not pretend to
build one.** A vote table prints no independent total, and the dominant extraction error
is a column permutation, which is INVARIANT under sums. Measured on 57 filings: a
"total <= shares outstanding" bound is computable on only 9 (16%); per-nominee totals are
computable on 96% and hold on 91%; meeting-level totals are computable on 100% and hold
on 81%. Combining all three gives recall 0.56 / precision 0.56 — **7 of 16 known-bad
filings pass clean**. So the per-nominee sum lands as a FLAG (`nominee_sum_matches`), a
monitor and not a filter. Prior-year say-on-pay in the proxy narrative is not an
independent check either: it is lossy (Merck says "approximately 94%" where its own 8-K
says 93.50%) and the denominator convention varies by state of incorporation (XOM notes
New Jersey excludes abstentions from votes cast).

What replaces a gate is three hard rules, all of them the LLM's own measured failure modes:

1. **The fabrication guard.** The model invented an entire table ("John Doe" / "Jane
   Smith", 250,000,000 votes) for a filing whose `item_text` was truncated, and a fake
   row for another. So: emit nothing unless the text is long enough AND contains a
   comma-grouped number, and drop any nominee whose name is not in the source or whose
   numbers are all absent from it. See `rejection_reason` / `_grounded_nominees`.
2. **Never "latest wins" on an amendment.** Of 190 multi-filing meetings, 135 (71%) have
   an amendment carrying no vote numbers at all, so "latest wins" is right on 33/190
   (17%) while "union the group" is right on 173/190 (91%). Every filing's rows are
   therefore stored under its own accession and NOTHING is deduped across accessions; the
   reader unions on `(ticker, period_of_report)`. The trap worth naming: **in a contested
   election the FIRST 8-K is the preliminary one** — DIS `0000950157-24-000595` states
   "estimated preliminary voting results ... do not include shares voted on the blue
   proxy card distributed by Trian" and the 8-K/A carries the Inspector of Election's
   final results, switching from Against to Withhold on the way.
3. **A per-line tally is never relabelled For/Against.** A say-on-pay FREQUENCY vote is
   printed `1 Year | 2 Years | 3 Years | Abstain | Broker Non-Votes` (Agilent writes
   `Every 1 Year`), and folding three year buckets onto For/Against/Abstain shifts every
   column along one, drops the broker non-votes off the end, and leaves `votes_against`
   reading as opposition when it counts shareholders who wanted a BIENNIAL vote. Neither
   guard above can see it — the shifted numbers are all genuinely printed in the source,
   so both grounding tests pass. So the year buckets go to `nominee_votes_json` and
   `votes_for` / `votes_against` are NULLED in `flatten`, not merely discouraged in the
   prompt. Same shape of defect as the dropped-header column permutation, same defence:
   the layer that knows the proposal's TYPE is the one that has to enforce it.

Role categorisation joins each nominee to the nearest prior proxy (`def14a_llm`,
`def14a_executive_comp`, `def14a_director_comp`) on the same `lastname|firstinitial` key
the DEF 14A gender consensus uses, so a nominee that matches there matches here. The
per-filing `unmatched` count is stored, not swallowed: that number IS the join's error
rate and it has to stay visible.

Accepted losses, recorded rather than repaired:
  * **62 rows corpus-wide (1.0%)** store a truncated `item_text` because the filer
    numbered its proposals `Item 1.` / `Item 2.` and edgartools' item splitter cut there
    (HWM 2019 stores 508 chars where the live filing has 10,922 and four tables). We do not
    re-fetch the HTML to repair them. Note what the LENGTH floor can and cannot do here:
    the shortest GENUINE tally in the baseline is 554 chars, so a floor high enough to
    reject a 508-char stub would destroy two correct filings. A stub cut before the tables
    is caught because it carries no share count, not because it is short.
  * **8.8% of Item 5.07 filings carry no vote table at all** — a 5.07(d) board-response
    filing discloses only the board's reaction to a say-on-pay frequency vote. Zero rows
    is a NORMAL outcome here, logged as such.
  * **9 of 190 multi-filing meetings (4.7%)** give no signal beyond the form type as to
    which filing supersedes which; only 14 filings corpus-wide say "preliminary", so
    `mentions_preliminary` resolves 8 of the 17 genuine restatements and no more.
"""
from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.sec_utils import existing_filings
from src.data_extract.utils.schemas.vote_schema import Item507Extract
from src.data_extract.utils.structure.votes.flatten import _prepare_frame, _proposal_rows
from src.data_extract.utils.structure.votes.guard import rejection_reason
from src.data_extract.utils.structure.votes.roles import _role_map, _role_source
from src.data_store.schema import Table, Tables
# `gpt_extract` is a shared service, like `src/utils/` -- the sanctioned cross-import. It
# owns the model, the keys, the prompts (`prompt_templates/sec8k_votes_*.md`) and the
# thread pool; the fabrication guard and the role map are this package's business.
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

logger = logging.getLogger(__name__)

#: The 8-K item code this module reads. Anything else in `sec_8k` is somebody else's row.
_ITEM = "5.07"

#: Columns read out of `sec_8k`. A narrow projection is mandatory (AGENTS.md): the table
#: holds ~197k item rows and `item_text` is the widest column in it, so an unprojected
#: read would pull every item code's narrative to select 3.4% of them.
_SOURCE_COLS = ("ticker", "cik", "accession_number", "form", "filing_date",
                "period_of_report", "is_amendment", "item_text")

#: Concurrent LLM calls, for the same measured reason as the DEF 14A path: the work is pure
#: network wait on an API that accepts parallel requests, and a serial universe run is not
#: finishable. An Item 5.07 narrative is far smaller than a proxy (the longest in the
#: 333-filing baseline is 17,032 chars), so the per-call latency is lower -- but there are
#: 6,657 of them. `config.gpt.threads` is the live knob; this is the no-config fallback.
_LLM_WORKERS = 12


def _result_frames(result: LlmResult, tally: dict) -> dict[Table, pd.DataFrame]:
    """One answer -> the `sec_8k_votes` rows that survive the fabrication guard.

    Runs on the MAIN thread, after the pool has joined. `roles` / `titles` were resolved at
    task-build time off the three frames read once per ticker, so no worker ever needed the
    store to categorise a nominee.
    """
    meta = result.task.meta
    rows, rejected = _proposal_rows(meta["ticker"], meta["filing"], result.parsed,
                                    meta["text"], meta["roles"], meta["titles"])
    tally["rejected"] += rejected
    if not rows:
        # NOT the same as a failure: a 5.07(d) board-response filing correctly yields zero
        # rows. The reason is what separates an accepted loss from a correct empty answer.
        tally["skips"]["no proposals survived the guard"] = \
            tally["skips"].get("no proposals survived the guard", 0) + 1
        return {}
    tally["rows"].extend(rows)
    return {Tables.sec_8k_votes: _prepare_frame(rows)}


def fetch_8k_votes_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    model: str | None = None,
    max_chars: int | None = None,
    cache: bool | None = None,
    workers: int = _LLM_WORKERS,
) -> None:
    """Build/refresh `sec_8k_votes` from the stored Item 5.07 narratives, ticker by ticker.

    Reads `sec_8k` (never unprojected), skips accessions already stored, and upserts each
    ticker's rows before starting the next -- LLM calls are paid for, so nothing is
    batched. Skips gracefully when OPENAI_API_KEY is absent.

    `max_chars` defaults to `config.gpt.max_chars.sec8k_votes` (40k, not the DEF 14A
    path's 130k) because the input is one already-carved item, not a whole proxy: the
    longest Item 5.07 narrative in the 333-filing baseline is 17,032 chars, so 40k
    truncates nothing while keeping a runaway `item_text` from turning into a runaway bill.

    `model` / `cache` default to `config.gpt`; pass an explicit keyword to pin one for
    research without touching config.
    """
    config = with_gpt_overrides(config, "sec8k_votes", model=model, max_chars=max_chars,
                                cache=cache)
    if not context.store.exists(Tables.sec_8k):
        context.log.warning("sec_8k does not exist yet — run the 8-K fetcher first; "
                            "Item 5.07 votes skipped")
        return

    source = context.store.load(Tables.sec_8k, columns=list(_SOURCE_COLS),
                                where={"item": _ITEM, "ticker": list(tickers)})
    if source is None or source.empty:
        context.log.info("no stored Item 5.07 narratives for the %d requested ticker(s)",
                         len(tickers))
        return

    seen = set(existing_filings(context, Tables.sec_8k_votes))
    todo = source[~source["accession_number"].isin(seen)]
    context.log.info("Item 5.07: %d stored filing(s) for %d ticker(s), %d already parsed, "
                     "%d to read", len(source), source["ticker"].nunique(),
                     len(source) - len(todo), len(todo))
    if todo.empty:
        return

    try:
        extractor = LLMExtractor(context, config, action="sec8k_votes", threads=workers)
    except EnvironmentError as e:
        context.log.warning("Item 5.07 vote extraction skipped: %s", e)
        return

    total_rows, total_rejected = 0, 0
    skips: dict[str, int] = {}
    for ticker, group in tqdm(todo.groupby("ticker"), desc="8-K votes"):
        # read ONCE per ticker, on this thread; `_role_map` is pure, so the per-meeting map
        # is resolved here too and a worker never needs the database to categorise a nominee
        role_source = _role_source(context, str(ticker))

        tasks: list[LlmTask] = []
        for _, f in group.iterrows():
            text = f.get("item_text")
            reason = rejection_reason(text)
            if reason is not None:
                # short-circuits BEFORE a task exists, so a truncated or tally-free
                # narrative costs nothing
                skips[reason] = skips.get(reason, 0) + 1
                continue
            roles, titles = _role_map(role_source, f.get("period_of_report"))
            tasks.append(LlmTask(seq=len(tasks), payload=text, schema=Item507Extract,
                                 table=Tables.sec_8k_votes,
                                 meta={"ticker": str(ticker), "filing": f, "text": text,
                                       "roles": roles, "titles": titles}))

        tally: dict = {"rejected": 0, "rows": [], "skips": {}}
        results = extractor.run_extraction(
            tasks, flatten=lambda r: _result_frames(r, tally),
            group_key=lambda t: str(t.meta["ticker"]))

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
            context.log.info("%s: +%d vote row(s) from %d filing(s); %d/%d nominees unmatched",
                             ticker, len(ticker_rows), len(group), int(unmatched), int(nominees))

    context.log.info("Item 5.07: %d row(s) written, %d row(s) rejected by the fabrication "
                     "guard", total_rows, total_rejected)
    for reason, n in sorted(skips.items(), key=lambda kv: -kv[1]):
        # every one of these is a filing that produced NO rows; the reason is what
        # separates an accepted loss from a correct empty answer
        context.log.info("Item 5.07: %d filing(s) skipped — %s", n, reason)

