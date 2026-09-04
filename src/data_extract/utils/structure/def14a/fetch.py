"""
fetch.py  (src/data_extract/utils/structure/def14a/fetch.py)
-----------------------------------------------------------------
Extract structured governance data from SEC DEF 14A proxy statements using an
LLM with structured output (Def14AExtract schema).

Per ticker, it fetches that ticker's DEF 14A filings from EDGAR, sends targeted
sections to the OpenAI Responses API (constrained to the Def14AExtract Pydantic
schema, prompt caching on), then **immediately upserts that
ticker's rows into the `def14a_llm` Postgres table** before moving to the next
ticker — so an interrupted run never loses the (expensive) LLM calls already made.

Per-filing incremental (gap-filling): each ticker's FULL `years_history` window of DEF 14A
filings is listed, and the LLM is (re-)run ONLY on filings whose `accession_number` is NOT already
in the `def14a_llm` table. So any MISSING year/filing — including a hole in the middle of the
history, not just after the latest — is filled, while every already-extracted filing is skipped
(no repeat LLM cost). Tickers with no rows yet get the whole window; already-complete tickers make
no LLM calls at all.

Requires OPENAI_API_KEY (or OPEN_AI_API_KEY) in the .env file.
If the key is absent the function logs a warning and returns whatever exists.

Output columns (DB table `def14a_llm`), scalar summaries + raw JSON:
    keys        ticker, as_of, period, accession_number, company_name, fiscal_year_extract
    board       n_directors, board_size, avg_director_age, avg_board_tenure,
                pct_independent_directors, pct_female_directors,
                avg_other_public_boards, pct_gender_stated,
                n_women_directors_vs_inferred
    ceo         ceo_name_proxy, ceo_age, ceo_since_year, ceo_is_founder,
                ceo_is_board_chair, ceo_salary, ceo_bonus, ceo_stock_awards,
                ceo_option_awards, ceo_non_equity_incentive, ceo_all_other_comp,
                ceo_total_comp, ceo_equity_pay_pct
    neos        n_neos, total_neo_comp, sct_years
    ownership   insider_ownership_pct, ceo_ownership_pct, n_five_percent_holders,
                n_ownership_rows
    governance  independent_chair, lead_independent_director, classified_board,
                dual_class_shares, poison_pill, majority_voting,
                say_on_pay_support_pct, ceo_pay_ratio, median_employee_pay
    auditor     auditor_name, auditor_since_year, auditor_fees, audit_fees_audit,
                audit_fees_audit_related, audit_fees_tax, audit_fees_other,
                auditor_fees_prior
    counts      n_director_comp_rows, n_ownership_rows
    def14a_json (full Def14AExtract as JSON for downstream use)

`n_technology_directors` / `pct_technology_directors` / `technology_committee` were REMOVED:
they were an opinion, not an extraction (mean |delta| of 1.06 directors between consecutive
filings of the same company, only 38.8% unchanged).

FOUR CHILD TABLES are written alongside, flattened out of the same paid extract:
    def14a_executive_comp   one row per NEO per fiscal year (Item 402(c), ~3 years/filing)
    def14a_director_comp    one row per non-employee director (Item 402(k), single-year)
    def14a_ownership        one row per beneficial holder (Item 403)
    def14a_directors        one row per director -- and the substrate the cross-filing gender
                            consensus pass groups over (see def14a_gender.py)
"""
from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping, sec_get
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_sections
from src.data_extract.utils.structure.def14a.flatten import _NUMERIC_COLS, _result_frames
from src.data_extract.utils.structure.def14a.gender import (
    basis_distribution, consensus, log_consensus, recompute_parent_gender,
)
from src.data_store.schema import Tables
# `gpt_extract` is a shared service, like `src/utils/` -- the sanctioned cross-import. It
# owns the model, the keys, the prompts (`prompt_templates/def14a_*.md`) and the thread
# pool; the anchor carve and the flatten are this package's business.
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmTask

logger = logging.getLogger(__name__)

#: Concurrent LLM calls. Not an optimisation -- it is what makes a universe run possible at
#: all. MEASURED on the Phase-6 validation set: one modern proxy is a ~130k-char payload and
#: takes **~94 seconds** on `gpt-5-mini` (a reasoning model), so 8,700 proxies serially is
#: **~9.5 days**. The work is pure network wait on an API that accepts parallel requests,
#: and 12 was measured to draw no 429s. `config.gpt.threads` is the live knob; this is the
#: fallback for a caller that passes no config.
_LLM_WORKERS = 12

def _fetch_filing_html(context: Context, filing: pd.Series) -> str:
    """The filing's raw markup, retrying the `<accession>.txt` full submission when the primary
    document 404s.

    `primaryDocument` names a file that is genuinely ABSENT from the archive on 7 of 663
    measured DEF 14A filings (all 2000-08..2001-03, all naming `"0001.txt"`). Those produce no
    row at all without this retry, because the raise propagates out of `_payload_for` and the
    filing never becomes a task -- a loss invisible in the "pre-2001 rows are NULL" count
    since there is no row to be null. The `.txt` carries the real proxy (53,661-165,380 chars
    on the four spot-checked).
    """
    try:
        return sec_get(context, filing["doc_url"]).text
    except Exception:
        txt_url = filing.get("txt_url")
        if not txt_url or txt_url == filing["doc_url"]:
            raise
        logger.info("%s: primary document unavailable, falling back to the full submission",
                    filing.get("accession_number", ""))
        return sec_get(context, txt_url).text


def _payload_for(context: Context, ticker: str, filing: pd.Series) -> str | None:
    """The carved `=== LABEL ===` text one filing contributes, or None if it cannot be read.

    Fetching and carving happen on the MAIN thread, before any task is queued: a worker
    receives text and a schema, never a `Context`. The SEC fetch is disk-cached and rate
    limited anyway, so the ~94s LLM call is what the pool is for.
    """
    try:
        raw_html = _fetch_filing_html(context, filing)
        return prepare_def14a_sections(raw_html, html_to_text(raw_html))
    except Exception as e:                          # noqa: BLE001 -- one filing, not the run
        logger.warning("%s %s: DEF 14A filing could not be read (%s)",
                       ticker, filing.get("filing_date", ""), e)
        return None


def _is_up_to_date(context: Context, requested_tickers: list[str]) -> bool:
    """Up to date only when EVERY requested ticker already has rows in the DB AND
    the shared extraction manifest (`run_manifest.py`) was refreshed today. The old
    check compared a DATE + a stored COUNT (`universe_size`), so a same-day rerun
    skipped tickers that were never actually extracted -- the '~15 tickers then it
    stops' bug. Checking per-ticker coverage (tickers x date) makes a rerun pick up
    the still-missing names; the per-ticker loop then skips already-done filings
    via `seen` (no re-LLM)."""
    if not context.store.exists(Tables.def14a_llm):
        return False
    entry = get_entry(context, Tables.def14a_llm)
    if entry is None or entry.get("last_run_date") != pd.Timestamp.today().strftime(DATE_FORMAT):
        return False
    have = set(context.store.load(Tables.def14a_llm, columns=["ticker"])["ticker"].dropna())
    return set(requested_tickers).issubset(have)



def _finalise_gender(context: Context) -> None:
    """Cross-ticker gender consensus, run ONCE after the per-ticker loop.

    It cannot live inside the loop: a director recurs across COMPANIES as well as years, so the
    consensus needs every ticker's rows before it can group on people. Cheap on a routine rerun
    -- DEF 14A is a yearly filing, so an incremental day adds ~0 rows and the pass is a narrow
    read plus a no-op write.

    Both reads are projected to the three columns the consensus needs; both writes carry only
    the columns they change (AGENTS.md: never read a large table unprojected).
    """
    directors = context.store.load(
        Tables.def14a_directors,
        columns=["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"],
        optional=True)
    if directors is None or directors.empty:
        return

    before = basis_distribution(directors)
    resolved, stats = consensus(directors)
    after = basis_distribution(resolved)
    log_consensus(context.log, stats, before, after)

    if not (stats["filled"] or stats["overturned"]):
        return   

    context.store.save(Tables.def14a_directors,
                       resolved[["ticker", "accession_number", "name", "as_of",
                                 "gender", "gender_basis"]],
                       pk=["ticker", "accession_number", "name"])

    # `pct_female_directors` keeps its existing precedence -- the filing's own
    # `n_women_directors` first, this ratio as the FALLBACK -- so a consensus correction makes
    # the fallback better rather than overriding a stated count.
    parent = recompute_parent_gender(resolved)
    if not parent.empty:
        context.store.save(Tables.def14a_llm, parent, pk=["ticker", "accession_number"])
        context.log.info("gender consensus: refreshed pct_female_directors / pct_gender_stated "
                         "on %d filings", len(parent))


def fetch_def14a_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    model: str | None = None,
    max_chars: int | None = None,
    cache: bool | None = None,
    workers: int = _LLM_WORKERS,
) -> None:
    """Build/refresh the DEF 14A LLM governance extract, one ticker at a time.

    For each ticker only filings AFTER its latest stored `as_of` are sent to the
    LLM (year-incremental), and the ticker's rows are upserted to Postgres
    immediately. Skips gracefully when OPENAI_API_KEY is absent.

    `model` / `max_chars` / `cache` default to `config.gpt`; pass an explicit keyword to
    pin one for research without touching config (how a prior measurement ran
    `gpt-4o-mini` while production had been running `gpt-5-mini` all along).
    """
    config = with_gpt_overrides(config, "def14a", model=model, max_chars=max_chars,
                                cache=cache)
    years = context.config.data_extract.years_history
    de = context.config.data_extract

    cik_map = load_cik_mapping(context, tickers)

    if _is_up_to_date(context, cik_map["ticker"].tolist()):
        existing = context.store.load(Tables.def14a_llm)
        context.log.info("DEF 14A LLM already up to date — every requested ticker present "
                         "(%d rows) — skipping", len(existing))
        return existing

    # accessions already extracted -> never re-LLM (accession-only dedup, same convention as
    # fetch_8k_edgar.py / fetch_13d_edgar.py / fetch_def14a_edgar.py's `existing_filings`)
    # a mutable copy: this fetcher is serial and adds each accession as it extracts it,
    # so a ticker filing twice in one run is not sent to the LLM twice
    seen = set(existing_filings(context, Tables.def14a_llm))

    # Manifest-driven listing window (see run_manifest.py): a routine run only lists
    # filings from the last run's date onward; a ticker-count change or the
    # `manifest_full_rescan_days` self-heal window falls back to the FULL `years`
    # window (gap-filling, same self-heal rationale as the other 4 EDGAR fetchers).
    # `list_filings`'s own `since` cutoff is STRICTLY AFTER the date passed, so we
    # step back one day to keep the last run's date itself inclusive.
    rescan_days = int(getattr(de, "manifest_full_rescan_days", 30))
    manifest_since, is_full_rescan = manifest_window(
        context, Tables.def14a_llm, len(cik_map),
        fallback_since=pd.Timestamp.today() - pd.DateOffset(years=years),
        full_rescan_days=rescan_days)
    list_since = None if is_full_rescan else (manifest_since - pd.Timedelta(days=1))

    try:
        extractor = LLMExtractor(context, config, action="def14a", threads=workers)
    except EnvironmentError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        existing = context.store.load(Tables.def14a_llm, optional=True)
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    total_new, tickers_touched, total_skipped = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        # `list_since=None` (full-rescan runs) lists the FULL years_history window so a MISSING
        # filing anywhere in the history is discovered; otherwise only filings from the manifest's
        # last run date onward are listed. The accession skip below then sends ONLY the
        # not-yet-stored filings to the LLM (gap-filling, per ticker / per date).
        try:
            filings = list_filings(context, cik, DEF14A_FORMS, years, company, since=list_since)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue

        # Resolve the work FIRST, then extract it. Deciding what to send before sending any of
        # it is what lets the LLM calls run concurrently, and the local `done` set covers a
        # ticker that lists the same accession twice in one window (the old loop relied on
        # mutating `seen` mid-iteration, which a pool cannot do safely).
        todo: list[pd.Series] = []
        done: set[str] = set()
        for _, f in filings.iterrows():
            accession = f["accession_number"]
            if accession in seen or accession in done:
                continue
            done.add(accession)
            todo.append(f)
        skipped = len(filings) - len(todo)
        total_skipped += skipped

        # Fetch and carve on THIS thread, then hand the pool text and a schema. A filing
        # whose HTML cannot be read never becomes a task, so it costs nothing.
        tasks: list[LlmTask] = []
        for f in todo:
            payload = _payload_for(context, ticker, f)
            if payload:
                tasks.append(LlmTask(seq=len(tasks), payload=payload, schema=Def14AExtract,
                                     table=Tables.def14a_llm,
                                     meta={"ticker": ticker, "filing": f}))

        # One call per ticker: the pool fills every schema, then THIS thread saves the five
        # frames once. LLM calls are paid for, so a ticker is persisted before the next
        # starts and an interrupted run loses at most one ticker's tokens.
        results = extractor.run_extraction(tasks, flatten=_result_frames,
                                           group_key=lambda t: str(t.meta["ticker"]))
        extracted = [r for r in results if r.ok]
        for r in extracted:
            seen.add(r.task.meta["filing"]["accession_number"])

        if extracted:
            total_new += len(extracted)
            tickers_touched += 1
            context.log.info("%s: +%d new DEF 14A filing(s) sent to the LLM (%d already in table)",
                             ticker, len(extracted), skipped)

    # A cross-ticker consensus needs every ticker's rows, so this is the only thing that
    # cannot run inside the loop. Skipped entirely when nothing new was extracted.
    if total_new:
        _finalise_gender(context)

    record_run(context, Tables.def14a_llm, len(cik_map), total_new, is_full_rescan=is_full_rescan)
   
