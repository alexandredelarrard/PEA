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
If the key is absent the function logs a warning and returns.

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
from typing import cast

import pandas as pd
from edgar import Filing
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.registrant import (
    Registrant,
    header_subject_ciks,
    issuer_ciks,
    load_registrants,
)
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping, sec_get
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_sections
from src.data_extract.utils.structure.def14a.flatten import (
    _DEF14A_EVIDENCE_COLUMNS,
    _has_extract_evidence,
    _has_parent_evidence,
    _result_frames,
)
from src.data_extract.utils.structure.def14a.gender import (
    basis_distribution,
    consensus,
    log_consensus,
    recompute_parent_gender,
)
from src.data_store.schema import Tables

# `gpt_extract` is a shared service, like `src/utils/` -- the sanctioned cross-import. It
# owns the model, the keys, the prompts (`prompt_templates/def14a_*.md`) and the thread
# pool; the anchor carve and the flatten are this package's business.
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmTask
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


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
        logger.info("%s: primary document unavailable, falling back to the full submission", filing.get("accession_number", ""))
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
    except Exception as e:  # noqa: BLE001 -- one filing, not the run
        logger.warning("%s %s: DEF 14A filing could not be read (%s)", ticker, filing.get("filing_date", ""), e)
        return None


def _filing_subject_ciks(filing: pd.Series) -> frozenset[str]:
    """Read a listed filing's SGML subject CIKs through the shared header reader."""
    candidate = Filing(
        cik=int(str(filing["cik"])),
        company=str(filing.get("company_name", "")),
        form=str(filing["form"]),
        filing_date=pd.Timestamp(filing["filing_date"]).date().isoformat(),
        accession_no=str(filing["accession_number"]),
    )
    return header_subject_ciks(candidate)


def _subject_is_accepted(
    context: Context,
    ticker: str,
    filing: pd.Series,
    accepted_subject_ciks: frozenset[str],
) -> bool:
    """Reject only a known subject that is outside the accepted registrant entity."""
    accession = str(filing["accession_number"])
    filer_cik = pad_cik(filing["cik"])
    try:
        context.ensure_edgar_identity()
        subjects = _filing_subject_ciks(filing)
    except Exception as exc:  # noqa: BLE001 -- an unknown header follows the existing path
        context.log.info(
            "%s: DEF 14A accession %s subject header unavailable (%s); continuing",
            ticker,
            accession,
            exc,
        )
        return True
    if not subjects or not subjects.isdisjoint(accepted_subject_ciks):
        return True
    context.log.warning(
        "%s: rejecting DEF 14A accession %s before extraction; filer CIK %s; "
        "subject CIK(s) %s; accepted entity CIK(s) %s; reason=subject_cik_disjoint",
        ticker,
        accession,
        filer_cik,
        ",".join(sorted(subjects)),
        ",".join(sorted(accepted_subject_ciks)),
    )
    return False


def _list_across_registrants(
    context: Context,
    ticker: str,
    cik: str,
    company: str,
    years: int,
    since: pd.Timestamp | None,
    cutovers: dict[str, Registrant],
) -> pd.DataFrame:
    """That ticker's DEF 14A filings, across a registrant boundary when it has one.

    ⚠ THIS MODULE IS THE ONLY PIPELINE IN THE REPO THAT RESOLVES BY CIK. Every other EDGAR
    fetcher goes through `registrant.resolve_registrant_filings`, i.e. `Company(ticker)`. That makes
    `sp500_tickers.cik` a single-pipeline dependency -- and it is why a wrong or superseded CIK
    shows up as a governance-only hole while prices, fundamentals and 8-K stay clean.

    XOM is the measured case (2026-09-09). EDGAR remapped the XOM ticker to ExxonMobil Holdings
    Corp (CIK 2115436), whose first filing is an 8-K12B on 2026-07-01 and which holds no proxy
    forms at all, so this loop listed ZERO proxies and XOM carried 0 rows in all five
    `def14a_*` tables while its 4,092 `cube_part_governance` rows held no non-null governance
    feature. The full proxy history is under the predecessor, CIK 34088.

    ⚠ THE SPLIT IS DATED, NEVER A UNION OF CIKS, and `DEF14A_FORMS` is declared SPLIT in
    `registrant.FORM_POLICY` for it. Two legal entities can file concurrently, and
    concatenating both CIKs blends a subsidiary's disclosures into the parent's -- on this
    table that means two boards and two pay tables for one company-year, which corrupts the
    governance grain rather than merely duplicating a row. Each segment contributes only
    filings inside `[valid_from, valid_to)`, so the sets are disjoint by construction.

    N SEGMENTS, NOT TWO: the register is a chain, and a two-CIK loop gave PSKY one hop.

    ⚠ THE PER-FILING CIK COMES OUT RIGHT FOR FREE, and that is the point of listing per
    registrant rather than post-labelling. `list_filings` stamps each row with the CIK whose
    submissions document it parsed, so a row's `cik` is the CIK that actually filed it. Stamping
    the roster CIK onto filings resolved by TICKER is how 521 XOM 8-K rows came to carry a CIK
    holding 29 filings.
    """
    entry = cutovers.get(ticker)
    if entry is None:
        return list_filings(context, cik, DEF14A_FORMS, years, company, since=since)

    frames = []
    for segment in entry.segments:
        part = list_filings(context, segment.cik, DEF14A_FORMS, years, company, since=since)
        if part is None or part.empty:
            continue
        filed = pd.to_datetime(part["filing_date"])
        part = part[filed.map(segment.covers)]
        if not part.empty:
            frames.append(part)
            context.log.info(
                "%s: %d DEF 14A filing(s) from CIK %s (%s .. %s)",
                ticker,
                len(part),
                segment.cik,
                segment.valid_from.date() if segment.valid_from else "start",
                segment.valid_to.date() if segment.valid_to else "now",
            )
    if not frames:
        return pd.DataFrame(columns=["ticker", "cik", "accession_number", "filing_date"])
    out = pd.concat(frames, ignore_index=True)
    dupes = int(out["accession_number"].duplicated().sum())
    if dupes:
        # The dated split makes this impossible; if it fires, the register is wrong rather
        # than the data, and silently deduping would hide that.
        context.log.warning("%s: %d duplicate accession(s) across the %s chain", ticker, dupes, " -> ".join(entry.all_ciks()))
    return out


def _is_up_to_date(context: Context, requested_tickers: list[str]) -> bool:
    """Up to date only when the manifest entry was refreshed today AND every requested ticker
    already has rows in `def14a_llm` (per-ticker coverage, so a same-day rerun still picks up a
    missing name)."""
    if not context.store.exists(Tables.def14a_llm):
        return False
    entry = get_entry(context, Tables.def14a_llm)
    if entry is None or entry.get("last_run_date") != pd.Timestamp.today().strftime(DATE_FORMAT):
        return False
    have = set(context.store.distinct(Tables.def14a_llm, "ticker", where={"ticker": list(requested_tickers)}))
    return set(requested_tickers).issubset(have)


def _completed_accessions(context: Context) -> set[str]:
    """Accessions whose parent contains real extracted evidence, not merely a PK.

    The projection deliberately excludes the JSON blob and metadata-only columns. Historical
    empty parents stay in the table for auditability, but no longer suppress a later repair.
    """
    if not context.store.exists(Tables.def14a_llm):
        return set()
    stored = context.store.load(
        Tables.def14a_llm,
        columns=["accession_number", *_DEF14A_EVIDENCE_COLUMNS],
        optional=True,
    )
    if stored is None or stored.empty:
        return set()
    return {
        str(row["accession_number"])
        for row in stored.to_dict(orient="records")
        if row.get("accession_number") is not None and _has_parent_evidence(row)
    }


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
        Tables.def14a_directors, columns=["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"], optional=True
    )
    if directors is None or directors.empty:
        return

    before = basis_distribution(directors)
    resolved, stats = consensus(directors)
    after = basis_distribution(resolved)
    log_consensus(context.log, stats, before, after)

    if not (stats["filled"] or stats["overturned"]):
        return

    context.store.save(
        Tables.def14a_directors,
        resolved[["ticker", "accession_number", "name", "as_of", "gender", "gender_basis"]],
        pk=["ticker", "accession_number", "name"],
    )

    # `pct_female_directors` keeps its existing precedence -- the filing's own
    # `n_women_directors` first, this ratio as the FALLBACK -- so a consensus correction makes
    # the fallback better rather than overriding a stated count.
    parent = recompute_parent_gender(resolved)
    if not parent.empty:
        context.store.save(Tables.def14a_llm, parent, pk=["ticker", "accession_number"])
        context.log.info("gender consensus: refreshed pct_female_directors / pct_gender_stated on %d filings", len(parent))


def _ticker_tasks(context: Context, ticker: str, filings: pd.DataFrame, seen: set[str], accepted_subjects: frozenset[str]) -> list[LlmTask]:
    """One LLM task per listed proxy not in `seen` (nor listed twice), whose subject is accepted and
    whose document carves to a payload.

    Fetching and carving run on this thread, so the pool only ever receives text and a schema;
    a filing that cannot be read never becomes a task.
    """
    tasks: list[LlmTask] = []
    queued: set[str] = set()
    for _, filing in filings.iterrows():
        accession = str(filing["accession_number"])
        if accession in seen or accession in queued:
            continue
        if accepted_subjects and not _subject_is_accepted(context, ticker, filing, accepted_subjects):
            continue
        queued.add(accession)
        payload = _payload_for(context, ticker, filing)
        if payload:
            tasks.append(
                LlmTask(seq=len(tasks), payload=payload, schema=Def14AExtract, table=Tables.def14a_llm, meta={"ticker": ticker, "filing": filing})
            )
    return tasks


def _extract_ticker(context: Context, extractor: LLMExtractor, ticker: str, tasks: list[LlmTask]) -> tuple[list[str], int]:
    """Run one ticker's tasks (the extractor saves that ticker's five frames once) and return the
    accessions whose answer carries domain evidence plus the count of evidence-free answers, which
    save no completion row and stay retryable."""
    results = extractor.run_extraction(tasks, flatten=_result_frames, group_key=lambda t: str(t.meta["ticker"]))
    evidenced = [r for r in results if r.ok and isinstance(r.parsed, Def14AExtract) and _has_extract_evidence(r.parsed)]
    semantic_empty = sum(1 for r in results if r.ok) - len(evidenced)
    if semantic_empty:
        context.log.warning(
            "%s: %d DEF 14A result(s) contained no domain evidence; no completion row was saved and they remain retryable", ticker, semantic_empty
        )
    return [str(cast(pd.Series, r.task.meta["filing"])["accession_number"]) for r in evidenced], semantic_empty


def fetch_def14a_llm(
    context: Context,
    config: DictConfig,
    tickers: list[str],
    model: str | None = None,
    max_chars: int | None = None,
    cache: bool | None = None,
    workers: int | None = None,
    full: bool = False,
) -> None:
    """Build/refresh the DEF 14A LLM governance extract, one ticker at a time.

    Lists each ticker's proxies across its registrant chain over the manifest window and sends
    only accessions without stored evidence to the LLM; each ticker's rows are upserted before
    the next starts. Skips when no OpenAI key is configured.

    `model` / `max_chars` / `cache` default to `config.gpt` and `workers` (concurrent LLM calls)
    to `config.gpt.threads`; an explicit keyword pins one without touching config. `full`
    bypasses the run-wide up-to-date gate and lists the whole `years_history` window.
    """
    config = with_gpt_overrides(config, "def14a", model=model, max_chars=max_chars, cache=cache)
    de = context.config.data_extract
    cik_map = load_cik_mapping(context, tickers)
    requested = cik_map["ticker"].astype(str).tolist()
    if not full and _is_up_to_date(context, requested):
        context.log.info("DEF 14A LLM already up to date — every requested ticker present — skipping")
        return
    try:
        extractor = LLMExtractor(context, config, action="def14a", threads=workers)
    except OSError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        return

    # Accessions with real extracted evidence are never re-sent; an evidence-free parent is absent so a later listing repairs it.
    seen = _completed_accessions(context)
    years = int(de.years_history)
    since, is_full_rescan = manifest_window(
        context,
        Tables.def14a_llm,
        requested,
        fallback_since=pd.Timestamp.today() - pd.DateOffset(years=years),
        full_rescan_days=int(getattr(de, "manifest_full_rescan_days", 30)),
    )
    # `list_filings` keeps filings STRICTLY AFTER its `since`, so the inclusive cutoff steps back one day; None lists all `years`.
    list_since = None if (full or is_full_rescan) else since - pd.Timedelta(days=1)
    # The curated registrant register (a dated SPLIT chain per ticker); `{}` when the file is absent.
    cutovers = load_registrants(str(context.config_dir))
    if cutovers:
        context.log.info("DEF 14A: %d registrant cutover(s) in force: %s", len(cutovers), ", ".join(sorted(cutovers)))

    total_new, total_semantic_empty = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = str(r["ticker"]), str(r["cik"]), str(r.get("company_name", ""))
        try:
            filings = _list_across_registrants(context, ticker, cik, company, years, list_since, cutovers)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue
        accepted_subjects = issuer_ciks(ticker, cik, cutovers) if ticker in cutovers else frozenset()
        tasks = _ticker_tasks(context, ticker, filings, seen, accepted_subjects)
        extracted, semantic_empty = _extract_ticker(context, extractor, ticker, tasks)
        seen.update(extracted)
        total_new += len(extracted)
        total_semantic_empty += semantic_empty
        if extracted:
            context.log.info("%s: +%d new DEF 14A filing(s) extracted", ticker, len(extracted))

    # The gender consensus groups directors across tickers, so it runs once after the loop.
    if total_new:
        _finalise_gender(context)
    if total_semantic_empty:
        context.log.warning("DEF 14A: %d semantic-empty result(s) left retryable", total_semantic_empty)
    record_run(context, Tables.def14a_llm, len(cik_map), total_new, is_full_rescan=is_full_rescan, tickers=requested)
