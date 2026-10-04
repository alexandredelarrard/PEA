"""Extract structured governance data from SEC DEF 14A proxies with an LLM (`Def14AExtract` schema).

Per ticker: list its DEF 14A filings over `years_history` (across its registrant chain), carve the
relevant sections, send only accessions with no saved `def14a_llm` row to the LLM,
and upsert that ticker's rows into `def14a_llm` plus four child tables (`def14a_executive_comp`,
`def14a_director_comp`, `def14a_ownership`, `def14a_directors`) before the next ticker. An answer
without evidence from a proxy that was read is stored as one `def14a_llm` marker; a failed read or
LLM call writes nothing and is listed again next run. `full` re-sends the saved rows without
evidence. A cross-ticker gender consensus runs once
after the loop. Skips with a warning when no OpenAI key is configured.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import cast

import pandas as pd
from edgar import Filing
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants.constants import DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.empty_markers import drop_markers_over_data
from src.data_extract.utils.common.registrant import (
    Registrant,
    header_subject_ciks,
    issuer_ciks,
    load_registrants,
)
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get
from src.data_extract.utils.common.sec_utils import load_cik_mapping
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
from src.data_store.schema import Table, Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


def _fetch_filing_html(context: Context, filing: pd.Series) -> str:
    """The filing's raw markup, falling back to the `<accession>.txt` full submission when the primary
    document is not served; re-raises a transient failure, and when there is no distinct `.txt` URL."""
    try:
        return sec_get(context, filing["doc_url"]).text
    except TransientReadError:
        raise
    except Exception:
        txt_url = filing.get("txt_url")
        if not txt_url or txt_url == filing["doc_url"]:
            raise
        logger.info("%s: primary document unavailable, falling back to the full submission", filing.get("accession_number", ""))
        return sec_get(context, txt_url).text


def _payload_for(context: Context, ticker: str, filing: pd.Series) -> str | None:
    """The carved `=== LABEL ===` text one filing contributes, or None if it cannot be read.

    Runs on the main thread before any task is queued: a worker receives text and a schema, never a `Context`.
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
    """Reject only a known subject that is outside the accepted registrant entity. A header SEC could
    not serve skips the filing this run (it is not queued, so a later run reads it again)."""
    accession = str(filing["accession_number"])
    filer_cik = pad_cik(filing["cik"])
    try:
        context.ensure_edgar_identity()
        subjects = _filing_subject_ciks(filing)
    except TransientReadError as exc:
        context.log.warning("%s: DEF 14A accession %s subject header unreadable (%s); skipped this run", ticker, accession, exc)
        return False
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
    """That ticker's DEF 14A filings by CIK, walking every segment of its registrant chain when it has one.

    This is the only EDGAR fetcher that resolves by CIK (`sp500_tickers.cik`), not `Company(ticker)`.
    The chain is a dated SPLIT, never a union (`DEF14A_FORMS` is SPLIT in `registrant.FORM_POLICY`): each
    segment contributes only filings inside `[valid_from, valid_to)`, so segments are disjoint and a
    company-year never blends two registrants' boards. Each row's `cik` is the CIK that actually filed it.
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
        # The dated split makes this impossible, so a duplicate means the register is wrong; do not dedupe it away.
        context.log.warning("%s: %d duplicate accession(s) across the %s chain", ticker, dupes, " -> ".join(entry.all_ciks()))
    return out


def _completed_accessions(context: Context) -> set[str]:
    """Accessions whose parent row carries real extracted evidence, not merely a PK.

    Evidence-free parents stay in the table but do not count as completed, so a `full` run re-extracts them.
    A table created by a marker-only save may lack evidence columns; only the stored ones are read.
    """
    stored_columns = set(context.store.columns(Tables.def14a_llm))
    if not stored_columns:
        return set()
    stored = context.store.load(
        Tables.def14a_llm,
        columns=["accession_number", *(c for c in _DEF14A_EVIDENCE_COLUMNS if c in stored_columns)],
        optional=True,
    )
    if stored is None or stored.empty:
        return set()
    return {
        str(row["accession_number"])
        for row in stored.to_dict(orient="records")
        if row.get("accession_number") is not None and _has_parent_evidence(row)
    }


def _stored_accessions(context: Context) -> set[str]:
    """Every accession with a saved `def14a_llm` row: evidence-backed, evidence-free or a marker."""
    return {str(a) for a in context.store.distinct(Tables.def14a_llm, "accession_number")}


def _frames_off_data(context: Context, result: LlmResult) -> dict[Table, pd.DataFrame]:
    """`_result_frames` for one answer, minus any marker whose key already holds a real parent row."""
    return {table: drop_markers_over_data(context.store, table, df, log=context.log) for table, df in _result_frames(result).items()}


def _finalise_gender(context: Context) -> None:
    """Cross-ticker gender consensus over `def14a_directors`, run once after the per-ticker loop.

    A director recurs across companies, so the consensus needs every ticker's rows. Rewrites
    `def14a_directors` gender columns and the parent's gender ratios only when something changed.
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

    # The filing's stated `n_women_directors` still wins; this ratio only improves the fallback.
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
    accessions whose answer carries domain evidence plus the count of evidence-free answers, each
    saved as an empty-filing marker unless its key already holds a parent row."""
    results = extractor.run_extraction(tasks, flatten=partial(_frames_off_data, context), group_key=lambda t: str(t.meta["ticker"]))
    evidenced = [r for r in results if r.ok and isinstance(r.parsed, Def14AExtract) and _has_extract_evidence(r.parsed)]
    semantic_empty = sum(1 for r in results if r.ok) - len(evidenced)
    if semantic_empty:
        context.log.info("%s: %d DEF 14A result(s) contained no domain evidence; marked as read", ticker, semantic_empty)
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

    Lists each ticker's proxies across its registrant chain over `years_history` and sends only
    accessions with no saved row to the LLM; each ticker's rows are upserted before the next starts. Skips when no OpenAI key is configured.

    `model` / `max_chars` / `cache` default to `config.gpt` and `workers` (concurrent LLM calls)
    to `config.gpt.threads`; an explicit keyword pins one without touching config. `full` also
    re-sends the saved accessions without evidence (markers included); a proxy with stored evidence
    is never re-sent.
    """
    config = with_gpt_overrides(config, "def14a", model=model, max_chars=max_chars, cache=cache)
    de = context.config.data_extract
    cik_map = load_cik_mapping(context, tickers)
    try:
        extractor = LLMExtractor(context, config, action="def14a", threads=workers)
    except OSError as e:
        context.log.warning("DEF 14A LLM extraction skipped: %s", e)
        return

    # A saved row is done; `full` re-sends every accession without evidence (legacy evidence-free rows and markers).
    seen = _completed_accessions(context) if full else _stored_accessions(context)
    years = int(de.years_history)
    # The curated registrant register (a dated SPLIT chain per ticker); `{}` when the file is absent.
    cutovers = load_registrants(str(context.config_dir))
    if cutovers:
        context.log.info("DEF 14A: %d registrant cutover(s) in force: %s", len(cutovers), ", ".join(sorted(cutovers)))

    total_new, total_semantic_empty = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, cik, company = str(r["ticker"]), str(r["cik"]), str(r.get("name", ""))
        try:
            filings = _list_across_registrants(context, ticker, cik, company, years, None, cutovers)
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
    context.log.info("DEF 14A: %d filing(s) extracted, %d evidence-free answer(s) marked as read", total_new, total_semantic_empty)
