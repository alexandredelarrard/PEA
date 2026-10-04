"""Extract structured governance data from SEC DEF 14A proxies with an LLM (`Def14AExtract` schema).

Per ticker: list its DEF 14A filings over the manifest window (one listing per CIK window of its scope), carve
the relevant sections, send only accessions without stored evidence to the LLM, and upsert that
ticker's rows into `def14a_llm` plus four child tables (`def14a_executive_comp`, `def14a_director_comp`,
`def14a_ownership`, `def14a_directors`) before the next ticker. A cross-ticker gender consensus runs
once after the loop. Skips with a warning when no OpenAI key is configured.
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
from src.data_extract.utils.common.identity import FilingScope, load_identity
from src.data_extract.utils.common.registrant import header_subject_ciks
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run, scope_changed_tickers
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

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmTask
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


def _fetch_filing_html(context: Context, filing: pd.Series) -> str:
    """The filing's raw markup, falling back to the `<accession>.txt` full submission when the primary
    document cannot be fetched; re-raises when there is no distinct `.txt` URL."""
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


def _list_scope_windows(
    context: Context,
    scope: FilingScope,
    company: str,
    years: int,
    since: pd.Timestamp | None,
) -> pd.DataFrame:
    """That ticker's DEF 14A filings, listed by CIK for each window of its filing scope.

    A dated SPLIT (`DEF14A_FORMS` is SPLIT in `registrant.FORM_POLICY`): each CIK contributes only
    filings its seam-widened window admits, so a company-year never blends two registrants' boards.
    Each row's `cik` is the CIK that filed it.
    """
    frames = []
    for window in scope.windows:
        part = list_filings(context, window.cik, DEF14A_FORMS, years, company, since=since)
        if part is None or part.empty:
            continue
        part = part[pd.to_datetime(part["filing_date"]).map(window.admits)]
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["ticker", "cik", "accession_number", "filing_date"])
    if len(frames) > 1:
        context.log.info("%s: DEF 14A listed across %s", scope.ticker, ", ".join(f"{len(part)} from {part['cik'].iloc[0]}" for part in frames))
    return pd.concat(frames, ignore_index=True).drop_duplicates("accession_number", keep="first")


def accepted_subjects(scope: FilingScope) -> frozenset[str]:
    """Subject CIKs a multi-window ticker's proxy may name (its event CIKs); empty, i.e. no header check, otherwise."""
    return frozenset(scope.event_ciks) if len(scope.windows) > 1 else frozenset()


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
    """Accessions whose parent row carries real extracted evidence, not merely a PK.

    Evidence-free parents stay in the table but do not count as completed, so a later run re-extracts them.
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

    Lists each ticker's proxies per CIK window of its filing scope over the manifest window (the whole
    window for a ticker whose lineage scope changed since the table's last run) and sends
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
    identity = load_identity(context)
    changed = scope_changed_tickers(get_entry(context, Tables.def14a_llm), {t: identity.filing_scope(t).scope_changed_at for t in requested})
    if not full and not changed and _is_up_to_date(context, requested):
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
    if changed:
        context.log.info("DEF 14A LLM: %d ticker lineage scope(s) changed -> full-window relist: %s", len(changed), ", ".join(sorted(changed)))

    total_new, total_semantic_empty = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker, company = str(r["ticker"]), str(r.get("name", ""))
        scope = identity.filing_scope(ticker)
        try:
            filings = _list_scope_windows(context, scope, company, years, None if ticker in changed else list_since)
        except Exception as e:
            context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
            continue
        tasks = _ticker_tasks(context, ticker, filings, seen, accepted_subjects(scope))
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
