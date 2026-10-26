"""Daily five-domain DEF 14A extraction with free replay and one durable section retry."""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd
from edgar import Filing
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.identity import FilingScope, load_identity
from src.data_extract.utils.common.registrant import header_subject_ciks
from src.data_extract.utils.common.resume import document_floor
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads, prepare_def14a_sections
from src.data_extract.utils.structure.def14a.gender import (
    basis_distribution,
    consensus,
    log_consensus,
    recompute_parent_gender,
)
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive, source_for_answer
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_extract.utils.structure.def14a.work import (
    FilingWork,
    load_archives,
    load_work,
    matched_batches,
    replay_candidates,
    requests_for,
    reserve,
    save_archive,
    save_section,
)
from src.data_store.schema import Resume, Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
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


def _list_scope_windows(
    context: Context,
    scope: FilingScope,
    company: str,
    years: int,
    since: pd.Timestamp | None,
) -> pd.DataFrame:
    """That ticker's DEF 14A filings, listed by CIK for each window of its filing scope.

    A dated SPLIT (the table's registered proxy forms use SPLIT in `registrant.FORM_POLICY`): each CIK contributes only
    filings its seam-widened window admits, so a company-year never blends two registrants' boards.
    Each row's `cik` is the CIK that filed it.
    """
    frames = []
    for window in scope.windows:
        part = list_filings(context, window.cik, cast(Resume, Tables.def14a_llm.resume).forms, years, company, since=since)
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


def _accepted_filings(context: Context, ticker: str, filings: pd.DataFrame, subjects: frozenset[str]) -> pd.DataFrame:
    """Apply the existing multi-entity subject guard before any counter or fact write."""
    if not subjects or filings.empty:
        return filings
    accepted = [index for index, filing in filings.iterrows() if _subject_is_accepted(context, ticker, filing, subjects)]
    return filings.loc[accepted]


def _read_proxy(context: Context, ticker: str, filing: pd.Series) -> str | None:
    """Read once; unavailable documents do not consume a domain attempt."""
    try:
        return _fetch_filing_html(context, filing)
    except (NameError, TypeError, KeyError, AttributeError, AssertionError):
        raise
    except Exception as exc:
        context.log.warning("%s %s: DEF 14A source unavailable (%s)", ticker, filing["accession_number"], exc)
        return None


def _section_extractors(context: Context, config: DictConfig, overrides: dict[str, Any], workers: int | None) -> dict[str, LLMExtractor]:
    """Bind each existing service to its own action, prompt, schema and unchanged worker setting."""
    return {
        section.key: LLMExtractor(context, with_gpt_overrides(config, section.action, **overrides), action=section.action, threads=workers)
        for section in SECTIONS
    }


def _listed_proxies(context: Context, ticker: str, scope: FilingScope, company: str, years: int, floor: pd.Timestamp) -> pd.DataFrame | None:
    """Preserve source-failure handling and registrant guards over the full rolling window."""
    try:
        filings = _list_scope_windows(context, scope, company, years, floor)
    except (NameError, TypeError, KeyError, AttributeError, AssertionError):
        raise
    except Exception as exc:
        context.log.warning("%s: DEF 14A filing list failed (%s)", ticker, exc)
        return None
    return _accepted_filings(context, ticker, filings, accepted_subjects(scope))


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
    """Extract new sections and repair source-present gaps once; full never resets attempts."""
    # Read the ticker map and preserve each registrant's filing windows.
    cik_map = load_cik_mapping(context, tickers)
    identity = load_identity(context)
    years = int(config.data_extract.years_history)
    floor = document_floor(Tables.def14a_llm, pd.Timestamp.today(), years)
    try:
        extractors = _section_extractors(context, config, {"model": model, "max_chars": max_chars, "cache": cache}, workers)
    except OSError as exc:
        context.log.warning("DEF 14A LLM extraction skipped: %s", exc)
        return
    updated = 0

    for _, company in tqdm(cik_map.iterrows(), total=len(cik_map), desc="DEF 14A LLM"):
        ticker = str(company["ticker"])
        scope = identity.filing_scope(ticker)
        filings = _listed_proxies(context, ticker, scope, str(company.get("name", "")), years, floor)
        if filings is None:
            continue
        works = load_work(context, ticker, filings, floor)
        load_archives(context, works)

        # Reuse paid answers before deciding whether another request is necessary.
        for work, section, answer in replay_candidates(works):
            reserve(context, work, section, work.attempts[section.key] or 1)
            processed = postprocess_section(section, answer, work.stamp, work.saved, work.parent, source_for_answer(work.raw, section, answer))
            save_section(context, work, processed)
            work.raw = refresh_archive(work.raw, section, processed.legacy, processed.corrections)
            save_archive(context, work)
            updated += 1

        # Deduce only unread sections or source-supported missing keys within the rolling window.
        requests = []
        for work in works:
            if not any(key in work.pending and attempt < 2 for key, attempt in work.attempts.items()):
                continue
            html = _read_proxy(context, ticker, work.filing)
            if html is None:
                continue
            payloads = prepare_def14a_payloads(html, html_to_text(html))
            requests.extend(requests_for(work, payloads))
            empty_sections = [section for section in SECTIONS if not work.attempts[section.key] and not payloads.get(section.key, "").strip()]
            for section in empty_sections:
                reserve(context, work, section, 1)

        # Reserve the owned attempt, call its matching prompt/schema, and get the answer.
        for section, tasks in matched_batches(requests, workers or int(config.gpt.threads)):
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
                work.raw = append_answer(work.raw, section, result.parsed, result.task.payload)
                save_archive(context, work)

                # Postprocess source units and retained facts, then save this section only.
                processed = postprocess_section(section, result.parsed, work.stamp, work.saved, work.parent, result.task.payload)
                save_section(context, work, processed)
                work.raw = refresh_archive(work.raw, section, processed.legacy, processed.corrections)
                save_archive(context, work)
                updated += 1

    # Preserve the existing cross-ticker gender reconciliation after saved updates.
    if updated:
        _finalise_gender(context)
    context.log.info("DEF 14A: %d section answer(s) replayed or extracted; retry=2 sections excluded", updated)
