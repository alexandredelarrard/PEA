"""Scoped reads, the per-ticker section calls and durable saves for the five proxy domains."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.context import Context
from src.data_extract.utils.structure.def14a.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.postprocess import ProcessedSection, append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section, reservation_frame
from src.data_store.schema import Table, Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmTask
from src.utils.data_policy import source_read_window, window_policy
from src.utils.window import Window


@dataclass
class FilingWork:
    """One walked proxy; counters belong to the saved domain, not this working copy. `payloads` holds the stored
    uncut `sec_def14a_text` payload per section (a section the proxy lacks is absent: that domain is empty)."""

    stamp: dict[str, Any]
    saved: dict[Table, pd.DataFrame]
    parent: dict[str, Any]
    attempts: dict[str, int]
    raw: str | None = None
    pending: set[str] = field(default_factory=set)
    payloads: dict[str, str] = field(default_factory=dict)
    cached: dict[str, BaseModel] = field(default_factory=dict)


def load_work(context: Context, ticker: str, filings: pd.DataFrame) -> list[FilingWork]:
    """Read each scoped fact table once, without the large paid JSON archive."""
    # load: each section table's rows of the ticker in the window, markers included
    if filings.empty:
        return []
    dates = Window(filings.filing_date.min(), filings.filing_date.max())
    frames: dict[Table, pd.DataFrame] = {}
    policies = {section.table: window_policy(context.config, section.table, context.run.as_of) for section in SECTIONS}
    for section in SECTIONS:
        table = section.table
        available = context.store.columns(table)
        wanted = (
            tuple(dict.fromkeys((*table.pk, "as_of", "period", "retry", *(column for item in SECTIONS for column in item.parent_columns))))
            if table is Tables.def14a_llm
            else available
        )
        columns = [column for column in wanted if column in available and column != "def14a_json"]
        source = source_read_window(context.config, table, context.run.as_of)
        window = dates.intersection(source) if source is not None else None
        loaded = (
            context.store.load(table, columns=columns, where={"ticker": ticker}, since=window.start, until=window.end, markers=True, optional=True)
            if window is not None
            else None
        )
        frames[table] = loaded if loaded is not None else pd.DataFrame(columns=columns)

    # Keep the walked proxy's entity and filing date separate from its fiscal-year rows.
    works: list[FilingWork] = []
    for _, filing in filings.drop_duplicates("accession_number").iterrows():
        accession = str(filing["accession_number"])
        saved = {table: frame.loc[frame.accession_number.eq(accession)].copy() for table, frame in frames.items()}
        parent_rows = saved[Tables.def14a_llm]
        parent = cast(dict[str, Any], parent_rows.iloc[0].to_dict()) if not parent_rows.empty else {}
        attempts = {
            section.key: int(pd.to_numeric(saved[section.table].retry, errors="coerce").max())
            if not saved[section.table].empty and saved[section.table].retry.notna().any()
            else 0
            for section in SECTIONS
        }
        stamp = {
            "ticker": ticker,
            "accession_number": accession,
            "as_of": pd.Timestamp(filing["filing_date"]),
            "cik": filing.get("cik"),
            "company_name": filing.get("company_name"),
        }
        work = FilingWork(stamp, saved, parent, attempts)
        work.pending = {
            section.key
            for section in SECTIONS
            if (policy := policies[section.table]) is not None
            and policy.requested.start <= stamp["as_of"] <= policy.requested.end
            and (saved[section.table].empty or 0 < attempts[section.key] < 2 or section.table is Tables.def14a_llm and not attempts[section.key])
        }
        works.append(work)
    return works


def _archived_answers(raw: str) -> dict[str, BaseModel]:
    """Existing schema-valid answers, including empty answers, prove inference completion."""
    if raw == "_empty":
        return {section.key: Def14AExtract.model_validate({}) for section in SECTIONS}
    try:
        document = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(document, dict):
        return {}
    metadata = document.get("_def14a_split")
    legacy = metadata.get("legacy_json") if isinstance(metadata, dict) else raw
    if isinstance(legacy, str):
        try:
            answer = Def14AExtract.model_validate_json("{}" if legacy == "_empty" else legacy)
        except ValidationError:
            pass
        else:
            return {section.key: answer for section in SECTIONS}
    responses = metadata.get("responses", {}) if isinstance(metadata, dict) else {}
    completed = {}
    for section in SECTIONS:
        recorded = responses.get(section.key, []) if isinstance(responses, dict) else []
        if not recorded:
            continue
        try:
            answer = section.schema.model_validate(recorded[-1].get("response"))
        except (ValidationError, AttributeError):
            continue
        completed[section.key] = answer
    return completed


def load_archives(context: Context, works: list[FilingWork]) -> None:
    """Resolve known incomplete reservations using archived response evidence, never economic NULL fields."""
    # load: the archives of the proxies with a gap
    accessions = [work.stamp["accession_number"] for work in works if work.pending]
    saved = (
        context.store.load(
            Tables.def14a_llm,
            columns=["ticker", "accession_number", "def14a_json"],
            where={"ticker": works[0].stamp["ticker"], "accession_number": accessions},
            markers=True,
            optional=True,
        )
        if accessions
        else None
    )
    archives = dict(zip(saved.accession_number, saved.def14a_json, strict=True)) if saved is not None else {}

    # Failed reservations have no successful response; valid NULL/empty answers remain complete.
    for work in works:
        raw = archives.get(work.stamp["accession_number"])
        work.raw = raw if isinstance(raw, str) and raw else None
        if not work.raw:
            continue
        completed = _archived_answers(work.raw)
        work.cached = {
            section.key: completed[section.key]
            for section in SECTIONS
            if section.key in work.pending and section.key in completed and work.saved[section.table].empty
        }
        work.pending.difference_update(completed)


def open_sections(work: FilingWork) -> list[Section]:
    """Missing or incomplete sections below the two-attempt limit."""
    return [section for section in SECTIONS if section.key in work.pending and work.attempts[section.key] < 2]


def ticker_works(context: Context, ticker: str, df_filings: pd.DataFrame) -> list[FilingWork]:
    """The ticker's walked proxies with their saved facts and archives, and the uncut payloads of those with an open
    section."""
    # load: the saved facts and archives, then the payloads of the proxies that need them
    works = load_work(context, ticker, df_filings)
    load_archives(context, works)
    wanted = [work.stamp["accession_number"] for work in works if open_sections(work) or work.cached]
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


def materialize_cached(context: Context, works: list[FilingWork]) -> int:
    """Reuse paid answers only for structurally missing child rows or completion markers."""
    repaired = 0
    for work in works:
        for section in (item for item in SECTIONS if item.key in work.cached):
            save_answer(context, work, section, work.cached[section.key])
            repaired += 1
    return repaired


def requests_for(work: FilingWork, budget: int) -> list[tuple[Section, LlmTask]]:
    """One task per open section whose stored payload is not empty; the prompt gets its first `budget` characters."""
    return [
        (section, LlmTask(seq=0, payload=work.payloads[section.key][:budget], schema=section.schema, table=section.table, meta={"work": work}))
        for section in open_sections(work)
        if work.payloads.get(section.key, "").strip()
    ]


def section_requests(context: Context, works: list[FilingWork], budget: int) -> list[tuple[Section, LlmTask]]:
    """The prompt task of every open section with a payload, cut to `budget`; an open section the proxy lacks consumes
    its attempt."""
    requests = [request for work in works for request in requests_for(work, budget)]
    absent = [(work, section) for work in works for section in open_sections(work) if not work.payloads.get(section.key, "").strip()]
    for work, section in absent:
        if not work.attempts[section.key]:
            reserve(context, work, section, 1)
        work.pending.discard(section.key)
    return requests


def matched_batches(requests: list[tuple[Section, LlmTask]], size: int) -> Iterator[tuple[Section, list[LlmTask]]]:
    """Bound in-flight answers using the existing worker count, without capping the daily worklist."""
    for section in SECTIONS:
        tasks = [task for owner, task in requests if owner == section]
        for start in range(0, len(tasks), max(1, size)):
            batch = tasks[start : start + max(1, size)]
            yield section, [replace(task, seq=seq) for seq, task in enumerate(batch)]


def call_sections(
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
            save_answer(context, work, section, result.parsed)
            answered += 1
    return answered


def save_answer(context: Context, work: FilingWork, section: Section, answer: BaseModel) -> None:
    """Materialize one validated response using the existing domain and archive writes."""
    processed = postprocess_section(section, answer, work.stamp, work.saved, work.parent, work.payloads.get(section.key, ""))
    save_section(context, work, processed)
    work.pending.discard(section.key)
    work.raw = refresh_archive(work.raw, section, processed.legacy, processed.corrections)
    save_archive(context, work)


def reserve(context: Context, work: FilingWork, section: Section, attempt: int) -> None:
    """Write the owned counter before submission, retaining partial facts and monotonic attempts."""
    _require_writes(context, work, (section.table, Tables.def14a_llm))
    frame = reservation_frame(section.table, work.stamp, work.saved[section.table], attempt)
    context.store.save(section.table, frame)
    work.attempts[section.key] = int(frame.retry.max())
    if section.table is Tables.def14a_llm:
        work.parent["retry"] = int(frame.retry.max())
    saved = work.saved[section.table]
    work.saved[section.table] = saved.assign(retry=int(frame.retry.max())) if not saved.empty else frame


def save_archive(context: Context, work: FilingWork) -> None:
    """Save raw paid output independently of any later postprocessing failure."""
    _require_writes(context, work, (Tables.def14a_llm,))
    row = {column: work.stamp[column] for column in (*Tables.def14a_llm.pk, "as_of")}
    context.store.save(Tables.def14a_llm, pd.DataFrame([{**row, "def14a_json": work.raw}]))


def save_section(context: Context, work: FilingWork, processed: ProcessedSection) -> None:
    """Save survivors before deleting exact obsolete holder keys or hidden placeholders."""
    _require_writes(context, work, processed.frames)
    for table, frame in processed.frames.items():
        context.store.save(table, frame)
        if table is Tables.def14a_llm:
            work.parent.update(cast(dict[str, Any], frame.iloc[0].to_dict()))
            work.saved[table] = pd.DataFrame([work.parent])
            continue
        if table is Tables.def14a_ownership and not work.saved[table].empty:
            _delete_obsolete_holders(context, work.saved[table], frame)
        work.saved[table] = frame
        marker_column, marker = cast(tuple[str, str], table.empty_marker)
        if frame[marker_column].ne(marker).any():
            context.store.delete(
                table, where={"ticker": work.stamp["ticker"], "accession_number": work.stamp["accession_number"], marker_column: marker}
            )


def _require_writes(context: Context, work: FilingWork, tables: Iterable[Table]) -> None:
    """Check every destination before reserving or saving a filing's facts."""
    for table in set(tables):
        policy = window_policy(context.config, table, context.run.as_of)
        if policy is None or not policy.requested.start <= work.stamp["as_of"] <= policy.requested.end:
            raise ValueError(f"{table.name}: proxy write outside its configured Window")


def _delete_obsolete_holders(context: Context, df_saved: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Delete each saved `def14a_ownership` holder key the new frame no longer holds."""
    keys = [*Tables.def14a_ownership.pk, "as_of"]
    df_previous = df_saved[keys].assign(as_of=pd.to_datetime(df_saved["as_of"]))
    df_current = frame[keys].assign(as_of=pd.to_datetime(frame["as_of"]))
    obsolete = df_previous.merge(df_current, on=keys, how="left", indicator=True)
    for row in obsolete.loc[obsolete._merge.eq("left_only"), keys].to_dict("records"):
        context.store.delete(Tables.def14a_ownership, where=cast(dict[str, object], row))
