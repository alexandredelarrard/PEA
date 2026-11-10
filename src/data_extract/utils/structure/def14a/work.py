"""Scoped reads, the per-ticker section calls and durable saves for the five proxy domains."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.structure.def14a.postprocess import ProcessedSection, append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section, reservation_frame
from src.data_store.schema import Table, Tables

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmTask


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


def has_gaps(section: Section, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Key-field check: a section is a gap when it has no row, or none of its key fields is populated."""
    if section.table is Tables.def14a_llm:
        return any(
            pd.isna(parent.get(column))
            for column in (
                "auditor_name",
                "auditor_fees",
                "auditor_fees_prior",
                "audit_fees_audit",
                "audit_fees_audit_related",
                "audit_fees_tax",
                "audit_fees_other",
            )
        )
    marker_column, marker = cast(tuple[str, str], section.table.empty_marker)
    rows = saved.loc[saved[marker_column].ne(marker)] if marker_column in saved else saved
    if rows.empty:
        return True
    if section.key == "governance":
        return pd.isna(parent.get("ceo_name_proxy"))
    keys = {
        "executive_pay": ("total", "salary"),
        "director_pay": ("total", "fees_earned"),
        "ownership": ("shares", "percent_of_class"),
    }[section.key]
    return all(column not in rows or rows[column].isna().all() for column in keys)


def load_work(context: Context, ticker: str, filings: pd.DataFrame, floor: pd.Timestamp) -> list[FilingWork]:
    """Read each scoped fact table once, without the large paid JSON archive."""
    # load: each section table's rows of the ticker in the window, markers included
    if filings.empty:
        return []
    frames: dict[Table, pd.DataFrame] = {}
    for section in SECTIONS:
        table = section.table
        available = context.store.columns(table)
        wanted = (
            tuple(dict.fromkeys((*table.pk, "as_of", "period", "retry", *(column for item in SECTIONS for column in item.parent_columns))))
            if table is Tables.def14a_llm
            else available
        )
        columns = [column for column in wanted if column in available and column != "def14a_json"]
        loaded = context.store.load(table, columns=columns, where={"ticker": ticker}, since=floor, markers=True, optional=True)
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
        work.pending = {section.key for section in SECTIONS if has_gaps(section, saved[section.table], parent)}
        works.append(work)
    return works


def load_archives(context: Context, works: list[FilingWork]) -> None:
    """Load only candidate archives, and inherit legacy all-domain attempts."""
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

    # A legacy whole-proxy response, including its empty marker, proves a first attempt for all five sections.
    for work in works:
        raw = archives.get(work.stamp["accession_number"])
        work.raw = raw if isinstance(raw, str) and raw else None
        if not work.raw:
            continue
        try:
            document = json.loads(work.raw) if work.raw != "_empty" else {}
        except (ValueError, TypeError):
            document = {}
        metadata = document.get("_def14a_split") if isinstance(document, dict) else None
        legacy = metadata.get("legacy_json") if isinstance(metadata, dict) else work.raw
        if legacy:
            work.attempts = {key: max(attempt, 1) for key, attempt in work.attempts.items()}


def open_sections(work: FilingWork) -> list[Section]:
    """Sections with a key gap and fewer than two attempts: the only ones worth a source read."""
    return [section for section in SECTIONS if section.key in work.pending and work.attempts[section.key] < 2]


def ticker_works(context: Context, ticker: str, df_filings: pd.DataFrame, floor: pd.Timestamp) -> list[FilingWork]:
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
    requests = []
    for work in works:
        requests.extend(requests_for(work, budget))
        for section in (item for item in open_sections(work) if not work.payloads.get(item.key, "").strip()):
            reserve(context, work, section, work.attempts[section.key] + 1)
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
            processed = postprocess_section(section, result.parsed, work.stamp, work.saved, work.parent, work.payloads.get(section.key, ""))
            save_section(context, work, processed)
            work.raw = refresh_archive(work.raw, section, processed.legacy, processed.corrections)
            save_archive(context, work)
            answered += 1
    return answered


def reserve(context: Context, work: FilingWork, section: Section, attempt: int) -> None:
    """Write the owned counter before submission, retaining partial facts and monotonic attempts."""
    frame = reservation_frame(section.table, work.stamp, work.saved[section.table], attempt)
    context.store.save(section.table, frame)
    work.attempts[section.key] = int(frame.retry.max())
    if section.table is Tables.def14a_llm:
        work.parent["retry"] = int(frame.retry.max())
    saved = work.saved[section.table]
    work.saved[section.table] = saved.assign(retry=int(frame.retry.max())) if not saved.empty else frame


def save_archive(context: Context, work: FilingWork) -> None:
    """Save raw paid output independently of any later postprocessing failure."""
    row = {column: work.stamp[column] for column in (*Tables.def14a_llm.pk, "as_of")}
    context.store.save(Tables.def14a_llm, pd.DataFrame([{**row, "def14a_json": work.raw}]))


def save_section(context: Context, work: FilingWork, processed: ProcessedSection) -> None:
    """Save survivors before deleting exact obsolete holder keys or hidden placeholders."""
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
    work.pending = {section.key for section in SECTIONS if has_gaps(section, work.saved[section.table], work.parent)}


def _delete_obsolete_holders(context: Context, df_saved: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Delete each saved `def14a_ownership` holder key the new frame no longer holds."""
    keys = [*Tables.def14a_ownership.pk, "as_of"]
    df_previous = df_saved[keys].assign(as_of=pd.to_datetime(df_saved["as_of"]))
    df_current = frame[keys].assign(as_of=pd.to_datetime(frame["as_of"]))
    obsolete = df_previous.merge(df_current, on=keys, how="left", indicator=True)
    for row in obsolete.loc[obsolete._merge.eq("left_only"), keys].to_dict("records"):
        context.store.delete(Tables.def14a_ownership, where=cast(dict[str, object], row))
