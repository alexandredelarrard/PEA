"""Scoped reads, free replay and durable saves for the five proxy domains."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.schemas.def14a_schema import OwnershipExtract
from src.data_extract.utils.structure.def14a.evidence import layout_has_gap, prose_has_gap, table_has_gap
from src.data_extract.utils.structure.def14a.holders import holders_need_reconciliation
from src.data_extract.utils.structure.def14a.postprocess import ProcessedSection, archived_answers
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section, reservation_frame
from src.data_store.schema import Table, Tables
from src.gpt_extract.utils.schemas_gpt import LlmTask


@dataclass
class FilingWork:
    """One listed proxy; counters belong to the saved domain, not this working copy."""

    filing: pd.Series
    stamp: dict[str, Any]
    saved: dict[Table, pd.DataFrame]
    parent: dict[str, Any]
    attempts: dict[str, int]
    raw: str | None = None
    pending: set[str] = field(default_factory=set)


def has_gaps(section: Section, saved: pd.DataFrame, parent: dict[str, Any]) -> bool:
    """Conservative key check; the source decides whether a null needs a paid retry."""
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
    if section.key == "ownership" and holders_need_reconciliation(rows):
        return True
    keys = {
        "executive_pay": (
            "salary",
            "bonus",
            "stock_awards",
            "option_awards",
            "non_equity_incentive",
            "pension_change",
            "other_compensation",
            "total",
        ),
        "director_pay": ("fees_earned", "stock_awards", "option_awards", "non_equity_incentive", "pension_change", "other_compensation", "total"),
        "ownership": ("shares", "percent_of_class", "percent_of_voting_power"),
        "governance": ("age", "tenure_years", "is_independent"),
    }[section.key]
    if any(column not in rows or rows[column].isna().any() for column in keys):
        return True
    parent_keys = {
        "governance": ("ceo_name_proxy", "ceo_age", "ceo_since_year", "board_size", "ceo_pay_ratio", "say_on_pay_support_pct"),
        "ownership": ("insider_shares", "insider_ownership_pct", "insider_voting_pct"),
    }.get(section.key, ())
    return any(pd.isna(parent.get(column)) for column in parent_keys)


def load_work(context: Context, ticker: str, filings: pd.DataFrame, floor: pd.Timestamp) -> list[FilingWork]:
    """Read each scoped fact table once, without the large paid JSON archive."""
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

    # Keep the listing's entity, filing date and report period separate from its fiscal-year rows.
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
            "period": filing.get("period_of_report"),
            "company_name": filing.get("company_name"),
        }
        work = FilingWork(filing, stamp, saved, parent, attempts)
        work.pending = {section.key for section in SECTIONS if has_gaps(section, saved[section.table], parent)}
        works.append(work)
    return works


def load_archives(context: Context, works: list[FilingWork]) -> None:
    """Load only candidate archives, and inherit legacy all-domain attempts."""
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


def replay_candidates(works: list[FilingWork]) -> Iterator[tuple[FilingWork, Section, Any]]:
    """Yield already-paid domain answers; no source read or provider is involved."""
    for work in works:
        answers = archived_answers(work.raw) if work.pending else {}
        for section in (item for item in SECTIONS if item.key in work.pending):
            # Normalizing saved holders is free, including complete retry=2 rows without JSON.
            if section.key == "ownership" and holders_need_reconciliation(work.saved[section.table]):
                yield work, section, OwnershipExtract(percent_unit="fraction", shares_scale="units")
            if section.key not in work.pending:
                continue
            for answer in answers.get(section.key, []):
                yield work, section, answer


def requests_for(work: FilingWork, payloads: dict[str, str]) -> list[tuple[Section, LlmTask]]:
    """Charge only an unread section, or a first attempt with a disclosed missing key."""
    requests = []
    for section in SECTIONS:
        attempt = work.attempts[section.key]
        if attempt >= 2 or section.key not in work.pending:
            continue
        payload = payloads.get(section.key, "")
        if not payload.strip():
            continue
        if attempt and not (
            table_has_gap(section, payload, work.saved[section.table], work.parent)
            or prose_has_gap(section, payload, work.saved[section.table], work.parent)
            or layout_has_gap(section, payload, work.saved[section.table], work.parent)
        ):
            continue
        task = LlmTask(seq=0, payload=payload, schema=section.schema, table=section.table, meta={"work": work})
        requests.append((section, task))
    return requests


def matched_batches(requests: list[tuple[Section, LlmTask]], size: int) -> Iterator[tuple[Section, list[LlmTask]]]:
    """Bound in-flight answers using the existing worker count, without capping the daily worklist."""
    for section in SECTIONS:
        tasks = [task for owner, task in requests if owner == section]
        for start in range(0, len(tasks), max(1, size)):
            batch = tasks[start : start + max(1, size)]
            yield section, [replace(task, seq=seq) for seq, task in enumerate(batch)]


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
            keys = [*table.pk, "as_of"]
            df_previous = work.saved[table][keys].assign(as_of=pd.to_datetime(work.saved[table]["as_of"]))
            df_current = frame[keys].assign(as_of=pd.to_datetime(frame["as_of"]))
            obsolete = df_previous.merge(df_current, on=keys, how="left", indicator=True)
            for row in obsolete.loc[obsolete._merge.eq("left_only"), keys].to_dict("records"):
                context.store.delete(table, where=row)
        work.saved[table] = frame
        marker_column, marker = cast(tuple[str, str], table.empty_marker)
        if frame[marker_column].ne(marker).any():
            context.store.delete(
                table, where={"ticker": work.stamp["ticker"], "accession_number": work.stamp["accession_number"], marker_column: marker}
            )
    work.pending = {section.key for section in SECTIONS if has_gaps(section, work.saved[section.table], work.parent)}
