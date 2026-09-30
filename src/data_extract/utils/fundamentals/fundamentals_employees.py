"""Standalone SEC 10-K employee-headcount extraction.

The employee table owns its filing discovery, accession outcomes, continuity
state and resume frontier. It deliberately does not share the XBRL facts walk:
a successfully parsed fact accession says nothing about whether its prose
headcount was read.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.edgar_driver import (
    PROGRAMMING_ERRORS,
    IncompleteEdgarRunError,
    filed_by,
    period_of_report,
)
from src.data_extract.utils.common.edgar_extract import extract_employee_count, html_to_text
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import Registrant, load_registrants, resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import (
    get_entry,
    manifest_window,
    record_filing_outcomes,
    record_run,
)
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Tables

HEADCOUNT_FORMS: tuple[str, ...] = ("10-K", "10-K/A")
HEADCOUNT_CONTINUITY_MIN = 0.2
HEADCOUNT_CONTINUITY_MAX = 5.0

SAVED = "saved"
NO_HEADCOUNT = "no_headcount"
REJECTED_OUTLIER = "rejected_outlier"
PENDING_REGIME = "pending_regime"
TERMINAL_STATUSES = frozenset({SAVED, NO_HEADCOUNT, REJECTED_OUTLIER})
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmployeeCandidate:
    ticker: str
    accession_number: str
    cik: str
    form: str
    filing_date: pd.Timestamp
    report_date: pd.Timestamp
    count: int
    ordering: int

    def outcome(self, status: str) -> dict:
        row: dict = {
            "ticker": self.ticker,
            "accession_number": self.accession_number,
            "cik": self.cik,
            "form": self.form,
            "filing_date": self.filing_date.strftime("%Y-%m-%d"),
            "report_date": self.report_date.strftime("%Y-%m-%d"),
            "ordering": self.ordering,
            "status": status,
        }
        if status == PENDING_REGIME:
            row["candidate"] = self.count
        return row


@dataclass(frozen=True)
class EmployeeTickerResult:
    frame: pd.DataFrame
    outcomes: list[dict]


def is_headcount_form(form: str | None) -> bool:
    return str(form or "").upper() in HEADCOUNT_FORMS


def _recent_anchor(history: list[int]) -> float | None:
    recent = history[-3:]
    return None if not recent else float(sorted(recent)[len(recent) // 2])


def is_continuous(count: int, history: list[int]) -> bool:
    """Whether a count fits the ticker's trusted headcount regime."""
    anchor = _recent_anchor(history)
    if anchor is None:
        return True
    if anchor <= 0:
        return True
    return HEADCOUNT_CONTINUITY_MIN <= count / anchor <= HEADCOUNT_CONTINUITY_MAX


def history_by_ticker(rows: pd.DataFrame | None) -> dict[str, list[int]]:
    """Accepted counts by ticker, supporting employee-table and legacy fact shapes."""
    if rows is None or rows.empty or "ticker" not in rows.columns:
        return {}
    date_col = "as_of" if "as_of" in rows.columns else "filing_date"
    value_col = "employees" if "employees" in rows.columns else "value"
    if date_col not in rows.columns or value_col not in rows.columns:
        return {}
    selected = rows[["ticker", date_col, value_col]].copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce")
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    selected = selected.dropna(subset=[date_col, value_col]).sort_values(date_col)
    return {str(ticker): group[value_col].astype(int).tolist() for ticker, group in selected.groupby("ticker")}


def filing_body_text(filing) -> str:
    raw = filing.html()
    if raw:
        return html_to_text(raw)
    return filing.text() or ""


def _filing_key(filing) -> tuple[pd.Timestamp, int, str]:
    return (
        pd.Timestamp(filing.filing_date).normalize(),
        1 if str(getattr(filing, "form", "")).upper() == "10-K/A" else 0,
        str(filing.accession_number),
    )


def _candidate_from_outcome(outcome: dict) -> EmployeeCandidate:
    return EmployeeCandidate(
        ticker=str(outcome["ticker"]),
        accession_number=str(outcome["accession_number"]),
        cik=str(outcome["cik"]),
        form=str(outcome["form"]),
        filing_date=pd.Timestamp(outcome["filing_date"]).normalize(),
        report_date=pd.Timestamp(outcome["report_date"]).normalize(),
        count=int(outcome["candidate"]),
        ordering=int(outcome.get("ordering", 0)),
    )


def _resolve_continuity(
    candidates: list[EmployeeCandidate],
    history: list[int],
) -> tuple[list[EmployeeCandidate], list[dict]]:
    accepted_values = list(history)
    saved: list[EmployeeCandidate] = []
    outcomes: list[dict] = []
    pending: list[EmployeeCandidate] = []

    for candidate in candidates:
        if is_continuous(candidate.count, accepted_values):
            anchor = _recent_anchor(accepted_values)
            for prior in pending:
                outcomes.append(prior.outcome(REJECTED_OUTLIER))
                logger.warning(
                    "employees decision reason=rejected_outlier ticker=%s accession=%s "
                    "count=%d anchor=%s ratio=%s; accession %s returned to the trusted regime",
                    prior.ticker,
                    prior.accession_number,
                    prior.count,
                    f"{anchor:.0f}" if anchor is not None else "none",
                    f"{prior.count / anchor:.3f}" if anchor and anchor > 0 else "n/a",
                    candidate.accession_number,
                )
            pending.clear()
            saved.append(candidate)
            accepted_values.append(candidate.count)
            outcomes.append(candidate.outcome(SAVED))
            continue
        if pending and is_continuous(candidate.count, [pending[-1].count]):
            old_anchor = _recent_anchor(accepted_values)
            for prior in pending[:-1]:
                outcomes.append(prior.outcome(REJECTED_OUTLIER))
                logger.warning(
                    "employees decision reason=rejected_outlier ticker=%s accession=%s count=%d anchor=%s ratio=%s; a later regime was corroborated",
                    prior.ticker,
                    prior.accession_number,
                    prior.count,
                    f"{old_anchor:.0f}" if old_anchor is not None else "none",
                    f"{prior.count / old_anchor:.3f}" if old_anchor and old_anchor > 0 else "n/a",
                )
            prior = pending[-1]
            saved.extend((prior, candidate))
            outcomes.extend((prior.outcome(SAVED), candidate.outcome(SAVED)))
            accepted_values = [prior.count, candidate.count]
            logger.info(
                "employees decision reason=new_regime ticker=%s old_anchor=%s "
                "first=%s/%d first_ratio=%s second=%s/%d second_ratio=%s mutual_ratio=%.3f",
                candidate.ticker,
                f"{old_anchor:.0f}" if old_anchor is not None else "none",
                prior.accession_number,
                prior.count,
                f"{prior.count / old_anchor:.3f}" if old_anchor and old_anchor > 0 else "n/a",
                candidate.accession_number,
                candidate.count,
                f"{candidate.count / old_anchor:.3f}" if old_anchor and old_anchor > 0 else "n/a",
                candidate.count / prior.count,
            )
            pending.clear()
            continue
        pending.append(candidate)

    for candidate in pending:
        anchor = _recent_anchor(accepted_values)
        ratio = candidate.count / anchor if anchor and anchor > 0 else float("nan")
        logger.warning(
            "employees decision reason=pending_regime ticker=%s accession=%s count=%d anchor=%s ratio=%s",
            candidate.ticker,
            candidate.accession_number,
            candidate.count,
            f"{anchor:.0f}" if anchor is not None else "none",
            f"{ratio:.3f}" if pd.notna(ratio) else "n/a",
        )
        outcomes.append(candidate.outcome(PENDING_REGIME))
    return saved, outcomes


def build_ticker_employees(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    registrants: dict[str, Registrant] | None,
    history: list[int],
    pending_outcomes: list[dict],
) -> EmployeeTickerResult:
    """Parse one ticker's annual filings and classify every readable accession."""
    filings = resolve_registrant_filings(
        ticker,
        HEADCOUNT_FORMS,
        since=since,
        done_accessions=done_accessions,
        registrants=registrants,
    )
    candidates = [_candidate_from_outcome(outcome) for outcome in pending_outcomes]
    outcomes: list[dict] = []

    for filing in sorted(filings, key=_filing_key):
        ordering = len(candidates) + len(outcomes)
        filing_date = pd.Timestamp(filing.filing_date).normalize()
        report = period_of_report(filing)
        provenance = {
            "ticker": ticker,
            "accession_number": str(filing.accession_number),
            "cik": filed_by(filing, cik),
            "form": str(filing.form),
            "filing_date": filing_date.strftime("%Y-%m-%d"),
            "report_date": None if report is None else pd.Timestamp(report).strftime("%Y-%m-%d"),
            "ordering": ordering,
        }
        if report is None:
            outcomes.append({**provenance, "status": NO_HEADCOUNT})
            logger.info(
                "employees decision reason=no_headcount ticker=%s accession=%s cik=%s form=%s filing_date=%s report_date=none",
                ticker,
                filing.accession_number,
                provenance["cik"],
                filing.form,
                provenance["filing_date"],
            )
            continue
        count = extract_employee_count(filing_body_text(filing))
        if count is None:
            outcomes.append({**provenance, "status": NO_HEADCOUNT})
            logger.info(
                "employees decision reason=no_headcount ticker=%s accession=%s cik=%s form=%s filing_date=%s report_date=%s",
                ticker,
                filing.accession_number,
                provenance["cik"],
                filing.form,
                provenance["filing_date"],
                provenance["report_date"],
            )
            continue
        candidates.append(
            EmployeeCandidate(
                ticker=ticker,
                accession_number=str(filing.accession_number),
                cik=filed_by(filing, cik),
                form=str(filing.form),
                filing_date=filing_date,
                report_date=pd.Timestamp(report).normalize(),
                count=int(count),
                ordering=ordering,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.filing_date,
            1 if candidate.form.upper() == "10-K/A" else 0,
            candidate.accession_number,
        ),
    )
    ordered = [EmployeeCandidate(**{**candidate.__dict__, "ordering": ordering}) for ordering, candidate in enumerate(ordered)]
    by_filing_date: dict[pd.Timestamp, EmployeeCandidate] = {}
    superseded: list[EmployeeCandidate] = []
    for candidate in ordered:
        prior = by_filing_date.get(candidate.filing_date)
        if prior is not None:
            superseded.append(prior)
        by_filing_date[candidate.filing_date] = candidate
    ordered = list(by_filing_date.values())
    anchor = _recent_anchor(history)
    for candidate in superseded:
        outcomes.append(candidate.outcome(REJECTED_OUTLIER))
        logger.warning(
            "employees decision reason=rejected_outlier ticker=%s accession=%s count=%d anchor=%s ratio=%s; same-date form/accession precedence",
            candidate.ticker,
            candidate.accession_number,
            candidate.count,
            f"{anchor:.0f}" if anchor is not None else "none",
            f"{candidate.count / anchor:.3f}" if anchor and anchor > 0 else "n/a",
        )
    saved, continuity_outcomes = _resolve_continuity(ordered, history)
    frame = pd.DataFrame(
        [
            {
                "ticker": candidate.ticker,
                "as_of": candidate.filing_date,
                "employees": float(candidate.count),
                "_ordering": candidate.ordering,
            }
            for candidate in saved
        ],
        columns=["ticker", "as_of", "employees", "_ordering"],
    )
    if not frame.empty:
        frame = (
            frame.sort_values("_ordering").drop_duplicates(subset=["ticker", "as_of"], keep="last").drop(columns="_ordering").reset_index(drop=True)
        )
    else:
        frame = frame.drop(columns="_ordering")
    return EmployeeTickerResult(
        frame=frame,
        outcomes=sorted(
            [*outcomes, *continuity_outcomes],
            key=lambda outcome: (
                str(outcome.get("filing_date", "")),
                int(outcome.get("ordering", 0)),
                str(outcome["accession_number"]),
            ),
        ),
    )


def fetch_fundamentals_employees(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
) -> None:
    """Fetch employee disclosures with an independent completeness frontier."""
    context.ensure_edgar_identity()
    cik_map = load_cik_mapping(context, tickers)
    fallback_since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    entry = get_entry(context, Tables.fundamentals_employees) or {}
    if full or not entry.get("coverage_complete"):
        since, is_full_rescan = fallback_since, True
    else:
        since, is_full_rescan = manifest_window(
            context,
            Tables.fundamentals_employees,
            len(cik_map),
            fallback_since=fallback_since,
            full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days),
        )

    stored = context.store.load(
        Tables.fundamentals_employees,
        columns=["ticker", "as_of", "employees"],
        where={"ticker": tickers},
        optional=True,
    )
    histories = history_by_ticker(stored)
    prior_outcomes = list(entry.get("filing_outcomes", []))
    relevant = [outcome for outcome in prior_outcomes if str(outcome.get("ticker")) in set(tickers)]
    if full:
        done = frozenset(str(outcome["accession_number"]) for outcome in relevant if outcome.get("status") == SAVED)
    else:
        done = frozenset(
            str(outcome["accession_number"])
            for outcome in relevant
            if outcome.get("status") in TERMINAL_STATUSES or outcome.get("status") == PENDING_REGIME
        )
    pending_by_ticker: dict[str, list[dict]] = {}
    if not full:
        for outcome in relevant:
            if outcome.get("status") == PENDING_REGIME:
                pending_by_ticker.setdefault(str(outcome["ticker"]), []).append(outcome)

    registrants = load_registrants(str(context.config_dir))
    create_lock = threading.Lock()
    created = False

    def _save(frame: pd.DataFrame) -> None:
        nonlocal created
        if created:
            context.store.save(Tables.fundamentals_employees, frame)
            return
        with create_lock:
            context.store.save(Tables.fundamentals_employees, frame)
            created = True

    def _worker(ticker: str, cik: str) -> EmployeeTickerResult | None:
        try:
            result = build_ticker_employees(
                ticker,
                cik,
                since=since,
                done_accessions=done,
                registrants=registrants,
                history=histories.get(ticker, []),
                pending_outcomes=pending_by_ticker.get(ticker, []),
            )
            if not result.frame.empty:
                _save(result.frame)
            return result
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 -- one ticker must not hide batch progress
            context.log.warning("fundamentals employees: %s failed (%s)", ticker, exc)
            return None

    results = run_per_ticker(
        cik_map,
        _worker,
        desc="fundamentals employees",
        max_workers=int(context.config.data_extract.fundamentals_workers),
    )
    successful = [result for result in results if result is not None]
    record_filing_outcomes(
        context,
        Tables.fundamentals_employees,
        [outcome for result in successful for outcome in result.outcomes],
    )
    failed = len(results) - len(successful)
    rows_added = sum(len(result.frame) for result in successful)
    context.log.info(
        "fundamentals employees: %d/%d ticker(s) ok, %d failed -> +%d rows",
        len(successful),
        len(cik_map),
        failed,
        rows_added,
    )
    if failed:
        raise IncompleteEdgarRunError(
            f"fundamentals employees: {failed}/{len(cik_map)} ticker(s) failed; rows and "
            "accession outcomes already saved remain idempotent, but no run manifest was "
            "advanced because coverage is incomplete"
        )
    record_run(
        context,
        Tables.fundamentals_employees,
        len(cik_map),
        rows_added,
        is_full_rescan=is_full_rescan,
        coverage_complete=True,
    )
