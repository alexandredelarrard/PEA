"""Read-only invariants for an explicitly recovered 13F version cohort."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.context import Context
from src.data_store.schema import Table, Tables
from src.utils.institutional_vintages import filing_public_dates, public_books
from src.validate.result import CheckResult, Finding


def check_institutionals(
    context: Context,
    *,
    ciks: Sequence[str],
    start_period: str,
    end_period: str,
    filed_through: str,
    trading_index: pd.DatetimeIndex | None = None,
    table: Table = Tables.sec13f_manager_holdings,
) -> CheckResult:
    """Validate complete books, source totals, clocks and historical-prefix invariance."""
    manifest_columns = [
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "amendment_type",
        "amendment_number",
        "report_type",
        "accepted_at",
        "restricted",
        "parse_status",
        "parsed_row_count",
        "source_row_count",
        "declared_row_count",
        "source_value_total",
        "declared_value_total",
        "source_checksum",
    ]
    scope: dict[str, Any] = {
        "ciks": list(ciks),
        "report_period_start": start_period,
        "report_period_end": end_period,
        "filing_cutoff": filed_through,
        "production_activation": False,
    }
    if not set(manifest_columns) <= set(context.store.columns(table)):
        return CheckResult.abstained("institutionals", table.name, "No verified accession schema", **scope)
    m = context.store.load(
        table,
        columns=manifest_columns,
        where={"cik": list(ciks), "cusip": table.empty_marker[1], "parse_status": "complete"},
        markers=True,
        since=start_period,
        until=end_period,
        date_col="period",
        optional=True,
    )
    if m is None:
        return CheckResult.abstained("institutionals", table.name, "No recovered accessions in scope", **scope)
    m = m.loc[pd.to_datetime(m.filing_date).le(pd.Timestamp(filed_through))].reset_index(drop=True)
    if m.empty:
        return CheckResult.abstained("institutionals", table.name, "No accessions public by cutoff", **scope)
    columns = [
        "cik",
        "accession_number",
        "period",
        "filing_date",
        "cusip",
        "issuer_name",
        "title_of_class",
        "position_type",
        "class_evidence",
        "shares",
        "value_usd",
        "call_shares",
        "call_value",
        "put_shares",
        "put_value",
        "debt_prn",
        "debt_value",
        "other_value",
        "raw_value_usd",
        "raw_call_value",
        "raw_put_value",
        "raw_debt_value",
        "raw_other_value",
    ]
    parts = list(
        context.store.iter_load(
            table,
            columns=columns,
            where={"cik": list(ciks), "accession_number": m.accession_number.tolist()},
            since=start_period,
            until=end_period,
            date_col="period",
        )
    )
    h = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if trading_index is None:
        calendar = context.store.load(
            Tables.prices, columns=["date"], where={"ticker": "AAPL"}, since=pd.to_datetime(m.period).min(), until=filed_through
        )
        trading_index = pd.DatetimeIndex(pd.to_datetime(calendar.date).drop_duplicates()).sort_values()
    findings: list[Finding] = []
    duplicates = int(m.accession_number.duplicated().sum() + h.duplicated(["cik", "accession_number", "cusip"]).sum())
    if duplicates:
        findings.append(Finding.at(9, f"{duplicates} duplicate vintage keys", "Unique accession/CUSIP grain"))
    counts = h.groupby("accession_number").size().reindex(m.accession_number, fill_value=0).to_numpy()
    raw_columns = [column for column in columns if column.startswith("raw_")]
    totals = h.groupby("accession_number")[raw_columns].sum().sum(axis=1).reindex(m.accession_number, fill_value=0).to_numpy()
    broken = int(
        (
            (counts != m.parsed_row_count.to_numpy())
            | (m.source_row_count.to_numpy() != m.declared_row_count.to_numpy())
            | ~np.isclose(totals, m.source_value_total.to_numpy(dtype=float), rtol=0.005, atol=1)
            | ~np.isclose(m.source_value_total.to_numpy(dtype=float), m.declared_value_total.to_numpy(dtype=float), rtol=0.005, atol=1)
        ).sum()
    )
    if broken:
        findings.append(Finding.at(9, f"{broken} unreconciled complete accessions", "Manifest and actual source books reconcile"))
    dates = filing_public_dates(m, trading_index)
    prefix_checks = 0
    try:
        for day in dates.dropna().drop_duplicates().sort_values():
            prior_sessions = trading_index[trading_index < day]
            if not len(prior_sessions):
                continue
            cutoff = prior_sessions[-1]
            prefix = m.loc[dates.le(cutoff)]
            hp = h.loc[h.accession_number.isin(prefix.accession_number)]
            a, sa = public_books(prefix, hp, cutoff, trading_index)
            b, sb = public_books(m, h, cutoff, trading_index)
            pd.testing.assert_frame_equal(a, b)
            pd.testing.assert_frame_equal(sa, sb)
            prefix_checks += 1
        _, states = public_books(m, h, filed_through, trading_index)
    except (AssertionError, ValueError) as exc:
        findings.append(Finding.at(9, str(exc), "Future filings cannot change any historical publication"))
        states = pd.DataFrame()
    metrics = {
        "accessions": len(m),
        "holding_rows": len(h),
        "duplicate_keys": duplicates,
        "unreconciled_books": broken,
        "future_prefix_checks": prefix_checks,
        "ambiguous_books": int(states.resolution_status.eq("ambiguous_additions").sum()) if len(states) else 0,
        "inferred_position_patches": int(states.resolution_status.eq("inferred_position_patch").sum()) if len(states) else 0,
        "books_without_exit_support": int((~states.zero_eligible).sum()) if len(states) else 0,
    }
    scope["source_filing_bounds"] = [str(pd.to_datetime(m.filing_date).min().date()), str(pd.to_datetime(m.filing_date).max().date())]
    scope["calendar_end"] = str(trading_index.max().date())
    scope["units_and_share_basis"] = "Source vintage validation; P1/P4 economic checks are separate"
    context.log.info(
        "SANITY: %d recovered accessions, %d rows, %d causal prefix checks; %d ambiguous books; %d invariant findings",
        len(m),
        len(h),
        prefix_checks,
        metrics["ambiguous_books"],
        len(findings),
    )
    return CheckResult.measured("institutionals", table.name, findings, scope=scope, metrics=metrics)
