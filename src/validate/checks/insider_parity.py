"""Quarterly-ZIP versus daily-EDGAR insider reconciliation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InsiderParityThresholds:
    accession_coverage: float
    exact_structure: float
    categorical_agreement: float
    share_agreement: float
    economic_within_tolerance: float
    identity_agreement: float
    numeric_relative_tolerance: float
    aggregate_value_relative_tolerance: float
    feature_cell_agreement: float
    edge_rank_correlation: float


_CATEGORICAL = (
    "transaction_date",
    "transaction_code",
    "acquired_disposed",
    "direct_indirect",
    "security_type",
    "security_title",
)
_IDENTITY = (
    "ticker",
    "issuer_cik",
    "owner_cik",
    "owner_name",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "officer_title",
    "is_10b5_1",
)
_SHARES = ("shares", "shares_owned_after")
_PRICE_VALUE = ("price_per_share", "value_usd")
_DATE_COLUMNS = {"transaction_date", "filing_date", "period_of_report"}
_CIK_COLUMNS = {"issuer_cik", "owner_cik"}
_FLAG_COLUMNS = {"is_director", "is_officer", "is_ten_pct_owner", "is_10b5_1"}
_REQUIRED_COLUMNS = {
    "accession_number",
    "security_type",
    "filing_date",
    *_CATEGORICAL,
    *_IDENTITY,
    *_SHARES,
    *_PRICE_VALUE,
}


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("accession_number", "security_type"):
        out[column] = out[column].astype("string")
    sort_columns = [
        column
        for column in (
            "accession_number",
            "security_type",
            "transaction_date",
            "transaction_code",
            "acquired_disposed",
            "security_title",
            "shares",
            "price_per_share",
            "shares_owned_after",
            "direct_indirect",
            "exercise_price",
            "underlying_shares",
        )
        if column in out
    ]
    out = out.sort_values(sort_columns, kind="stable", na_position="last")
    out["_source_ordinal"] = out.groupby(["accession_number", "security_type"], dropna=False).cumcount()
    return out


def _normalized(series: pd.Series, column: str) -> pd.Series:
    if column in _DATE_COLUMNS:
        values = pd.to_datetime(series, errors="coerce")
        return values.dt.strftime("%Y-%m-%d").astype("string").fillna("<NULL>")
    if column in _FLAG_COLUMNS:
        values = pd.to_numeric(series, errors="coerce")
        return values.map(lambda value: "<NULL>" if pd.isna(value) else f"{float(value):g}").astype("string")
    values = series.astype("string").str.strip().str.upper()
    if column in _CIK_COLUMNS:
        values = values.str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return values.fillna("<NULL>")


def _agreement(merged: pd.DataFrame, columns: tuple[str, ...]) -> float:
    scores: list[np.ndarray] = []
    for column in columns:
        left, right = f"{column}_bulk", f"{column}_live"
        if left not in merged or right not in merged:
            continue
        a = _normalized(merged[left], column)
        b = _normalized(merged[right], column)
        scores.append(a.eq(b).to_numpy())
    return float(np.concatenate(scores).mean()) if scores else float("nan")


def _agreements(merged: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    return {
        column: float(_normalized(merged[f"{column}_bulk"], column).eq(_normalized(merged[f"{column}_live"], column)).mean())
        for column in columns
        if f"{column}_bulk" in merged and f"{column}_live" in merged
    }


def _within(merged: pd.DataFrame, columns: tuple[str, ...], tolerance: float) -> float:
    scores: list[np.ndarray] = []
    for column in columns:
        left, right = f"{column}_bulk", f"{column}_live"
        if left not in merged or right not in merged:
            continue
        a = pd.to_numeric(merged[left], errors="coerce").to_numpy(dtype="float64")
        b = pd.to_numeric(merged[right], errors="coerce").to_numpy(dtype="float64")
        both_null = np.isnan(a) & np.isnan(b)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
        close = np.abs(a - b) <= tolerance * scale
        scores.append(both_null | close)
    return float(np.concatenate(scores).mean()) if scores else float("nan")


def _withins(
    merged: pd.DataFrame,
    columns: tuple[str, ...],
    tolerance: float,
) -> dict[str, float]:
    return {column: _within(merged, (column,), tolerance) for column in columns if f"{column}_bulk" in merged and f"{column}_live" in merged}


def reconcile_transactions(
    bulk: pd.DataFrame,
    live: pd.DataFrame,
    thresholds: InsiderParityThresholds,
) -> dict[str, object]:
    """Compare normalized sources without relying on either source's row identifier."""
    if bulk.empty:
        raise ValueError("bulk quarter is empty")
    missing_bulk_columns = sorted(_REQUIRED_COLUMNS - set(bulk.columns))
    missing_live_columns = sorted(_REQUIRED_COLUMNS - set(live.columns))
    bulk_accessions = set(bulk["accession_number"].dropna().astype(str))
    live_accessions = set(live["accession_number"].dropna().astype(str))
    matched_accessions = bulk_accessions & live_accessions
    coverage = len(matched_accessions) / len(bulk_accessions)

    count_keys = ["accession_number", "security_type"]
    bulk_matched = bulk[bulk["accession_number"].astype(str).isin(matched_accessions)]
    live_matched = live[live["accession_number"].astype(str).isin(matched_accessions)]
    counts = (
        bulk_matched.groupby(count_keys)
        .size()
        .rename("bulk")
        .to_frame()
        .join(live_matched.groupby(count_keys).size().rename("live"), how="outer")
        .fillna(0)
    )
    structure = float(counts["bulk"].eq(counts["live"]).mean()) if len(counts) else 0.0

    keys = [*count_keys, "_source_ordinal"]
    merged = _ordered(bulk_matched).merge(
        _ordered(live_matched),
        on=keys,
        how="inner",
        suffixes=("_bulk", "_live"),
    )
    categorical = _agreement(merged, _CATEGORICAL)
    identity = _agreement(merged, _IDENTITY)
    share_agreement = _within(
        merged,
        _SHARES,
        thresholds.numeric_relative_tolerance,
    )
    economic = _within(
        merged,
        _PRICE_VALUE,
        thresholds.numeric_relative_tolerance,
    )

    bulk_value = float(pd.to_numeric(bulk.get("value_usd"), errors="coerce").sum())
    live_value = float(pd.to_numeric(live.get("value_usd"), errors="coerce").sum())
    value_difference = abs(live_value - bulk_value) / max(abs(bulk_value), 1.0)
    matched_bulk_value = pd.to_numeric(merged.get("value_usd_bulk"), errors="coerce").fillna(0.0)
    matched_live_value = pd.to_numeric(merged.get("value_usd_live"), errors="coerce").fillna(0.0)
    value_weighted_difference = float((matched_live_value - matched_bulk_value).abs().sum() / max(matched_bulk_value.abs().sum(), 1.0))
    missing = sorted(bulk_accessions - live_accessions)
    missing_rows = bulk[bulk["accession_number"].astype(str).isin(missing)].copy()
    missing_details: list[dict[str, object]] = []
    if not missing_rows.empty:
        for accession, rows in missing_rows.groupby("accession_number", sort=True):
            filing_date = pd.to_datetime(rows["filing_date"], errors="coerce").min() if "filing_date" in rows else pd.NaT
            form = rows.get("document_type")
            missing_details.append(
                {
                    "accession_number": str(accession),
                    "document_type": (str(form.dropna().iloc[0]) if form is not None and not form.dropna().empty else None),
                    "filing_date": (filing_date.strftime("%Y-%m-%d") if pd.notna(filing_date) else None),
                }
            )
    passed = bool(
        not missing_bulk_columns
        and not missing_live_columns
        and coverage >= thresholds.accession_coverage
        and structure >= thresholds.exact_structure
        and categorical >= thresholds.categorical_agreement
        and share_agreement >= thresholds.share_agreement
        and economic >= thresholds.economic_within_tolerance
        and identity >= thresholds.identity_agreement
        and value_difference <= thresholds.aggregate_value_relative_tolerance
        and value_weighted_difference <= thresholds.aggregate_value_relative_tolerance
    )
    return {
        "passed": passed,
        "missing_bulk_columns": missing_bulk_columns,
        "missing_live_columns": missing_live_columns,
        "bulk_accessions": len(bulk_accessions),
        "live_accessions": len(live_accessions),
        "matched_rows": len(merged),
        "accession_coverage": coverage,
        "exact_structure": structure,
        "categorical_agreement": categorical,
        "categorical_agreement_by_field": _agreements(merged, _CATEGORICAL),
        "share_agreement": share_agreement,
        "share_agreement_by_field": _withins(
            merged,
            _SHARES,
            thresholds.numeric_relative_tolerance,
        ),
        "economic_within_tolerance": economic,
        "economic_within_tolerance_by_field": _withins(
            merged,
            _PRICE_VALUE,
            thresholds.numeric_relative_tolerance,
        ),
        "identity_agreement": identity,
        "identity_agreement_by_field": _agreements(merged, _IDENTITY),
        "aggregate_value_relative_difference": value_difference,
        "value_weighted_relative_difference": value_weighted_difference,
        "missing_accessions": missing,
        "missing_accession_details": missing_details,
        "extra_live_accessions": sorted(live_accessions - bulk_accessions),
    }


def reconcile_feature_panels(
    bulk_panel: pd.DataFrame,
    live_panel: pd.DataFrame,
    thresholds: InsiderParityThresholds,
) -> dict[str, object]:
    """Compare final insider feature cells, including the availability mask."""
    keys = ["date", "ticker"]
    feature_columns = sorted((set(bulk_panel) & set(live_panel)) - set(keys))
    bulk_keys = set(map(tuple, bulk_panel[keys].itertuples(index=False, name=None)))
    live_keys = set(map(tuple, live_panel[keys].itertuples(index=False, name=None)))
    row_key_agreement = len(bulk_keys & live_keys) / max(len(bulk_keys | live_keys), 1)
    merged = bulk_panel[keys + feature_columns].merge(
        live_panel[keys + feature_columns],
        on=keys,
        how="inner",
        suffixes=("_bulk", "_live"),
    )
    null_scores: list[np.ndarray] = []
    value_scores: list[np.ndarray] = []
    feature_metrics: dict[str, dict[str, float | int | None]] = {}
    for column in feature_columns:
        a = pd.to_numeric(merged[f"{column}_bulk"], errors="coerce").to_numpy()
        b = pd.to_numeric(merged[f"{column}_live"], errors="coerce").to_numpy()
        same_null = np.isnan(a) == np.isnan(b)
        null_scores.append(same_null)
        comparable = ~np.isnan(a) & ~np.isnan(b)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
        within = np.ones(len(a), dtype=bool)
        within[comparable] = np.abs(a[comparable] - b[comparable]) <= thresholds.numeric_relative_tolerance * scale[comparable]
        value_scores.append(within & same_null)
        feature_metrics[column] = {
            "cells": len(a),
            "null_mask_agreement": float(same_null.mean()),
            "cell_agreement": float((within & same_null).mean()),
            "edge_rank_correlation": None,
        }
    null_agreement = float(np.concatenate(null_scores).mean()) if null_scores else 0.0
    cell_agreement = float(np.concatenate(value_scores).mean()) if value_scores else 0.0
    edge_correlations: dict[str, float] = {}
    if not merged.empty:
        edge = merged[merged["date"].eq(merged["date"].max())]
        for column in feature_columns:
            pair = edge[[f"{column}_bulk", f"{column}_live"]].dropna()
            if len(pair) < 2 or pair.nunique().min() < 2:
                continue
            correlation = pair.corr(method="spearman").iloc[0, 1]
            if pd.notna(correlation):
                edge_correlations[column] = float(correlation)
                feature_metrics[column]["edge_rank_correlation"] = float(correlation)
    minimum_edge_rank = min(edge_correlations.values(), default=1.0)
    return {
        "passed": bool(
            row_key_agreement == 1.0
            and null_agreement == 1.0
            and cell_agreement >= thresholds.feature_cell_agreement
            and minimum_edge_rank >= thresholds.edge_rank_correlation
        ),
        "row_key_agreement": row_key_agreement,
        "shared_rows": len(merged),
        "feature_columns": len(feature_columns),
        "null_mask_agreement": null_agreement,
        "cell_agreement": cell_agreement,
        "minimum_edge_rank_correlation": minimum_edge_rank,
        "edge_rank_correlations": edge_correlations,
        "by_feature": feature_metrics,
    }
