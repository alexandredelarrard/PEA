"""Dedicated earnings-call coverage and feature-quality validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import EARNINGS_CALL_FEATURES, EARNINGS_REPORT_TO_QUARTER_LAG_DAYS
from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.utils.text_metrics import assess_earnings_call_sections
from src.validate.result import CheckResult, Finding, full_table_only

CHECK = "earnings_calls"
_BOUNDS = {
    "f_ec_tone": (-1.0, 1.0),
    "f_ec_qa_gap": (-2.0, 2.0),
    "f_ec_uncertainty": (0.0, 1.0),
    "f_ec_qa_coherence_mean": (-1.0, 1.0),
    "f_ec_tone_delta": (-2.0, 2.0),
    "f_ec_qa_qq_distance": (0.0, 2.0),
    "f_ec_prep_qq_distance": (0.0, 2.0),
}


def _quarter_index(date: object) -> int | None:
    value = pd.to_datetime(date, errors="coerce")
    if pd.isna(value):
        return None
    value -= pd.Timedelta(days=EARNINGS_REPORT_TO_QUARTER_LAG_DAYS)
    return int(value.year) * 4 + int(value.quarter) - 1


def _coverage(context: Context) -> tuple[dict[str, Any], dict[str, float]]:
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker"])
    releases = context.store.load(Tables.earnings_surprises, columns=["ticker", "earnings_date"], optional=True)
    assert roster is not None
    release_ranges: dict[str, tuple[int, int]] = {}
    if releases is not None:
        for ticker, group in releases.groupby("ticker", sort=False):
            indices = [value for value in group["earnings_date"].map(_quarter_index) if value is not None]
            if indices:
                release_ranges[str(ticker)] = (max(2006 * 4, min(indices)), max(indices))

    ratios: dict[str, float] = {}
    malformed = valid_calls = observed_calls = 0
    for ticker in roster["ticker"].astype(str):
        sections = context.store.load(
            Tables.earnings_call_sections,
            columns=["ticker", "quarter", "tag", "text"],
            where={"ticker": ticker},
            optional=True,
        )
        valid: set[str] = set()
        if sections is not None:
            for (_, quarter), call in sections.groupby(["ticker", "quarter"], sort=False):
                observed_calls += 1
                quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
                if quality.valid:
                    valid.add(str(quarter))
                    valid_calls += 1
                else:
                    malformed += 1
        bounds = release_ranges.get(ticker)
        if bounds is None or bounds[1] < bounds[0]:
            continue
        expected = bounds[1] - bounds[0] + 1
        present = 0
        for quarter in valid:
            if len(quarter) == 6 and quarter[4] == "Q" and quarter[:4].isdigit() and quarter[5] in "1234":
                index = int(quarter[:4]) * 4 + int(quarter[5]) - 1
                present += bounds[0] <= index <= bounds[1]
        ratios[ticker] = present / expected

    values = pd.Series(ratios, dtype="float64")
    summary = {
        "denominator": "current roster; expected quarters from first to latest recorded earnings release, floored at 2006Q1",
        "tickers_measured": int(len(values)),
        "coverage_100pct": int((values == 1.0).sum()),
        "coverage_70_to_lt100pct": int(((values >= 0.70) & (values < 1.0)).sum()),
        "coverage_50_to_lt70pct": int(((values >= 0.50) & (values < 0.70)).sum()),
        "coverage_lt50pct": int((values < 0.50).sum()),
        "median_coverage": float(values.median()) if len(values) else None,
        "observed_calls": observed_calls,
        "valid_calls": valid_calls,
        "malformed_calls": malformed,
    }
    return summary, ratios


def _feature_sample(context: Context, table: Table, columns: list[str]) -> pd.DataFrame:
    samples = []
    for number, chunk in enumerate(context.store.iter_load(table, columns=["date", "ticker", *columns], chunksize=100_000)):
        samples.append(chunk.sample(n=min(5_000, len(chunk)), random_state=17 + number))
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=["date", "ticker", *columns])


def check_earnings_calls(
    context: Context,
    table: Table | str,
    *,
    config: Any = None,
    cache: str | None = None,
    out: str | None = None,
    tickers: list[str] | None = None,
) -> CheckResult:
    """Measure transcript coverage, malformed calls, distributions, redundancy, and drift."""
    del config, cache, out
    spec = resolve(table)
    subset = full_table_only(CHECK, spec.name, tickers)
    if subset is not None:
        return subset

    expected = [f"f_{name}" for name in EARNINGS_CALL_FEATURES]
    live = set(context.store.columns(spec))
    missing = sorted(set(expected) - live)
    legacy = sorted(column for column in live if column.startswith("f_ec_") and column not in expected)
    findings: list[Finding] = []
    if missing or legacy:
        findings.append(
            Finding.at(
                9,
                f"missing={missing}; legacy={legacy}",
                "exactly the 12 approved raw/issuer-history earnings-call columns",
                missing=missing,
                legacy=legacy,
            )
        )

    coverage, ratios = _coverage(context)
    available = [column for column in expected if column in live]
    sample = _feature_sample(context, spec, available) if available else pd.DataFrame()
    distributions: dict[str, Any] = {}
    if not sample.empty:
        sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
        for column in available:
            values = pd.to_numeric(sample[column], errors="coerce")
            finite = values[np.isfinite(values)]
            distributions[column] = {
                "n": int(len(finite)),
                "missing_share": float(values.isna().mean()),
                "zero_share_of_observed": float((finite == 0).mean()) if len(finite) else None,
                "min": float(finite.min()) if len(finite) else None,
                "p01": float(finite.quantile(0.01)) if len(finite) else None,
                "median": float(finite.median()) if len(finite) else None,
                "p99": float(finite.quantile(0.99)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
            }
            if column in _BOUNDS and len(finite):
                low, high = _BOUNDS[column]
                violations = int(((finite < low) | (finite > high)).sum())
                if violations:
                    findings.append(
                        Finding.at(8, f"{violations} sampled values outside [{low}, {high}]", "no theoretical-bound violations", field=column)
                    )

    correlation: dict[str, float] = {}
    drift: dict[str, float] = {}
    if not sample.empty and available:
        numeric = sample[available].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr(min_periods=100)
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                value = corr.at[left, right]
                if pd.notna(value) and abs(value) >= 0.90:
                    correlation[f"{left}__{right}"] = float(value)
                    if abs(value) >= 0.985:
                        findings.append(Finding.at(6, f"sample Pearson r={value:.4f}", "|r| < 0.985", field=f"{left} / {right}"))
        cutoff = sample["date"].dropna().sort_values().drop_duplicates()
        cutoff_date = cutoff.iloc[-252] if len(cutoff) >= 252 else (cutoff.iloc[0] if len(cutoff) else pd.NaT)
        recent = numeric[sample["date"] >= cutoff_date]
        history = numeric[sample["date"] < cutoff_date]
        for column in available:
            pooled = pd.concat([recent[column], history[column]]).std()
            if pd.notna(pooled) and pooled > 0 and recent[column].notna().any() and history[column].notna().any():
                drift[column] = float((recent[column].mean() - history[column].mean()) / pooled)

    metrics = {
        "coverage": coverage,
        "coverage_by_ticker": ratios,
        "distributions": distributions,
        "high_correlations": correlation,
        "recent_vs_history_standardized_mean": drift,
        "sample_rows": int(len(sample)),
    }
    scope = {
        "rows": int(len(sample)),
        "tickers": int(sample["ticker"].nunique()) if not sample.empty else 0,
        "first_date": sample["date"].min() if not sample.empty else None,
        "last_date": sample["date"].max() if not sample.empty else None,
        "feature_columns": len(available),
    }
    return CheckResult.measured(CHECK, spec.name, findings, scope=scope, metrics=metrics)
