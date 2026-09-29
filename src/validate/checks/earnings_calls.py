"""Dedicated earnings-call coverage and feature-quality validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import (
    EARNINGS_CALL_FEATURES,
    EARNINGS_CALL_SCORED_TAGS,
    EARNINGS_CALL_SIGNAL_SESSIONS,
    EARNINGS_REPORT_TO_QUARTER_LAG_DAYS,
)
from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.utils.text_metrics import assess_earnings_call_sections
from src.validate.result import CheckResult, Finding, full_table_only

CHECK = "earnings_calls"


def _settings(config: Any) -> Any:
    validate = getattr(config, "validate", None)
    if validate is None or getattr(validate, "earnings_calls", None) is None:
        raise ValueError("validate.earnings_calls must be declared in configs/validate.yml")
    return validate.earnings_calls


def _population_stability_index(history: pd.Series, recent: pd.Series) -> float | None:
    """PSI on history-decile bins; missingness is measured separately."""
    left = pd.to_numeric(history, errors="coerce").dropna()
    right = pd.to_numeric(recent, errors="coerce").dropna()
    if len(left) < 100 or len(right) < 100 or left.nunique() < 2:
        return None
    edges = np.unique(left.quantile(np.linspace(0, 1, 11)).to_numpy(float))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    expected = pd.cut(left, edges, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy(float)
    actual = pd.cut(right, edges, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy(float)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _quarter_index(date: object) -> int | None:
    value = pd.to_datetime(date, errors="coerce")
    if pd.isna(value):
        return None
    value -= pd.Timedelta(days=EARNINGS_REPORT_TO_QUARTER_LAG_DAYS)
    return int(value.year) * 4 + int(value.quarter) - 1


def _calendar_quarter_index(date: object) -> int | None:
    value = pd.to_datetime(date, errors="coerce")
    return None if pd.isna(value) else int(value.year) * 4 + int(value.quarter) - 1


def _coverage_buckets(values: pd.Series) -> dict[str, Any]:
    return {
        "tickers_measured": int(len(values)),
        "coverage_100pct": int((values == 1.0).sum()),
        "coverage_70_to_lt100pct": int(((values >= 0.70) & (values < 1.0)).sum()),
        "coverage_50_to_lt70pct": int(((values >= 0.50) & (values < 0.70)).sum()),
        "coverage_lt50pct": int((values < 0.50).sum()),
        "median_coverage": float(values.median()) if len(values) else None,
    }


def _coverage(context: Context) -> tuple[dict[str, Any], dict[str, float], dict[str, list[pd.Timestamp]]]:
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker"])
    releases = context.store.load(Tables.earnings_surprises, columns=["ticker", "earnings_date"], optional=True)
    tenure = context.store.load(
        Tables.symbol_tenure,
        columns=["symbol", "valid_from", "valid_to"],
        optional=True,
    )
    assert roster is not None
    release_ranges: dict[str, tuple[int, int]] = {}
    if releases is not None:
        release_date = pd.to_datetime(releases["earnings_date"], errors="coerce")
        releases = releases[release_date.le(pd.Timestamp.today().normalize())]
        for ticker, group in releases.groupby("ticker", sort=False):
            indices = [value for value in group["earnings_date"].map(_quarter_index) if value is not None]
            if indices:
                release_ranges[str(ticker)] = (max(2006 * 4, min(indices)), max(indices))

    ratios: dict[str, float] = {}
    fixed_ratios: dict[str, float] = {}
    eligible_ratios: dict[str, float] = {}
    malformed = valid_calls = observed_calls = 0
    rejected_reasons: dict[str, int] = {}
    valid_dates: dict[str, list[pd.Timestamp]] = {}
    roster_tickers = roster["ticker"].astype(str).tolist()
    valid_by_ticker: dict[str, set[str]] = {ticker: set() for ticker in roster_tickers}
    section_columns = ["ticker", "quarter", "tag", "text"]
    if "as_of" in context.store.columns(Tables.earnings_call_sections):
        section_columns.insert(3, "as_of")
    for start in range(0, len(roster_tickers), 25):
        batch = roster_tickers[start : start + 25]
        sections = context.store.load(
            Tables.earnings_call_sections,
            columns=section_columns,
            where={"ticker": batch, "tag": list(EARNINGS_CALL_SCORED_TAGS)},
            optional=True,
        )
        if sections is not None:
            for (ticker, quarter), call in sections.groupby(["ticker", "quarter"], sort=False):
                observed_calls += 1
                quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
                if quality.valid:
                    valid_by_ticker[str(ticker)].add(str(quarter))
                    valid_calls += 1
                    if "as_of" in call:
                        call_date = pd.to_datetime(call["as_of"], errors="coerce").dropna()
                        if not call_date.empty:
                            valid_dates.setdefault(str(ticker), []).append(pd.Timestamp(call_date.iloc[0]))
                else:
                    malformed += 1
                    reason = quality.reason or "unknown"
                    if reason.startswith("only "):
                        reason = "below minimum cleaned words"
                    elif reason.startswith("missing sections:"):
                        reason = "missing required section"
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for ticker in roster_tickers:
        valid = valid_by_ticker[ticker]
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

        fixed_start = 2006 * 4
        fixed_present = sum(
            fixed_start <= int(quarter[:4]) * 4 + int(quarter[5]) - 1 <= bounds[1]
            for quarter in valid
            if len(quarter) == 6 and quarter[4] == "Q" and quarter[:4].isdigit() and quarter[5] in "1234"
        )
        fixed_ratios[ticker] = fixed_present / (bounds[1] - fixed_start + 1)

        eligible_quarters: set[int] = set()
        ticker_tenure = pd.DataFrame() if tenure is None else tenure[tenure["symbol"].astype(str) == ticker]
        for row in ticker_tenure.itertuples(index=False):
            start = _calendar_quarter_index(row.valid_from)
            end_date = pd.to_datetime(row.valid_to, errors="coerce")
            end = bounds[1] if pd.isna(end_date) else _calendar_quarter_index(end_date - pd.Timedelta(days=1))
            if start is not None and end is not None:
                eligible_quarters.update(range(max(fixed_start, start), min(bounds[1], end) + 1))
        if not eligible_quarters:
            eligible_quarters.update(range(bounds[0], bounds[1] + 1))
        present_eligible = sum(
            int(quarter[:4]) * 4 + int(quarter[5]) - 1 in eligible_quarters
            for quarter in valid
            if len(quarter) == 6 and quarter[4] == "Q" and quarter[:4].isdigit() and quarter[5] in "1234"
        )
        eligible_ratios[ticker] = present_eligible / len(eligible_quarters) if eligible_quarters else np.nan

    values = pd.Series(ratios, dtype="float64")
    fixed_values = pd.Series(fixed_ratios, dtype="float64").dropna()
    eligible_values = pd.Series(eligible_ratios, dtype="float64").dropna()
    summary = {
        "denominator": "current roster; expected quarters from first to latest recorded earnings release, floored at 2006Q1",
        **_coverage_buckets(values),
        "fixed_since_2006": _coverage_buckets(fixed_values),
        "symbol_tenure_eligible_life": _coverage_buckets(eligible_values),
        "observed_calls": observed_calls,
        "valid_calls": valid_calls,
        "malformed_calls": malformed,
        "rejected_by_reason": rejected_reasons,
    }
    return summary, ratios, valid_dates


def _feature_sample(context: Context, table: Table, columns: list[str], recent_sessions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = []
    latest = pd.DataFrame(columns=["date", "ticker", *columns])
    for number, chunk in enumerate(context.store.iter_load(table, columns=["date", "ticker", *columns], chunksize=100_000)):
        samples.append(chunk.sample(n=min(5_000, len(chunk)), random_state=17 + number))
        combined = pd.concat([latest, chunk], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        dates = combined["date"].dropna().drop_duplicates().sort_values().tail(recent_sessions)
        latest = combined[combined["date"].isin(dates)].copy()
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=["date", "ticker", *columns])
    return sample, latest


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
    del cache, out
    spec = resolve(table)
    subset = full_table_only(CHECK, spec.name, tickers)
    if subset is not None:
        return subset

    expected = [f"f_{name}" for name in EARNINGS_CALL_FEATURES]
    live = set(context.store.columns(spec))
    missing = sorted(set(expected) - live)
    legacy = sorted(column for column in live if column.startswith("f_ec_") and column not in expected)
    findings: list[Finding] = []
    settings = _settings(config)
    configured_sessions = int(settings.signal_sessions)
    if configured_sessions != EARNINGS_CALL_SIGNAL_SESSIONS:
        findings.append(
            Finding.at(
                9,
                f"runtime={EARNINGS_CALL_SIGNAL_SESSIONS}; validation config={configured_sessions}",
                "one shared 66-session signal lifetime",
            )
        )
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

    coverage, ratios, valid_dates = _coverage(context)
    available = [column for column in expected if column in live]
    recent_sessions = int(settings.recent_sessions)
    sample, latest = _feature_sample(context, spec, available, recent_sessions) if available else (pd.DataFrame(), pd.DataFrame())
    distributions: dict[str, Any] = {}
    if not sample.empty:
        sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
        for column in available:
            values = pd.to_numeric(sample[column], errors="coerce")
            finite = values[np.isfinite(values)]
            distributions[column] = {
                "count": int(len(finite)),
                "null_count": int(values.isna().sum()),
                "missing_share": float(values.isna().mean()),
                "genuine_zero_count": int((finite == 0).sum()),
                "zero_share_of_observed": float((finite == 0).mean()) if len(finite) else None,
                "min": float(finite.min()) if len(finite) else None,
                "p01": float(finite.quantile(0.01)) if len(finite) else None,
                "median": float(finite.median()) if len(finite) else None,
                "p99": float(finite.quantile(0.99)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
                "mean": float(finite.mean()) if len(finite) else None,
                "std": float(finite.std()) if len(finite) else None,
            }
            bounds = config.validate.tables[spec.name].get("bounds", {})
            if column in bounds and len(finite):
                low, high = bounds[column]
                violations = int(((finite < low) | (finite > high)).sum())
                if violations:
                    findings.append(
                        Finding.at(8, f"{violations} sampled values outside [{low}, {high}]", "no theoretical-bound violations", field=column)
                    )

    correlation: dict[str, float] = {}
    drift: dict[str, dict[str, float | None]] = {}
    if not sample.empty and available:
        numeric = sample[available].apply(pd.to_numeric, errors="coerce")
        redundancy_limit = float(settings.redundancy_spearman_abs)
        redundancy_fail = float(settings.redundancy_fail_abs)
        corr = numeric.corr(method="spearman", min_periods=100)
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                value = corr.at[left, right]
                if pd.notna(value) and abs(value) >= redundancy_limit:
                    correlation[f"{left}__{right}"] = float(value)
                    if abs(value) >= redundancy_fail:
                        findings.append(
                            Finding.at(
                                6,
                                f"sample Spearman rho={value:.4f}",
                                f"|rho| < {redundancy_fail}",
                                field=f"{left} / {right}",
                            )
                        )
        train_end = pd.Timestamp(str(config.train.end_date))
        recent_mask = sample["date"] > train_end
        recent = numeric[recent_mask]
        history = numeric[~recent_mask]
        psi_limit = float(settings.drift_psi_warn)
        missing_limit = float(settings.missingness_delta_warn)
        for column in available:
            pooled = pd.concat([recent[column], history[column]]).std()
            mean_shift = None
            if pd.notna(pooled) and pooled > 0 and recent[column].notna().any() and history[column].notna().any():
                mean_shift = float((recent[column].mean() - history[column].mean()) / pooled)
            psi = _population_stability_index(history[column], recent[column])
            missing_delta = float(recent[column].isna().mean() - history[column].isna().mean())
            drift[column] = {
                "standardized_mean_shift": mean_shift,
                "psi": psi,
                "missingness_delta": missing_delta,
            }
            if psi is not None and psi > psi_limit:
                findings.append(Finding.at(7, f"PSI={psi:.4f}", f"PSI <= {psi_limit}", field=column))
            if abs(missing_delta) > missing_limit:
                findings.append(
                    Finding.at(
                        7,
                        f"recent-minus-train missingness={missing_delta:.4f}",
                        f"absolute delta <= {missing_limit}",
                        field=column,
                    )
                )

    latest_metrics: list[dict[str, Any]] = []
    if not latest.empty and available:
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
        for date, group in latest.groupby("date", sort=True):
            numeric = group[available].apply(pd.to_numeric, errors="coerce")
            ages = []
            active = group[numeric.notna().any(axis=1)]
            for row in active.itertuples(index=False):
                prior = [value for value in valid_dates.get(str(row.ticker), []) if value < pd.Timestamp(date)]
                if prior:
                    ages.append((pd.Timestamp(date) - max(prior)).days)
            latest_metrics.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "rows": int(len(group)),
                    "null_cells": int(numeric.isna().sum().sum()),
                    "zero_cells": int((numeric == 0).sum().sum()),
                    "source_age_days_median": float(np.median(ages)) if ages else None,
                    "source_age_days_max": int(max(ages)) if ages else None,
                }
            )

    metrics = {
        "coverage": coverage,
        "coverage_by_ticker": ratios,
        "distributions": distributions,
        "high_spearman_correlations": correlation,
        "train_vs_recent": drift,
        "train_end": str(config.train.end_date),
        "latest_sessions": latest_metrics,
        "signal_sessions": EARNINGS_CALL_SIGNAL_SESSIONS,
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
