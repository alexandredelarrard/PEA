"""Dedicated earnings-call coverage, source-grain, split-quality and feature-quality validation.

Reads `earnings_call_sections` at its paragraph grain (projected, 25 tickers per read), splits
every call with `src/utils/earnings_call_split.split_call` and applies the shared quality gate,
so the coverage it reports is the coverage the features can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import (
    EARNINGS_CALL_FEATURES,
    EARNINGS_CALL_SIGNAL_SESSIONS,
    EARNINGS_REPORT_TO_QUARTER_LAG_DAYS,
    NO_EARNINGS_CALL_TICKERS,
)
from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.utils.earnings_call_split import split_call
from src.utils.text_metrics import assess_earnings_call_sections, word_count
from src.validate.result import CheckResult, Finding, full_table_only

CHECK = "earnings_calls"
_PARAGRAPH_COLUMNS = ["ticker", "quarter", "paragraph", "as_of", "speaker", "content"]
_SPLIT_STATUSES = ("ok", "no_qa", "no_prepared", "empty")
_GRAIN_KEYS = (
    "rows",
    "calls",
    "duplicate_keys",
    "null_paragraph_rows",
    "null_as_of_rows",
    "calls_with_multiple_as_of",
    "calls_missing_first_paragraph",
    "calls_non_contiguous_paragraphs",
    "ticker_dates_with_multiple_calls",
)
_FLOOR_QUARTER = 2006 * 4  # quarter index of 2006Q1, the coverage floor


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


def _grain_counts(paragraphs: pd.DataFrame) -> dict[str, int]:
    """Paragraph-grain integrity of one read: unique (ticker, quarter, paragraph), one non-null
    `as_of` per call, paragraph numbers 1..n per call (the extractor diffs stored calls on
    paragraph 1 only, so a call without it is invisible to the re-issue check), and one call per
    (ticker, as_of) (a provider fiscal relabel can put a second label on a stored call's date).
    `paragraphs` must hold every call of its tickers: the reads are batched by ticker."""
    keyed = paragraphs.dropna(subset=["paragraph"])
    calls = keyed.groupby(["ticker", "quarter"], sort=False).agg(
        first=("paragraph", "min"),
        last=("paragraph", "max"),
        distinct=("paragraph", "nunique"),
    )
    dates = paragraphs.groupby(["ticker", "quarter"], sort=False)["as_of"].nunique()
    call_dates = (
        paragraphs.assign(as_of=pd.to_datetime(paragraphs["as_of"], errors="coerce").dt.normalize())
        .dropna(subset=["as_of"])
        .groupby(["ticker", "quarter"], sort=False)["as_of"]
        .first()
        .reset_index()
    )
    calls_per_date = call_dates.groupby(["ticker", "as_of"], sort=False)["quarter"].size()
    return {
        "rows": int(len(paragraphs)),
        "calls": int(paragraphs[["ticker", "quarter"]].drop_duplicates().shape[0]),
        "duplicate_keys": int(keyed.duplicated(["ticker", "quarter", "paragraph"]).sum()),
        "null_paragraph_rows": int(paragraphs["paragraph"].isna().sum()),
        "null_as_of_rows": int(paragraphs["as_of"].isna().sum()),
        "calls_with_multiple_as_of": int((dates > 1).sum()),
        "calls_missing_first_paragraph": int((calls["first"] != 1).sum()),
        "calls_non_contiguous_paragraphs": int((calls["last"] - calls["first"] + 1 != calls["distinct"]).sum()),
        "ticker_dates_with_multiple_calls": int((calls_per_date > 1).sum()),
    }


def _release_gap_buckets(gaps: list[int | None]) -> dict[str, int]:
    """Valid calls by `as_of` minus the nearest `earnings_surprises` release date of the issuer."""
    known = [gap for gap in gaps if gap is not None]
    return {
        "same_day": sum(gap == 0 for gap in known),
        "one_day": sum(abs(gap) == 1 for gap in known),
        "two_to_seven_days": sum(2 <= abs(gap) <= 7 for gap in known),
        "beyond_seven_days": sum(abs(gap) > 7 for gap in known),
        "no_release": len(gaps) - len(known),
    }


def _coverage_buckets(values: pd.Series) -> dict[str, Any]:
    return {
        "tickers_measured": int(len(values)),
        "coverage_100pct": int((values == 1.0).sum()),
        "coverage_70_to_lt100pct": int(((values >= 0.70) & (values < 1.0)).sum()),
        "coverage_50_to_lt70pct": int(((values >= 0.50) & (values < 0.70)).sum()),
        "coverage_lt50pct": int((values < 0.50).sum()),
        "median_coverage": float(values.median()) if len(values) else None,
    }


def _event_entity(tenure_by_symbol: dict[str, pd.DataFrame], symbol: str, date: object) -> str | None:
    """Resolve one event through the same half-open symbol-tenure rule as features."""
    value = pd.to_datetime(date, errors="coerce")
    candidates = tenure_by_symbol.get(symbol)
    if pd.isna(value) or candidates is None:
        return None
    candidates = candidates[candidates["valid_from"].le(value)]
    candidates = candidates[candidates["valid_to"].isna() | candidates["valid_to"].gt(value)]
    identifiers = candidates["entity_id"].dropna().astype(str).unique()
    return str(identifiers[0]) if len(identifiers) == 1 else None


@dataclass
class _Identity:
    """Point-in-time issuer identity: tenure rows (with `entity_id` when lineage resolves them),
    the current entity of each roster ticker, every symbol of an entity, tenure rows per symbol."""

    tenure: pd.DataFrame | None
    roster_entity: dict[str, str] = field(default_factory=dict)
    aliases_by_entity: dict[str, set[str]] = field(default_factory=dict)
    tenure_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class _CallScan:
    """Per-call split and quality-gate tallies of the paragraph source."""

    observed_calls: int = 0
    valid_calls: int = 0
    malformed: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_SPLIT_STATUSES, 0))
    prepared_shares: list[float] = field(default_factory=list)
    grain: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_GRAIN_KEYS, 0))
    valid_dates_by_ticker: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    call_dates_by_ticker: dict[str, list[pd.Timestamp]] = field(default_factory=dict)


@dataclass
class _TickerCoverage:
    """Per roster ticker coverage ratios and the per-quarter / release-gap tallies."""

    ratios: dict[str, float] = field(default_factory=dict)
    fixed_ratios: dict[str, float] = field(default_factory=dict)
    eligible_ratios: dict[str, float] = field(default_factory=dict)
    valid_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    tickers_by_quarter: dict[int, set[str]] = field(default_factory=dict)
    valid_tickers_by_quarter: dict[int, set[str]] = field(default_factory=dict)
    release_gap_days: list[int | None] = field(default_factory=list)


def _identity(tenure: pd.DataFrame | None, lineage: pd.DataFrame | None, measured_tickers: list[str]) -> _Identity:
    """Attach the lineage entity to every tenure row (CIKs with one entity only) and resolve the
    roster tickers' current entity; an empty identity when either table is missing."""
    if tenure is None or tenure.empty or lineage is None or lineage.empty:
        return _Identity(tenure)
    unique_lineage = lineage.dropna(subset=["cik", "entity_id"]).copy()
    unique_lineage["cik"] = unique_lineage["cik"].astype(str)
    unique_lineage = unique_lineage.groupby("cik", as_index=False).filter(lambda group: group["entity_id"].astype(str).nunique() == 1)
    tenure = tenure.assign(issuer_cik=tenure["issuer_cik"].astype(str)).merge(
        unique_lineage[["cik", "entity_id"]].drop_duplicates("cik"),
        left_on="issuer_cik",
        right_on="cik",
        how="left",
    )
    tenure["valid_from"] = pd.to_datetime(tenure["valid_from"], errors="coerce")
    tenure["valid_to"] = pd.to_datetime(tenure["valid_to"], errors="coerce")
    identity = _Identity(tenure)
    identity.tenure_by_symbol = {str(symbol): group for symbol, group in tenure.groupby(tenure["symbol"].astype(str), sort=False)}
    for entity_id, group in tenure.dropna(subset=["entity_id"]).groupby("entity_id", sort=False):
        identity.aliases_by_entity[str(entity_id)] = set(group["symbol"].astype(str))
    today = pd.Timestamp.today().normalize()
    valid_from = pd.to_datetime(tenure["valid_from"], errors="coerce")
    valid_to = pd.to_datetime(tenure["valid_to"], errors="coerce")
    active = tenure[valid_from.le(today) & (valid_to.isna() | valid_to.gt(today))]
    for ticker in measured_tickers:
        entities = active.loc[active["symbol"].astype(str).eq(ticker), "entity_id"]
        if entities.notna().all() and entities.astype(str).nunique() == 1:
            identity.roster_entity[ticker] = str(entities.iloc[0])
    return identity


def _release_dates_by_ticker(releases: pd.DataFrame | None) -> dict[str, list[pd.Timestamp]]:
    """Past `earnings_surprises` release dates per ticker (future releases excluded)."""
    if releases is None:
        return {}
    release_date = pd.to_datetime(releases["earnings_date"], errors="coerce")
    releases = releases[release_date.le(pd.Timestamp.today().normalize())]
    dates_by_ticker: dict[str, list[pd.Timestamp]] = {}
    for ticker, group in releases.groupby("ticker", sort=False):
        dates = pd.to_datetime(group["earnings_date"], errors="coerce").dropna().tolist()
        if dates:
            dates_by_ticker[str(ticker)] = [pd.Timestamp(date) for date in dates]
    return dates_by_ticker


def _rejection_reason(scan: _CallScan, call: pd.DataFrame) -> str | None:
    """Split one call and apply the quality gate; None when the call is valid."""
    split = split_call(call[["paragraph", "speaker", "content"]].to_dict("records"))
    scan.status_counts[split.status] += 1
    if split.status != "ok":
        return f"split status {split.status}"
    prepared_words, qa_words = word_count(split.prepared_remarks), word_count(split.qa)
    scan.prepared_shares.append(prepared_words / (prepared_words + qa_words))
    quality = assess_earnings_call_sections({"prepared_remarks": split.prepared_remarks, "qa": split.qa})
    return None if quality.valid else quality.reason or "unknown"


def _reason_bucket(reason: str) -> str:
    """Pool the per-call word-count and missing-section messages into one bucket each."""
    if reason.startswith("only "):
        return "below minimum cleaned words"
    if reason.startswith("missing sections:"):
        return "missing required section"
    return reason


def _scan_call(scan: _CallScan, ticker: str, call: pd.DataFrame) -> None:
    """Record one call's date, split status and validity."""
    scan.observed_calls += 1
    call_date = pd.to_datetime(call["as_of"], errors="coerce").dropna()
    date = pd.Timestamp(call_date.iloc[0]) if not call_date.empty else None
    if date is not None:
        scan.call_dates_by_ticker.setdefault(ticker, []).append(date)
    reason = _rejection_reason(scan, call)
    if reason is None:
        scan.valid_calls += 1
        if date is not None:
            scan.valid_dates_by_ticker.setdefault(ticker, []).append(date)
        return
    scan.malformed += 1
    bucket = _reason_bucket(reason)
    scan.rejected_reasons[bucket] = scan.rejected_reasons.get(bucket, 0) + 1


def _scan_calls(context: Context, symbols: list[str]) -> _CallScan:
    """Grain counts and per-call tallies over `earnings_call_sections`, 25 tickers per read."""
    scan = _CallScan()
    for start in range(0, len(symbols), 25):
        paragraphs = context.store.load(
            Tables.earnings_call_sections,
            columns=_PARAGRAPH_COLUMNS,
            where={"ticker": symbols[start : start + 25]},
            order_by=["ticker", "quarter", "paragraph"],
            optional=True,
        )
        if paragraphs is None:
            continue
        for key, value in _grain_counts(paragraphs).items():
            scan.grain[key] += value
        for (ticker, _quarter), call in paragraphs.groupby(["ticker", "quarter"], sort=False):
            _scan_call(scan, str(ticker), call)
    return scan


def _linked(dates_by_symbol: dict[str, list[pd.Timestamp]], symbols: set[str], entity_id: str | None, identity: _Identity) -> list[pd.Timestamp]:
    """Event dates of the roster ticker's symbols that belong to its issuer at the event date."""
    return [
        date
        for symbol in symbols
        for date in dates_by_symbol.get(symbol, [])
        if entity_id is None or _event_entity(identity.tenure_by_symbol, symbol, date) == entity_id
    ]


def _ticker_tenure(tenure: pd.DataFrame | None, ticker: str, entity_id: str | None) -> pd.DataFrame:
    """Tenure rows of the ticker's issuer (entity when resolved, else the symbol itself)."""
    if tenure is None:
        return pd.DataFrame()
    if entity_id is None or "entity_id" not in tenure:
        return tenure[tenure["symbol"].astype(str) == ticker]
    return tenure[tenure["entity_id"].astype(str) == entity_id]


def _eligible_quarters(ticker_tenure: pd.DataFrame, bounds: tuple[int, int]) -> set[int]:
    """Calendar quarters inside the issuer's tenure, the floor and the last release; the release
    window when the tenure yields none."""
    eligible: set[int] = set()
    for row in ticker_tenure.itertuples(index=False):
        start = _calendar_quarter_index(row.valid_from)
        end_date = pd.to_datetime(row.valid_to, errors="coerce")
        end = bounds[1] if pd.isna(end_date) else _calendar_quarter_index(end_date - pd.Timedelta(days=1))
        if start is not None and end is not None:
            eligible.update(range(max(_FLOOR_QUARTER, start), min(bounds[1], end) + 1))
    if not eligible:
        eligible.update(range(bounds[0], bounds[1] + 1))
    return eligible


def _measure_ticker(
    coverage: _TickerCoverage, ticker: str, identity: _Identity, scan: _CallScan, release_dates_by_ticker: dict[str, list[pd.Timestamp]]
) -> None:
    """Per-quarter presence, release gaps and the three coverage ratios of one roster ticker."""
    entity_id = identity.roster_entity.get(ticker)
    symbols = identity.aliases_by_entity.get(entity_id, {ticker}) if entity_id is not None else {ticker}
    for date in _linked(scan.call_dates_by_ticker, symbols, entity_id, identity):
        coverage.tickers_by_quarter.setdefault(cast(int, _calendar_quarter_index(date)), set()).add(ticker)
    ticker_valid_dates = _linked(scan.valid_dates_by_ticker, symbols, entity_id, identity)
    for date in ticker_valid_dates:
        coverage.valid_tickers_by_quarter.setdefault(cast(int, _calendar_quarter_index(date)), set()).add(ticker)
    release_dates = _linked(release_dates_by_ticker, symbols, entity_id, identity)
    for date in ticker_valid_dates:
        coverage.release_gap_days.append(min(((date - release).days for release in release_dates), key=abs) if release_dates else None)
    indices = [index for index in (_quarter_index(date) for date in release_dates) if index is not None]
    bounds = (max(_FLOOR_QUARTER, min(indices)), max(indices)) if indices else None
    if bounds is None or bounds[1] < bounds[0]:
        return
    coverage.valid_dates[ticker] = ticker_valid_dates
    valid_indices = {index for index in (_quarter_index(date) for date in ticker_valid_dates) if index is not None}
    present = sum(bounds[0] <= index <= bounds[1] for index in valid_indices)
    coverage.ratios[ticker] = present / (bounds[1] - bounds[0] + 1)
    fixed_present = sum(_FLOOR_QUARTER <= index <= bounds[1] for index in valid_indices)
    coverage.fixed_ratios[ticker] = fixed_present / (bounds[1] - _FLOOR_QUARTER + 1)
    eligible_quarters = _eligible_quarters(_ticker_tenure(identity.tenure, ticker, entity_id), bounds)
    present_eligible = sum(index in eligible_quarters for index in valid_indices)
    coverage.eligible_ratios[ticker] = present_eligible / len(eligible_quarters) if eligible_quarters else np.nan


def _coverage_summary(roster_tickers: list[str], measured_tickers: list[str], scan: _CallScan, coverage: _TickerCoverage) -> dict[str, Any]:
    """The coverage metrics block of the check."""
    structural_no_call = sorted(set(roster_tickers) & set(NO_EARNINGS_CALL_TICKERS))
    shares = pd.Series(scan.prepared_shares, dtype="float64")
    by_quarter = [
        {
            "quarter": f"{index // 4}Q{index % 4 + 1}",
            "tickers_with_call": len(coverage.tickers_by_quarter.get(index, set())),
            "tickers_with_valid_call": len(coverage.valid_tickers_by_quarter.get(index, set())),
            "roster_measured": len(measured_tickers),
            "share_with_call": len(coverage.tickers_by_quarter.get(index, set())) / len(measured_tickers) if measured_tickers else None,
        }
        for index in sorted(set(coverage.tickers_by_quarter) | set(coverage.valid_tickers_by_quarter))
    ]
    return {
        "denominator": "current roster tickers with earnings calls; call and release dates mapped to calendar reporting quarters and point-in-time issuer identity; predecessor symbols linked through CIK lineage; future releases excluded; floored at 2006Q1",
        "roster_tickers": len(roster_tickers),
        "structural_no_call_count": len(structural_no_call),
        "structural_no_call_tickers": structural_no_call,
        **_coverage_buckets(pd.Series(coverage.ratios, dtype="float64")),
        "fixed_since_2006": _coverage_buckets(pd.Series(coverage.fixed_ratios, dtype="float64").dropna()),
        "issuer_linked_insider_filing_tenure_sensitivity": _coverage_buckets(pd.Series(coverage.eligible_ratios, dtype="float64").dropna()),
        "tenure_sensitivity_note": "symbol_tenure starts at the first observed insider filing, not listing; use only as a sensitivity, not company-life coverage",
        "observed_calls": scan.observed_calls,
        "valid_calls": scan.valid_calls,
        "malformed_calls": scan.malformed,
        "rejected_by_reason": scan.rejected_reasons,
        "split_status_counts": scan.status_counts,
        "split_ok_rate": scan.status_counts["ok"] / scan.observed_calls if scan.observed_calls else None,
        "prepared_share_quantiles": {f"q{round(q * 100):02d}": float(shares.quantile(q)) for q in (0.05, 0.5, 0.95)} if len(shares) else {},
        "grain": scan.grain,
        "coverage_by_calendar_quarter": by_quarter,
        "as_of_vs_release_days": _release_gap_buckets(coverage.release_gap_days),
    }


def _coverage(context: Context) -> tuple[dict[str, Any], dict[str, float], dict[str, list[pd.Timestamp]]]:
    """(coverage summary, coverage ratio per roster ticker, valid call dates per roster ticker)."""
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker"])
    releases = context.store.load(Tables.earnings_surprises, columns=["ticker", "earnings_date"], optional=True)
    tenure = context.store.load(
        Tables.symbol_tenure,
        columns=["symbol", "issuer_cik", "valid_from", "valid_to"],
        optional=True,
    )
    lineage = context.store.load(Tables.entity_lineage, columns=["cik", "entity_id"], optional=True)
    assert roster is not None
    roster_tickers = roster["ticker"].astype(str).tolist()
    measured_tickers = [ticker for ticker in roster_tickers if ticker not in NO_EARNINGS_CALL_TICKERS]
    identity = _identity(tenure, lineage, measured_tickers)
    analysis_symbols = sorted(
        set(measured_tickers) | {symbol for entity in identity.roster_entity.values() for symbol in identity.aliases_by_entity.get(entity, set())}
    )
    release_dates_by_ticker = _release_dates_by_ticker(releases)
    scan = _scan_calls(context, analysis_symbols)
    coverage = _TickerCoverage()
    for ticker in measured_tickers:
        _measure_ticker(coverage, ticker, identity, scan, release_dates_by_ticker)
    return _coverage_summary(roster_tickers, measured_tickers, scan, coverage), coverage.ratios, coverage.valid_dates


def _source_findings(coverage: dict[str, Any], settings: Any) -> list[Finding]:
    """Grain and split-quality findings on the paragraph source."""
    grain = coverage["grain"]
    findings: list[Finding] = []
    for key, score, expected in (
        ("duplicate_keys", 9, "unique (ticker, quarter, paragraph)"),
        ("null_paragraph_rows", 9, "every row carries its source paragraph number"),
        ("null_as_of_rows", 9, "every paragraph carries the call date"),
        ("calls_with_multiple_as_of", 8, "one as_of per call"),
        ("calls_missing_first_paragraph", 6, "every call starts at paragraph 1"),
        ("calls_non_contiguous_paragraphs", 3, "paragraphs 1..n per call"),
        ("ticker_dates_with_multiple_calls", 8, "one call per (ticker, as_of)"),
    ):
        if grain[key]:
            findings.append(Finding.at(score, f"{key}={grain[key]} of {grain['calls']} calls / {grain['rows']} rows", expected, field=key))
    ok_rate, ok_min = coverage["split_ok_rate"], float(settings.split_ok_rate_min)
    if ok_rate is not None and ok_rate < ok_min:
        findings.append(
            Finding.at(7, f"split ok rate={ok_rate:.4f}", f">= {ok_min}", field="split_status", status_counts=coverage["split_status_counts"])
        )
    q05, q05_min = coverage["prepared_share_quantiles"].get("q05"), float(settings.prepared_share_q05_min)
    if q05 is not None and q05 < q05_min:
        findings.append(Finding.at(6, f"prepared word share q05={q05:.4f}", f">= {q05_min}", field="prepared_share"))
    return findings


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


def _schema_findings(expected: list[str], live: set[str], settings: Any) -> list[Finding]:
    """Signal-lifetime agreement and the exact earnings-call column set."""
    missing = sorted(set(expected) - live)
    legacy = sorted(column for column in live if column.startswith("f_ec_") and column not in expected)
    findings: list[Finding] = []
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
    return findings


def _distribution(values: pd.Series) -> dict[str, Any]:
    """Observed-value distribution of one sampled feature column."""
    finite = values[np.isfinite(values)]
    return {
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


def _distributions(sample: pd.DataFrame, available: list[str], bounds: Any) -> tuple[dict[str, Any], list[Finding]]:
    """Per-column distributions of the sample and its theoretical-bound violations."""
    distributions: dict[str, Any] = {}
    findings: list[Finding] = []
    for column in available:
        values = pd.to_numeric(sample[column], errors="coerce")
        distributions[column] = _distribution(values)
        finite = values[np.isfinite(values)]
        if column not in bounds or not len(finite):
            continue
        low, high = bounds[column]
        violations = int(((finite < low) | (finite > high)).sum())
        if violations:
            findings.append(Finding.at(8, f"{violations} sampled values outside [{low}, {high}]", "no theoretical-bound violations", field=column))
    return distributions, findings


def _redundancy(numeric: pd.DataFrame, available: list[str], settings: Any) -> tuple[dict[str, float], list[Finding]]:
    """Sample Spearman pairs at or above the report limit; a finding at or above the fail limit."""
    redundancy_limit = float(settings.redundancy_spearman_abs)
    redundancy_fail = float(settings.redundancy_fail_abs)
    corr = numeric.corr(method="spearman", min_periods=100)
    correlation: dict[str, float] = {}
    findings: list[Finding] = []
    for left, right in combinations(available, 2):
        value = corr.at[left, right]
        if pd.isna(value) or abs(value) < redundancy_limit:
            continue
        correlation[f"{left}__{right}"] = float(value)
        if abs(value) >= redundancy_fail:
            findings.append(Finding.at(6, f"sample Spearman rho={value:.4f}", f"|rho| < {redundancy_fail}", field=f"{left} / {right}"))
    return correlation, findings


def _column_drift(recent: pd.Series, history: pd.Series) -> dict[str, float | None]:
    """Standardized mean shift, PSI and missingness delta of recent vs training-period values."""
    pooled = pd.concat([recent, history]).std()
    mean_shift = None
    if pd.notna(pooled) and pooled > 0 and recent.notna().any() and history.notna().any():
        mean_shift = float((recent.mean() - history.mean()) / pooled)
    return {
        "standardized_mean_shift": mean_shift,
        "psi": _population_stability_index(history, recent),
        "missingness_delta": float(recent.isna().mean() - history.isna().mean()),
    }


def _drift(sample: pd.DataFrame, numeric: pd.DataFrame, available: list[str], train_end: pd.Timestamp, settings: Any) -> tuple[dict, list[Finding]]:
    """Train-vs-recent drift per column, split at `train_end`, with PSI and missingness findings."""
    recent_mask = sample["date"] > train_end
    recent = numeric[recent_mask]
    history = numeric[~recent_mask]
    psi_limit = float(settings.drift_psi_warn)
    missing_limit = float(settings.missingness_delta_warn)
    drift: dict[str, dict[str, float | None]] = {}
    findings: list[Finding] = []
    for column in available:
        drift[column] = _column_drift(recent[column], history[column])
        psi, missing_delta = drift[column]["psi"], cast(float, drift[column]["missingness_delta"])
        if psi is not None and psi > psi_limit:
            findings.append(Finding.at(7, f"PSI={psi:.4f}", f"PSI <= {psi_limit}", field=column))
        if abs(missing_delta) > missing_limit:
            findings.append(Finding.at(7, f"recent-minus-train missingness={missing_delta:.4f}", f"absolute delta <= {missing_limit}", field=column))
    return drift, findings


def _source_ages(group: pd.DataFrame, numeric: pd.DataFrame, valid_dates: dict[str, list[pd.Timestamp]], date: pd.Timestamp) -> list[int]:
    """Days since the latest valid call before `date`, per row carrying any feature value."""
    ages = []
    for row in group[numeric.notna().any(axis=1)].itertuples(index=False):
        prior = [value for value in valid_dates.get(str(row.ticker), []) if value < date]
        if prior:
            ages.append((date - max(prior)).days)
    return ages


def _latest_metrics(latest: pd.DataFrame, available: list[str], valid_dates: dict[str, list[pd.Timestamp]]) -> list[dict[str, Any]]:
    """Null / zero cells and source age on each of the most recent sessions."""
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    metrics: list[dict[str, Any]] = []
    for date, group in latest.groupby("date", sort=True):
        numeric = group[available].apply(pd.to_numeric, errors="coerce")
        ages = _source_ages(group, numeric, valid_dates, pd.Timestamp(date))
        metrics.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "rows": int(len(group)),
                "null_cells": int(numeric.isna().sum().sum()),
                "zero_cells": int((numeric == 0).sum().sum()),
                "source_age_days_median": float(np.median(ages)) if ages else None,
                "source_age_days_max": int(max(ages)) if ages else None,
            }
        )
    return metrics


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
    settings = _settings(config)
    findings = _schema_findings(expected, live, settings)
    coverage, ratios, valid_dates = _coverage(context)
    findings.extend(_source_findings(coverage, settings))
    available = [column for column in expected if column in live]
    recent_sessions = int(settings.recent_sessions)
    sample, latest = _feature_sample(context, spec, available, recent_sessions) if available else (pd.DataFrame(), pd.DataFrame())

    distributions: dict[str, Any] = {}
    correlation: dict[str, float] = {}
    drift: dict[str, dict[str, float | None]] = {}
    if not sample.empty:
        sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
        distributions, bound_findings = _distributions(sample, available, config.validate.tables[spec.name].get("bounds", {}))
        numeric = sample[available].apply(pd.to_numeric, errors="coerce")
        correlation, redundancy_findings = _redundancy(numeric, available, settings)
        drift, drift_findings = _drift(sample, numeric, available, pd.Timestamp(str(config.train.end_date)), settings)
        findings += bound_findings + redundancy_findings + drift_findings
    latest_metrics = _latest_metrics(latest, available, valid_dates) if not latest.empty and available else []

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
