"""
fundamentals_audit.py  (src/utils/fundamentals_audit.py)
-----------------------------------------------------------
Top-level orchestrator for the `fundamentals_history` extraction-cleanliness audit.
Composes the four signal sources that already exist as separate, single-purpose
diagnostics -- rather than reinventing any of their logic -- into two things neither
had on its own:

  1. A Tiingo -> Yahoo -> "no external validation" FALLBACK CHAIN across the whole
     analysis universe (`tiingo_comparison.run_tiingo_audit`, then
     `yahoo_comparison.run_non_tiingo_audit` for whatever Tiingo's plan didn't cover --
     confirmed live to be gated well below the full universe, see
     `tiingo_comparison.fetch_tiingo_statements`'s docstring). A ticker covered by
     NEITHER source is logged to `no_external_validation.csv`, never treated as a
     failure (per design: absence of an external check is informational, not
     blocking).
  2. ONE ranked review queue combining that external cross-check with the internal,
     full-universe, free diagnostics (`fundamentals_tag_ledger.detect_tag_switch_breaks`,
     `analyze_history.detect_level_outliers`/`run_audit`) -- the gap
     `fundamentals_tag_ledger.py`'s own docstring names explicitly: "Neither existing
     check carries a MAGNITUDE, so their output cannot be ranked ... 675 rows that all
     looked equally urgent."

Priority is a documented HEURISTIC, not a calibrated statistic (the four sources report
fundamentally different units -- a percent delta, a MAD z-score, a level ratio): sort
first by how many INDEPENDENT sources flag the same (ticker, field, fiscal_year,
fiscal_period) cell (agreement across sources is the strongest signal this pass can
produce without new data), then by a 0-100 per-source severity score, then by the tag
ledger's own systemic-vs-one-off signal (`n_tickers_same_switch`). Good enough to work a
queue worst-first; not a substitute for reading the actual filing before writing a
correction (see the plan's verification step).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants.constants import (
    FUNDAMENTALS_FINDINGS_RANKED_FILENAME, NO_EXTERNAL_VALIDATION_FILENAME,
    TIINGO_EXACT_MATCH_TOLERANCE_FLOW, TIINGO_EXACT_MATCH_TOLERANCE_LEVEL,
    YAHOO_EXACT_MATCH_TOLERANCE_FLOW, YAHOO_EXACT_MATCH_TOLERANCE_LEVEL,
)
from src.context import Context
from src.utils import tiingo_comparison, yahoo_comparison
from src.utils.analyze_history import DEFAULT_AUDIT_FIELDS, run_audit
from src.utils.fundamentals_tag_ledger import build_tag_ledger, detect_tag_switch_breaks

_LOG: logging.Logger = logging.getLogger(__name__)

__all__ = ["run_universe_audit", "build_ranked_findings", "run_full_audit"]

FINDINGS_COLS = ["source", "ticker", "field", "fiscal_year", "fiscal_period",
                 "severity", "n_tickers_same_switch", "agreement_count",
                 "priority_score", "detail"]


def run_universe_audit(
    context: Context, tickers: list[str] | None = None, *, api_key: str,
    threshold: float = 3.5, tiingo_cache_dir: Path | None = None,
    yahoo_cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Tiingo first (deeper history, when its plan covers the ticker), Yahoo for
    whatever Tiingo returned no data for (shallower ~4-5Q depth, broader coverage),
    and a plain list for whatever neither source has -- one bad/uncovered ticker never
    blocks the rest of the universe, same guarantee each individual audit already
    gives per-ticker."""
    if tickers is None:
        from src.utils.universe import load_universe_tickers
        tickers = load_universe_tickers(context)
    tickers = list(tickers)

    tiingo_result = tiingo_comparison.run_tiingo_audit(
        context, tickers, threshold=threshold, api_key=api_key, cache_dir=tiingo_cache_dir)
    tiingo_covered = (set(tiingo_result["comparison"]["ticker"].unique())
                      if not tiingo_result["comparison"].empty else set())
    _LOG.info("fundamentals_audit: Tiingo covered %d/%d tickers", len(tiingo_covered), len(tickers))

    remaining = [t for t in tickers if t not in tiingo_covered]
    yahoo_result = yahoo_comparison.run_non_tiingo_audit(
        context, remaining, threshold=threshold, cache_dir=yahoo_cache_dir)
    yahoo_covered = (set(yahoo_result["comparison"]["ticker"].unique())
                     if not yahoo_result["comparison"].empty else set())
    _LOG.info("fundamentals_audit: Yahoo covered %d/%d Tiingo-uncovered tickers",
             len(yahoo_covered), len(remaining))

    uncovered = sorted(set(remaining) - yahoo_covered)
    if uncovered:
        _LOG.warning("fundamentals_audit: %d tickers have NO external validation "
                    "available (neither Tiingo nor Yahoo) -- logged, not blocking",
                    len(uncovered))

    return {
        "tiingo": tiingo_result, "yahoo": yahoo_result,
        "no_external_validation": pd.DataFrame({"ticker": uncovered}),
    }


def _period_from_date(date) -> tuple[int, str]:
    d = pd.Timestamp(date)
    return d.year, f"Q{(d.month - 1) // 3 + 1}"


def _bucket_a_misses(comparison: pd.DataFrame, source: str,
                     tol_flow: float, tol_level: float) -> pd.DataFrame:
    """Bucket-a rows that missed their own exact-match tolerance -- the counterpart to
    `ratio_outlier_check`'s bucket-b/c flags, for the fields that are SUPPOSED to
    match exactly."""
    if comparison.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    a = comparison[(comparison["bucket"] == "a") & comparison["delta_pct"].notna()].copy()
    if a.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    tol = np.where(a["kind"].isin(["flow", "flow_abs"]), tol_flow * 100, tol_level * 100)
    beyond = a[a["delta_pct"].abs() > tol].copy()
    if beyond.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    fy_fp = beyond["quarter"].map(_period_from_date)
    beyond["fiscal_year"] = fy_fp.map(lambda t: t[0])
    beyond["fiscal_period"] = fy_fp.map(lambda t: t[1])
    beyond["source"] = source
    beyond["severity"] = beyond["delta_pct"].abs().clip(upper=100.0)
    beyond["n_tickers_same_switch"] = 0
    beyond["detail"] = beyond["delta_pct"].map(
        lambda d: f"bucket-a exact-match miss: delta_pct={d:.2f}%")
    return beyond[["source", "ticker", "field", "fiscal_year", "fiscal_period", "severity",
                  "n_tickers_same_switch", "detail"]]


def _ratio_outliers_to_findings(ratio_outliers: pd.DataFrame, source: str) -> pd.DataFrame:
    """Shared shape for `tiingo_comparison.ratio_outlier_check`,
    `yahoo_comparison.ratio_outlier_check` and `analyze_history.run_audit`'s own
    outliers -- all three already emit `ticker/field/fiscal_year/fiscal_period/
    level_z_score` (or `value`+outlier flags), just reused unmodified here."""
    if ratio_outliers.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    out = ratio_outliers.copy()
    out["source"] = source
    out["severity"] = out["level_z_score"].abs().clip(upper=100.0)
    out["n_tickers_same_switch"] = 0
    out["detail"] = out.apply(
        lambda r: f"{'YoY' if r.get('is_yoy_outlier') else 'level'} outlier, "
                 f"z={r['level_z_score']:.2f}", axis=1)
    return out[["source", "ticker", "field", "fiscal_year", "fiscal_period", "severity",
               "n_tickers_same_switch", "detail"]]


def _tag_breaks_to_findings(breaks: pd.DataFrame) -> pd.DataFrame:
    """`detect_tag_switch_breaks` is already worst-first ranked by `level_ratio` --
    reused as this source's severity, just rescaled onto the shared 0-100 band via
    log2 (a 4x jump and a 0.25x drop are equally severe, a plain ratio is not
    symmetric around 1.0)."""
    if breaks.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    out = breaks.copy()
    fy_fp = pd.to_datetime(out["boundary_period_end"]).map(_period_from_date)
    out["fiscal_year"] = fy_fp.map(lambda t: t[0])
    out["fiscal_period"] = fy_fp.map(lambda t: t[1])
    out["source"] = "tag_switch_break"
    ratio = out["level_ratio"].abs().replace(0, np.nan)
    out["severity"] = (np.log2(ratio.clip(lower=1e-6)).abs() * 25).clip(upper=100.0).fillna(100.0)
    out["n_tickers_same_switch"] = out["n_tickers_same_switch"].fillna(0)
    out["detail"] = out.apply(
        lambda r: f"tag switch {r['from_tag']} -> {r['to_tag']}: level_ratio={r['level_ratio']:.2f}",
        axis=1)
    return out[["source", "ticker", "field", "fiscal_year", "fiscal_period", "severity",
               "n_tickers_same_switch", "detail"]]


def _tag_misalignment_to_findings(misalignment: pd.DataFrame) -> pd.DataFrame:
    """`detect_source_tag_misalignment` carries NO magnitude at all (the exact gap
    `fundamentals_tag_ledger.py` was built to fix for tag-switch breaks specifically) --
    fixed low severity so it always ranks below anything with a real magnitude, never
    silently dropped."""
    if misalignment.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)
    out = misalignment.copy()
    out["fiscal_period"] = "FY"
    out["source"] = "tag_misalignment"
    out["severity"] = 10.0
    out["n_tickers_same_switch"] = 0
    out["detail"] = out.apply(
        lambda r: f"period-end tag {r['period_end_source_tag']!r} vs interim "
                 f"{r['interim_source_tags']}", axis=1)
    return out[["source", "ticker", "field", "fiscal_year", "fiscal_period", "severity",
               "n_tickers_same_switch", "detail"]]


def build_ranked_findings(
    *, tiingo_comparison_df: pd.DataFrame, tiingo_ratio_outliers: pd.DataFrame,
    yahoo_comparison_df: pd.DataFrame, yahoo_ratio_outliers: pd.DataFrame,
    tag_breaks: pd.DataFrame, tag_misalignment: pd.DataFrame,
    internal_outliers: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize all seven flag sources into one tidy frame and rank worst-first.

    Cross-source AGREEMENT (how many distinct `source`s flag the same (ticker, field,
    fiscal_year, fiscal_period) cell) is the primary sort key -- a cell an external
    vendor AND an internal check both flag independently is far higher-confidence than
    either alone, and is exactly the signal a single-source view cannot produce.
    `fiscal_period` here is a CALENDAR-quarter approximation for Tiingo/Yahoo/tag-break
    rows (derived from a date) vs. the filer's OWN fiscal labels for
    `analyze_history`-sourced rows -- a non-calendar-fiscal-year filer's own labels
    won't line up with the calendar approximation, which under-counts agreement for
    those filers rather than fabricating a false match (a conservative failure mode,
    not a false-positive one)."""
    frames = [
        _bucket_a_misses(tiingo_comparison_df, "tiingo",
                        TIINGO_EXACT_MATCH_TOLERANCE_FLOW, TIINGO_EXACT_MATCH_TOLERANCE_LEVEL),
        _ratio_outliers_to_findings(tiingo_ratio_outliers, "tiingo_ratio_outlier"),
        _bucket_a_misses(yahoo_comparison_df, "yahoo",
                        YAHOO_EXACT_MATCH_TOLERANCE_FLOW, YAHOO_EXACT_MATCH_TOLERANCE_LEVEL),
        _ratio_outliers_to_findings(yahoo_ratio_outliers, "yahoo_ratio_outlier"),
        _tag_breaks_to_findings(tag_breaks),
        _tag_misalignment_to_findings(tag_misalignment),
        _ratio_outliers_to_findings(internal_outliers, "internal_outlier"),
    ]
    findings = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame(columns=FINDINGS_COLS[:-2])
    if findings.empty:
        return pd.DataFrame(columns=FINDINGS_COLS)

    key = ["ticker", "field", "fiscal_year", "fiscal_period"]
    findings["agreement_count"] = findings.groupby(key)["source"].transform("nunique")
    findings["priority_score"] = (
        findings["agreement_count"] * 1000
        + findings["severity"]
        + findings["n_tickers_same_switch"].clip(upper=50)
    )
    return findings.sort_values("priority_score", ascending=False).reset_index(drop=True)[FINDINGS_COLS]


def run_full_audit(
    context: Context, tickers: list[str] | None = None, *, api_key: str,
    threshold: float = 3.5, tiingo_cache_dir: Path | None = None,
    yahoo_cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """End-to-end: universe-wide external audit (Tiingo -> Yahoo -> uncovered) plus a
    fresh run of the internal, full-universe diagnostics on `fundamentals_facts`
    (already full-scope, see `analyze_history.py`'s own `__main__`), combined into one
    ranked findings queue. The single entry point `__main__` below calls."""
    external = run_universe_audit(context, tickers, api_key=api_key, threshold=threshold,
                                  tiingo_cache_dir=tiingo_cache_dir, yahoo_cache_dir=yahoo_cache_dir)

    # Scoped by `where=` when an explicit ticker subset is given (a smoke test, e.g.) --
    # `fundamentals_facts` is accession-grain across the whole universe and reading it
    # unscoped for a 10-ticker smoke test is exactly the unscoped-large-table read
    # CLAUDE.md's data conventions warn against.
    facts = context.store.load("fundamentals_facts",
                               where={"ticker": list(tickers)} if tickers is not None else None)
    audit_tickers = tickers if tickers is not None else sorted(facts["ticker"].unique().tolist())
    internal = run_audit(facts, audit_tickers, list(DEFAULT_AUDIT_FIELDS), threshold=threshold)
    ledger = build_tag_ledger(facts)
    breaks = detect_tag_switch_breaks(ledger, facts)

    ranked = build_ranked_findings(
        tiingo_comparison_df=external["tiingo"]["comparison"],
        tiingo_ratio_outliers=external["tiingo"]["ratio_outliers"],
        yahoo_comparison_df=external["yahoo"]["comparison"],
        yahoo_ratio_outliers=external["yahoo"]["ratio_outliers"],
        tag_breaks=breaks,
        tag_misalignment=internal["tag_misalignment"],
        internal_outliers=internal["outliers"],
    )
    return {**external, "tag_ledger": ledger, "tag_breaks": breaks,
           "internal_outliers": internal["outliers"],
           "tag_misalignment": internal["tag_misalignment"], "ranked_findings": ranked}


if __name__ == "__main__":
    import os

    from src.context import get_config_context
    from src.constants.constants import TIINGO_CACHE_DIRNAME, YAHOO_CACHE_DIRNAME

    _, context = get_config_context(config_path="./configs", use_cache=True, save=False)
    api_key = os.getenv("TIIGO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TIIGO_API_KEY is not set. Add it to your .env file to run the "
                          "fundamentals extraction-cleanliness audit.")

    out_dir = context.paths["DATA_STORE"] / "gaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_full_audit(
        context, api_key=api_key,
        tiingo_cache_dir=out_dir / TIINGO_CACHE_DIRNAME,
        yahoo_cache_dir=out_dir / YAHOO_CACHE_DIRNAME,
    )

    result["tiingo"]["comparison"].to_csv(out_dir / "tiingo_comparison.csv", index=False)
    result["tiingo"]["ratio_outliers"].to_csv(out_dir / "tiingo_ratio_outliers.csv", index=False)
    result["yahoo"]["comparison"].to_csv(out_dir / "yahoo_comparison.csv", index=False)
    result["yahoo"]["ratio_outliers"].to_csv(out_dir / "yahoo_ratio_outliers.csv", index=False)
    result["no_external_validation"].to_csv(out_dir / NO_EXTERNAL_VALIDATION_FILENAME, index=False)
    result["ranked_findings"].to_csv(out_dir / FUNDAMENTALS_FINDINGS_RANKED_FILENAME, index=False)

    print("\n=== SANITY CHECK: fundamentals extraction-cleanliness audit (full universe) ===")
    n_uncovered = len(result["no_external_validation"])
    n_findings = len(result["ranked_findings"])
    n_agree2plus = int((result["ranked_findings"]["agreement_count"] >= 2).sum()) if n_findings else 0
    print(f"  tickers with no external validation available: {n_uncovered}")
    print(f"  total ranked findings: {n_findings} ({n_agree2plus} flagged by 2+ independent sources)")
    print("  Validated." if n_findings == 0 else "  Review data/gaps/fundamentals_findings_ranked.csv, worst first.")
