"""
fundamentals_tag_ledger.py  (src/utils/fundamentals_tag_ledger.py)
-------------------------------------------------------------------
Which XBRL concept actually produced each segment of each `fundamentals_facts`
series, and where swapping concepts mid-history spliced two different MEASURES
into one column.

Lives in `src/utils/` beside `analyze_history.py` (the audit entry point that
calls it) rather than in the `data_extract` package, for the same reason stated
there: a read-only diagnostic over an already-persisted table must not make
`src/utils` import from `src/data_extract`. Nothing here needs the extractor --
only the table's own columns.

FLAG-ONLY: nothing here mutates a value, exactly like
`fundamentals_validation.reconcile_fundamentals_facts`. The output is a review
queue; the fix always lands either in a candidate list (`fundamentals_tags.py`,
when the right answer is global) or in `FIELD_TAG_DENYLIST` (when it is one
filer's own tagging defect).

Why this exists alongside the two checks that already look at `source_tag`:

  * `analyze_history.detect_source_tag_misalignment` compares a fiscal year's
    PERIOD-END tag against its own INTERIM quarters' tags, and by design does
    NOT flag a clean cross-year cutover (see its docstring). So the whole class
    of "filer migrates to a different concept and never goes back, but the new
    concept measures something else" is invisible to it. That is the shape of
    the confirmed MetLife ASC-606 revenue defect (the contract element captured
    only fee income, ~48x too small, permanently, with every period WITHIN each
    year agreeing).
  * `reconcile_fundamentals_facts`' `large_discontinuity` check sees the level
    jump but knows nothing about tags, so it also fires on every real business
    event and is emitted at severity 'info'. Conditioning the same jump ON a tag
    boundary is what turns it from noise into a specific, actionable claim.

Neither existing check carries a MAGNITUDE, so their output cannot be ranked:
the 2026-08 audit produced 675 misalignment rows that all looked equally urgent.
Every row here is scored, worst first.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.constants.constants import (
    TAG_SWITCH_BASELINE_PERIODS, TAG_SWITCH_LEVEL_BREAK_RATIO,
    TAG_SWITCH_MAX_BOUNDARY_GAP_DAYS,
)
from src.context import Context

__all__ = ["build_tag_ledger", "detect_tag_switch_breaks", "write_tag_ledger",
           "TAG_LEDGER_FILENAME", "TAG_BREAKS_FILENAME"]

# `Context` exposes no logger attribute, so this follows `utils/step.py`'s convention.
_LOG: logging.Logger = logging.getLogger(__name__)

TAG_LEDGER_FILENAME = "fundamentals_tag_ledger.csv"
TAG_BREAKS_FILENAME = "fundamentals_tag_breaks.csv"
_GAPS_DIRNAME = "gaps"

LEDGER_COLS = ["ticker", "field", "duration_type", "source_tag", "era_index", "n_eras",
               "first_period_end", "last_period_end", "n_periods",
               "first_value", "last_value", "median_value"]
BREAK_COLS = ["ticker", "field", "duration_type", "check", "severity", "from_tag", "to_tag",
              "tag_pair", "boundary_period_end", "level_before", "level_after", "level_ratio",
              "boundary_gap_days", "n_periods_before", "n_periods_after",
              "n_boundaries", "n_tickers_same_switch", "detail"]

TAG_SWITCH_LEVEL_BREAK = "tag_switch_level_break"


def build_tag_ledger(facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse a `fundamentals_facts`-shaped frame into contiguous tag ERAS: one row
    per (ticker, field, run of consecutive periods sharing a `source_tag`).

    Two reductions happen first, both load-bearing:

      * DERIVED rows are dropped. They carry no `source_tag` by construction (a flow
        field's Q4 is unconditionally derived -- see `fundamentals_periods.
        decumulate_quarterly_flow` -- as are the cross-field fills like
        `derive_missing_total_liabilities`), and they are 9.5% of the live table.
        Keeping them would split one real era into three around every Q4.
      * One row per (ticker, field, period_end), taking the LATEST `filing_date`.
        `fundamentals_facts` is accession-grain, so an amended or later filing
        legitimately restates the same period; the most recent filing is the
        authority. Same rule as `analyze_history._latest_per_period`.

    Era boundaries come from a cumulative sum over `source_tag != source_tag.shift()`
    within each (ticker, field, duration_type), so a filer that ALTERNATES between two
    concepts (confirmed on DTE `shortTermDebt`: `ShortTermBorrowings` in its 10-Qs,
    `DebtCurrent` in its 10-K, every single year) produces many short eras rather
    than two -- which is the honest description of that series and lets
    `detect_tag_switch_breaks` score each swap.

    Scoped by duration_type, not just (ticker, field): a flow field's annual and
    quarterly buckets legitimately resolve DIFFERENT tags by design (`depAmort`'s
    quarterly `Depreciation` vs its annual `DepreciationDepletionAndAmortization`;
    `totalRevenue`'s FY `Revenues` alongside a quarterly-only `Revenues` ->
    `RevenuesNetOfInterestExpense` cutover for JPM). Pooling annual rows into the
    same chronological series as quarterly rows makes FY alternate with the
    surrounding quarters' tag every single year -- a switch manufactured by the
    duration split itself, not by the filer, and it also collided in
    `drop_duplicates` below: a calendar-year filer's FY and Q4 share one
    `period_end`, so without duration_type in the subset one of the two was
    silently dropped as a "duplicate" of the other.
    """
    if facts is None or facts.empty or "source_tag" not in facts.columns:
        return pd.DataFrame(columns=LEDGER_COLS)

    d = facts.copy()
    if "derived" in d.columns:
        d = d[pd.to_numeric(d["derived"], errors="coerce").fillna(0.0) != 1.0]
    d = d.dropna(subset=["source_tag", "period_end"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"])
    if d.empty:
        return pd.DataFrame(columns=LEDGER_COLS)
    d["duration_type"] = d["duration_type"].fillna("unknown") if "duration_type" in d.columns else "unknown"

    d["period_end"] = pd.to_datetime(d["period_end"], errors="coerce")
    d = d.dropna(subset=["period_end"])
    group_key = ["ticker", "field", "duration_type"]
    sort_cols = group_key + ["period_end"]
    if "filing_date" in d.columns:
        d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
        sort_cols = sort_cols + ["filing_date"]
    d = (d.sort_values(sort_cols)
         .drop_duplicates(subset=group_key + ["period_end"], keep="last"))

    grp = d.groupby(group_key, sort=False)["source_tag"]
    d["_era"] = (grp.transform(lambda s: (s != s.shift()).cumsum())).astype("int64")

    eras = (d.groupby(group_key + ["_era"], sort=True)
            .agg(source_tag=("source_tag", "first"),
                 first_period_end=("period_end", "min"),
                 last_period_end=("period_end", "max"),
                 n_periods=("value", "size"),
                 first_value=("value", "first"),
                 last_value=("value", "last"),
                 median_value=("value", "median"))
            .reset_index())
    # `_era` is already 1-based and gapless per (ticker, field, duration_type) -- expose
    # it under the public name and publish the era COUNT beside it so a reader can tell
    # "1 of 1" (a clean single-concept history) from "3 of 11" at a glance.
    eras = eras.rename(columns={"_era": "era_index"})
    eras["n_eras"] = eras.groupby(group_key)["era_index"].transform("max")
    return eras[LEDGER_COLS].sort_values(group_key + ["era_index"]).reset_index(drop=True)


def _pooled_level(values: pd.Series, *, from_end: bool) -> float:
    """Median of up to `TAG_SWITCH_BASELINE_PERIODS` periods taken from one end of an
    era. A MEDIAN over a window, not the single boundary value, because a balance-sheet
    level is genuinely volatile inside one tag (DTE's short-term borrowings swing $0 ->
    $1,131M quarter-over-quarter with no concept change at all) and one noisy boundary
    quarter would otherwise dominate the verdict."""
    window = values.iloc[-TAG_SWITCH_BASELINE_PERIODS:] if from_end else values.iloc[:TAG_SWITCH_BASELINE_PERIODS]
    return float(window.median())


def detect_tag_switch_breaks(ledger: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Score every consecutive era boundary in `ledger`, returning one
    `tag_switch_level_break` row per boundary whose pooled level moves by more than
    `TAG_SWITCH_LEVEL_BREAK_RATIO` in either direction -- i.e. the concept changed AND
    the number changed, so two different MEASURES are spliced into one column.

    `facts` is needed as well as `ledger` because the pooled level is a median over the
    periods INSIDE each era, which the collapsed ledger no longer carries.

    Two boundaries are deliberately left UNSCORED rather than reported:

      * a gap wider than `TAG_SWITCH_MAX_BOUNDARY_GAP_DAYS`, where the two levels are
        separated by missing periods as well as by the tag change, so nothing can be
        attributed to the switch;
      * a level of exactly 0 on either side (see `_level_ratio`).

    `n_tickers_same_switch` is what makes the output triageable, and it is the one signal
    that distinguishes the two ROOT CAUSES rather than just the symptom. A from_tag ->
    to_tag transition made by MANY tickers is a US-GAAP taxonomy migration (measured on
    the live table: ASC 842 lease maturities, ASU 2016-18 cash, CECL credit losses, the
    X -> XNet deprecations), so a level break there indicts the FIELD's candidate list
    and the fix is global. A transition only ONE ticker makes is that filer's own
    mis-tagging, and the fix is a `FIELD_TAG_DENYLIST` entry. Note this deliberately does
    NOT downgrade severity: a common migration can still change the measure for a subset
    of filers (confirmed: the ASC-606 contract element captures only fee income for
    insurers and REITs, which is how MetLife's revenue came out ~48x too small).
    """
    if ledger is None or ledger.empty or facts is None or facts.empty:
        return pd.DataFrame(columns=BREAK_COLS)

    values = facts.copy()
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    values["period_end"] = pd.to_datetime(values["period_end"], errors="coerce")
    values = values.dropna(subset=["value", "period_end"])
    values["duration_type"] = (values["duration_type"].fillna("unknown")
                               if "duration_type" in values.columns else "unknown")

    rows: list[dict] = []
    for (ticker, field, duration_type), eras in ledger.groupby(
            ["ticker", "field", "duration_type"], sort=True):
        if len(eras) < 2:
            continue
        eras = eras.sort_values("era_index").to_dict("records")
        series = values[(values["ticker"] == ticker) & (values["field"] == field)
                        & (values["duration_type"].fillna("unknown") == duration_type)]
        for prev, nxt in zip(eras, eras[1:]):
            gap_days = int((nxt["first_period_end"] - prev["last_period_end"]).days)
            if gap_days > TAG_SWITCH_MAX_BOUNDARY_GAP_DAYS:
                continue
            before = _pooled_level(_era_values(series, prev), from_end=True)
            after = _pooled_level(_era_values(series, nxt), from_end=False)
            ratio = _level_ratio(before, after)
            if not np.isfinite(ratio) or ratio <= TAG_SWITCH_LEVEL_BREAK_RATIO:
                continue
            rows.append(_break_row(
                ticker, field, duration_type, prev, nxt, gap_days, TAG_SWITCH_LEVEL_BREAK,
                "warning", before, after, ratio,
                detail=f"pooled level moves {before:,.0f} -> {after:,.0f} ({ratio:.2f}x) "
                       f"across the {prev['source_tag']} -> {nxt['source_tag']} switch"))

    out = pd.DataFrame(rows, columns=BREAK_COLS)
    if out.empty:
        return out
    out = _collapse_recurring_swaps(out)
    out["n_tickers_same_switch"] = out.groupby(
        ["field", "duration_type", "tag_pair"])["ticker"].transform("nunique")
    return (out.sort_values("level_ratio", ascending=False, na_position="last")
            .reset_index(drop=True))


def _collapse_recurring_swaps(breaks: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, field, unordered tag PAIR), keeping the worst boundary.

    A filer that ALTERNATES between two concepts crosses the same boundary once per
    filing, in both directions, and each crossing is the same single finding. Measured on
    the live table this is the dominant duplication: Citigroup's `allowanceCreditLosses`
    swap between a ~$1M and a ~$12.8B concept produced EIGHT rows -- four in each
    direction -- for one defect, and DTE's annual 10-K/10-Q `shortTermDebt` swap behaves
    the same way. Left uncollapsed the queue is ~10x longer than the number of things to
    actually decide, and the loudest filer buries everyone else.

    `n_boundaries` is kept rather than discarded: how OFTEN a swap recurs separates a
    one-time cutover (1) from a systematic per-filing measure swap (many), which is the
    difference between a taxonomy migration to verify and a filer whose 10-K and 10-Qs
    disagree with each other every single year."""
    pair = breaks[["from_tag", "to_tag"]].apply(
        lambda r: " <-> ".join(sorted((str(r["from_tag"]), str(r["to_tag"])))), axis=1)
    breaks = breaks.assign(tag_pair=pair)
    key = ["ticker", "field", "duration_type", "tag_pair"]
    worst = breaks.loc[breaks.groupby(key)["level_ratio"].idxmax()].copy()
    worst["n_boundaries"] = breaks.groupby(key)["level_ratio"].size().reindex(
        pd.MultiIndex.from_frame(worst[key])).to_numpy()
    return worst


def _era_values(series: pd.DataFrame, era: dict) -> pd.Series:
    """The era's own periods, chronologically. Bounded by the era's dates rather than
    re-deriving the era membership, so the ledger stays the single source of truth for
    where an era begins and ends."""
    inside = series[series["period_end"].between(era["first_period_end"], era["last_period_end"])]
    return inside.sort_values("period_end")["value"]


def _level_ratio(before: float, after: float) -> float:
    """Symmetric magnitude ratio: >= 1 whichever direction the level moved, so a halving
    and a doubling score the same. Zero on either side makes the ratio undefined rather
    than infinite -- a level of exactly 0 is a real, common reading for a revolver
    balance (DTE reports $0 short-term borrowings in several quarters) and must not be
    reported as an infinite break."""
    a, b = abs(before), abs(after)
    if not np.isfinite(a) or not np.isfinite(b) or a == 0.0 or b == 0.0:
        return np.nan
    return max(a / b, b / a)


def _break_row(ticker: str, field: str, duration_type: str, prev: dict, nxt: dict,
               gap_days: int, check: str, severity: str, before: float, after: float,
               ratio: float, *, detail: str) -> dict:
    return {
        "ticker": ticker, "field": field, "duration_type": duration_type,
        "check": check, "severity": severity,
        "from_tag": prev["source_tag"], "to_tag": nxt["source_tag"],
        "boundary_period_end": prev["last_period_end"],
        "level_before": before, "level_after": after, "level_ratio": ratio,
        "boundary_gap_days": gap_days,
        "n_periods_before": prev["n_periods"], "n_periods_after": nxt["n_periods"],
        # filled once every boundary is known (see `_collapse_recurring_swaps`)
        "tag_pair": None, "n_boundaries": pd.NA, "n_tickers_same_switch": pd.NA,
        "detail": detail,
    }


def write_tag_ledger(context: Context, facts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build both frames and write them beside the existing audit CSVs
    (`data/gaps/`, resolved through `context.paths` rather than hardcoded).

    CSV rather than a `context.store` table on purpose: this is a derived diagnostic
    rebuilt from `fundamentals_facts` on demand, so persisting it would add DDL to
    `sql/schema.sql` for something that is never read back by the pipeline."""
    ledger = build_tag_ledger(facts)
    breaks = detect_tag_switch_breaks(ledger, facts)

    out_dir = context.paths["DATA_STORE"] / _GAPS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_dir / TAG_LEDGER_FILENAME, index=False)
    breaks.to_csv(out_dir / TAG_BREAKS_FILENAME, index=False)

    multi = ledger[ledger["n_eras"] > 1] if not ledger.empty else ledger
    _LOG.info("fundamentals tag ledger: %d eras over %d (ticker, field) pairs, "
             "%d pairs switch concept mid-history",
             len(ledger), ledger.groupby(["ticker", "field"]).ngroups if not ledger.empty else 0,
             multi.groupby(["ticker", "field"]).ngroups if not multi.empty else 0)
    if not breaks.empty:
        counts = breaks["check"].value_counts().to_dict()
        _LOG.info("fundamentals tag ledger: %s", counts)
    return {"ledger": ledger, "breaks": breaks}
