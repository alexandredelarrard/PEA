"""
gap_check.py  (src/data_extract/utils/fundamentals_sharadar/gap_check.py)
------------------------------------------------------------------------------------
Where Sharadar and the SEC layer DISAGREE, on the dates where both published -- and which of
those disagreements is a BASIS CONFLICT rather than a restatement.

This is the merged table's instrument, and it exists because it had to replace one. Reason
codes stay with the SEC table (D24), so `unexplained_null` stops being a universal
zero-ceiling gate on `fundamentals_history`. Nothing else measures the merged table's
per-column truth, so this does.

## Why `is_systematic` is the column that decides

A one-date gap is a restatement, a rounding difference, or a filing the other source missed.
A gap that holds on MOST dates is a definitional fork, and only a definitional fork is worth
an override. AXP's `totalRevenue` was 6.6-8.1% low on **all 11** shared dates -- that
persistence is the entire signal, and it is what separates it from JPM's `totalRevenue`,
which matched the repo EXACTLY on all 11.

## The expected forks are named, not rediscovered

Eight columns are DESIGNED to differ (`SHARADAR_GAP_EXPECTED_FIELDS`) -- phase 3 chose those
bases on purpose and `sharadar_field_map.json` states why for each. They are reported as
`is_expected` so they do not drown the signal. **Anything gapping that is not on that list is
the real finding.**

## This is NOT the validator (D25)

Nothing here registers a check, writes a `fundamentals_check` row, or imports `src/validate/`.
It writes one markdown report and, with `--propose`, candidate entries in the override
register -- every one of them inert until a human sets `approved`.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_GAP_EXPECTED_FIELDS, SHARADAR_GAP_MIN_DATES,
    SHARADAR_GAP_RELATIVE_THRESHOLD, SHARADAR_GAP_SYSTEMATIC_SHARE,
    SHARADAR_OVERRIDE_APPROVED_KEY, SHARADAR_OVERRIDE_SOURCE_SEC,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    DEFAULT_CONFIG_DIR, HISTORY_STATEMENT_ORDER)
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ, build_ttm
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    FieldMap, load_field_map, translate)
from src.data_extract.utils.fundamentals_sharadar.merge_history import (
    _KEY_FROM_VENDOR, collapse_same_date, load_overrides, write_overrides)
from src.data_store.schema import Tables

log = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = ("reports/planning/active-tasks/2026-08-26-sharadar-integration/"
                       "phase-4-gap-check.md")

#: The three floor classes, and how a field is assigned one. Read off the FIELD MAP's own
#: declarations -- `op: ratio`/`ratio_minus_one` is a ratio, `split_basis: count` is a share
#: count -- rather than from a name pattern, so a new column classifies itself.
MONEY, SHARES, RATIO = "money", "shares", "ratio"

#: How many `(ticker, field)` rows the markdown lists in full. The count is always printed
#: beside it, so the tail is visible without being enumerated.
WORST_ROWS = 30


def field_class(name: str, field_map: FieldMap) -> str:
    spec = field_map.outputs.get(name)
    if spec is None:
        return MONEY
    if spec.op in ("ratio", "ratio_minus_one"):
        return RATIO
    if spec.split_basis == "count":
        return SHARES
    return MONEY


def inherited_from_expected(name: str, field_map: FieldMap) -> str:
    """Which EXPECTED basis fork a derived column's gap is mechanically inherited from.

    Informational only -- it does NOT reclassify anything, and a row it names is still a
    candidate. `returnOnEquity` is `netIncome / stockholdersEquity`, and `stockholdersEquity`
    is a designed-in fork, so a gap there is the same decision showing up one level down.
    Without saying so, six of the flagged rows read as six independent findings.

    ⚠ It reaches DERIVED columns only. `totalDebt` gaps for exactly this reason -- both its
    legs are expected forks -- but the map has it as a DIRECT column from `debt`, so nothing
    here can know that. It stays a finding, which is the right default: the alternative is a
    hand-maintained list of implied relationships that goes stale silently.
    """
    spec = field_map.outputs.get(name)
    if spec is None or spec.kind != "derived":
        return ""
    return ", ".join(sorted(set(spec.inputs) & SHARADAR_GAP_EXPECTED_FIELDS))


def comparable_fields(field_map: FieldMap) -> list[str]:
    """Every field BOTH sources carry, in statement order.

    The Sharadar-owned half of the contract intersected with the SEC table's own vocabulary.
    The 15 SEC-owned columns are excluded because comparing them would compare the SEC layer
    against itself; the 25 extras and the 3 `null` columns because the SEC table has no such
    column at all.
    """
    owned = {n for n, s in field_map.outputs.items() if s.kind in ("direct", "derived")}
    return [n for n in HISTORY_STATEMENT_ORDER if n in owned]


def sharadar_history(vendor_arq: pd.DataFrame, field_map: FieldMap,
                     actions: pd.DataFrame | None) -> pd.DataFrame:
    """The Sharadar side on the merged table's grain, WITHOUT the SEC block.

    Deliberately not `merge_history.build_frame`: that one has already joined the SEC values
    in and applied the overrides, so measuring a gap on it would compare the SEC layer with
    itself wherever the answer matters most.
    """
    frame = build_ttm(translate(vendor_arq, field_map), field_map, actions=actions)
    frame = frame.rename(columns=_KEY_FROM_VENDOR)
    for column in ("as_of", "fiscal_end"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").astype("datetime64[ns]")
    collapsed, _ = collapse_same_date(frame)
    return collapsed


def measure_gaps(context, tickers: Sequence[str] | None = None, *,
                 config_dir: str = DEFAULT_CONFIG_DIR) -> pd.DataFrame:
    """One row per `(ticker, field)` both sources carry, over their SHARED `as_of` dates.

    An EXACT date join here, not the merge's backward as-of: a gap check must compare the two
    sources' statements about the SAME publication, and an as-of carry would manufacture
    differences out of a one-day filing-date disagreement.
    """
    field_map = load_field_map(config_dir)
    floors = {klass: float(value) for klass, value
              in context.config.data_extract.sharadar_gap_floor.items()}

    where = {"ticker": sorted(tickers)} if tickers else {}
    vendor = context.store.load(Tables.sharadar_fundamentals, project=True,
                                where={**where, "dimension": ARQ}, optional=True)
    if vendor is None or vendor.empty:
        raise RuntimeError("gap check: no stored Sharadar ARQ rows to measure")
    actions = context.store.load(Tables.sharadar_actions, project=True, optional=True)
    fields = comparable_fields(field_map)
    sec = context.store.load(Tables.fundamentals_history_sec,
                             columns=["ticker", "as_of", *fields],
                             where=where or None, optional=True)
    if sec is None:
        raise RuntimeError("gap check: fundamentals_history_sec has no rows for this scope")
    sec["as_of"] = pd.to_datetime(sec["as_of"]).astype("datetime64[ns]")

    shar = sharadar_history(vendor, field_map, actions)
    overlap = sorted(set(shar["ticker"]) & set(sec["ticker"]))
    log.info("gap check: %d overlapping ticker(s) of %d Sharadar / %d SEC; %d comparable "
             "field(s)", len(overlap), shar["ticker"].nunique(), sec["ticker"].nunique(),
             len(fields))
    joined = shar.merge(sec, on=["ticker", "as_of"], suffixes=("_shar", "_sec"))
    if joined.empty:
        raise RuntimeError("gap check: the two sources share no (ticker, as_of) at all -- "
                           "that is a grain bug, not a gap")

    rows = []
    for name in fields:
        klass = field_class(name, field_map)
        floor = floors[klass]
        pair = joined[["ticker", "as_of", f"{name}_shar", f"{name}_sec"]].dropna()
        if pair.empty:
            continue
        left = pair[f"{name}_shar"].astype("float64")
        right = pair[f"{name}_sec"].astype("float64")
        delta = left - right
        pct = (delta.abs() / right.abs().replace(0.0, np.nan))
        flagged = (pct > SHARADAR_GAP_RELATIVE_THRESHOLD) & (delta.abs() > floor)
        for ticker, block in pair.assign(_pct=pct, _delta=delta,
                                         _flag=flagged).groupby("ticker", sort=True):
            n_dates = len(block)
            n_flagged = int(block["_flag"].sum())
            share = n_flagged / n_dates
            rows.append({
                "ticker": ticker, "field": name, "class": klass,
                "n_dates": n_dates, "n_flagged": n_flagged,
                "flagged_share": round(share, 3),
                "median_pct_gap": float(block["_pct"].median()),
                "min_pct_gap": float(block["_pct"].min()),
                "max_pct_gap": float(block["_pct"].max()),
                "median_abs_gap": float(block["_delta"].abs().median()),
                "is_systematic": bool(share >= SHARADAR_GAP_SYSTEMATIC_SHARE
                                      and n_dates >= SHARADAR_GAP_MIN_DATES
                                      and n_flagged > 0),
                "is_expected": name in SHARADAR_GAP_EXPECTED_FIELDS,
                "inherits_from": inherited_from_expected(name, field_map),
            })
    gaps = pd.DataFrame(rows)
    if gaps.empty:
        return gaps
    return gaps.sort_values(["is_systematic", "is_expected", "median_pct_gap"],
                            ascending=[False, True, False]).reset_index(drop=True)


def candidates(gaps: pd.DataFrame) -> pd.DataFrame:
    """The rows worth an override: SYSTEMATIC, and not one of the designed-in forks.

    Both filters matter and for different reasons. Without `is_systematic` a single
    restatement proposes a permanent source change; without `is_expected` the eight bases
    phase 3 chose on purpose come back as findings every single run, and a report whose
    findings are always the same is a report nobody reads.
    """
    if gaps.empty:
        return gaps
    return gaps[gaps["is_systematic"] & ~gaps["is_expected"]].reset_index(drop=True)


def propose(gaps: pd.DataFrame, *, config_dir: str = DEFAULT_CONFIG_DIR) -> tuple[Path, int]:
    """Merge candidate entries into the register with `approved: null`, keeping every
    existing entry EXACTLY as it stands.

    A proposer that could overwrite a reviewed decision would make the review worthless, so
    an existing `(ticker, field)` is never touched -- not its reason, not its approval date,
    not even a re-measured gap. Re-measuring is the report's job.
    """
    existing = load_overrides(config_dir)
    entries: dict[str, dict[str, dict]] = {}
    for (ticker, field), entry in {**existing.approved, **existing.pending}.items():
        entries.setdefault(ticker, {})[field] = entry
    added = 0
    for row in candidates(gaps).to_dict("records"):
        ticker, field = row["ticker"], row["field"]
        if field in entries.get(ticker, {}):
            continue
        entries.setdefault(ticker, {})[field] = {
            "source": SHARADAR_OVERRIDE_SOURCE_SEC,
            "reason": (f"PROPOSED {date.today().isoformat()}: Sharadar differs from the SEC "
                       f"layer on {row['n_flagged']} of {row['n_dates']} shared dates "
                       f"(median {row['median_pct_gap']:.1%}). Systematic, so a basis fork "
                       f"rather than a restatement. REPLACE THIS with what the filer's own "
                       f"caption says before approving."),
            "measured_gap_pct": round(row["median_pct_gap"], 4),
            "n_dates": int(row["n_dates"]),
            SHARADAR_OVERRIDE_APPROVED_KEY: None,
        }
        added += 1
    path = write_overrides(entries, _README, config_dir=config_dir)
    log.warning("gap check: %d new proposal(s) written to %s with `approved: null` -- they "
                "change NOTHING until a human adjudicates them", added, path)
    return path, added


_README: list[str] = [
    "PER-(TICKER, FIELD) SOURCE OVERRIDES for the merged `fundamentals_history` (D22).",
    "",
    "MACHINE-PROPOSED, HUMAN-APPROVED. `sharadar-gap-check --propose` writes candidates here",
    "with `approved: null`; `merge_history.load_overrides` IGNORES those and logs how many are",
    "awaiting a decision. An unapproved proposal must never silently change data. Set",
    "`approved` to the date you adjudicated it -- and replace the proposed `reason` with what",
    "you actually found, because the generated one only says a gap exists, not what it is.",
    "",
    "⚠ A GAP DOES NOT SAY WHICH SIDE IS WRONG, and the proposer cannot tell. Measured on the",
    "first run: MCD `depAmort` gaps by ~480% because the SEC layer stores ~460M against MCD's",
    "own ~2.2bn annual D&A -- the SHARADAR value is the correct one there, so approving that",
    "proposal would import the defect. Read the filing before setting `approved`; a proposal",
    "is a question, not an answer.",
    "",
    "THE ONLY LEGAL DIRECTION is `\"source\": \"sec\"`. Moving a column the other way is not an",
    "override but a field-BLOCK change (D14) and belongs in `sharadar_field_map.json`.",
    "",
    "⚠ COVERAGE COST. An override moves a (ticker, field) to a source with the SEC roster's",
    "coverage. If that ticker is outside the roster the column becomes NULL -- it does NOT",
    "fall back to Sharadar, because a per-row fallback is exactly the mid-series source switch",
    "D14 forbids. The merge logs the cost per entry at build time.",
    "",
    "⚠ EIGHT FIELDS ARE EXPECTED TO GAP and are never proposed: stockholdersEquity, ppeNet,",
    "shortTermDebt, longTermDebt, accountsReceivable, accountsPayable, cash, ebitda. Those are",
    "the bases phase 3 chose on purpose; `sharadar_field_map.json` states why for each.",
    "",
    "Unlike the other two Sharadar registers this file needs NO `_APPROVED` block: approval is",
    "per ENTRY here, so a fresh proposal is inert rather than fatal.",
]


# --------------------------------------------------------------------------- #
# the report                                                                   #
# --------------------------------------------------------------------------- #
def format_report(gaps: pd.DataFrame, *, overlap: int, fields: int) -> str:
    """The markdown. Ordered so the only actionable section is first."""
    found = candidates(gaps)
    expected = gaps[gaps["is_systematic"] & gaps["is_expected"]] if not gaps.empty else gaps
    clean = gaps[~gaps["is_systematic"]] if not gaps.empty else gaps
    show = ["ticker", "field", "class", "n_dates", "n_flagged", "median_pct_gap",
            "min_pct_gap", "max_pct_gap", "median_abs_gap", "inherits_from"]

    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_none_\n"
        head = frame.head(WORST_ROWS)[show].copy()
        for column in ("median_pct_gap", "min_pct_gap", "max_pct_gap"):
            head[column] = head[column].map(lambda v: f"{v:.2%}")
        head["median_abs_gap"] = head["median_abs_gap"].map(lambda v: f"{v:,.0f}")
        return head.to_markdown(index=False) + (
            f"\n\n_{len(frame)} row(s) total; {WORST_ROWS} shown._\n"
            if len(frame) > WORST_ROWS else "\n")

    return "\n".join([
        "# Phase 4 — Sharadar vs SEC gap check",
        "",
        f"Scope: **{overlap} overlapping tickers**, **{fields} comparable fields**, every "
        f"shared `as_of`. Flagged when |Δ|/|sec| > "
        f"{SHARADAR_GAP_RELATIVE_THRESHOLD:.0%} **and** |Δ| exceeds the class floor; "
        f"**systematic** when that holds on ≥ {SHARADAR_GAP_SYSTEMATIC_SHARE:.0%} of at "
        f"least {SHARADAR_GAP_MIN_DATES} shared dates.",
        "",
        "⚠ This is not the validator (D25). It registers no check and writes no "
        "`fundamentals_check` row.",
        "",
        "## 1. Override candidates — systematic, and NOT a designed-in fork",
        "",
        "**These are the findings.** Each is a basis conflict nobody has adjudicated. An "
        "override moves the field to a source with the SEC roster's coverage; a ticker "
        "outside that roster gets NULL, not a Sharadar fallback.",
        "",
        table(found),
        "",
        "## 2. Systematic and EXPECTED — the phase-3 basis forks, not defects",
        "",
        "Named so they do not drown section 1. `sharadar_field_map.json` states the reason "
        "for each.",
        "",
        table(expected),
        "",
        "## 3. Not systematic — restatements, roundings, one-offs",
        "",
        f"_{len(clean)} (ticker, field) pair(s)._ A gap on 1 of 11 dates is not an override "
        "candidate.",
        "",
    ])


def run_gap_check(context, tickers: Sequence[str] | None = None, *,
                  report_path: str | Path = DEFAULT_REPORT_PATH,
                  propose_overrides: bool = False,
                  config_dir: str = DEFAULT_CONFIG_DIR) -> pd.DataFrame:
    """Measure, write the markdown, and optionally write the proposals. Returns the frame so
    a test can assert on it without re-reading a file."""
    gaps = measure_gaps(context, tickers, config_dir=config_dir)
    if gaps.empty:
        context.log.warning("gap check: nothing comparable -- 0 rows measured")
        return gaps
    overlap = gaps["ticker"].nunique()
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(gaps, overlap=overlap,
                                  fields=gaps["field"].nunique()), encoding="utf-8")
    found = candidates(gaps)
    context.log.info("gap check: %d (ticker, field) pair(s) over %d ticker(s); %d systematic, "
                     "%d of them NOT expected -> %s", len(gaps), overlap,
                     int(gaps["is_systematic"].sum()), len(found), path)
    if not found.empty:
        context.log.warning("gap check candidates:\n%s",
                            found[["ticker", "field", "n_flagged", "n_dates",
                                   "median_pct_gap"]].to_string(index=False))
    if propose_overrides:
        propose(gaps, config_dir=config_dir)
    return gaps
