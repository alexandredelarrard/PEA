"""
institutionals.py  (src/validate/institutionals.py)
---------------------------------------------------
The Part 2 CORRECTNESS checks for `cube_part_institutionals` -- the check list in
`reports/planning/active-tasks/2026-09-07-informed-capital/part-2-validation.md`, executed as
Phase 2.8 of `part-2b-panels.md`.

Correctness, never predictiveness (D15): leak-free, point-in-time, in-range, reconciling to
source. IC-based selection belongs to modelling, on the modelling sample.

WHY THE CHECKS ARE PANEL-DRIVEN AND NOT LISTED BY NAME. Every group below enumerates the
columns the panel actually emitted and scores those. A hand-maintained list of the ~100
expected features would be a second declaration of the feature set, and the 2026-08 incident
is on record for exactly that: a `_sec` rename was never applied to consumers, leaving 46 dead
field names and 39 dead features with 68 tests green and blind. The one place a DECLARATION is
read is `D1`, which diffs each builder's own `EMISSION` map against the emitted columns -- so a
feature that stops building fails a check instead of vanishing quietly.

⚠ ALL-NAN IS NOT COVERAGE. `f_ic_act_percent_of_class_vs_peers` shipped declared, emitted and
0.0% non-null across 3,893,697 rows -- an existence assertion passes on a dead column. Every
coverage figure here is a non-null count, and `G2` lists what falls under the floor.

WHAT THIS MODULE CANNOT SCORE YET is not silently absent: `SKIPPED` carries one row per check
with the phase that must supply it. Running the suite against an incomplete part and reading a
green summary is the failure mode this register exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd

#: Coverage floor (D14). Applied to a feature's non-null share INSIDE its own availability
#: window, never over all history -- a 2024-12-17 feature scored from 1996 fails by arithmetic.
COVERAGE_FLOOR = 0.05

#: Redundancy (D14). Reported at 0.90, flagged at 0.95.
REDUNDANCY_REPORT = 0.90
REDUNDANCY_FLAG = 0.95

#: `XS_CLIP_PEER` is +/-8; a cell within this of the bound is AT it. Not `== 8.0`: the legs are
#: cast to float32 by `build_peer_relative_panel`, so the stored value is 7.9999998.
CLIP_EDGE = 7.999

#: C1 bands, set from the live measurement of all 118 `_vs_peers` columns of
#: `cube_part_fundamentals` (part-2-validation.md 10.3b): dense fields top out at 3.70% and the
#: repo's worst column, `f_pbo_to_mcap_vs_peers`, sits at 8.59%.
CLIP_NORMAL = 0.04
CLIP_ELEVATED = 0.09

#: C2. On a healthy field the `|z|>4` band runs 1.6-2.5x the clip rate. Near 1.0x is a mass AT
#: the boundary rather than a tail, which is the signature of a collapsing peer basket.
SHOULDER_HEALTHY = (1.6, 2.5)

#: C3. Above this share of "raw exists, peer leg NaN because pstd == 0" the feature is
#: peer-unrankable and belongs on `raw+xs`.
DEGENERATE_MAX = 0.20

#: 13F availability: the filing deadline is 45 calendar days after the period end, and the
#: panel must not move before it.
F13_LAG_DAYS = 45

#: L10 / R4. 2013-06-30 is the BROAD 13F coverage regime start (192 -> 3,046 filers), so its
#: QoQ deltas have no comparable prior quarter and must be NaN rather than a universe-wide
#: positive, and every LEVEL before it is nulled by D16. Imported from the builder rather than
#: retyped: two spellings of one date is how a guard and its check drift apart.
from src.data_aggregate.utils.institutionals.institutional_features import (  # noqa: E402
    COVERAGE_BREAK_DEFAULT, INST_LEVEL_FLOOR_PERIOD, _coverage_periods)

FIRST_13F_PERIOD = INST_LEVEL_FLOOR_PERIOD

PASS, FAIL, REPORT, SKIP = "PASS", "FAIL", "REPORT", "SKIP"

#: G2. A per-FEATURE availability floor, for the four legs whose window is narrower than their
#: family's. `FAMILY_SOURCES` keys on a prefix, but `f_ic_bo_` mixes event-shape features that
#: run to 1996 with NUMERIC ones that are ~0% filled before beneficial-ownership XML became
#: mandatory (Finding 1: every numeric, the event date, the CUSIP and the security title are
#: unavailable from edgartools before it). Scored from 1996 they read 0.24-4.72% and G2 lists
#: them as low-coverage -- a property of the CALENDAR, not of the feature. The builder already
#: masks these four to NaN before the same date; this is the check reading the same constant.
FEATURE_FLOORS: dict[str, pd.Timestamp] = {
    base: pd.Timestamp("2024-12-17") for base in (
        "ic_act_percent_of_class", "ic_act_delta_percent_class",
        "ic_bo_percent_of_class", "ic_bo_delta_percent_class")
}

#: Feature-name prefix -> the source table whose dates set that family's availability floor,
#: and how the floor is taken. Read by L1: a family may not produce a value before its source
#: could have been public.
#:
#: ⚠ ORDER MATTERS AND THE SPECIFIC PREFIXES COME FIRST. `_family_of` returns the first match,
#: so `f_ic_sig_insider_` must precede any shorter prefix that would also match it.
#: The two DERIVED families take the floor of the source they are derived FROM: a conditioning
#: feature cannot exist before the disclosure it dates from, and a cross-source count cannot
#: exist before the earliest of its four inputs (13G, 1996).
FAMILY_SOURCES: dict[str, tuple[str, str, int]] = {
    "f_ic_inst_": ("sec13f_hr", "period", F13_LAG_DAYS),
    "f_ic_super_": ("sec13f_manager_holdings", "period", F13_LAG_DAYS),
    "f_ic_insider_": ("insider_transactions", "filing_date", 0),
    "f_ic_act_": ("sec_13d", "filing_date", 0),
    "f_ic_bo_": ("sec_13g", "filing_date", 0),
    "f_ic_shortvol_": ("sec_short_interest", "date", 0),
    "f_ic_ftd_": ("sec_fails_to_deliver", "date", 0),
    "f_ic_sig_super_": ("sec13f_manager_holdings", "period", F13_LAG_DAYS),
    "f_ic_sig_insider_": ("insider_transactions", "filing_date", 0),
    "f_ic_sig_act_": ("sec_13d", "filing_date", 0),
    # ⚠ `sec_13d`, NOT `sec_13g`. The cross-source count exists as soon as ANY of its four
    # inputs does, and the earliest is the activist channel (13D, 1995-09-06), not the passive
    # one (13G, 1996-01-26). The comment here used to read "the earliest of its four inputs
    # (13G, 1996)" and was simply wrong about which is earliest, which failed the three bullish
    # `ic_xs_` legs on 326 legitimate cells apiece on the 2026-09-12 run.
    "f_ic_xs_": ("sec_13d", "filing_date", 0),
}

#: V1-V3. The declared plausible bound per feature BASE name (no `f_` prefix, no leg suffix).
#: Only the raw leg is scored: a `_vs_peers` z and an `_xs` percentile have their own bounds
#: (V9, V10) and are not in the feature's own units.
DECLARED_BOUNDS: dict[str, tuple[float, float]] = {
    # V1 -- signed ratios
    "ic_inst_net_options_ratio": (-1.0, 1.0),
    "ic_insider_net_buy_ratio_60d": (-1.0, 1.0),
    "ic_insider_net_buy_ratio_120d": (-1.0, 1.0),
    "ic_insider_net_buy_ratio_180d": (-1.0, 1.0),
    # V2 -- unit-interval shares
    "ic_inst_new_buyer_ratio": (0.0, 1.0),
    "ic_inst_exit_ratio": (0.0, 1.0),
    "ic_super_sp500_share": (0.0, 1.0),
    "ic_shortvol_ratio_5d": (0.0, 1.0),
    "ic_shortvol_ratio_20d": (0.0, 1.0),
    "ic_shortvol_ratio_60d": (0.0, 1.0),
    # V3 -- 13F holders cannot own more than the company; 5% slack for share-count timing
    # ⚠ 2.0, NOT 1.05, and the widening is a measurement rather than a concession. Securities
    # lending DOUBLE-COUNTS a position -- the lender still reports it on its 13F and the
    # borrower's buyer reports it too -- so a heavily-lent name legitimately reads 100-130% and
    # vendors publish such figures. The 1.05 bound failed 24,347 of 1,516,801 cells on the
    # 2026-09-12 build, of which 3,623 were above 1.30 and only 231 above 2.0. The builder now
    # NULLS above 2.0 (`institutional_features.OWNERSHIP_CEILING`), so this bound and that
    # ceiling are the same number stated twice on purpose: the guard removes the impossible
    # values and this check proves the guard ran.
    "ic_inst_ownership_pct": (0.0, 2.0),
    # 13D/13G percent-of-class is a disclosed percentage, and the group aggregate is taken as a
    # MAX never a SUM (R3) -- a value above 100 is the summing defect, measured at 4.02x mean
    # and 14.77x worst on live multi-reporting-person events.
    "ic_act_percent_of_class": (0.0, 100.0),
    "ic_bo_percent_of_class": (0.0, 100.0),
    # D28 redefinitions: a SHARE of the quarter's filers, and a difference of two such shares.
    "ic_inst_holders": (0.0, 1.0),
    "ic_inst_breadth_chg": (-1.0, 1.0),
    "ic_inst_cluster_buying": (-1.0, 1.0),
    # `self_history_z` clips at +/-8 by construction; a value outside it means the z was not
    # taken through that helper.
    "ic_shortvol_ratio_z252": (-8.0, 8.0),
    "ic_ftd_z252": (-8.0, 8.0),
    # a count of days inside a 30-day window
    "ic_ftd_persistence_30d": (0.0, 30.0),
    # RegSHO is OFF-EXCHANGE ONLY, so its reported volume is a SUBSET of the consolidated
    # tape: the coverage share cannot exceed 1. ⚠ This is the one bound here that is a claim
    # about two SOURCES agreeing rather than about arithmetic -- if it breaches, the tape
    # volume and the RegSHO total are not measuring the same thing, which is worth failing on.
    "ic_shortvol_market_coverage": (0.0, 1.0),
    # non-negative rates
    "ic_shortvol_turnover_20d": (0.0, np.inf),
    "ic_ftd_pct_so": (0.0, np.inf),
    "ic_ftd_to_adv20": (0.0, np.inf),
    # ic_sig_*: a day count is non-negative; a return is bounded below by -100%; the two
    # excursions are signed by construction (see `_capped_excursion`).
    "ic_sig_super_age_days": (0.0, np.inf),
    "ic_sig_insider_age_days": (0.0, np.inf),
    "ic_sig_act_age_days": (0.0, np.inf),
    "ic_sig_super_ret_since": (-1.0, np.inf),
    "ic_sig_insider_ret_since": (-1.0, np.inf),
    "ic_sig_act_ret_since": (-1.0, np.inf),
    "ic_sig_insider_price_vs_buy": (-1.0, np.inf),
    "ic_sig_insider_max_dd_since_buy": (-1.0, 0.0),
    "ic_sig_insider_max_runup_since_buy": (0.0, np.inf),
    # ic_xs_*: counts of families, bounded by how many families there are
    "ic_xs_bullish_family_count": (0.0, 4.0),
    "ic_xs_bearish_family_count": (0.0, 4.0),
    "ic_xs_conflict": (0.0, 4.0),
    "ic_xs_bullish_actor_count": (0.0, np.inf),
}

#: Checks that cannot run until the phase named here has shipped. One row each, printed in the
#: report: an absent check must be visible, not inferred from a short table.
SKIPPED: tuple[tuple[str, str, str, str], ...] = (
    ("L2", "leakage", "13F invisible on period+44d, visible on period+45d",
     "Implemented as L2 below; it is the only 13F leak check that runs today."),
    ("L5", "leakage", "`ic_insider_owner_surprise_120d` is truncation-invariant",
     "Needs a rebuild of the insider panel per truncation date. Deep check -- `--deep`."),
    ("L6", "leakage", "PIT manager selection is truncation-invariant",
     "Covered by tests/data_aggregate/test_manager_selection.py; not re-measured here."),
    # ⚠ THIS SKIP REASON USED TO PRESCRIBE A VACUOUS PROCEDURE, AND THE PROCEDURE HID A REAL
    # DEFECT. It said: run `build-institutionals -F`, then `build-institutionals`, and compare
    # the overlap. Run on 2026-09-12 that appended **+0 rows** -- sources and part both end on
    # the same date, so `write_part`'s strict append had nothing newer to write and the "after"
    # table was the "before" table. A two-run table diff proves nothing whenever the sources
    # have not moved, which is most of the time.
    #
    # THE PROCEDURE THAT WORKS: rebuild through `build_panel(full=False)` and difference it IN
    # MEMORY against the persisted full-build rows, at a float32 tolerance (rtol 1e-5 / atol
    # 1e-6) so the storage round-trip cannot masquerade as drift. Run that way it FAILED on
    # 2026-09-12: 76 of 127 columns drifted and the drift reached the last date, which is the
    # only row an append writes -- `f_ic_bo_new_holder` wrong on 490 of 491 tickers on
    # 2026-09-04, `f_ic_sig_insider_age_days` by a median 674 days and up to 4,412, inside a
    # 567-day window. Cause: event sources are read WHOLE but the daily GRID was bounded by
    # `window.since`, and `decay.snap_to_grid` moves an event onto the first trading day >= its
    # date, so every event predating the window landed ON the window's first day. The
    # age/new-holder/expanding-window families have no finite look-back for a warm-up to cover.
    #
    # FIXED the same day: `StepCubeInstitutionals._load_frames` takes no `since`, so the part
    # computes over the full trading calendar and lets `write_part` slice the tail. Re-measured
    # by the same procedure against the rebuilt table: **127 of 127 columns identical**, 191,979
    # shared rows, 0 rows missing on either side, and the newest date's 491 rows exact.
    ("L7", "leakage", "Incremental == full on the overlapping window",
     "MEASURED GREEN on 2026-09-12 after the grid fix -- 127/127 columns identical over a "
     "567-day window including the newest date (was 76/127 drifting). Not run from here "
     "because it needs a SECOND build of the part and this suite already builds one, so it "
     "stays a `--deep` check; the reproduction is `build_panel(full=False)` differenced "
     "against the stored rows at rtol 1e-5 / atol 1e-6."),
    ("L8", "leakage", "No future prices in the conditioning layer",
     "Covered by tests/data_aggregate/test_signal_conditioning.py::"
     "test_no_future_prices_reach_a_conditioning_value -- it rebuilds the panel with the price "
     "frame truncated and asserts every earlier value is identical. Re-measuring it here would "
     "mean a second full build per truncation date."),
    ("R1", "reconciliation", "Superinvestor weights sum to 1 over the WHOLE book",
     "Needs the builder's per-manager weight frame, not a panel column. Owned by "
     "tests/data_aggregate/test_superinvestor_features.py."),
    ("R2", "reconciliation", "Old-vs-new superinvestor weights track value coverage",
     "The OLD weights were overwritten by the Part 2a rebuild (D8) -- there is nothing left "
     "to difference against. The ratio table in README Finding 2 is the record."),
    ("R6", "reconciliation", "discretionary + planned sell == total S value, 2023q3+",
     "Needs the insider builder's intermediate window sums. Owned by test_insider_features.py."),
    ("V4", "value", "A synthetic 20:1 share jump yields NaN, not +1900%",
     "Synthetic by definition -- a unit test, not a panel measurement."),
    ("V5", "value", "Split-guard trigger count on real data is reported",
     "REPORTED, but by the BUILD rather than by this suite: all three restatement sites log "
     "their trigger count (`institutional_features` the broad-13F pairs, `superinvestor_"
     "features` the manager-quarters, `signal_conditioning._insider_price_legs` the cost-anchor "
     "purchases). The counts are a property of a build, not of a panel column, so they are read "
     "from the run log and quoted in the DoD; there is nothing in the panel to score."),
    ("G5", "gate", "Measured IC sign vs the declared monotone direction",
     "Needs the modelling label from cube_part_targets joined to this part, and the declared "
     "direction from the model configs' monotone block -- which G4 is what writes. Run it "
     "after the G4 survivors land in configs/models/*.yml, so the comparison is against a "
     "direction somebody declared rather than one this module invented."),
)


@dataclass
class CheckResult:
    """One check. `detail` is the table that goes into the DoD report.

    `status` is four-valued on purpose. A `REPORT` check has no pass condition -- C4's outlier
    profile and G1's coverage sheet are evidence for a human, and scoring them PASS would imply
    a threshold nobody set. Only `blocking` FAILs stop the phase.
    """
    check_id: str
    group: str
    title: str
    status: str
    measured: str = ""
    expected: str = ""
    detail: list[dict] = dataclass_field(default_factory=list)
    blocking: bool = False

    def summary(self) -> str:
        line = f"{self.check_id} [{self.status}] {self.title}"
        if self.measured:
            line += f" -- {self.measured}"
        if self.expected and self.status in (FAIL,):
            line += f" (expected {self.expected})"
        return line


@dataclass
class InstitutionalsReport:
    checks: list[CheckResult] = dataclass_field(default_factory=list)
    rows: int = 0
    tickers: int = 0
    features: int = 0
    columns: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == FAIL and c.blocking]

    def to_markdown(self) -> str:
        head = [
            "# Part 2 validation -- `cube_part_institutionals`",
            "",
            f"Panel: **{self.rows:,} rows x {self.columns} columns**, {self.features} feature "
            f"legs, {self.tickers} tickers, "
            f"{self.date_min.date() if self.date_min is not None else '?'} -> "
            f"{self.date_max.date() if self.date_max is not None else '?'}.",
            "",
            "Correctness only (D15) -- leak-free, point-in-time, in-range, reconciling to "
            "source. Predictive selection belongs to modelling.",
            "",
            f"**{len(self.blocking_failures)} blocking failure(s)**, "
            f"{sum(c.status == FAIL for c in self.checks)} failure(s) total, "
            f"{sum(c.status == SKIP for c in self.checks)} check(s) not runnable yet.",
            "",
        ]
        for group in ("leakage", "reconciliation", "value", "saturation", "behavioural", "gate"):
            rows = [c for c in self.checks if c.group == group]
            if not rows:
                continue
            head += [f"## {group.capitalize()}", ""]
            head += ["| check | status | measured | expected |", "|---|---|---|---|"]
            head += [f"| **{c.check_id}** {c.title} | {c.status} | {c.measured} | {c.expected} |"
                     for c in rows]
            head += [""]
            for c in rows:
                if c.detail:
                    head += [f"### {c.check_id} detail", "", _table(c.detail), ""]
        return "\n".join(head)


def _table(rows: list[dict]) -> str:
    frame = pd.DataFrame(rows)
    return frame.to_markdown(index=False) if not frame.empty else "_(empty)_"


# --------------------------------------------------------------------------- helpers

def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("f_")]


def split_leg(column: str) -> tuple[str, str]:
    """`f_ic_bo_percent_of_class_vs_peers` -> `("ic_bo_percent_of_class", "vs_peers")`."""
    base = column[2:] if column.startswith("f_") else column
    for suffix, leg in (("_vs_peers", "vs_peers"), ("_xs", "xs")):
        if base.endswith(suffix):
            return base[: -len(suffix)], leg
    return base, "raw"


def _family_of(column: str) -> str | None:
    for prefix in FAMILY_SOURCES:
        if column.startswith(prefix):
            return prefix
    return None


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _first_last(series: pd.Series, dates: pd.Series) -> tuple[object, object]:
    live = dates[series.notna()]
    if live.empty:
        return None, None
    return live.min(), live.max()


# --------------------------------------------------------------------------- D1 / gates

def check_declared_vs_emitted(panel: pd.DataFrame,
                              declared: dict[str, dict[str, str]]) -> CheckResult:
    """D1 -- every field a builder DECLARES emits at least one non-null column, and no leg
    appears that no builder declared.

    Not in the plan's list; it is the Risk Mitigation row "renaming silently drops a feature"
    turned into a check. `build_peer_relative_panel` already raises on an `emission` key that
    no field produced, which catches the declaration side inside one builder -- this catches
    the case that got past it: the field builds, the emission map names it, and the column is
    emitted ALL-NAN because its input never resolved.
    """
    columns = set(feature_columns(panel))
    expected: dict[str, list[str]] = {}
    for module, emission in declared.items():
        for name, mode in emission.items():
            legs = ["raw"]
            if mode == "raw+xs":
                legs.append("xs")
            elif mode == "raw+peers":
                legs.append("vs_peers")
            expected[name] = legs

    missing, dead = [], []
    for name, legs in expected.items():
        for leg in legs:
            suffix = "" if leg == "raw" else f"_{leg}"
            column = f"f_{name}{suffix}"
            if column not in columns:
                missing.append({"feature": column, "problem": "declared, not emitted"})
            elif not panel[column].notna().any():
                dead.append({"feature": column, "problem": "emitted, 0 non-null cells"})

    declared_columns = {f"f_{n}{'' if leg == 'raw' else '_' + leg}"
                        for n, legs in expected.items() for leg in legs}
    undeclared = [{"feature": c, "problem": "emitted, not declared by any EMISSION map"}
                  for c in sorted(columns - declared_columns) if _family_of(c) in FAMILY_SOURCES]

    detail = missing + dead + undeclared
    return CheckResult(
        "D1", "gate", "Declared features == emitted, non-empty columns",
        FAIL if (missing or dead) else (REPORT if undeclared else PASS),
        measured=f"{len(expected)} fields declared; {len(missing)} absent, {len(dead)} all-NaN, "
                 f"{len(undeclared)} emitted without a declaration",
        expected="0 absent, 0 all-NaN", detail=detail, blocking=True)


def _distribution(series: pd.Series) -> dict[str, float | None]:
    """The five-number summary the coverage sheet carries beside the coverage number.

    ⚠ EVERY leg gets one, including the legs `DECLARED_BOUNDS` says nothing about. V1-V3 only
    scores features with a declared bound and only their raw leg, so without this a z-leg, an
    `_xs` percentile or any unbounded feature shipped with NO range evidence at all -- which is
    the shape of the `f_ic_act_percent_of_class_vs_peers` incident, where a column was declared,
    emitted and never once looked at. p1/p99 rather than min/max alone because a single bad cell
    moves an extremum and tells you nothing about the body of the distribution.
    """
    values = _finite(series)          # a numpy array: NaN AND +/-inf are already gone
    if not len(values):
        return {"min": None, "p1": None, "p50": None, "p99": None, "max": None, "pct_zero": None}
    p1, p50, p99 = np.quantile(values, [0.01, 0.5, 0.99])
    return {"min": round(float(values.min()), 4), "p1": round(float(p1), 4),
            "p50": round(float(p50), 4), "p99": round(float(p99), 4),
            "max": round(float(values.max()), 4),
            "pct_zero": round(100.0 * float(np.mean(values == 0.0)), 2)}


def value_universe_scope(panel: pd.DataFrame, universe: set[str] | None) -> CheckResult:
    """V12 -- every ticker in the part is a ticker `cube_part_prices` has.

    ⚠ MEASURED AS A REAL DEFECT ON THE 2026-09-11 BUILD: 503 tickers against 491 in prices,
    betas and governance. `PanelMerger.to_long` is an outer-aligned concat and the step dropped
    its `_grid` marker unused, so any (date, ticker) a source table carried and the price grid
    did not was appended -- `EA`, `EQR`, `AVB` (absent from `prices` entirely) plus nine recent
    spin-offs, 14,982 rows.

    It is scored rather than reported because the consequence is not only the extra rows.
    `_xs` is a same-day percentile over the cross-section (D26), so an off-universe column in
    a builder's wide frame silently changed the denominator of every `_xs` leg in the part,
    and `peer_relative` resolved baskets partly out of names no model will ever see.
    """
    if universe is None:
        return CheckResult("V12", "value", "Part tickers are a subset of the price universe",
                           SKIP, measured="cube_part_prices is not available to compare against")
    present = set(panel["ticker"].astype(str).unique())
    extra = sorted(present - universe)
    rows = [{"ticker": t, "rows": int((panel["ticker"].astype(str) == t).sum())} for t in extra]
    return CheckResult("V12", "value", "Part tickers are a subset of the price universe",
                       FAIL if extra else PASS,
                       measured=f"{len(present)} ticker(s) in the part, {len(universe)} in "
                                f"cube_part_prices, {len(extra)} not in it",
                       expected="0 off-universe tickers", detail=rows, blocking=True)


def value_degenerate_legs(panel: pd.DataFrame) -> CheckResult:
    """V11 -- no emitted leg is constant, and none is a single value on > 99.9% of its cells.

    D1 catches the all-NaN column; this catches the one below it. A leg that is 0.0 everywhere
    passes D1, passes its declared bound, passes the peer-z cap and carries exactly zero
    information into the model. It is also the observable signature of a real defect class here
    -- a `decay_events` input whose magnitudes are all zero reads as "the event never happened"
    -- so it is scored, not merely reported.
    """
    rows = []
    for column in feature_columns(panel):
        values = _finite(panel[column])
        if len(values) < 100:
            continue
        # float32 legs carry representation noise, so round before counting -- otherwise a
        # genuinely constant column reads as thousands of distinct values differing at 1e-9.
        uniques, counts = np.unique(np.round(values, 9), return_counts=True)
        top_at = int(counts.argmax())
        share = float(counts[top_at]) / float(len(values))
        if len(uniques) <= 1:
            rows.append({"feature": column, "problem": "constant",
                         "value": round(float(uniques[0]), 6), "share": 100.0,
                         "non_null": int(len(values))})
        elif share > 0.999:
            rows.append({"feature": column, "problem": "> 99.9% one value",
                         "value": round(float(uniques[top_at]), 6),
                         "share": round(100.0 * share, 3), "non_null": int(len(values))})
    return CheckResult("V11", "value", "No constant / single-valued leg",
                       FAIL if rows else PASS,
                       measured=f"{len(rows)} degenerate leg(s) of "
                                f"{len(feature_columns(panel))}",
                       expected="0", detail=rows, blocking=True)


def gate_coverage(panel: pd.DataFrame, floor: float = COVERAGE_FLOOR,
                  family_floors: dict[str, pd.Timestamp] | None = None
                  ) -> tuple[CheckResult, CheckResult]:
    """G1 the coverage sheet, G2 the dropped-for-coverage list.

    ⚠ THE FLOOR IS APPLIED INSIDE THE AVAILABILITY WINDOW, not over all history. Four of the
    beneficial-ownership percent features start at the 2024-12-17 XML mandate; scored from
    1996 they read ~1% and would be cut for a property of the calendar rather than of the
    feature. `family_floors` supplies the window start per family; without it the window is the
    feature's own first non-null date, which is the same number for a class-D feature and
    generous for a sparse one.
    """
    dates = panel["date"]
    family_floors = family_floors or {}
    sheet, dropped = [], []
    for column in feature_columns(panel):
        series = panel[column]
        first, last = _first_last(series, dates)
        family = _family_of(column)
        # Per-FEATURE floor first, then the family's, then the leg's own first non-null. The
        # order matters: the four mandate-gated numerics have a family floor (1995/1996) that
        # is 28 years wider than their real window.
        window_start = FEATURE_FLOORS.get(split_leg(column)[0])
        if window_start is None:
            window_start = family_floors.get(family) if family else None
        if window_start is None:
            window_start = first
        in_window = panel.loc[dates >= window_start, column] if window_start is not None else series
        non_null = int(series.notna().sum())
        share = float(in_window.notna().mean()) if len(in_window) else 0.0
        row = {"feature": column, "non_null": non_null,
               "pct_non_null_in_window": round(100.0 * share, 2),
               "first_non_null": str(first.date()) if first is not None else "-",
               "last_non_null": str(last.date()) if last is not None else "-",
               "tickers": int(panel.loc[series.notna(), "ticker"].nunique()),
               **_distribution(series)}
        sheet.append(row)
        if share < floor:
            dropped.append({**row, "window_start": str(pd.Timestamp(window_start).date())
                            if window_start is not None else "-"})

    g1 = CheckResult("G1", "gate", "Coverage sheet", REPORT,
                     measured=f"{len(sheet)} feature legs scored", detail=sheet)
    g2 = CheckResult("G2", "gate", f"Dropped for coverage (< {_pct(floor)} in window)",
                     REPORT if dropped else PASS,
                     measured=f"{len(dropped)} of {len(sheet)} legs below the floor",
                     expected=f">= {_pct(floor)}", detail=dropped)
    return g1, g2


def gate_redundancy(panel: pd.DataFrame, report_at: float = REDUNDANCY_REPORT,
                    flag_at: float = REDUNDANCY_FLAG, sample: int = 120_000) -> CheckResult:
    """G3 -- every pair at |Spearman| > 0.90, flagged above 0.95.

    Sampled rows, not all of them: a rank correlation over ~100 columns x 3.9M rows is a
    10-minute job for a number that does not move in the fourth decimal. The sample is a
    deterministic stride so the matrix is reproducible run to run.
    """
    columns = feature_columns(panel)
    if len(columns) < 2:
        return CheckResult("G3", "gate", "Redundancy matrix", PASS, measured="< 2 features")
    step = max(1, len(panel) // sample)
    corr = panel.iloc[::step][columns].corr(method="spearman", min_periods=500)
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            rho = corr.at[a, b]
            if pd.notna(rho) and abs(rho) > report_at:
                pairs.append({"a": a, "b": b, "spearman": round(float(rho), 4),
                              "flagged": bool(abs(rho) > flag_at)})
    pairs.sort(key=lambda r: -abs(r["spearman"]))
    flagged = sum(p["flagged"] for p in pairs)
    return CheckResult("G3", "gate", f"Redundancy pairs |rho| > {report_at}", REPORT,
                       measured=f"{len(pairs)} pair(s) above {report_at}, {flagged} above "
                                f"{flag_at} (on {len(panel.iloc[::step]):,} sampled rows)",
                       detail=pairs[:60])


# --------------------------------------------------------------------------- value sanity

def value_declared_bounds(panel: pd.DataFrame) -> CheckResult:
    """V1-V3 -- the raw leg of every feature with a declared bound stays inside it."""
    rows = []
    for column in feature_columns(panel):
        base, leg = split_leg(column)
        if leg != "raw" or base not in DECLARED_BOUNDS:
            continue
        low, high = DECLARED_BOUNDS[base]
        values = _finite(panel[column])
        if not len(values):
            rows.append({"feature": column, "bound": f"[{low}, {high}]", "non_null": 0,
                         "out_of_range": 0, "pct_out": 0.0, "min": None, "max": None})
            continue
        bad = int(((values < low) | (values > high)).sum())
        rows.append({"feature": column, "bound": f"[{low}, {high}]", "non_null": len(values),
                     "out_of_range": bad, "pct_out": round(100.0 * bad / len(values), 4),
                     "min": round(float(values.min()), 4), "max": round(float(values.max()), 4)})
    breached = [r for r in rows if r["out_of_range"]]
    return CheckResult("V1-V3", "value", "Declared bounds on the raw leg",
                       FAIL if breached else PASS,
                       measured=f"{len(breached)} of {len(rows)} bounded features breach",
                       expected="0 breaches", detail=rows, blocking=True)


def value_peer_z(panel: pd.DataFrame) -> CheckResult:
    """V9 -- no +/-inf anywhere, and |z| <= 8 on every `_vs_peers` column."""
    rows = []
    inf_total = 0
    for column in feature_columns(panel):
        arr = np.asarray(panel[column], dtype="float64")
        n_inf = int(np.isinf(arr).sum())
        inf_total += n_inf
        base, leg = split_leg(column)
        if leg != "vs_peers":
            if n_inf:
                rows.append({"feature": column, "problem": "+/-inf", "cells": n_inf})
            continue
        finite = arr[np.isfinite(arr)]
        over = int((np.abs(finite) > 8.0 + 1e-6).sum()) if len(finite) else 0
        if n_inf or over:
            rows.append({"feature": column, "problem": "+/-inf" if n_inf else "|z| > 8",
                         "cells": n_inf or over})
    return CheckResult("V9", "value", "No +/-inf; |z| <= 8 on every peer leg",
                       FAIL if rows else PASS,
                       measured=f"{inf_total} infinite cell(s), {len(rows)} offending column(s)",
                       expected="0", detail=rows, blocking=True)


def value_xs_unit(panel: pd.DataFrame) -> CheckResult:
    """V10 -- every `_xs` column is a percentile in [0, 1]."""
    rows = []
    for column in feature_columns(panel):
        if split_leg(column)[1] != "xs":
            continue
        values = _finite(panel[column])
        if not len(values):
            continue
        bad = int(((values < 0.0) | (values > 1.0)).sum())
        if bad:
            rows.append({"feature": column, "out_of_unit": bad,
                         "min": round(float(values.min()), 6),
                         "max": round(float(values.max()), 6)})
    return CheckResult("V10", "value", "Every `_xs` leg in [0, 1]", FAIL if rows else PASS,
                       measured=f"{len(rows)} offending column(s)", expected="0",
                       detail=rows, blocking=True)


def value_decay_behaviour(panel: pd.DataFrame, max_features: int = 40) -> CheckResult:
    """V6 + V7 -- a decayed (class-S) column is NaN before a ticker's first event, never 0, and
    strictly decreases on a day with no new event.

    THE CLASS-S SET IS INFERRED, not listed. A column is treated as decayed if, on the sampled
    tickers, its non-null series is non-increasing on more than 60% of consecutive day pairs --
    which is the observable property the check is about. A hand-kept list of class-S names
    would be one more declaration to drift, and the inference reports what it found so a
    feature that stops decaying shows up as a membership change rather than a silent pass.
    """
    rows = []
    columns = [c for c in feature_columns(panel)[:max_features] if split_leg(c)[1] == "raw"]
    if not columns:
        return CheckResult("V6-V7", "value", "Decay behaviour", SKIP, measured="no raw legs")
    # Sample the tickers BEFORE sorting: 12 tickers is ~0.1% of the panel, and sorting the
    # whole thing to read 12 groups is the copy this check does not need.
    sample = sorted(panel["ticker"].dropna().unique())[:12]
    narrow = panel.loc[panel["ticker"].isin(sample), ["ticker", "date"] + columns]
    grouped = narrow.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
    for column in columns:
        down = flat = up = 0
        zeros_before_first = 0
        for ticker in sample:
            series = grouped.get_group(ticker)[column]
            live = series.dropna()
            if len(live) < 20:
                continue
            diff = live.diff().dropna()
            down += int((diff < 0).sum())
            flat += int((diff == 0).sum())
            up += int((diff > 0).sum())
            # ⚠ `first_valid_index` is the wrong anchor here: 0.0 IS a valid value, so on a
            # zero-filled prefix it returns row 0 and the check inspects nothing. The defect
            # being looked for is a leading run of exact zeros standing in for "no event has
            # ever happened", so the anchor is the first NON-ZERO observation.
            moved = series.ne(0.0) & series.notna()
            if moved.any():
                zeros_before_first += int((series[: moved.idxmax()] == 0.0).sum())
        total = down + flat + up
        if not total:
            continue
        decayed = down / total > 0.60
        if decayed:
            rows.append({"feature": column, "pct_decreasing": round(100.0 * down / total, 1),
                         "pct_flat": round(100.0 * flat / total, 1),
                         "pct_increasing": round(100.0 * up / total, 1),
                         "zeros_before_first_event": zeros_before_first})
    bad = [r for r in rows if r["zeros_before_first_event"]]
    return CheckResult("V6-V7", "value", "Decay: NaN (never 0) before the first event, "
                                         "decreasing between events",
                       FAIL if bad else (REPORT if rows else SKIP),
                       measured=f"{len(rows)} column(s) inferred class-S on 12 sampled "
                                f"tickers; {len(bad)} carry a pre-event zero",
                       expected="0 pre-event zeros", detail=rows, blocking=True)


# --------------------------------------------------------------------------- saturation (C)

def saturation_profile(panel: pd.DataFrame) -> list[CheckResult]:
    """C1 clip rate, C2 shoulder ratio, C3 degenerate-basket NaN rate, C4 raw-leg outlier
    profile, C5 `_xs` tie mass.

    C1's bands come from the live measurement of the 118 fundamentals peer legs, NOT from a
    round number: an earlier draft set them at <3% / 3-12% / >12% and seven healthy live fields
    exceeded 3%. Where a clip rate IS high the fix is a definition or coverage change, never a
    wider clip -- raising `XS_CLIP_PEER` hides the diagnostic.
    """
    c1_rows, c2_rows, c3_rows, c4_rows, c5_rows = [], [], [], [], []
    columns = feature_columns(panel)
    for column in columns:
        base, leg = split_leg(column)
        arr = np.asarray(panel[column], dtype="float64")
        finite = arr[np.isfinite(arr)]
        if leg == "vs_peers" and len(finite):
            clip = float((np.abs(finite) >= CLIP_EDGE).mean())
            shoulder = float((np.abs(finite) > 4.0).mean())
            band = ("normal" if clip < CLIP_NORMAL
                    else "elevated" if clip < CLIP_ELEVATED else "investigate")
            c1_rows.append({"feature": column, "non_null": len(finite),
                            "pct_at_clip": round(100.0 * clip, 3), "band": band})
            c2_rows.append({"feature": column, "pct_at_clip": round(100.0 * clip, 3),
                            "pct_abs_z_gt_4": round(100.0 * shoulder, 3),
                            "shoulder_ratio": round(shoulder / clip, 2) if clip else None,
                            "healthy": (SHOULDER_HEALTHY[0] <= (shoulder / clip)
                                        <= SHOULDER_HEALTHY[1]) if clip else None})
            raw = f"f_{base}"
            if raw in panel.columns:
                raw_live = panel[raw].notna()
                degenerate = float((raw_live & panel[column].isna()).sum() / max(int(raw_live.sum()), 1))
                c3_rows.append({"feature": column, "raw_non_null": int(raw_live.sum()),
                                "pct_raw_present_peer_nan": round(100.0 * degenerate, 2),
                                "over_limit": degenerate > DEGENERATE_MAX})
        elif leg == "raw" and len(finite):
            low, high = DECLARED_BOUNDS.get(base, (None, None))
            outside = (int(((finite < low) | (finite > high)).sum())
                       if low is not None else None)
            p50 = float(np.percentile(finite, 50))
            p999 = float(np.percentile(finite, 99.9))
            c4_rows.append({"feature": column, "non_null": len(finite),
                            "min": round(float(finite.min()), 4),
                            "p50": round(p50, 4),
                            "p99": round(float(np.percentile(finite, 99)), 4),
                            "p99.9": round(p999, 4),
                            "max": round(float(finite.max()), 4),
                            "p99.9_over_p50": round(p999 / p50, 2) if p50 else None,
                            "outside_declared_bound": outside})
        elif leg == "xs" and len(finite):
            modal = pd.Series(finite).round(6).value_counts()
            c5_rows.append({"feature": column, "non_null": len(finite),
                            "modal_value": round(float(modal.index[0]), 6),
                            "pct_at_mode": round(100.0 * float(modal.iloc[0]) / len(finite), 2),
                            "distinct_ranks": int(pd.Series(finite).round(6).nunique())})

    worst = max((r["pct_at_clip"] for r in c1_rows), default=0.0)
    c1_bad = [r for r in c1_rows if r["band"] == "investigate"]
    c3_bad = [r for r in c3_rows if r["over_limit"]]
    return [
        CheckResult("C1", "saturation", "Peer-z clip rate", FAIL if c1_bad else PASS,
                    measured=f"{len(c1_rows)} peer leg(s); worst {worst:.2f}%",
                    expected=f"< {_pct(CLIP_NORMAL)} normal, > {_pct(CLIP_ELEVATED)} investigate",
                    detail=c1_rows, blocking=False),
        # ⚠ A LEG THAT NEVER CLIPS HAS NO SHOULDER RATIO, AND THAT IS THE BEST OUTCOME, NOT A
        # MISS. `healthy` is None for those (0/0), and counting None as "not healthy" made the
        # headline read "1 of 5 in the healthy band" on a panel whose worst clip rate is 1.90%
        # -- three of the five legs clip ZERO cells. The summary now names the three populations
        # separately, because D29's diagnostic is a ratio near 1.0x (a mass AT the boundary),
        # and no boundary mass at all cannot be that.
        CheckResult("C2", "saturation", "Shoulder ratio pct(|z|>4) / pct(clip)", REPORT,
                    measured=f"{sum(1 for r in c2_rows if r['healthy'] is True)} in the "
                             f"{SHOULDER_HEALTHY[0]}-{SHOULDER_HEALTHY[1]}x healthy band, "
                             f"{sum(1 for r in c2_rows if r['healthy'] is False)} outside it, "
                             f"{sum(1 for r in c2_rows if r['healthy'] is None)} with no clipped "
                             f"cell at all (ratio undefined)",
                    detail=c2_rows),
        CheckResult("C3", "saturation", "Degenerate-basket NaN rate",
                    FAIL if c3_bad else PASS,
                    measured=f"{len(c3_bad)} peer leg(s) above {_pct(DEGENERATE_MAX)}",
                    expected=f"< {_pct(DEGENERATE_MAX)}", detail=c3_rows),
        CheckResult("C4", "saturation", "Raw-leg outlier profile (D27: unwinsorized, unclipped)",
                    REPORT, measured=f"{len(c4_rows)} raw leg(s) profiled", detail=c4_rows),
        CheckResult("C5", "saturation", "`_xs` tie mass", REPORT,
                    measured=f"{len(c5_rows)} percentile leg(s) profiled", detail=c5_rows),
    ]


# --------------------------------------------------------------------------- leakage (L)

def leak_availability(panel: pd.DataFrame,
                      family_floors: dict[str, pd.Timestamp]) -> CheckResult:
    """L1 + L9 -- no feature carries a value before its family's source could have been public.

    The floor is MEASURED from the source table, never declared here: 13F families take
    `min(period) + 45 calendar days`, filing-space families take `min(filing_date)`. A
    hand-written effective-start table is a second declaration and would be wrong the first
    time a source is backfilled.
    """
    dates = panel["date"]
    rows = []
    for column in feature_columns(panel):
        family = _family_of(column)
        floor = family_floors.get(family) if family else None
        if floor is None:
            continue
        first, _ = _first_last(panel[column], dates)
        if first is None:
            continue
        early = int((panel.loc[dates < floor, column].notna()).sum())
        rows.append({"feature": column, "family": family,
                     "availability_floor": str(pd.Timestamp(floor).date()),
                     "first_non_null": str(pd.Timestamp(first).date()),
                     "cells_before_floor": early})
    bad = [r for r in rows if r["cells_before_floor"]]
    return CheckResult("L1/L9", "leakage", "No feature precedes its family's availability date",
                       FAIL if bad else PASS,
                       measured=f"{len(bad)} of {len(rows)} feature legs carry a pre-floor value",
                       expected="0", detail=bad or rows[:40], blocking=True)


def leak_13f_lag(panel: pd.DataFrame, periods: pd.Series, cohort: float = 0.20,
                 tolerance: int = 3, daily_share: float = 0.50) -> CheckResult:
    """L2 -- a 13F feature may only STEP on a filing deadline (`period + 45 days`).

    ⚠ THE PLAN'S WORDING IS NOT TESTABLE AS WRITTEN. "NaN on period+44d, non-NaN on +45d" is
    not what a forward-filled panel does: the feature carries the PREVIOUS quarter's value
    throughout and is non-null all along. What a leak would actually look like is a STEP on the
    wrong day, so the check finds the days on which a cohort of tickers changed value at once
    and asserts each one sits within `tolerance` trading days of some deadline.

    A price-scaled 13F feature (`*_to_mcap`) moves every day by construction, because its
    denominator is a close. Those are separated out as `daily` and REPORTED, not failed -- a
    daily-moving feature has no step dates to place, and calling it a leak would be wrong.

    ⚠ THE TWO 13F FAMILIES OBEY DIFFERENT RULES, and scoring them alike was a check defect that
    failed six correct features on the 2026-09-12 run. `ic_inst_*` is stamped on the DEADLINE
    (the documented departure from registry 0.5 -- a 3,000-filer aggregate cannot go on a
    per-filer availability grid), so its steps must land ON a deadline. `ic_super_*` is stamped
    on the AVAILABILITY GRID, `max(deadline, filing_date)`, because Phase 2.2 measured the
    deadline stamp leaking on 16.5% of filings / 36.7% of value for that family and fixed it.
    A late elite filer therefore SHOULD produce a step after the deadline -- 2015-07-23 for a
    2015-03-31 period, whose deadline was 2015-05-15. What must never happen there is a step
    BEFORE the deadline, so that is what is scored: on-deadline for `inst`, not-early for
    `super`.
    """
    columns = [c for c in feature_columns(panel)
               if c.startswith(("f_ic_inst_", "f_ic_super_")) and split_leg(c)[1] == "raw"]
    quarters = sorted(pd.to_datetime(pd.Series(periods).dropna().unique()))
    if not columns or not quarters:
        return CheckResult("L2", "leakage", "13F steps only on period+45d", SKIP,
                           measured="no 13F feature legs, or no periods in the source")
    deadlines = pd.DatetimeIndex([pd.Timestamp(p) + pd.Timedelta(days=F13_LAG_DAYS)
                                  for p in quarters]).sort_values()
    grid = pd.DatetimeIndex(sorted(panel["date"].unique()))
    #: A deadline that is not a trading day is honoured on the next one, so the allowed step
    #: dates are the deadlines SNAPPED FORWARD onto the grid -- the same rule `decay_events`
    #: uses, and without it every Saturday deadline reads as a violation.
    snapped = grid[np.clip(grid.searchsorted(deadlines, side="left"), 0, len(grid) - 1)]

    # Narrowed BEFORE the sort: a `sort_values` over the whole ~100-column panel copies it,
    # and at 3.9M rows that is the difference between 200MB and 1.5GB. Same reason
    # `step_cube_institutionals` loads one source at a time.
    ordered = panel[["ticker", "date"] + columns].sort_values(["ticker", "date"])
    rows = []
    for column in columns:
        live = ordered[column].notna()
        moved = ordered.groupby("ticker", sort=False)[column].diff().abs() > 0
        per_day = pd.DataFrame({"date": ordered["date"].to_numpy(),
                                "moved": (moved & live).to_numpy(),
                                "live": live.to_numpy()}).groupby("date").sum()
        share = per_day["moved"] / per_day["live"].replace(0, np.nan)
        steps = pd.DatetimeIndex(share.index[share.fillna(0.0) > cohort])
        live_days = int((per_day["live"] > 0).sum())
        if not live_days:
            continue
        if len(steps) > daily_share * live_days:
            rows.append({"feature": column, "cohort_steps": len(steps), "off_deadline": 0,
                         "verdict": "daily by construction (price-scaled)"})
            continue
        elite = column.startswith("f_ic_super_")
        if elite:
            # The elite rule: a step may be LATE (a late filer becoming public) but never
            # EARLY. `d` is early iff no deadline at or before it exists within tolerance --
            # i.e. its nearest PRECEDING deadline is missing entirely.
            off = [d for d in steps
                   if not len(snapped)
                   or grid.searchsorted(d) + tolerance < grid.searchsorted(snapped[0])
                   or (snapped <= d + pd.Timedelta(days=1)).sum() == 0]
            verdict = "clean (late steps allowed: availability grid)" if not off \
                else "STEPS BEFORE A DEADLINE"
        else:
            off = [d for d in steps
                   if not len(snapped) or abs(grid.searchsorted(d)
                                              - grid.searchsorted(snapped)).min() > tolerance]
            verdict = "clean" if not off else "STEPS OFF A DEADLINE"
        rows.append({"feature": column, "cohort_steps": len(steps), "off_deadline": len(off),
                     "first_off_deadline": str(off[0].date()) if off else "-",
                     "rule": "not-early (availability grid)" if elite else "on-deadline",
                     "verdict": verdict})
    bad = [r for r in rows if r["off_deadline"]]
    return CheckResult("L2", "leakage", "13F features step only on a filing deadline",
                       FAIL if bad else (PASS if rows else SKIP),
                       measured=f"{len(rows) - len(bad)}/{len(rows)} 13F raw legs clean "
                                f"({sum(r['verdict'].startswith('daily') for r in rows)} "
                                f"daily by construction)",
                       expected="every cohort step within "
                                f"{tolerance} trading days of a period+{F13_LAG_DAYS}d deadline",
                       detail=rows, blocking=True)


def leak_event_date_stamping(check_id: str, family: str, source_columns: list[str],
                             event_column: str) -> CheckResult:
    """L3 / L4 -- a family must be stamped on `filing_date`, never on the event date.

    Scored STRUCTURALLY where it can be: if the family's source projection does not read the
    event-date column at all, no feature can be stamped on it, and the projection in
    `utils/common/sources.py` is the evidence. That is a stronger proof than a sampled
    change-date test and it costs nothing -- `sec_13d` / `sec_13g` never read `date_of_event`,
    so L4 holds by construction. Where the column IS read (insider reads `transaction_date` to
    link an exercise-and-sell package) the check falls back to REPORT, because reading it is
    legitimate and only a change-date test on the built panel can settle the stamping.
    """
    reads = event_column in source_columns
    return CheckResult(
        check_id, "leakage", f"{family} stamped on filing_date, never {event_column}",
        REPORT if reads else PASS,
        measured=(f"the {family} projection READS {event_column} -- legitimate (transaction "
                  f"linkage) but the stamping needs a change-date test on the panel"
                  if reads else
                  f"the {family} projection does not read {event_column}: stamping on it is "
                  f"structurally impossible"),
        expected=f"{event_column} unread, or a clean change-date test",
        detail=[{"projection": family, "columns_read": ", ".join(source_columns)}],
        blocking=False)


def leak_first_period_delta(panel: pd.DataFrame) -> CheckResult:
    """L10 -- the first 13F period has no prior quarter, so its QoQ deltas must be NaN on the
    deadline rather than a universe-wide positive.

    D17 is on record for this: a delta against a missing prior quarter reads as every holder in
    the universe initiating at once.

    ⚠ `ic_inst_*` ONLY, and scoring `ic_super_*` here was a check defect that failed four
    correct legs on the 2026-09-12 run. `FIRST_13F_PERIOD` is 2013-06-30 -- the BROAD family's
    coverage regime start (192 -> 3,046 filers), which is why its deltas have no comparable
    predecessor. The elite family reads `sec13f_manager_holdings`, a per-manager walk with no
    such break, and its own history starts 2011-11-14; by 2013-08-14 it has six prior quarters
    and a delta there is the correct value, not a phantom.
    """
    deadline = FIRST_13F_PERIOD + pd.Timedelta(days=F13_LAG_DAYS)
    columns = [c for c in feature_columns(panel)
               if c.startswith("f_ic_inst_")
               and any(k in c for k in ("_chg", "_delta", "_qoq"))]
    day = panel.loc[panel["date"] == panel.loc[panel["date"] >= deadline, "date"].min()]
    if not columns or day.empty:
        return CheckResult("L10", "leakage", f"No delta spike on {deadline.date()}", SKIP,
                           measured="no 13F delta legs, or the panel starts after the deadline")
    rows = [{"feature": c, "non_null_on_deadline": int(day[c].notna().sum()),
             "cross_sectional_mean": (round(float(day[c].mean()), 6)
                                      if day[c].notna().any() else None)}
            for c in columns]
    bad = [r for r in rows if r["non_null_on_deadline"] > 0]
    return CheckResult("L10", "leakage", f"No universe-wide delta spike on {deadline.date()}",
                       FAIL if bad else PASS,
                       measured=f"{len(bad)} of {len(rows)} delta legs non-null on the first "
                                f"13F deadline", expected="0 (no prior quarter to difference)",
                       detail=rows, blocking=True)


# --------------------------------------------------------------------------- reconciliation

def reconcile_holder_count(panel: pd.DataFrame, holdings: pd.DataFrame | None,
                           samples: int = 5) -> CheckResult:
    """R4 -- `ic_inst_holders` reconciles to a direct `COUNT(DISTINCT cik)` on `sec13f_hr`,
    read on the first trading day at or after the 45-day deadline.

    ⚠ **`ic_inst_holders` IS A SHARE, NOT A COUNT** (D28: `holders(ticker, q) / filers(q)`),
    so the reconciliation is `panel_value x universe_filers(q) == distinct cik`, not
    `panel_value == distinct cik`. Comparing the two directly is what made this check report
    0/5 on a correct panel on the 2026-09-12 run -- the redefinition landed in Phase 2.5 and
    the check still spoke the old vocabulary. D28 exists precisely because the raw count jumps
    41 -> 556 per ticker across the 2013-06-30 fetch break; a check that insists on the count
    is asking for the number the design removed.

    ⚠ AND THE SAMPLE MUST COME FROM THE POST-FLOOR ERA. Three of the five pairs sampled on that
    run were pre-2013 quarters, where D16 nulls the level on purpose, so `None != 1` was
    scored as a mismatch when it was the guard working. Periods below `INST_LEVEL_FLOOR` are
    excluded from the draw and the exclusion is reported.

    ⚠ AND THE HOLE QUARTERS TOO, FOR THE SAME REASON ONE STEP FURTHER ON. D17 nulls the LEVEL on
    a hole quarter; the panel is forward-filled; so at that quarter's deadline the panel still
    carries the PRIOR quarter's share, and comparing it against this quarter's source count is a
    cross-quarter comparison that must fail. It did, on `FIX` / 2023-12-31: source 35 filers,
    panel 0.064469, implied 29.85 -- and 0.064469 is byte-identical to the 2023-09-30 value,
    held flat from 2023-11-14 to 2024-05-14 straight across the hole. A non-integer implied
    count is the signature: where the check is comparing like with like the product lands on a
    whole number, as the other four samples did (811, 713, 1210, 2288, all exact).

    The exclusion uses the BUILDER's own `_coverage_periods`, not a re-derived rule and not the
    two dates written down, so the check and the guard cannot drift apart. Breaks are left IN
    deliberately: D17 nulls only the DELTAS on a break, so the level a break quarter carries is
    its own and must still reconcile.
    """
    column = next((c for c in panel.columns
                   if c in ("f_ic_inst_holders", "f_ic_inst_holder_count")), None)
    if column is None or holdings is None or holdings.empty:
        return CheckResult("R4", "reconciliation", "`ic_inst_holders` == COUNT(DISTINCT cik)",
                           SKIP, measured="no holder-count leg in the panel, or no 13F source")
    src = holdings.copy()
    src["period"] = pd.to_datetime(src["period"], errors="coerce")
    filers = src.groupby("period")["cik"].nunique()          # the D28 denominator
    holes, _breaks, _cov = _coverage_periods(src, COVERAGE_BREAK_DEFAULT)
    src = src.loc[(src["period"] >= INST_LEVEL_FLOOR_PERIOD) & (~src["period"].isin(holes))]
    counts = src.groupby(["ticker", "period"])["cik"].nunique()
    if counts.empty:
        return CheckResult("R4", "reconciliation", f"`{column}` reconciles to the source", SKIP,
                           measured=f"no 13F period at or after the D16 level floor "
                                    f"({INST_LEVEL_FLOOR_PERIOD.date()})")
    rows = []
    for (ticker, period), expected in counts.sample(
            min(samples, len(counts)), random_state=7).items():
        deadline = period + pd.Timedelta(days=F13_LAG_DAYS)
        slice_ = panel.loc[(panel["ticker"] == ticker) & (panel["date"] >= deadline)]
        if slice_.empty:
            continue
        got = slice_.iloc[0][column]
        denom = float(filers.get(period, np.nan))
        implied = float(got) * denom if pd.notna(got) and denom > 0 else np.nan
        rows.append({"ticker": ticker, "period": str(period.date()),
                     "source_distinct_cik": int(expected), "universe_filers": int(denom)
                     if pd.notna(denom) else None,
                     "panel_share": None if pd.isna(got) else round(float(got), 6),
                     "implied_count": None if pd.isna(implied) else round(implied, 2),
                     "match": bool(pd.notna(implied) and abs(implied - expected) < 0.5)})
    bad = [r for r in rows if not r["match"]]
    return CheckResult("R4", "reconciliation",
                       f"`{column}` x universe filers == COUNT(DISTINCT cik) on source",
                       FAIL if bad else (PASS if rows else SKIP),
                       measured=f"{len(rows) - len(bad)}/{len(rows)} sampled (ticker, quarter) "
                                f"pairs reconcile (drawn at or after the D16 level floor "
                                f"{INST_LEVEL_FLOOR_PERIOD.date()}, excluding the "
                                f"{len(holes)} D17 hole quarter(s) "
                                f"{sorted(str(p.date()) for p in holes)} where the level is "
                                f"nulled on purpose and the panel carries the prior quarter)",
                       expected="all", detail=rows, blocking=False)


def value_signal_age(panel: pd.DataFrame, max_tickers: int = 40) -> CheckResult:
    """V8 -- `ic_sig_*_age_days` is NaN before the family's first event and increases by
    EXACTLY 1 per trading day between events.

    The step of +1 is what makes the whole conditioning layer auditable: it says the clock is
    running on the trading grid rather than on calendar days (which would step by 3 over a
    weekend) and that the panel's date axis has no holes. A drop to 0 is an event; any other
    change is a defect.
    """
    columns = [c for c in feature_columns(panel)
               if c.startswith("f_ic_sig_") and c.endswith("_age_days")]
    if not columns:
        return CheckResult("V8", "value", "`ic_sig_*_age_days` steps by exactly 1 per day",
                           SKIP, measured="no `ic_sig_*_age_days` legs in the panel")
    sample = sorted(panel["ticker"].dropna().unique())[:max_tickers]
    narrow = panel.loc[panel["ticker"].isin(sample), ["ticker", "date"] + columns]
    ordered = narrow.sort_values(["ticker", "date"])
    rows = []
    for column in columns:
        step = ordered.groupby("ticker", sort=False)[column].diff()
        live = ordered[column].notna() & ordered.groupby("ticker", sort=False)[column].shift(
            1).notna()
        moves = step[live]
        resets = int((ordered[column][live] == 0).sum())
        bad = int(((moves != 1.0) & (ordered[column][live] != 0)).sum())
        # a zero after a non-null value is an EVENT (the clock restarts), not a bad step
        zeros_before_first = 0
        for ticker in sample:
            series = ordered.loc[ordered["ticker"] == ticker, column]
            first = series.first_valid_index()
            if first is not None:
                zeros_before_first += int((series.loc[:first].iloc[:-1] == 0).sum())
        rows.append({"feature": column, "steps_checked": int(live.sum()),
                     "bad_steps": bad, "event_resets": resets,
                     "zeros_before_first_event": zeros_before_first})
    bad_rows = [r for r in rows if r["bad_steps"] or r["zeros_before_first_event"]]
    return CheckResult("V8", "value",
                       "`ic_sig_*_age_days`: +1 per trading day, NaN (never 0) before the "
                       "first event", FAIL if bad_rows else PASS,
                       measured=f"{len(rows) - len(bad_rows)}/{len(rows)} age legs clean on "
                                f"{len(sample)} sampled tickers",
                       expected="0 bad steps, 0 pre-event zeros", detail=rows, blocking=True)


def behavioural_moves_without_a_filing(panel: pd.DataFrame, span: int = 30,
                                       pairs: int = 10) -> CheckResult:
    """B1 -- THE defining check (part-2-validation.md 10.4, report acceptance test #13): on a
    span with NO filing of any kind, the conditioning and decayed columns move every trading
    day while the 13F level features hold flat.

    The "no filing" spans are found FROM the panel rather than from the sources, and
    `ic_sig_*_age_days` is what makes that possible: it increments by exactly 1 on every day
    with no event of its family, so a span where all three age legs increment monotonically is
    a span with no 13F, Form 4 or 13D filing, by construction. That is the same fact the
    source tables would give, read off the object under test.
    """
    ages = [c for c in feature_columns(panel)
            if c.startswith("f_ic_sig_") and c.endswith("_age_days")]
    moving = [c for c in feature_columns(panel)
              if c.startswith("f_ic_sig_") and split_leg(c)[1] == "raw"
              and not c.endswith("_age_days")]
    flat = [c for c in feature_columns(panel)
            if c.startswith("f_ic_inst_") and split_leg(c)[1] == "raw"]
    if not ages or not moving:
        return CheckResult("B1", "behavioural", "The panel moves on a day with no filing",
                           SKIP, measured="no `ic_sig_*` family in the panel")

    cols = ages + moving + flat
    ordered = panel[["ticker", "date"] + cols].sort_values(["ticker", "date"])
    rows = []
    for ticker, group in ordered.groupby("ticker", sort=False):
        if len(rows) >= pairs or len(group) < span + 2:
            continue
        steps = group[ages].diff()
        quiet = (steps == 1.0).all(axis=1)          # every family's clock just ticked
        if not quiet.any():
            continue
        run = quiet.rolling(span).sum()
        ends = run[run >= span].index
        if not len(ends):
            continue
        end = ends[-1]
        window = group.loc[:end].tail(span)
        moved = {c: int((window[c].diff().abs() > 0).sum()) for c in moving}
        held = {c: int((window[c].diff().abs() > 0).sum()) for c in flat}
        rows.append({"ticker": ticker,
                     "span": f"{window['date'].iloc[0].date()} -> {window['date'].iloc[-1].date()}",
                     "conditioning_legs_moving_every_day":
                         sum(1 for c, n in moved.items() if n == span - 1),
                     "conditioning_legs": len(moving),
                     "inst_legs_flat": sum(1 for c, n in held.items() if n == 0),
                     "inst_legs": len(flat)})
    if not rows:
        return CheckResult("B1", "behavioural", "The panel moves on a day with no filing",
                           SKIP, measured=f"no {span}-day filing-free span found")
    ok = [r for r in rows if r["conditioning_legs_moving_every_day"] > 0]
    return CheckResult("B1", "behavioural",
                       f"The panel MOVES on a {span}-day span with no filing (acceptance #13)",
                       PASS if len(ok) == len(rows) else FAIL,
                       measured=f"{len(ok)}/{len(rows)} filing-free spans have conditioning "
                                f"legs moving every trading day",
                       expected="every span", detail=rows, blocking=True)


def reconcile_group_summing(sec_13d: pd.DataFrame | None) -> CheckResult:
    """R3 -- a co-filer group's aggregate is taken as a MAX, never a SUM.

    Measured on the SOURCE rather than asserted on the panel, because the panel only shows the
    corrected number: the evidence that the trap is real is the `sum / max` ratio on live
    multi-reporting-person events, and V1-V3's 0-100 bound on `ic_act_percent_of_class` is what
    catches a regression.
    """
    if sec_13d is None or sec_13d.empty or "percent_of_class" not in sec_13d.columns:
        return CheckResult("R3", "reconciliation", "13D group aggregate not summed", SKIP,
                           measured="no `sec_13d` source")
    live = sec_13d.dropna(subset=["percent_of_class"])
    grouped = live.groupby(["ticker", "accession_number", "cusip"], dropna=False)["percent_of_class"]
    agg = pd.DataFrame({"mx": grouped.max(), "sm": grouped.sum(), "n": grouped.size()})
    multi = agg[agg["n"] > 1]
    if multi.empty:
        return CheckResult("R3", "reconciliation", "13D group aggregate not summed", PASS,
                           measured="no multi-reporting-person event carries a stake")
    ratio = multi["sm"] / multi["mx"]
    return CheckResult("R3", "reconciliation",
                       "13D group aggregate is a MAX (summing would inflate it)", REPORT,
                       measured=f"{len(multi):,} multi-filer event(s) with a stake; summing "
                                f"would inflate by mean {ratio.mean():.2f}x, worst "
                                f"{ratio.max():.2f}x",
                       detail=[{"multi_filer_events": len(multi),
                                "mean_sum_over_max": round(float(ratio.mean()), 3),
                                "worst_sum_over_max": round(float(ratio.max()), 3)}])


# --------------------------------------------------------------------------- driver

def family_floors(store, log=None) -> dict[str, pd.Timestamp]:
    """Per-family availability floor, MEASURED from each source table."""
    from src.data_store.schema import Tables

    by_name = {t.name: t for t in vars(Tables).values() if hasattr(t, "name")}
    floors: dict[str, pd.Timestamp] = {}
    for prefix, (table_name, column, lag) in FAMILY_SOURCES.items():
        table = by_name.get(table_name)
        if table is None or not store.exists(table):
            continue
        frame = store.load(table, columns=[column], optional=True)
        if frame is None or frame.empty:
            continue
        earliest = pd.to_datetime(frame[column], errors="coerce").min()
        if pd.notna(earliest):
            floors[prefix] = earliest + pd.Timedelta(days=lag)
    if log is not None:
        log.info("Availability floors: %s",
                 {k: str(v.date()) for k, v in sorted(floors.items())})
    return floors


def run_institutionals_validation(context, config, *, panel: pd.DataFrame | None = None,
                                  floor: float = COVERAGE_FLOOR) -> InstitutionalsReport:
    """Build (or take) the part panel and score every runnable check.

    `panel=None` rebuilds it through `StepCubeInstitutionals.build_panel` -- the step's OWN
    merge chain, not a copy of it. Reading the persisted table instead would score whatever
    was last written, which is how a stale-row false green happens; `validate prices` keeps the
    same rule.
    """
    from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
    from src.data_store.schema import Tables

    store = context.store
    if panel is None:
        context.log.info("Building cube_part_institutionals in memory (full history)...")
        panel, _ = StepCubeInstitutionals(context, config).build_panel(full=True)
    else:
        # Only a CALLER's frame is copied. The built one is ours and a defensive copy of a
        # ~100-column, multi-million-row panel doubles peak RSS for nothing.
        panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    declared = _declared_emission_maps()
    floors = family_floors(store, context.log)
    holdings = (store.load(Tables.sec13f_hr, columns=["ticker", "period", "cik"], optional=True)
                if store.exists(Tables.sec13f_hr) else None)
    # ⚠ R4 MUST READ THE SAME SOURCE SCOPE THE BUILDER READS. The step now cuts `sec13f_hr` to
    # the universe before aggregating, so its D28 denominator is the filer count over
    # IN-UNIVERSE holdings. Re-counting here over the whole table made the denominator 3 filers
    # too large (filers whose only holdings are off-universe names), and AMAT/2017-12-31 came
    # back as 1,124.8 against 1,124 -- a mismatch of 0.8 that was the check's own basis, not
    # the panel's arithmetic. This is the "two declarations of one fact" failure in check form.
    if holdings is not None and not holdings.empty:
        universe = set(panel["ticker"].astype(str).unique())
        holdings = holdings.loc[holdings["ticker"].astype(str).isin(universe)]
    sec_13d = (store.load(Tables.sec_13d, optional=True)
               if store.exists(Tables.sec_13d) else None)
    # V12's reference set. Projected to the one column: the price part is ~3.9M rows and this
    # check needs its ticker axis, not its prices.
    price_universe = None
    if store.exists(Tables.cube_part_prices):
        price_universe = set(store.load(Tables.cube_part_prices, columns=["ticker"])["ticker"]
                             .astype(str).unique())

    from src.data_aggregate.utils.common.sources import SOURCE_COLUMNS

    checks: list[CheckResult] = [
        leak_availability(panel, floors),
        leak_13f_lag(panel, holdings["period"] if holdings is not None else pd.Series(dtype=object)),
        leak_event_date_stamping("L3", "insider_transactions",
                                 SOURCE_COLUMNS.get("insider_transactions", []),
                                 "transaction_date"),
        leak_event_date_stamping("L4", "sec_13d/sec_13g",
                                 SOURCE_COLUMNS.get("sec_13d", []) +
                                 SOURCE_COLUMNS.get("sec_13g", []), "date_of_event"),
        leak_first_period_delta(panel),
        reconcile_group_summing(sec_13d),
        reconcile_holder_count(panel, holdings),
        value_declared_bounds(panel),
        value_peer_z(panel),
        value_xs_unit(panel),
        value_universe_scope(panel, price_universe),
        value_degenerate_legs(panel),
        value_decay_behaviour(panel),
        value_signal_age(panel),
        behavioural_moves_without_a_filing(panel),
        *saturation_profile(panel),
        check_declared_vs_emitted(panel, declared),
    ]
    g1, g2 = gate_coverage(panel, floor=floor, family_floors=floors)
    checks += [g1, g2, gate_redundancy(panel)]
    checks += [CheckResult(cid, grp, title, SKIP, measured=reason)
               for cid, grp, title, reason in SKIPPED]

    return InstitutionalsReport(
        checks=checks, rows=len(panel), tickers=int(panel["ticker"].nunique()),
        features=len(feature_columns(panel)), columns=len(panel.columns),
        date_min=panel["date"].min(), date_max=panel["date"].max())


def _declared_emission_maps() -> dict[str, dict[str, str]]:
    """Each builder's own `EMISSION` map, by module name -- all seven now declare one, so D1
    scores every family.

    A builder without one contributes nothing rather than raising, which is what let this
    check run at all while `institutional_features` and the short-flow module were still
    undeclared: D1 reported their columns as "emitted without a declaration" instead of
    pretending they had been checked.
    """
    import importlib

    out: dict[str, dict[str, str]] = {}
    for module_name in ("cross_source_features", "insider_features",
                        "institutional_features", "ownership_features",
                        "short_flow_features", "signal_conditioning",
                        "superinvestor_features"):
        module = importlib.import_module(
            f"src.data_aggregate.utils.institutionals.{module_name}")
        emission = getattr(module, "EMISSION", None)
        if isinstance(emission, dict):
            out[module_name] = emission
    return out


__all__ = ["CLIP_EDGE", "COVERAGE_FLOOR", "CheckResult", "DECLARED_BOUNDS", "FAMILY_SOURCES",
           "InstitutionalsReport", "SKIPPED", "behavioural_moves_without_a_filing",
           "check_declared_vs_emitted", "family_floors",
           "feature_columns", "gate_coverage", "gate_redundancy", "leak_13f_lag",
           "leak_availability", "leak_event_date_stamping", "leak_first_period_delta",
           "reconcile_group_summing", "reconcile_holder_count", "run_institutionals_validation",
           "saturation_profile", "split_leg", "value_declared_bounds", "value_decay_behaviour",
           "value_degenerate_legs", "value_peer_z", "value_signal_age", "value_universe_scope",
           "value_xs_unit"]
