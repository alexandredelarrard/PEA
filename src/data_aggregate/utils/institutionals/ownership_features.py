"""
ownership_features.py (src/data_aggregate/utils/institutionals/ownership_features.py)
--------------------------------------------------------------------------------------
Beneficial-ownership panel: Schedule 13D (`ic_act_*`, activist) and Schedule 13G (`ic_bo_*`,
passive >5%) event history. Sources: `sec_13d`, `sec_13g`. `sec_13d_transactions` (the Item
5(c) 60-day trade log) is deliberately NOT read here -- the registry marks it "not used" for
this feature set (18 tickers total), and a projection with no reader is not free -- so
its registry entry declares no `read_columns`; it stays available in the DB for a future
consumer.

CANONICAL EVENT CONSTRUCTION (the group-summing trap). Both tables are at reporting-person
grain (`rp_seq`): a joint filing repeats the SAME `aggregate_amount` / `percent_of_class` on
every co-filer's row. Summing across `rp_seq` turns a 10%-stake filing by four co-filers into
40%. Every event here is first collapsed to one row per `(ticker, accession_number, cusip)` --
`percent_of_class` and `aggregate_amount` taken as `max` within the group, never `sum` --
before any feature reads it.

`filer_id` -- the identity a position is tracked BY over time (new-holder, escalation,
holder-count, the summed-percent features) -- is the lexicographically smallest non-null
`reporting_person_cik` in the canonical group (falling back to `reporting_person_name` when no
co-filer has a CIK). This is a deliberate simplification: it assumes a joint filing's
membership is stable enough, across its own amendments, to stand in for "the same beneficial
owner" over time. Building true group-membership resolution is out of scope for this pass.

`ic_bo_holder_count` HAS NO EXIT SIGNAL, so activity stands in for holding: a filer counts as a
holder for `HOLDER_ACTIVE_DAYS` after each 13G and then lapses (see that constant for the Rule
13d-2(b) cadence it rests on). D28 also redefines the feature: the count is divided by the
universe-wide distinct-13G-filer count over THE SAME trailing window, which is what makes it a
share. `emit: raw` only, per D28.

⚠ BOTH HALVES OF THAT RATIO HAD TO CHANGE AND THE EVIDENCE WAS THE RATIO ITSELF. Built with a
never-releasing numerator over a calendar-year denominator it reached **1.23** -- a "share" of
more than everything, because the numerator accumulated 30 years of filers while the
denominator counted one year of them, and because the newest calendar year is always partial
(measured 2026-09-08: 102 filers year-to-date against 140 for all of 2025), which also made
every current-year value get revised as the year filled in. One shared trailing window fixes
both: max **1.00**, p50 0.0149, p99 0.167, and nothing is ever revised after the fact.

SOURCES AS MEASURED (2026-09-11, live tables):

    sec_13d   4,705 rows | 274 tickers | 3,854 accessions | 1995-09-06 -> 2026-08-28
              is_amendment 100% filled, only 0.0/1.0 -> 626 initial events, 3,228 amendments
              cusip 100% | rp_cik 87.1% (all 10-digit zero-padded) | rp_name 98.8%
              item4 text 68.8% of canonical events | 1.22 reporting persons/event, max 22
    sec_13g  30,199 rows | 499 tickers | 28,926 accessions | 1996-01-26 -> 2026-09-08
              cusip 15.4% | rp_cik 97.1% | 1.04 reporting persons/event, max 29

⚠ THE GROUP-SUMMING TRAP IS REAL AND BIG, measured rather than assumed: on the multi-reporting
person events that carry a `percent_of_class`, `sum / max` averages **4.02x** and reaches
**14.77x** on `sec_13d` (2.51x / 14.98x on `sec_13g`). Canonicalizing with `max` is not
defensive tidying -- summing would have reported a 15% stake as 220%.

⚠ `cusip` IS 84.6% NULL ON `sec_13g` and it is part of the canonical key. Measured, that costs
exactly ONE split event (28,927 canonical events against 28,926 accessions), because the null
is all-or-nothing within a filing -- it is the pre-mandate header-only parse, the same 15.4%
that `percent_of_class` has. Grouping on the null-as-"" is therefore safe here, but it is safe
by a data condition rather than by construction.

MANDATE FLOOR. `percent_of_class` is ~0% filled on `sec_13d`/`sec_13g` before the 2024-12-17
XML beneficial-ownership mandate (edgartools builds pre-mandate filings from the SGML header
alone). Measured fill by filing year -- `sec_13d`: 0.0% through 2023, **17.0%** 2024, **98.0%**
2025, **99.8%** 2026; `sec_13g`: 0.0% through 2023, **1.7%** 2024, **100%** 2025 and 2026. The
four features built from it -- `ic_act_percent_of_class`, its delta, and the two `ic_bo_*`
twins -- are explicitly masked to NaN before `MANDATE_FLOOR` even though the input is already
null there, matching the defensive floor-masking used elsewhere in this family (e.g. insider's
`TEN_B5_1_FLOOR`). Verified on the built panel: **0** pre-mandate non-null values on all four,
against 16,607 / 7,764 / 155,023 / 77,140 post-mandate.

`ic_act_disclosure_lag_days` is NOT built: `date_of_event` is 0% filled pre-2024-12 and the
post-mandate sample alone fails the coverage floor. `rule_designation` / `is_passive_investor`
(stored in `sec_13g`, post-mandate only, near-constant on this universe) are not featurized.

ITEM 4 KEYWORD CATEGORIES ARE DETERMINISTIC ONLY (report Sec 8.1 puts LLM classification out
of scope). Each category is a separate binary event; they are never collapsed into an
activist score. `sec_13g` has no Item 4 text at all, so the two purpose features are a
`sec_13d`-only concept.

THE TWO DELTA FEATURES ARE PER-FILER SUMS OF PER-FILER CHANGES, HELD FORWARD -- the registry's
"minus the same filer's prior filing", taken literally. Two earlier constructions were wrong
and each was caught by a measurement, not by reading the code:

  1. `level.diff()` on the forward-filled level is a ONE-DAY SPIKE: non-zero on ~200 ticker-
     days in a 7,803-day panel, a dead column wearing a class-D label. The fix is to record
     the change on the filing day and hold it forward (`_delta_state`), which also lets a
     re-filing at an unchanged stake read a true 0.
  2. Diffing the SUMMED level reads FILER-SET GROWTH as buying. The summed 13G level steps up
     whenever a new filer is first observed, so its diff had a median of **+5.1 percentage
     points** -- a second index fund appearing in the table, not anyone adding to a position.
     Per-filer diffs are NaN until that filer's second filing; median moved to **-1.6pp** and
     p99 from 14.4 to 4.49. The same artifact inflated the 13D side's p99 from 2.6 to 38.7.

Half-lives (`act` and `bo`, both default 126 trading days -- "campaigns run for quarters, not
weeks") come from `build_cube.institutionals.decay_halflife`.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.institutionals.decay import (
    days_since_last_true, decay_events, snap_to_grid)
from src.data_aggregate.utils.common.price_frames import PriceFrames


#: D5: every builder answers an absent source with the SAME empty frame. A fresh object each
#: call, never a module-level constant -- `PanelMerger.add` and several callers reindex or
#: assign onto what they get back, and a shared instance would be mutated across builds.
def _EMPTY_PANEL() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker"])


#: The columns `_canonicalize` reads off EITHER schedule without checking first. The
#: canonical-event key (`ticker`, `accession_number`, `cusip`) plus the only legal stamp
#: (`filing_date` -- `date_of_event` is never projected, which is what makes L4 structural)
#: plus the two ownership/identity legs.
_NEED = {"ticker", "accession_number", "cusip", "filing_date", "percent_of_class",
         "reporting_person_cik"}


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """True when `df` cannot be built from: missing, empty, or short a required column.

    The three-part test is the D5 entry contract stated once. `need` is the set the builder
    dereferences unconditionally -- a column it only uses `if present` does NOT belong here,
    or an optional projection turns into an empty panel.
    """
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)


#: D28: 13G filing volume is not constant over 15 years, so the raw filer count is meaningless
#: on its own -- see the module docstring.
ACT_HALFLIFE_DEFAULT = 126.0
BO_HALFLIFE_DEFAULT = 126.0

#: XBRL beneficial-ownership data became mandatory on this date; `percent_of_class` reads are
#: ~0% filled before it. NaN, never 0, before the floor -- a 0 would claim a disclosed stake of
#: nothing rather than "not disclosed this way yet".
MANDATE_FLOOR = pd.Timestamp("2024-12-17")

#: How long a 13G filing keeps its filer counted as a holder, in TRADING days (~18 months).
#: The schema carries no "no longer holds >5%" signal, so activity has to stand in for
#: holding, and Rule 13d-2(b) sets the cadence it stands on: a 13G holder owed an annual
#: amendment within 45 days of year-end (quarterly since the 2024 amendments), so a still-live
#: >5% position refiles at least yearly and 18 months tolerates one late or skipped cycle.
#: Both sides of `holder_count` use this ONE window -- an ever-accumulating numerator over a
#: recent-flow denominator is not a share of anything, and measured that way the "share"
#: reached 1.23.
HOLDER_ACTIVE_DAYS = 378

_BOARD_RE = re.compile(
    r"board seat|board representation|board of directors|nominat|director representation",
    re.IGNORECASE)
_STRATEGIC_RE = re.compile(
    r"strategic alternative|strategic review|sale of the (?:issuer|company)|"
    r"business combination|merger|explore.{0,20}alternative",
    re.IGNORECASE)

#: Emission, on the Phase 2.2b TIE-FRACTION criterion, measured on the panel this module
#: builds from the live tables (7,803 trading days x the 500-name universe, 2026-09-11):
#:
#:     feature                          tickers  cov_all%  cov_2025%  ties/date%      p50    p99
#:     initial_13d                          257     33.05      50.56         0.3   0.0000  1.247
#:     amendment_intensity                  243     31.46      47.97         0.2   0.0075  4.798
#:     campaign_age_days                    257     33.05      50.56         2.9   1911.0   6967
#:     repeat_activist                       36      3.21       7.20         2.2   0.0001  0.934
#:     purpose_board                        211     24.93      41.52         0.0   0.0004  2.772
#:     purpose_strategic                    172     20.67      33.57         0.1   0.0000  2.575
#:     act_percent_of_class                  60      0.43       7.89         2.7   13.100  62.60
#:     act_delta_percent_class               36      0.20       3.70        13.4  -0.2000  2.600
#:     bo_holder_count                      499     44.48      94.56        70.3   0.0149  0.167
#:     bo_new_holder                        499     44.98      99.30         2.9   0.1733  3.552
#:     bo_percent_of_class                  499      3.97      73.75        24.8   13.540  51.02
#:     bo_delta_percent_class               366      1.98      36.72        17.2  -1.6000  4.490
#:     escalation_13g_to_13d                 11      0.58       2.16         0.0   0.0017  1.244
#:     de_escalation_13d_to_13g              20      1.64       4.00        23.4   0.0002  0.941
#:
#: ⚠ THREE FEATURES FAIL THE 5% COVERAGE FLOOR even on the 2025 cross-section --
#: `repeat_activist` (7.20% in 2025 but 3.21% over the whole panel), `act_delta_percent_class`
#: (3.70%), `escalation_13g_to_13d` (2.16%) and `de_escalation_13d_to_13g` (4.00%). They are
#: BUILT and reported here rather than dropped silently, because the cut is Phase 2.8's gate
#: to make with the rest of the part in front of it -- and because the escalation pair is the
#: report's #3 post-extraction priority and its 14 events are individually real (Berkshire ->
#: DVA 2012-10-10 13G then 2017-08-11 13D; Paulson -> TMUS 2013-02-14 then 2013-03-01, both
#: confirmed against EDGAR).
EMISSION: dict[str, str] = {
    "ic_act_initial_13d":              "raw",       # decayed occurrence; 0.3% ties, no drift
    "ic_act_amendment_intensity":      "raw",       # 0.2% ties
    "ic_act_campaign_age_days":        "raw+xs",    # 2.9% ties AND a real drift -- see below
    "ic_act_repeat_activist":          "raw",       # 2.2% ties, 36 tickers ever
    "ic_act_purpose_board":            "raw",       # 0.0% ties
    "ic_act_purpose_strategic":        "raw",       # 0.1% ties
    "ic_act_percent_of_class":         "raw+xs",    # NOT peers -- measured 0.0% non-null, below
    "ic_act_delta_percent_class":      "raw+xs",    # 13.4% ties, still informative
    "ic_bo_holder_count":              "raw",       # D28-normalized; 70.3% ties -> no xs leg
    "ic_bo_new_holder":                "raw",       # 2.9% ties
    "ic_bo_escalation_13g_to_13d":     "raw",
    "ic_bo_de_escalation_13d_to_13g":  "raw",       # 23.4% ties
    "ic_bo_percent_of_class":          "raw+peers",
    "ic_bo_delta_percent_class":       "raw+xs",    # 17.2% ties
}

#: ⚠ `ic_act_percent_of_class` TAKES NO `_vs_peers` LEG, AND THE DECLARED ONE WAS DEAD. Built
#: as `raw+peers` it emitted `f_ic_act_percent_of_class_vs_peers` at **0.0% non-null over
#: 3.89M panel rows** against its own raw leg's 0.4%: only 60 tickers ever carry a 13D
#: `percent_of_class`, so a peer basket almost never holds two of them at once, and
#: `peer_relative` returns NaN for a basket with no dispersion (D25, and `decay.py`'s reason
#: for existing). That is an ABSENT feature wearing a declared column name -- the failure mode
#: no test catches, because a test that asserts the column exists passes on an all-NaN column.
#: `ic_bo_percent_of_class` keeps its peer leg: 499 tickers carry it and it measures 3.7%
#: non-null against a 4.0% raw leg.
#:
#: ⚠ `ic_act_campaign_age_days` IS THE ONE FEATURE THAT TAKES AN `_xs` LEG ON DRIFT RATHER THAN
#: ON ITS TIE FRACTION, and it is the same argument the plan flags for insider's
#: `days_since_last_buy`. Measured on the last grid day: 257 names, MEDIAN 3,879 trading days
#: and a max of 7,800 -- i.e. the typical "campaign age" is 15 years, because the feature keeps
#: counting long after the campaign ended and the count is mechanically small early in the
#: sample only because the history is short. The raw leg stays (a day count is a quantity in
#: its own units), but the within-date percentile is what makes 2003 comparable to 2026.


def _canonicalize(df: pd.DataFrame, text_col: str | None = None,
                   has_amendment: bool = True) -> pd.DataFrame:
    """One row per `(ticker, accession_number, cusip)` -- see module docstring. `filer_id` is
    the group's identity for time-series tracking; `n_reporting_persons` is the co-filer count
    kept SEPARATE from any ownership number, exactly so nothing downstream is tempted to fold
    it back in."""
    cols = ["ticker", "accession_number", "cusip", "filing_date", "percent_of_class",
            "filer_id", "n_reporting_persons"]
    if has_amendment:
        cols.append("is_amendment")
    if text_col:
        cols.append("text")
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    d = df.copy()
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d.dropna(subset=["ticker", "accession_number", "filing_date"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["cusip"] = d["cusip"].fillna("") if "cusip" in d.columns else ""
    cik = d["reporting_person_cik"] if "reporting_person_cik" in d.columns else pd.Series(
        index=d.index, dtype=object)
    name = d["reporting_person_name"] if "reporting_person_name" in d.columns else pd.Series(
        index=d.index, dtype=object)
    cik = cik.astype(object).where(cik.notna() & (cik.astype(str).str.len() > 0), None)
    d["_filer_key"] = cik.where(cik.notna(), name)

    key = ["ticker", "accession_number", "cusip"]
    agg_map = {
        "filing_date": ("filing_date", "first"),
        "percent_of_class": ("percent_of_class", "max"),
        "filer_id": ("_filer_key", lambda s: (s.dropna().sort_values().iloc[0]
                                               if s.notna().any() else None)),
        "n_reporting_persons": ("_filer_key", "nunique"),
    }
    if has_amendment:
        agg_map["is_amendment"] = ("is_amendment", "first")
    if text_col and text_col in d.columns:
        agg_map["text"] = (text_col, lambda s: next(
            (x for x in s if isinstance(x, str) and x.strip()), None))
    return d.groupby(key, sort=False).agg(**agg_map).reset_index()


def _snap_to_grid(dates: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Snap each date onto the first trading day >= it -- see `decay.snap_to_grid`, which now
    owns the rule because the conditioning layer needs the same one."""
    return snap_to_grid(dates, idx)


def _days_since(bool_wide: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Trading days since the last True per column -- see `decay.days_since_last_true`."""
    return days_since_last_true(bool_wide.reindex(idx))


def _ticker_level_and_mask(canon: pd.DataFrame, value_col: str,
                            idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(level held forward, event mask)`, per ticker. No per-filer split -- the right shape
    for #48 (`ic_act_percent_of_class` is a per-ticker latest-reported value, not a sum).

    The MASK is the grid days a filing carrying a non-null `value_col` actually landed on, and
    it is what makes the delta (#49/#55) a STATE rather than a one-day spike: a plain
    `level.diff()` is non-zero on ~200 ticker-days in the whole 30-year panel, which is a dead
    column dressed as a class-D feature. It also keeps a re-filing at an UNCHANGED stake
    reading a true 0 instead of silently holding the previous change."""
    empty = pd.DataFrame(index=idx, dtype="float64")
    e = canon.dropna(subset=["filing_date", value_col]).sort_values("filing_date").copy()
    if e.empty:
        return empty, empty
    e["_grid_date"] = _snap_to_grid(e["filing_date"], idx)
    e = e.dropna(subset=["_grid_date"])
    if e.empty:
        return empty, empty
    wide = e.pivot_table(index="_grid_date", columns="ticker", values=value_col, aggfunc="last")
    raw = wide.reindex(idx)
    return raw.ffill(), raw.notna()


def _delta_state(level: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """The change recorded on each filing day, held forward until the next one. NaN until a
    ticker's SECOND filing: there is no prior disclosed stake to difference against."""
    if level.empty or mask.empty:
        return pd.DataFrame(index=level.index, dtype="float64")
    return level.diff().where(mask.reindex_like(level).fillna(False)).ffill()


def _filer_frame(canon: pd.DataFrame, value_col: str,
                  idx: pd.DatetimeIndex) -> pd.DataFrame | None:
    """The raw `(ticker, filer_id)`-columned frame of `value_col` on the grid, unfilled: a
    value on the days a filing landed, NaN elsewhere. `None` when nothing qualifies."""
    e = canon.dropna(subset=["filing_date", value_col, "filer_id"]).sort_values("filing_date")
    if e.empty:
        return None
    e = e.assign(_grid_date=_snap_to_grid(e["filing_date"], idx)).dropna(subset=["_grid_date"])
    if e.empty:
        return None
    wide = e.pivot_table(index="_grid_date", columns=["ticker", "filer_id"],
                         values=value_col, aggfunc="last")
    return wide.reindex(idx)


def _filer_delta_sum(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum over filers of each filer's OWN change, held forward -- the registry's "minus the
    same filer's prior filing", done per filer.

    ⚠ NOT the diff of the summed level, which is what makes this its own function. The summed
    level steps up every time a NEW filer is first observed, so its diff reads that SET GROWTH
    as somebody buying: measured on the post-mandate 13G panel, the diff-of-sum has a median
    of **+5.1 percentage points**, which is a second index fund appearing in the table rather
    than any holder adding to a position. A per-filer diff is NaN until that filer's SECOND
    filing, so a newly-observed holder contributes nothing until they actually move."""
    return _sum_over_filers(_delta_state(raw.ffill(), raw.notna()))


def _rolling_distinct(events: pd.DataFrame, idx: pd.DatetimeIndex,
                       window: int) -> pd.Series:
    """Distinct `filer_id` with an event in the trailing `window` TRADING DAYS, as of each
    grid day.

    THIS IS THE D28 DENOMINATOR AND IT HAS TO BE POINT-IN-TIME. A per-CALENDAR-YEAR distinct
    count is not: the newest year is always PARTIAL, so it reads low (measured 2026-09-08: 102
    filers year-to-date against 140 for all of 2025 and 187 for 2024) and every value computed
    inside that year is revised downward as the year fills in -- an inflated `holder_count`
    exactly where the predictions are made. A trailing window is stable the day it is computed
    and never revised. An exact rolling distinct count, not a decayed one (a decayed distinct
    count is not a distinct count).

    It shares `window` with the numerator's activity window on purpose: a ratio of an
    ever-accumulating numerator to a recent-flow denominator is not a share of anything, and
    measured that way `holder_count` reached **1.23**."""
    ev = events.dropna(subset=["filing_date", "filer_id"])
    if ev.empty or len(idx) == 0:
        return pd.Series(0.0, index=idx)
    ev = ev.assign(_grid_date=_snap_to_grid(ev["filing_date"], idx)).dropna(subset=["_grid_date"])
    if ev.empty:
        return pd.Series(0.0, index=idx)
    # Work in GRID POSITIONS, so one constant in trading days drives both sides of the ratio.
    pos = idx.get_indexer(pd.DatetimeIndex(ev["_grid_date"]))
    order = np.argsort(pos, kind="stable")
    pos, filers = pos[order], ev["filer_id"].astype(str).to_numpy()[order]
    active: Counter = Counter()
    out = np.zeros(len(idx), dtype="float64")
    enter = leave = 0
    for k in range(len(idx)):
        while enter < len(pos) and pos[enter] <= k:
            active[filers[enter]] += 1
            enter += 1
        while leave < len(pos) and pos[leave] <= k - window:
            key = filers[leave]
            active[key] -= 1
            if active[key] <= 0:
                del active[key]
            leave += 1
        out[k] = len(active)
    return pd.Series(out, index=idx)


def _sum_over_filers(wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse a `(ticker, filer_id)`-columned wide frame to `ticker`, NaN-preserving: a
    ticker with no active filer at all stays NaN (`min_count=1`) rather than reading as a
    measured zero."""
    return wide.T.groupby(level="ticker").sum(min_count=1).T


def _act_fields(canon: pd.DataFrame, idx: pd.DatetimeIndex,
                 halflife: float) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    is_amend = canon["is_amendment"].fillna(0).astype(float).eq(1.0)
    initial, amend = canon[~is_amend], canon[is_amend]

    out["ic_act_initial_13d"] = decay_events(initial, idx, halflife, date_col="filing_date")
    out["ic_act_amendment_intensity"] = decay_events(amend, idx, halflife, date_col="filing_date")

    occ = initial.assign(_flag=1.0, _grid_date=_snap_to_grid(initial["filing_date"], idx))
    occ = occ.dropna(subset=["_grid_date"])
    wide = occ.pivot_table(index="_grid_date", columns="ticker", values="_flag", aggfunc="max")
    out["ic_act_campaign_age_days"] = _days_since(wide.reindex(idx).notna(), idx)

    initial_sorted = initial.dropna(subset=["filer_id"]).sort_values("filing_date")
    prior_campaigns = initial_sorted.groupby("filer_id").cumcount()
    out["ic_act_repeat_activist"] = decay_events(
        initial_sorted[prior_campaigns >= 3], idx, halflife, date_col="filing_date")

    if "text" in canon.columns:
        has_text = canon["text"].notna()
        board = canon[has_text & canon["text"].str.contains(_BOARD_RE, na=False)]
        strat = canon[has_text & canon["text"].str.contains(_STRATEGIC_RE, na=False)]
        out["ic_act_purpose_board"] = decay_events(board, idx, halflife, date_col="filing_date")
        out["ic_act_purpose_strategic"] = decay_events(strat, idx, halflife, date_col="filing_date")

    pct, _ = _ticker_level_and_mask(canon, "percent_of_class", idx)
    out["ic_act_percent_of_class"] = pct
    # #49 is per-FILER by the registry's own wording, like #55 -- not the diff of #48's
    # per-ticker latest-reported level, which would also move when a DIFFERENT activist's
    # filing becomes the latest one.
    raw = _filer_frame(canon, "percent_of_class", idx)
    if raw is not None:
        out["ic_act_delta_percent_class"] = _filer_delta_sum(raw)
    return out


def _bo_fields(canon: pd.DataFrame, idx: pd.DatetimeIndex,
                halflife: float) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if canon.empty:
        return out
    ce = canon.dropna(subset=["filer_id"])

    occ = ce.assign(_flag=1.0, _grid_date=_snap_to_grid(ce["filing_date"], idx))
    occ = occ.dropna(subset=["_grid_date"])
    occ_wide = occ.pivot_table(index="_grid_date", columns=["ticker", "filer_id"],
                               values="_flag", aggfunc="max")
    # `limit=` is the whole exit policy: a filer counts as a holder for HOLDER_ACTIVE_DAYS
    # after each filing, then lapses. See the constant.
    occ_wide = occ_wide.reindex(idx).ffill(limit=HOLDER_ACTIVE_DAYS)
    active_count = _sum_over_filers(occ_wide)

    denom = _rolling_distinct(ce, idx, window=HOLDER_ACTIVE_DAYS)
    out["ic_bo_holder_count"] = active_count.divide(denom.where(denom > 0), axis=0)

    first_holder = ce.sort_values("filing_date").drop_duplicates(
        subset=["ticker", "filer_id"], keep="first")
    out["ic_bo_new_holder"] = decay_events(first_holder, idx, halflife, date_col="filing_date")

    raw = _filer_frame(ce, "percent_of_class", idx)
    if raw is not None:
        out["ic_bo_percent_of_class"] = _sum_over_filers(raw.ffill())
        out["ic_bo_delta_percent_class"] = _filer_delta_sum(raw)
    return out


def _cross_fields(canon_13d: pd.DataFrame, canon_13g: pd.DataFrame, idx: pd.DatetimeIndex,
                   halflife: float) -> dict[str, pd.DataFrame]:
    """13G<->13D escalation, keyed on each filer's FIRST-EVER filing of each type per ticker
    (an amendment does not re-trigger the transition)."""
    out: dict[str, pd.DataFrame] = {}
    if canon_13d.empty or canon_13g.empty:
        return out
    first_d = (canon_13d.dropna(subset=["filer_id"]).sort_values("filing_date")
               .drop_duplicates(subset=["ticker", "filer_id"], keep="first")
               [["ticker", "filer_id", "filing_date"]])
    first_g = (canon_13g.dropna(subset=["filer_id"]).sort_values("filing_date")
               .drop_duplicates(subset=["ticker", "filer_id"], keep="first")
               [["ticker", "filer_id", "filing_date"]])
    merged = first_d.merge(first_g, on=["ticker", "filer_id"], suffixes=("_d", "_g"))
    esc = (merged[merged["filing_date_g"] < merged["filing_date_d"]]
           .rename(columns={"filing_date_d": "filing_date"})[["ticker", "filing_date"]])
    deesc = (merged[merged["filing_date_d"] < merged["filing_date_g"]]
             .rename(columns={"filing_date_g": "filing_date"})[["ticker", "filing_date"]])
    out["ic_bo_escalation_13g_to_13d"] = decay_events(esc, idx, halflife, date_col="filing_date")
    out["ic_bo_de_escalation_13d_to_13g"] = decay_events(deesc, idx, halflife, date_col="filing_date")
    return out


def build_ownership_feature_panel(
    frames: PriceFrames,
    sec_13d: pd.DataFrame | None,
    sec_13g: pd.DataFrame | None,
    *,
    decay_halflife_act: float = ACT_HALFLIFE_DEFAULT,
    decay_halflife_bo: float = BO_HALFLIFE_DEFAULT,
    sink=None,
) -> pd.DataFrame:
    """Long-format beneficial-ownership panel (`f_<name>` per `EMISSION`). Empty when neither
    source has usable rows.

    `sink` is the optional `ConditioningSink`. The `act` family hands it EVERY 13D filing as
    its event dates (an amendment restates a live campaign, so it is news and the conditioning
    clock should restart on it) but only the INITIAL filings as bullish acts, since an
    amendment can as easily disclose a sale.

    ⚠ `frames` RATHER THAN TWO UNPACKED FIELDS. `peer_dict` and `trading_index` were all read
    off one `PriceFrames` at the call site. Collapsing them is not about the basis here -- this
    builder reads no wide price frame -- but about arity: three of the old parameters were one
    object at every call site, and unpacking them at 39 of those is what let them drift apart.

    ⚠ NO `frames.require(...)`: this builder dereferences no optional wide frame at all.
    `trading_index` and `peers` are non-Optional fields of `PriceFrames`, so requiring them
    would assert something the type already guarantees.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    # D5 entry guard, PER LEG. The two channels are independent fetchers -- a universe with
    # 13G coverage and no 13D still builds the `ic_bo_*` half -- so a leg that cannot be used
    # is nulled rather than failing the whole panel. `_NEED` is what `_canonicalize`
    # dereferences unconditionally; `is_amendment` and the Item 4 text are NOT in it, because
    # they are 13D-only and it already builds its column list around their absence.
    sec_13d = None if _absent(sec_13d, _NEED) else sec_13d
    sec_13g = None if _absent(sec_13g, _NEED) else sec_13g
    if sec_13d is None and sec_13g is None:
        return _EMPTY_PANEL()

    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    canon_13d = _canonicalize(sec_13d, text_col="item4_purpose_of_transaction",
                               has_amendment=True)
    canon_13g = _canonicalize(sec_13g, has_amendment=False)
    if canon_13d.empty and canon_13g.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    fields: dict[str, pd.DataFrame] = {}
    fields.update(_act_fields(canon_13d, idx, decay_halflife_act))
    fields.update(_bo_fields(canon_13g, idx, decay_halflife_bo))
    fields.update(_cross_fields(canon_13d, canon_13g, idx, decay_halflife_bo))

    floor = pd.Series(idx >= MANDATE_FLOOR, index=idx)
    for name in ("ic_act_percent_of_class", "ic_act_delta_percent_class",
                 "ic_bo_percent_of_class", "ic_bo_delta_percent_class"):
        if name in fields and fields[name] is not None and not fields[name].empty:
            fields[name] = fields[name].where(floor, axis=0)

    for name in list(fields):
        if fields[name] is None or fields[name].empty:
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    if sink is not None and not canon_13d.empty:
        sink.add_events("act", canon_13d[["ticker", "filing_date"]]
                        .rename(columns={"filing_date": "date"}).drop_duplicates())
        initial = canon_13d[~canon_13d["is_amendment"].fillna(0).astype(float).eq(1.0)]
        sink.add_actors("act", initial.dropna(subset=["filer_id"])
                        .assign(actor=lambda d: d["filer_id"])
                        [["ticker", "filing_date", "actor"]]
                        .rename(columns={"filing_date": "date"}))
    if sink is not None:
        sink.keep_signals(fields)

    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
