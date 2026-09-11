"""
def14a_impute.py  (src/data_aggregate/utils/governance/def14a_impute.py)
------------------------------------------------------------------------
Data-CLEANING step for the cube build: deduce / fill MISSING values in the `def14a_llm`
proxy archive and its per-NEO child `def14a_executive_comp`, before governance /
executive-pay features are computed (the LLM extraction leaves gaps on filings it couldn't
fully parse). Applied clean-on-read in `StepCubeGovernance` — the raw extraction tables are
never mutated.

STRICTLY non-destructive: a value is only ever written where it is currently NaN — a real
extracted number is never overwritten.

THREE KINDS OF MISSING, THREE TREATMENTS. Every fill in this module belongs to exactly one
of these, and no other kind of fill is allowed anywhere in the governance package:

| kind | how to recognise it | treatment | example |
|---|---|---|---|
| **A — structurally absent** | the disclosure regime did not exist at that `as_of` | **leave NaN, never fill.** The feature's history simply starts at the regime date | `ceo_total_comp` pre-2006 (Reg S-K), `say_on_pay_support_pct` pre-2011 (Dodd-Frank §951), `ceo_pay_ratio` pre-2018 (Item 402(u)) |
| **B — disclosed but unextracted** | era-flat or era-noisy fill, no legislative cliff | **deduce**: within-row identities, then a bounded forward CARRY from the last known observation | `avg_other_public_boards`, `lead_independent_director`, the 42,386 NULL NEO totals |
| **C — silent by choice (tri-state)** | the extraction deliberately writes NULL when the document says nothing | **carry the last known value forward**, bounded; never infer FALSE from silence | `poison_pill`, `majority_voting` |

⚠ **Kind A is NOT detected in code and must not be.** No date literal, no `if year < 2006`
branch: a regime cliff is visible in the coverage report and belongs in prose, not in a
filter. A hard-coded regime date would silently null a filer whose fiscal year straddles it.
Also forbidden here, by the same argument: cross-sectional / peer-median fills, `fillna(0)`,
and any fitted (model-based) imputation, which would leak cross-sectional information into a
point-in-time feature.

⚠ **NEVER READ A LATER FILING TO FILL AN EARLIER ROW.** Every temporal fill here is a FORWARD
carry: the value comes from the most recent observation at or before the row it lands on, and
nothing else. **The one exception is `_accrue_ceo_age`**, which fits a per-CEO birth year over
all of that person's disclosed ages, later ones included — admissible because a birth year is
TIME-INVARIANT, so a 2020 disclosure pins the same constant a 2015 one would and the 2015 age it
implies was knowable in 2015. That argument works for a constant and for nothing else; it is not
a licence to fit anything else across time.

This rule replaced linear interpolation on 2026-09-09, and the defect it closes was not subtle:
`limit_area="inside"` fills a gap by drawing a line between the observations either
side, so **every intermediate value was computed from a filing that did not exist yet**. LVS
`insider_ownership_pct`: 0.1080 filed 2020-04-01, silence, 0.0120 filed 2024-03-28; the shipped
2021 / 2022 / 2023 values were 0.0840 / 0.0600 / 0.0360, so the June-2021 feature asserted 8.4%
— arithmetic on a filing three years in its future. 12,607 interior gaps across 13 fields were
built that way, 46% of everything `f_board_busyness` shipped, and `03_pit` cannot see any of it
because the future value is laundered onto a row whose own `as_of` is legitimately in the past.

⚠ **A CARRY IS BOUNDED, AT `CARRY_MAX_DAYS` (1,095 days).** Not a taste question: `expire_stale`
dates a cell by the last filing that carried the field, so anything written here is dated to the
row it lands on and reads as age 0 downstream. An unbounded `ffill` would push 10,400 cells —
some copying a 30-year-old observation — straight through the level horizon.

⚠ **NEVER CARRY A LEVEL WHOSE YoY CHANGE IS ITSELF A FEATURE** without recording it, because the
fill *is* the change: a carried segment has a first difference of exactly zero, so the delta
asserts "nothing changed". That is why `CARRY_LEVELS` holds slow-moving structural ratios and NOT
`ceo_total_comp`: its 68.4% → 76.0% recovery comes entirely from the within-row component
identity. Two fields knowingly take the trade (`avg_other_public_boards`,
`pct_independent_directors`) and their deltas therefore partly measure the fill; the affected
share is flagged per cell in `DELTA_PROVENANCE_COLUMNS` rather than hidden.

Measured recovery on the live table, 2026-09-09 — 12,372 rows, 489 tickers,
1995-09-13 → 2026-09-04 (a moving target as `fetch_def14a_llm` runs):

| field | raw | after impute | recovered | modern (≥2011) fill | modern holes |
|---|---|---|---|---|---|
| `avg_other_public_boards` | 46.1% | 72.0% | +3,202 | 74.7% | 1,833 |
| `majority_voting` | 36.1% | 52.4% | +2,007 | 67.6% | 2,344 |
| `ceo_age` | 81.8% | 96.3% | +1,789 | 97.2% | 200 |
| `ceo_is_founder` | 78.1% | 92.0% | +1,728 | 96.7% | 237 |
| `lead_independent_director` | 50.1% | 62.5% | +1,535 | 82.7% | 1,254 |
| `ceo_since_year` | 79.6% | 91.2% | +1,429 | 94.3% | 415 |
| `pct_independent_directors` | 84.0% | 94.3% | +1,272 | 99.3% | 53 |
| `insider_ownership_pct` | 55.5% | 65.6% | +1,254 | 55.8% | 3,200 |
| `independent_chair` | 89.8% | 97.4% | +941 | 99.0% | 74 |
| `ceo_total_comp` | 68.4% | 76.0% | +932 | 95.9% | 298 |
| `say_on_pay_support_pct` | 37.8% | 44.8% | +861 | 75.8% | 1,749 |
| `avg_board_tenure` | 89.7% | 95.9% | +773 | 97.3% | 197 |
| `poison_pill` | 8.5% | 13.4% | +613 | 19.2% | 5,850 |
| `ceo_salary` | 91.4% | 95.4% | +504 | 98.1% | 136 |
| `ceo_is_board_chair` | 96.0% | 99.1% | +394 | 99.6% | 29 |
| `ceo_pay_ratio` | 32.2% | 33.4% | +145 | 56.9% | 3,122 |
| `board_size` | 99.3% | 100.0% | +78 | 100.0% | 1 |
| `ceo_name_proxy` | 99.2% | 99.2% | **+0** | 99.5% | 38 |

The single most important column is the second-to-last: **in the modern era the fields the
governance families need are 75-100% filled after impute**, and the remaining sparsity is
concentrated in eras where the disclosure did not legally exist. `poison_pill` at 8.5% raw is
kind C, not a defect — 91.5% "unknown" is the honest state of the world, and inferring FALSE
from silence is what made it degenerate (TRUE in 0.1% of rows) before the tri-state fix.

⚠ THIS TABLE FELL when the fill went forward-only on 2026-09-09, and the drop is the point,
not a regression. `avg_other_public_boards` went 84.9% → 72.0%, `ceo_salary` 97.4% → 95.4%,
`avg_board_tenure` 97.8% → 95.9%; `ceo_name_proxy` recovers **nothing** where it used to
recover 71. Two separate rules did that, and their costs are counted separately in `stats`:

  * **the 1,095-day cap** — 10,400 candidate cells copy an observation older than the level
    horizon, so they are refused here instead of being laundered past it downstream. The
    largest single decline is `ceo_ownership_pct` at 4,067 cells, a column whose old coverage
    was carried up to 30 years;
  * **forward-only itself** — a value the old rule computed from a LATER filing has no
    replacement, only the last known one, and where there is no last known one the cell stays
    NaN.

What was lost was mostly not information. A cell filled by interpolating across a 10-year gap
was an invented number that then read as freshly filed; a cell filled from a 3-year-old
disclosure is a stated estimate with a stated age. The one real loss is `ceo_name_proxy`'s 68
correct fills, given up to avoid 18 fabricated CEO identities — see `CARRY_FORBIDDEN`.

⚠ `recovered` here is `after − before` and is consistent with the fill columns beside it.
`test_impute_coverage.py` regenerates the whole table, which is the only defence against it
going stale — it moves every time `fetch_def14a_llm` runs.

Deductions:
  1. CEO pay identity. `ceo_total_comp` == sum of the six Summary-Comp-Table components
     (salary + bonus + stock + option + non-equity incentive + all-other). Fill the total
     from the components, or the single missing component from `total - others` (clipped
     >=0). Caveat: the schema omits the SCT "change in pension value" column, so a large
     deduced component may absorb it -- acceptable for a gap-fill.
  2. Board consistency. `n_directors == board_size` (either direction). The
     `pct_technology_directors == n_technology_directors / board_size` identity was removed
     with the fields themselves -- they were an opinion, not an extraction.
  3. Pay ratio. `ceo_pay_ratio == ceo_total_comp / median_employee_pay` -> fill the median
     employee pay (or the ratio) from the other two.
  4. Temporal gap-fill. Per ticker (sorted by filing date), carry the last KNOWN value
     forward into a gap -- ONE rule for levels/ratios and stable flags alike -- bounded by
     `CARRY_MAX_DAYS` from the `as_of` that sourced it. Leading gaps stay NaN because there
     is nothing behind them; TRAILING gaps are now filled, which the old interior-only rule
     refused, and that asymmetry was itself a defect: a gap at the live edge has no "after",
     so a backtest filled situations a live run structurally cannot. `CARRY_FORBIDDEN` blocks
     the carry entirely for `ceo_name_proxy` -- a wrong CEO name lets the turnover guard
     compare old-vs-old and compute pay growth across the succession it exists to catch, and
     forward-only cannot tell which gaps hide one. `IDENTITY_GATED_CARRY` applies a weaker,
     satisfiable version of that gate to `ceo_salary`, a property of a CONTRACT and not of a
     company: the CEO named on the source row must be the CEO named on the row being filled.
  5. Accrual. `ceo_age` is a CLOCK, not a level: it is recomputed from a per-CEO median
     birth-year anchor (`accrual.py`) rather than interpolated between neighbours.

`impute_def14a` never overwrites a present value — that invariant is enforced by
`test_impute_real_data_nondestructive`.

`impute_exec_comp` is the same job on the per-NEO child grain (D22): one identity, no
temporal fill at all, and a provenance flag. It lives in this module rather than its own
because it is the same job on a different grain and splitting it would duplicate `_fill`.

⚠ NOTHING here nulls a present-but-suspicious cell, and a plausibility FLOOR is the wrong
instrument for this table. The one that existed — 0.50 on `say_on_pay_support_pct` — was
measured wrong on 14 of 14 sampled values and deleted 61 real shareholder revolts (JPM 2023
disclosed 31% support, INTC 2023 34%, SPG 2024 11.1%), i.e. exactly the highest-signal
governance events in the archive. `test_def14a_say_on_pay.py` pins that those values survive.

`impute_def14a(df) -> (df, stats)` is pure (returns a copy + per-rule fill counts).
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.governance.accrual import accrual_anchor, accrue
from src.data_aggregate.utils.governance.names import ceo_identity_series
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS

CEO_COMP = ["ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
            "ceo_non_equity_incentive", "ceo_all_other_comp"]

#: How far a forward carry may reach, in days, measured from the `as_of` of the observation it
#: copies. **NOT a free parameter** -- it is `LEVEL_MAX_AGE_DAYS`, the same 1,095 days the daily
#: panel expires a level on, and it has to be, for a reason that is easy to miss:
#:
#: `staleness.expire_stale` drops NULL rows before it dates a cell, so it ages a value against
#: the last filing that CARRIED that field. A value written here onto a 2026 filing row is
#: therefore dated 2026 downstream -- age 0, indistinguishable from a real disclosure. An
#: unbounded `ffill` would launder stale values straight through the horizon phase 3 built:
#: measured 2026-09-09 on the live table, 21,518 carryable cells of which **10,400 (48%) copy an
#: observation more than 1,095 days old**, the worst reaching 10,984 days (30.1 years, on
#: `ceo_ownership_pct`). This is the last place the true age is still knowable, so it is the only
#: place the bound can be applied.
#:
#: ⚠ THE LINEAR INTERPOLATION THIS REPLACED LAUNDERED TOO, and that was recorded nowhere: its
#: interior gaps reach 10,593 days on `avg_other_public_boards`. The cap is therefore not a cost
#: of going forward-only -- it closes a second, independent defect that predates it.
CARRY_MAX_DAYS = LEVEL_MAX_AGE_DAYS

#: Levels and ratios that PERSIST between disclosures -> carry the last KNOWN value forward into
#: a gap, bounded by `CARRY_MAX_DAYS`.
#:
#: ⚠ FORWARD-ONLY, AND THAT IS THE WHOLE POINT. These were linearly interpolated until
#: 2026-09-09, via `limit_area="inside"`, which fills a gap by drawing a line between the
#: observation BEFORE it and the observation AFTER it -- so every intermediate value was computed
#: from a filing that did not exist yet. Worked example, LVS `insider_ownership_pct`: filed
#: 0.1080 on 2020-04-01, nothing for three years, then 0.0120 on 2024-03-28. The interpolation
#: shipped 0.0840 / 0.0600 / 0.0360 for 2021 / 2022 / 2023 -- a ramp in steps of -0.024 in which
#: the June-2021 feature asserts 8.4%, a number arithmetically derived from a filing three years
#: in its future. 12,607 interior gaps across these fields were built that way, 46% of everything
#: `f_board_busyness` shipped. `03_pit` cannot see it: the future value is laundered onto a row
#: whose own `as_of` is legitimately in the past, so the check looks one layer below the defect.
#:
#: A forward carry answers 0.1080 for all three years -- the last thing anyone could have known --
#: and, because it needs no later filing, it also fills the TRAILING gap the old rule refused
#: (LVS 2026-04-01 = 0.0120). That matters more than it sounds: under the old rule a gap at the
#: live edge could never be filled, because "after" does not exist yet, so a backtest filled
#: situations that a live run structurally cannot. The carry treats both the same.
#:
#: `ceo_age` was here and is NOT any more: it is an ACCRUAL, filled by `_accrue_ceo_age`.
CARRY_LEVELS = ["board_size", "n_directors", "avg_director_age", "avg_board_tenure",
                "pct_independent_directors", "pct_female_directors",
                "avg_other_public_boards", "insider_ownership_pct",
                "ceo_ownership_pct", "n_five_percent_holders", "say_on_pay_support_pct",
                "median_employee_pay", "ceo_pay_ratio"]
#: Carried ONLY when the CEO named on the source row is the CEO named on THIS row (D31). A salary
#: is a term of one person's CONTRACT, so carrying it across a succession states the outgoing
#: CEO's pay as the incoming one's.
#:
#: ⚠ THE GATE IS FORWARD-ONLY, unlike the one it replaced. The old test was "the same person
#: bounds the gap on BOTH sides", which read the later filing; the new one compares the source
#: row against the current row, both at or before the date being filled. It is also STRICTER: a
#: row that does not name its CEO can no longer be filled, because the old `bfill` leg supplied
#: an identity such a row does not have.
#:
#: `ceo_salary` earns a carry and the other pay components do not, on measurement. Median YoY
#: |change|: salary **3.8%**, stock awards 22.8%, non-equity incentive 29.3%, bonus **40.0%**
#: (p90 100%, only 20.4% of years within ±10%). A carry asserts "unchanged", so it is defensible
#: exactly to the extent the field is sticky -- 3.8% for salary, and the last three ARE the
#: performance-sensitive part of the package, where the assertion would manufacture precisely the
#: variation the pay features exist to measure.
#:
#: Cost, quantified rather than argued: salary is a mean 13.4% of CEO total comp (p90 23.6%), so
#: a carried salary summed with five real components moves a derived total by ~0.5%.
#: `comp_imputed` keeps that population stateable if the extraction's coverage shifts.
IDENTITY_GATED_CARRY = frozenset({"ceo_salary"})
#: ⚠ CARRIED columns whose YoY CHANGE also ships as a feature, and which therefore need
#: PROVENANCE. The LEVEL is worth filling -- a carried board average is a defensible estimate of
#: a standing fact -- while the DELTA across the fill is not, and the forward carry does not make
#: that better, only different in shape: a carried segment has a first difference of **exactly
#: zero**, so the delta asserts "this board changed nothing that year", which is a claim about
#: the company and not a measurement of it. (The interpolation it replaced had a constant
#: NON-zero first difference, so the delta reported the fill's slope. Both are fabricated; only
#: the fabricated number changed.)
#:
#: Rather than choose between the two, the fill happens and its footprint is recorded: each
#: column gains `<column>_imputed` (1.0 where the temporal fill wrote the value, else 0.0), and
#: `provisions_features._annual_delta` requires BOTH legs of a delta to be un-imputed. Measured
#: 2026-09-08 under the old interpolation, the share of adjacent pairs this rejects:
#:     avg_other_public_boards      65.5%  (6,556 of 10,004)
#:     pct_independent_directors    20.7%  (2,329 of 11,224)
#: The rejected share exceeds the filled-CELL count because one invented value invalidates up to
#: two deltas -- the one landing on it and the one whose earlier leg it is. ⚠ Both figures are
#: pre-carry and move with it; `test_impute_coverage.py` regenerates them.
#:
#: Same shape as `comp_imputed` (D31) and for the same reason: a flag keeps the population
#: STATEABLE instead of assumed, where a repair-in-place makes it unknowable. The columns are
#: always present, even when nothing was filled.
DELTA_PROVENANCE_COLUMNS: tuple[str, ...] = ("avg_other_public_boards",
                                             "pct_independent_directors")
# Stable per-company/CEO facts -> carry the last known value forward, bounded by
# `CARRY_MAX_DAYS`, exactly as `CARRY_LEVELS` now is. These were already a carry rather than an
# interpolation, but they were gated on `fwd.notna() & bwd.notna()` -- an INTERIOR test, so a gap
# was filled only when a LATER filing existed. The value written was never from the future; the
# DECISION to write it was, and it produced the same live/backtest asymmetry: a gap at the live
# edge has no "after" and so could never be filled.
#
# `poison_pill` and `majority_voting` are TRI-STATE at extraction (null when the proxy is
# silent), so a carry-forward here fills a genuine gap rather than propagating a fabricated
# FALSE -- which is what made `majority_voting` flip 21.2% year-over-year before.
FLAGS = ["ceo_is_founder", "ceo_is_board_chair", "independent_chair", "lead_independent_director",
         "classified_board", "dual_class_shares", "poison_pill", "majority_voting",
         "ceo_since_year", "ceo_name_proxy"]
#: Columns a forward carry must NOT touch at all, because the carry cannot be validated without
#: reading a LATER filing -- which is the thing this module stopped doing on 2026-09-09.
#:
#: `ceo_name_proxy` is the whole set, and it is here reluctantly. A carry-forward is a sound
#: prior for a bylaw -- "unchanged since the last disclosure" is what a provision usually is --
#: and it is even a defensible real-time BELIEF for a person ("as far as anyone knew, Ruiz still
#: ran AMD"). But a wrong CEO name does specific downstream damage: the turnover guard compares
#: old-vs-old, sees no change, and computes pay growth straight across the succession it exists
#: to catch. Unknown is the correct output for "we do not know who ran this company that year".
#:
#: Until 2026-09-09 the gap was filled when the same person bounded it on BOTH sides, which
#: separated the safe fills from the unsafe ones using the future. Forward-only cannot make that
#: distinction, so the question became binary, and it was settled by measuring rather than
#: arguing (live table, 2026-09-09): **87 interior gaps, 86 inside the carry cap, of which 18
#: have a DIFFERENT CEO on each side** -- a real transition inside the gap (ACGL Mosca->Appel,
#: AMD Ruiz->Meyer, CNC Neidorff->London). Carrying all 86 fabricates 18 identities; carrying
#: none loses 68 correct fills, 0.7% of the column, which stays 99.2% filled from the extraction
#: alone. 68 knowable cells is the measured price of 18 unknowable ones.
#:
#: ⚠ A PIT-VALID GATE WAS LOOKED FOR AND DOES NOT EXIST. `ceo_since_year` would give one -- if
#: this row's start year matches the source row's, the same person is in post, using only the
#: present and the past. It fills **0 of the 86**: a row that fails to name its CEO fails to
#: disclose the start year too, because both come from the same extraction. Recorded so the next
#: reader does not re-derive it.
CARRY_FORBIDDEN = frozenset({"ceo_name_proxy"})
INT_COLS = ["n_directors", "board_size", "ceo_age",
            "n_five_percent_holders", "n_neos", "ceo_since_year"]

#: The seven Item 402(c) components, in `def14a_executive_comp`'s flat column vocabulary.
#: ⚠ SEVEN, against `CEO_COMP`'s six: the child table carries `pension_change`, which the
#: parent's schema omits. That is why the parent's identity has to tolerate a deduced
#: component absorbing the pension column and this one does not.
EXEC_COMPONENTS = ["salary", "bonus", "stock_awards", "option_awards",
                   "non_equity_incentive", "pension_change", "other_compensation"]


def _fill(df: pd.DataFrame, col: str, cond: pd.Series, values, stats: dict, tag: str) -> None:
    """Set df[col] = values ONLY where col is currently NaN AND `cond` AND value is finite."""
    if col not in df.columns:
        return
    vals = values if isinstance(values, pd.Series) else pd.Series(values, index=df.index)
    m = df[col].isna() & cond.fillna(False) & vals.notna()
    n = int(m.sum())
    if n:
        df.loc[m, col] = vals[m]
        stats[tag] = stats.get(tag, 0) + n


def _reconcile_rows(df: pd.DataFrame, stats: dict) -> None:
    comps = [c for c in CEO_COMP if c in df.columns]
    if "ceo_total_comp" in df.columns and len(comps) == 6:
        _fill(df, "ceo_total_comp", df[comps].notna().all(axis=1),
              df[comps].sum(axis=1), stats, "ceo_total_comp = sum(components)")
        for c in comps:
            others = [x for x in comps if x != c]
            cond = df["ceo_total_comp"].notna() & df[others].notna().all(axis=1)
            _fill(df, c, cond, (df["ceo_total_comp"] - df[others].sum(axis=1)).clip(lower=0),
                  stats, "ceo component = total - others")
    if {"n_directors", "board_size"} <= set(df.columns):
        _fill(df, "n_directors", df["board_size"].notna(), df["board_size"], stats, "n_directors = board_size")
        _fill(df, "board_size", df["n_directors"].notna(), df["n_directors"], stats, "board_size = n_directors")
    if {"ceo_pay_ratio", "median_employee_pay", "ceo_total_comp"} <= set(df.columns):
        r, med, tot = df["ceo_pay_ratio"], df["median_employee_pay"], df["ceo_total_comp"]
        _fill(df, "median_employee_pay", (r > 0) & tot.notna(), tot / r, stats, "median_pay = total / ratio")
        _fill(df, "ceo_pay_ratio", (med > 0) & tot.notna(), tot / med, stats, "pay_ratio = total / median")


def _carry(df: pd.DataFrame, col: str, gk: pd.Series) -> tuple[pd.Series, pd.Series]:
    """`(value carried forward from the last known observation, its age in days)`.

    The two halves have to be produced together, because the age is measured against the
    `as_of` of the row that SOURCED the value -- not the previous row, and not the previous
    row that merely exists. `df["as_of"].where(df[col].notna())` then `ffill` carries the
    source DATE as a payload alongside the value, the same trick `staleness.expire_stale`
    uses with `_produced_at`, so the two agree on what "age" means.

    Reads only rows at or before each row: this is the point-in-time guarantee, and it is a
    property of `ffill` rather than of a guard that has to be remembered.
    """
    fwd = df[col].groupby(gk, sort=False).ffill()
    src = df["as_of"].where(df[col].notna()).groupby(gk, sort=False).ffill()
    return fwd, (df["as_of"] - src).dt.days


def _same_ceo_as_source(df: pd.DataFrame, col: str, gk: pd.Series) -> pd.Series:
    """True where the CEO named on THIS row is the CEO named on the row sourcing `col`'s carry.

    Both legs are at or before the row being filled, which is what makes the gate PIT. It
    replaces a `ffill`/`bfill` pair that asked whether the same person BOUNDED the gap -- a
    question only a later filing can answer.

    Judged on `ceo_identity` rather than the raw string so a respelling (`Timothy D. Cook` /
    `Tim Cook`) counts as agreement, with an explicit `notna` on both keys so two UNKEYABLE
    cells never agree by both being None. Read off the raw disclosed `ceo_name_proxy`, before
    the FLAGS loop touches it, so the gate is judged on what the filings actually said.

    ⚠ STRICTER THAN ITS PREDECESSOR, deliberately: a row that does not name its CEO now fails
    the gate, where the old `bfill` leg handed it an identity it does not have.
    """
    if "ceo_name_proxy" not in df.columns:
        return pd.Series(False, index=df.index)
    ident = ceo_identity_series(df["ceo_name_proxy"])
    at_source = ident.where(df[col].notna()).groupby(gk, sort=False).ffill()
    return ident.notna() & at_source.notna() & (ident == at_source)


def _accrue_ceo_age(df: pd.DataFrame, stats: dict) -> pd.Series:
    """Fill `ceo_age` from a per-(ticker, CEO) median birth-year anchor. Returns the mask.

    Replaces the plain interpolation `ceo_age` used to get, which closes a measured defect
    (D33): that interpolation had NO identity check, so some of its 1,597 fills interpolated
    an age straight ACROSS a CEO change -- a fictional number belonging to neither the
    outgoing nor the incoming person. Anchoring per PERSON makes that impossible by
    construction rather than by a guard that has to be remembered.

    ⚠ HOW MANY depends on what "across a CEO change" means, and the two defensible readings
    differ by an order of magnitude. Measured 2026-09-07 on the live table (whose 1,597 total
    reproduces exactly): by BOUNDING names -- the nearest disclosed name each side of the gap
    differs -- 17 raw / 13 keyed; by ADJACENT rows -- the two the interpolation actually ran
    BETWEEN name different CEOs -- 483 raw / 411 keyed. D33 publishes 103, which reproduces on
    neither, nor on the two looser bases also tried (516/444 and 1,526/1,489). Quote a basis,
    never the bare figure. The fix does not depend on the magnitude: a per-person anchor
    cannot span a succession at any of them.

    The anchor also fills LEADING gaps, which the forward carry cannot: it is the one fill in
    this module that legitimately runs backwards, because an age before a CEO's first disclosed
    one is not unknown, it is `first_age - elapsed_years`. A birth year is a CONSTANT, so the
    anchor is not an estimate that decays -- which is why it is exempt from `CARRY_MAX_DAYS`
    while every carry above it is bound by it.
    """
    if "ceo_age" not in df.columns or "ceo_name_proxy" not in df.columns:
        return pd.Series(False, index=df.index)
    # ⚠ The entity is (ticker, CEO), never the bare ticker -- see `accrual.py`. A NaN name
    # yields a NaN key, hence no anchor and no fill: an unknown CEO has an unknown age.
    ident = ceo_identity_series(df["ceo_name_proxy"])
    key = df["ticker"].astype(str).str.cat(ident, sep="|")
    obs = pd.DataFrame({"pk": key, "as_of": df["as_of"], "ceo_age": df["ceo_age"]})
    anchor = accrual_anchor(obs, "ceo_age", key="pk", date="as_of")
    if anchor.empty:
        return pd.Series(False, index=df.index)
    implied = accrue(df["as_of"], key, anchor)
    newly = df["ceo_age"].isna() & implied.notna()
    n = int(newly.sum())
    if n:
        df.loc[newly, "ceo_age"] = implied[newly]
        stats["accrue: ceo_age"] = n
    return newly


def _temporal_fill(df: pd.DataFrame, stats: dict) -> None:
    """Fill cross-filing gaps by carrying the last KNOWN value forward, and nothing else.

    ⚠ ONE RULE FOR BOTH BLOCKS, which was not true before 2026-09-09: the levels were linearly
    interpolated between the observations either side of a gap, and the flags were carried
    forward but only INSIDE a gap that a later filing closed. Both therefore read a filing dated
    after the row they wrote to -- the levels for the VALUE, the flags for the DECISION. Now both
    read only the past, and both are bounded by `CARRY_MAX_DAYS`.

    Order matters and is unchanged: the levels first, then the `ceo_age` accrual, then the flags.
    `ceo_name_proxy` is carried LAST (and, being in `CARRY_FORBIDDEN`, not at all) so every
    identity gate above it is judged on the raw disclosed name.
    """
    df.sort_values(["ticker", "as_of"], inplace=True)
    gk = df["ticker"]
    for col in CARRY_LEVELS + sorted(IDENTITY_GATED_CARRY):
        _carry_one(df, col, gk, stats)
    _accrue_ceo_age(df, stats)
    for col in FLAGS:
        _carry_one(df, col, gk, stats)


def _carry_one(df: pd.DataFrame, col: str, gk: pd.Series, stats: dict) -> None:
    """Carry `col` forward into its gaps, bounded and gated, and record what was declined.

    Every reason a candidate cell is NOT filled gets its own counter, because a fill count on
    its own cannot distinguish "nothing was missing" from "everything was refused". The three
    reasons are mutually exclusive by construction: the cap is applied before the identity
    gate, so no cell is reported twice.
    """
    if col not in df.columns:
        return
    fwd, age = _carry(df, col, gk)
    candidate = df[col].isna() & fwd.notna()
    if col in CARRY_FORBIDDEN:
        blocked = int((candidate & (age <= CARRY_MAX_DAYS)).sum())
        if blocked:
            stats[f"declined (carry cannot be validated): {col}"] = blocked
        return
    within = age <= CARRY_MAX_DAYS
    stale = int((candidate & ~within).sum())
    if stale:
        stats[f"declined (>{CARRY_MAX_DAYS}d stale): {col}"] = stale
    newly = candidate & within
    if col in IDENTITY_GATED_CARRY:
        same_ceo = _same_ceo_as_source(df, col, gk)
        declined = int((newly & ~same_ceo).sum())
        if declined:
            stats[f"declined (identity changed): {col}"] = declined
        newly &= same_ceo
    n = int(newly.sum())
    if n:
        df.loc[newly, col] = fwd[newly]
        stats[f"carry: {col}"] = n


def impute_def14a(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (imputed copy, stats) — fills only NaNs via the identities + temporal gap-fill."""
    if df is None or df.empty:
        return df, {}
    
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"], format="%Y-%m-%d", errors="coerce")
    was_na = {c: df[c].isna() for c in INT_COLS if c in df.columns}  # to round ONLY what we fill
    stats: dict[str, int] = {}
    _reconcile_rows(df, stats)          # within-row identities

    # PROVENANCE for D31: which `ceo_total_comp` cells were derived from an INTERPOLATED
    # component. Snapshot the component/total NaN pattern either side of the temporal fill --
    # a total is flagged only if it was absent before it, present after, and the row gained an
    # interpolated component in between. The flag does NOT gate the growth feature (D31); it
    # exists so the population can be stated instead of assumed.
    comps = [c for c in CEO_COMP if c in df.columns]
    comp_na_before = df[comps].isna() if comps else None
    # The same provenance snapshot for the two delta-feature levels. Taken here, immediately
    # before `_temporal_fill`, so it captures ONLY the temporal fill -- `_reconcile_rows` above
    # writes values deduced from a within-row identity, which are facts and not estimates.
    delta_cols = [c for c in DELTA_PROVENANCE_COLUMNS if c in df.columns]
    delta_na_before = df[delta_cols].isna() if delta_cols else None
    total_na_before = (df["ceo_total_comp"].isna() if "ceo_total_comp" in df.columns
                       else pd.Series(False, index=df.index))

    _temporal_fill(df, stats)           # cross-year interior gaps + the ceo_age accrual
    interp_comp = (pd.Series(False, index=df.index) if comps is None or not comps
                   else (comp_na_before & df[comps].notna()).any(axis=1))
    _reconcile_rows(df, stats)          # reconcile values the temporal fill unlocked

    if "ceo_total_comp" in df.columns:
        derived = total_na_before & df["ceo_total_comp"].notna() & interp_comp
        df["comp_imputed"] = derived.astype("float64")
        n = int(derived.sum())
        if n:
            stats["comp_imputed (total used an interpolated component)"] = n

    if delta_na_before is not None:
        for c in delta_cols:
            filled = delta_na_before[c] & df[c].notna()
            df[f"{c}_imputed"] = filled.astype("float64")
            n = int(filled.sum())
            if n:
                stats[f"{c}_imputed (carried -> delta legs excluded)"] = n

    for c, na in was_na.items():        # keep DEDUCED counts integral (never touch real values)
        filled = na & df[c].notna()
        df.loc[filled, c] = df.loc[filled, c].round()
    return df.sort_values(["ticker", "as_of"]).reset_index(drop=True), stats


def impute_exec_comp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill a NULL per-NEO `total` from its components. NON-DESTRUCTIVE: writes only where NULL.

    Measured on the live table (2026-09-07): 157,002 rows over 488 tickers and 11,316 filings,
    114,435 carrying a filer-stated `total` and 42,386 carrying NO total but at least one
    component. The identity is trustworthy where it is checkable — the extraction's own
    `reconciles` flag is 1 on 97.3% of the rows that carry both sides.

    WHY IT MATTERS, in one number: the exact top-5 CEO Pay Slice needs five NEO totals inside
    one filing. Filings clearing that bar go **8,026 → 10,695 (+33%)**; against a 3-NEO floor,
    8,422 → 11,232 of 11,316.

    Two deliberate restraints, both following the module's taxonomy:

      * `min_count=1` on the row sum, so a row with EVERY component NULL stays NULL instead of
        becoming a $0 pay package. `sum()` returning 0.0 for an all-NaN row is the single
        easiest way to fabricate a fact here, and `fillna(0)` is forbidden for the same reason.
      * **NO temporal interpolation of pay at all** — not even the gated kind `ceo_salary`
        gets on the parent grain. A NEO absent from one year's table is missing, and inventing
        their package would invent the very cross-sectional spread the CPS measures.

    `total_imputed` (1.0 where this function wrote the value, else 0.0) is stamped on every
    row and carried into the CPS builder's log, so a repaired denominator stays
    *distinguishable* from a filer-stated one — exactly as `reconciles` is a flag and not a
    repair. The column is always present, even when nothing was filled.

    Pure: returns `(copy, stats)`, matching `impute_def14a`'s contract. The raw table is never
    mutated.
    """
    return fill_total_from_components(df, EXEC_COMPONENTS)


def fill_total_from_components(df: pd.DataFrame,
                              components: list[str] | tuple[str, ...],
                              ) -> tuple[pd.DataFrame, dict]:
    """The `total = sum(components)` identity on a per-PERSON compensation table.

    Shared by `impute_exec_comp` (Item 402(c), seven components, `salary` first) and
    `impute_director_comp` (Item 402(k), six, `fees_earned` first — a director draws fees, not a
    salary). ONE implementation rather than two, because the restraints are what matter and they
    are identical on both grains:

      * `min_count=1`, so a row with EVERY component NULL stays NULL instead of becoming a $0
        pay package — `sum()` returning 0.0 for an all-NaN row is the single easiest way to
        fabricate a fact here;
      * NON-DESTRUCTIVE: written only where `total` is NULL, never over a filer-stated figure;
      * `total_imputed` stamped on every row, always present even when nothing was filled.

    Pure: returns `(copy, stats)`.
    """
    if df is None or df.empty:
        return df, {}

    out = df.copy()
    comps = [c for c in components if c in out.columns]
    if "total" not in out.columns or not comps:
        return out, {}

    out["total_imputed"] = 0.0
    stated = int(out["total"].notna().sum())
    summed = out[comps].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    fill = out["total"].isna() & summed.notna()
    n = int(fill.sum())
    if n:
        out.loc[fill, "total"] = summed[fill]
        out.loc[fill, "total_imputed"] = 1.0

    stats: dict[str, int] = {
        "rows": len(out),
        "total stated by the filer": stated,
        "total = sum(components)": n,
        "total still NULL (no component at all)": int(out["total"].isna().sum()),
        "components available": len(comps),
    }
    return out, stats
