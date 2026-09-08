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
| **B — disclosed but unextracted** | era-flat or era-noisy fill, no legislative cliff | **deduce**: within-row identities, then a bounded temporal fill between two KNOWN observations | `avg_other_public_boards`, `lead_independent_director`, the 42,386 NULL NEO totals |
| **C — silent by choice (tri-state)** | the extraction deliberately writes NULL when the document says nothing | **carry forward only between two known observations**; never infer FALSE from silence | `poison_pill`, `majority_voting` |

⚠ **Kind A is NOT detected in code and must not be.** No date literal, no `if year < 2006`
branch: a regime cliff is visible in the coverage report and belongs in prose, not in a
filter. A hard-coded regime date would silently null a filer whose fiscal year straddles it.
Also forbidden here, by the same argument: cross-sectional / peer-median fills, `fillna(0)`,
and any fitted (model-based) imputation, which would leak cross-sectional information into a
point-in-time feature.

⚠ **NEVER LINEARLY INTERPOLATE A LEVEL WHOSE YoY CHANGE IS ITSELF A FEATURE** — because the
interpolation *is* the change. Pay growth measured on an interpolated pay series measures the
fill, not the company. That is why `INTERP` holds slow-moving structural ratios and NOT
`ceo_total_comp`: its 68.6% → 76.0% recovery comes entirely from the within-row component
identity. Two fields knowingly take the trade (`avg_other_public_boards`,
`say_on_pay_support_pct`) and their deltas therefore partly measure the interpolation; the
affected share is measured and reported rather than hidden.

Measured recovery on the live table, 2026-09-07 — 12,343 rows, 488 tickers,
1995-09-13 → 2026-09-04 (a moving target as `fetch_def14a_llm` runs):

| field | raw | after impute | recovered | modern (≥2011) fill | modern holes |
|---|---|---|---|---|---|
| `avg_other_public_boards` | 45.8% | 84.9% | +4,828 | 88.5% | 829 |
| `majority_voting` | 36.0% | 58.0% | +2,714 | 72.4% | 1,990 |
| `lead_independent_director` | 50.1% | 66.9% | +2,080 | 87.3% | 915 |
| `ceo_is_founder` | 77.7% | 94.3% | +2,051 | 97.6% | 175 |
| `ceo_since_year` | 79.6% | 94.2% | +1,812 | 96.1% | 285 |
| `ceo_age` | 81.9% | 96.3% | +1,782 | 97.2% | 202 |
| `pct_independent_directors` | 83.6% | 94.9% | +1,388 | 99.2% | 59 |
| `independent_chair` | 89.7% | 98.0% | +1,024 | 99.2% | 57 |
| `avg_board_tenure` | 89.6% | 97.8% | +1,005 | 98.7% | 92 |
| `insider_ownership_pct` | 60.8% | 68.4% | +944 | 57.7% | 3,054 |
| `ceo_total_comp` | 68.6% | 76.0% | +915 | 95.9% | 294 |
| `say_on_pay_support_pct` | 37.9% | 45.1% | +895 | 76.0% | 1,737 |
| `poison_pill` | 8.5% | 14.6% | +763 | 20.0% | 5,776 |
| `ceo_salary` | 91.4% | 97.4% | +752 | 99.3% | 51 |
| `ceo_is_board_chair` | 95.8% | 99.3% | +425 | 99.7% | 22 |
| `board_size` | 99.3% | 100.0% | +76 | 100.0% | 3 |
| `ceo_name_proxy` | 99.2% | 99.8% | +71 | 99.9% | 8 |

The single most important column is the second-to-last: **in the modern era the fields the
governance families need are 88-100% filled after impute**, and the remaining sparsity is
concentrated in eras where the disclosure did not legally exist. `poison_pill` at 8.5% raw is
kind C, not a defect — 91.5% "unknown" is the honest state of the world, and inferring FALSE
from silence is what made it degenerate (TRUE in 0.1% of rows) before the tri-state fix.

⚠ `recovered` here is `after − before` and is consistent with the fill columns beside it
(45.8% → 84.9% over 12,343 rows IS ~4,828). The plan's §1.2 prints larger figures in that one
column that its own percentages contradict; the percentages are the half that reproduces.
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
  4. Temporal gap-fill. Per ticker (sorted by filing date), fill a value missing BETWEEN
     two filled years -- linear interpolation for levels/ratios, carry-forward for stable
     flags -- via `limit_area='inside'`, so leading/trailing gaps and special-meeting
     proxies at the edges are left untouched. `AGREEMENT_REQUIRED_FLAGS` narrows that
     carry-forward for a PERSON: an identity gap is filled only when the same human stands
     on both sides of it, because a bounded gap proves a value was disclosed either side,
     not that it was the same value. `IDENTITY_GATED_INTERP` applies the same gate to
     `ceo_salary`, which is a property of a CONTRACT and not of a company.
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

CEO_COMP = ["ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
            "ceo_non_equity_incentive", "ceo_all_other_comp"]
# levels / ratios that vary smoothly -> linear interpolate an interior gap.
# ⚠ `ceo_age` was here and is NOT any more: it is an ACCRUAL, filled by `_accrue_ceo_age`.
INTERP = ["board_size", "n_directors", "avg_director_age", "avg_board_tenure",
          "pct_independent_directors", "pct_female_directors",
          "avg_other_public_boards", "insider_ownership_pct",
          "ceo_ownership_pct", "n_five_percent_holders", "say_on_pay_support_pct",
          "median_employee_pay", "ceo_pay_ratio"]
#: Interior gaps interpolated ONLY when the same CEO stands on both sides (D31). A salary is a
#: term of one person's CONTRACT, so a gap spanning a succession has no smooth path across it.
#:
#: `ceo_salary` earns interpolation and the other pay components do not, on measurement.
#: Median YoY |change|: salary **3.8%**, stock awards 22.8%, non-equity incentive 29.3%,
#: bonus **40.0%** (p90 100%, only 20.4% of years within ±10%). The last three ARE the
#: performance-sensitive part of the package -- interpolating them would manufacture exactly
#: the variation the pay features exist to measure. Salary is the one component sticky enough.
#:
#: Cost, quantified rather than argued: salary is a mean 13.4% of CEO total comp (p90 23.6%),
#: so an interpolated salary summed with five real components moves a derived total by ~0.5%.
#: Measured 2026-09-07 on the live table, the realised cost is **zero**: 766 interior-fillable
#: salary cells (752 same-CEO, 14 declined), and NOT ONE of them sits on a row where salary is
#: the only absent component -- so no `ceo_total_comp` is unlocked, and the legacy
#: `ceo_pay_growth` is bit-identical. `comp_imputed` exists to keep that checkable if the
#: extraction's coverage shifts.
IDENTITY_GATED_INTERP = frozenset({"ceo_salary"})
#: ⚠ INTERP columns whose YoY CHANGE also ships as a feature, and which therefore need
#: PROVENANCE. This module's own headline rule is "never linearly interpolate a level whose YoY
#: change is itself a feature", and these two are the standing exceptions to it: the LEVEL is
#: worth filling (a carried board average is a defensible estimate of a standing fact) while the
#: DELTA across the fill is not, because a linearly-filled segment has a constant first
#: difference -- so the delta reports the fill's slope rather than the company's change.
#:
#: Rather than choose between the two, the fill happens and its footprint is recorded: each
#: column gains `<column>_imputed` (1.0 where the temporal fill wrote the value, else 0.0), and
#: `provisions_features._annual_delta` requires BOTH legs of a delta to be un-imputed. Measured
#: 2026-09-08, the share of adjacent pairs this rejects:
#:     avg_other_public_boards      65.5%  (6,556 of 10,004)
#:     pct_independent_directors    20.7%  (2,329 of 11,224)
#: The rejected share exceeds the interpolated-CELL count (4,828 and 1,388) because one invented
#: value invalidates up to two deltas -- the one landing on it and the one whose earlier leg it
#: is.
#:
#: Same shape as `comp_imputed` (D31) and for the same reason: a flag keeps the population
#: STATEABLE instead of assumed, where a repair-in-place makes it unknowable. The columns are
#: always present, even when nothing was filled.
DELTA_PROVENANCE_COLUMNS: tuple[str, ...] = ("avg_other_public_boards",
                                             "pct_independent_directors")
# stable per-company/CEO facts -> carry the last known value forward within an interior gap
# `poison_pill` and `majority_voting` are now TRI-STATE at extraction (null when the proxy is
# silent), so a carry-forward here fills a genuine gap rather than propagating a fabricated
# FALSE -- which is what made `majority_voting` flip 21.2% year-over-year before.
FLAGS = ["ceo_is_founder", "ceo_is_board_chair", "independent_chair", "lead_independent_director",
         "classified_board", "dual_class_shares", "poison_pill", "majority_voting",
         "ceo_since_year", "ceo_name_proxy"]
#: Columns whose interior gap is filled ONLY when the value before the gap and the value after
#: it are the same PERSON. A carry-forward is a sound prior for a bylaw -- "unchanged since the
#: last disclosure" is what a provision usually is -- but for a person it is a guess about who
#: held a job, and the plain `ffill` gets it wrong in the one case that matters most.
#:
#: Measured 2026-09-07: 85 interior `ceo_name_proxy` gaps, of which 14 have a DIFFERENT CEO on
#: each side, i.e. a real transition happened inside the gap (ACGL Mosca->Appel, AMD Ruiz->Meyer,
#: CNC Neidorff->London). An unconditional fill hands those 14 the OLD CEO's name, so the
#: turnover guard compares old-vs-old, sees no change, and computes pay growth straight across
#: the transition it exists to catch. Requiring agreement leaves them NaN = UNKNOWN, and unknown
#: is the correct output for "we do not know who ran this company that year".
#:
#: Agreement is judged on `ceo_identity`, never the raw string: that keeps 71 of the 85 against
#: the 63 a string comparison finds, because `Timothy D. Cook` and `Tim Cook` ARE agreement.
#: (71/14 rather than the 70/15 a BARE `person_key` gives -- `ceo_identity` splits a co-CEO cell
#: first, so one further gap reconciles. Reproduced by `test_governance_names.py`.)
AGREEMENT_REQUIRED_FLAGS = frozenset({"ceo_name_proxy"})
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


def _same_ceo_across_gap(df: pd.DataFrame, g) -> pd.Series:
    """True where the same CEO identity bounds the row on BOTH sides.

    Computed on the raw disclosed `ceo_name_proxy`, before the FLAGS carry-forward touches
    it, so the gate is judged on what the filings actually said. Judged on `ceo_identity`
    rather than the raw string so a respelling (`Timothy D. Cook` / `Tim Cook`) counts as
    agreement, and with an explicit `notna` on both keys so two UNKEYABLE cells never agree
    with each other by both being None.
    """
    if "ceo_name_proxy" not in df.columns:
        return pd.Series(False, index=df.index)
    k_fwd = ceo_identity_series(g["ceo_name_proxy"].ffill())
    k_bwd = ceo_identity_series(g["ceo_name_proxy"].bfill())
    return k_fwd.notna() & k_bwd.notna() & (k_fwd == k_bwd)


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

    The anchor also fills EDGE gaps, which `limit_area="inside"` refuses. That is correct for
    a clock and is why the fill count goes UP while the population gets stricter: an age
    before a CEO's first disclosed one is not unknown, it is `first_age - elapsed_years`.
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
    df.sort_values(["ticker", "as_of"], inplace=True)
    g = df.groupby("ticker", sort=False)
    same_ceo = _same_ceo_across_gap(df, g)
    for col in INTERP + sorted(IDENTITY_GATED_INTERP):
        if col in df.columns:
            filled = g[col].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
            newly = df[col].isna() & filled.notna()
            if col in IDENTITY_GATED_INTERP:
                declined = int((newly & ~same_ceo).sum())
                if declined:
                    stats[f"declined (identity changed): {col}"] = declined
                newly &= same_ceo
            n = int(newly.sum())
            if n:
                df.loc[newly, col] = filled[newly]
                stats[f"interp: {col}"] = n
    _accrue_ceo_age(df, stats)
    for col in FLAGS:
        if col in df.columns:
            fwd, bwd = g[col].ffill(), g[col].bfill()
            inside = df[col].isna() & fwd.notna() & bwd.notna()      # bounded by a known value each side
            if col in AGREEMENT_REQUIRED_FLAGS:
                # The two sides must be the same PERSON, not merely both present. Compared on
                # the identity key so a respelling counts as agreement, and with an explicit
                # `notna` on both keys so two unkeyable cells never agree by both being None.
                k_fwd, k_bwd = ceo_identity_series(fwd), ceo_identity_series(bwd)
                agree = k_fwd.notna() & k_bwd.notna() & (k_fwd == k_bwd)
                declined = int((inside & ~agree).sum())
                if declined:
                    stats[f"declined (identity changed): {col}"] = declined
                inside &= agree
            n = int(inside.sum())
            if n:
                df.loc[inside, col] = fwd[inside]
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
                stats[f"{c}_imputed (interpolated -> delta legs excluded)"] = n

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
