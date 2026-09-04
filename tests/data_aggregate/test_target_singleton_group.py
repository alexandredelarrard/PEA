"""
The PER-GROUP presence floor in `xs_project_out` (`MIN_GROUP_SIZE_FOR_NEUTRALIZATION`).

THE DEFECT it closes. `xs_group_dummies` builds a STATIC one-hot block over every ticker, so
a GICS group that happens to have exactly one PRESENT member on a day contributes a column
with a single 1 in that member's own row. OLS then fits that member's residual exactly and the
"neutralized" label is identically 0.0 -- not stale, not noisy, zero -- which re-percentiles to
a valid-looking `target_rank ~ 0.50`. Two present members split the column and each keeps
exactly half its label. Measured on the live h=60 panel before the fix: 5,230 cells exactly
zero (`F` lost 48.7% of its history, `CSGP` 21.5%) plus 16,960 cells at half the label. The
cause is that `sp500_tickers` carries CURRENT index membership, so 1996's
`Automobiles & Components` holds only `F` until `TSLA`'s first target in 2010.

`xs_project_out`'s existing whole-day guard cannot see it: `present.sum() > design.shape[1]`
asks whether the WHOLE fit is over-determined, and a day with 400 names and one solo group
passes that easily while the solo name is still fitted exactly by its own column.

These tests pin BOTH sides -- the defect as it was (no floor), and that the floor removes it
without nulling anything and without touching the >= 5 arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.xs import (
    MIN_GROUP_SIZE_FOR_NEUTRALIZATION, xs_group_dummies, xs_project_out,
)
from src.data_aggregate.utils.target.targets import build_targets_multi


# --------------------------------------------------------------------------- #
# a panel shaped like F / CSGP: one lonely name, then the rest of its group    #
# --------------------------------------------------------------------------- #
SOLO, LATE = "SOLO", ["L1", "L2", "L3", "L4", "L5"]
OTHERS = [f"O{i}" for i in range(12)]
N_SOLO_DAYS = 8                      # days on which only SOLO is present in its group


def _panel(seed: int = 11):
    """`SOLO` is alone in group `THIN` for `N_SOLO_DAYS`, then joined by 5 more names one at a
    time -- so the panel walks 1, 2, 3, 4, 5, 6 present members and every threshold case is
    exercised by the same frame. `OTHERS` is a fat control group present throughout."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=N_SOLO_DAYS + len(LATE) + 3)
    tickers = [SOLO, *LATE, *OTHERS]
    values = pd.DataFrame(rng.standard_normal((len(dates), len(tickers))),
                          index=dates, columns=tickers)
    for k, t in enumerate(LATE):                     # L1 arrives on day N, L2 on N+1, ...
        values.loc[values.index[: N_SOLO_DAYS + k], t] = np.nan
    groups = {SOLO: "THIN", **{t: "THIN" for t in LATE}, **{t: "FAT" for t in OTHERS}}
    return values, xs_group_dummies(groups, values.columns)


def _n_present_in_thin(values: pd.DataFrame) -> pd.Series:
    return values[[SOLO, *LATE]].notna().sum(axis=1)


def test_no_floor_reproduces_the_defect():
    """Documents the bug the floor exists for, so a regression is visible as a test flip."""
    values, dummies = _panel()
    out = xs_project_out(values, [], dummies)                 # min_group_size unset = today
    n_thin = _n_present_in_thin(values)

    solo_days = n_thin[n_thin == 1].index
    pair_days = n_thin[n_thin == 2].index
    assert len(solo_days) == N_SOLO_DAYS and len(pair_days) >= 1
    assert np.allclose(out.loc[solo_days, SOLO], 0.0, atol=1e-12), \
        "the un-floored projection is supposed to zero a solo group -- defect not reproduced"

    # the pair days keep exactly HALF the (demeaned) label each, the 2-member shape
    centered = values.loc[pair_days, [SOLO, "L1"]].sub(values.loc[pair_days].mean(axis=1),
                                                       axis=0)
    pair_frac = out.loc[pair_days, [SOLO, "L1"]].abs() / centered.abs()
    print("\n=== SANITY CHECK: the singleton-group defect, unfloored ===")
    print(f"  {len(solo_days)} solo days -> max |epsilon| = "
          f"{out.loc[solo_days, SOLO].abs().max():.2e} (identically zero)")
    print(f"  2-member days -> median surviving fraction of the label = "
          f"{float(np.nanmedian(pair_frac.to_numpy())):.3f} (~0.5, the halving)")
    print("  -> this is the defect `min_group_size` closes. Validated.")


def test_floor_stops_zeroing_thin_groups_without_nulling():
    values, dummies = _panel()
    floored = xs_project_out(values, [], dummies, min_group_size=5)
    unfloored = xs_project_out(values, [], dummies)
    n_thin = _n_present_in_thin(values)

    below = n_thin[(n_thin >= 1) & (n_thin < 5)].index
    assert len(below) >= 4, "panel must exercise 1..4 present members"
    kept = floored.loc[below, SOLO]
    assert kept.notna().all(), "the floor must not NULL a cell -- it only drops a column"
    assert (kept.abs() > 1e-9).all(), \
        f"a below-floor day is still being forced to ~0: {kept[kept.abs() <= 1e-9]}"
    # and the un-floored version really was degenerate on those same days
    assert (unfloored.loc[below, SOLO].abs() < kept.abs()).all()

    print("\n=== SANITY CHECK: the floor un-freezes thin-group days ===")
    print(f"  {len(below)} days with 1-4 present members in group THIN")
    print(f"  |epsilon| unfloored: max {unfloored.loc[below, SOLO].abs().max():.2e}")
    print(f"  |epsilon| floored:   min {kept.abs().min():.4f}, "
          f"max {kept.abs().max():.4f}; 0 cells nulled")
    print("  -> below the floor the name keeps a real residual, not a fabricated 0. Validated.")


def test_floor_is_inclusive_at_exactly_five_and_still_demeans_exactly():
    """`>=`, not `>`: a group with exactly `min_group_size` members is STILL neutralized, and
    the demean it gets is the same exact-zero-group-mean arithmetic as before the fix."""
    values, dummies = _panel()
    floored = xs_project_out(values, [], dummies, min_group_size=5)
    n_thin = _n_present_in_thin(values)

    at_or_above = n_thin[n_thin >= 5].index
    exactly_five = n_thin[n_thin == 5].index
    assert len(exactly_five) >= 1, "panel must contain a day with exactly 5 present members"

    # on these days EVERY group clears the floor, so the complement is empty, nothing is
    # folded in and the design is the same full block the unfloored path builds
    thin_means = floored.loc[at_or_above, [SOLO, *LATE]].mean(axis=1)
    fat_means = floored.loc[at_or_above, OTHERS].mean(axis=1)
    assert np.allclose(thin_means, 0.0, atol=1e-9), \
        "a group AT or above the floor must still demean to exactly zero"
    assert np.allclose(fat_means, 0.0, atol=1e-9), "the fat control group must be untouched"
    unfloored = xs_project_out(values, [], dummies)
    pd.testing.assert_frame_equal(floored.loc[at_or_above], unfloored.loc[at_or_above])

    print("\n=== SANITY CHECK: >= 5 is unchanged, exactly ===")
    print(f"  {len(exactly_five)} day(s) with exactly 5 present -> group mean "
          f"{float(thin_means.loc[exactly_five].abs().max()):.2e}")
    print(f"  all {len(at_or_above)} at-or-above days -> max |group mean| "
          f"{float(thin_means.abs().max()):.2e}; FAT control "
          f"{float(fat_means.abs().max()):.2e}")
    print("  -> floored and unfloored are bit-identical on those days; the floor removes "
          "ONLY the degenerate cases. Validated.")


def test_complement_cell_is_floored_too_and_the_block_can_fall_away():
    """The mechanism the naive column filter misses. `_day_residual` centers both sides, so the
    kept one-hot columns plus that implicit intercept PARTITION the day -- a dropped group's
    members land in the COMPLEMENT cell and get demeaned there. Dropping `THIN`'s column while
    `FAT` clears the floor would leave a complement of ONE and reproduce the exact zero.

    Here `FAT` (12 present) is therefore folded into the complement on the solo days, no group
    demean happens at all, and `SOLO` keeps a real residual. That is the correct answer: on a
    13-name day split 12/1, no partition exists with every cell at 5 or more."""
    values, dummies = _panel()
    floored = xs_project_out(values, [], dummies, min_group_size=5)
    naive = xs_project_out(values, [], dummies[[c for c in dummies.columns if c != "THIN"]])
    n_thin = _n_present_in_thin(values)
    solo_days = n_thin[n_thin == 1].index

    assert np.allclose(naive.loc[solo_days, SOLO], 0.0, atol=1e-12), \
        "the naive column-drop is supposed to leave a singleton COMPLEMENT -- premise gone"
    assert (floored.loc[solo_days, SOLO].abs() > 1e-9).all()
    # FAT is not demeaned on those days: the block fell away entirely
    fat_means = floored.loc[solo_days, OTHERS].mean(axis=1)
    assert (fat_means.abs() > 1e-9).any()

    print("\n=== SANITY CHECK: the complement cell is floored too ===")
    print(f"  drop THIN's column only -> SOLO max |epsilon| "
          f"{naive.loc[solo_days, SOLO].abs().max():.2e} (still exactly zero)")
    print(f"  cell-aware floor        -> SOLO min |epsilon| "
          f"{floored.loc[solo_days, SOLO].abs().min():.4f}")
    print(f"  on those {len(solo_days)} days FAT is folded in, so no group demean runs "
          f"(max |FAT mean| {fat_means.abs().max():.4f})")
    print("  -> no design cell is left below the floor. Validated.")


def test_default_none_is_bit_identical_to_today():
    """Every existing caller passes no `min_group_size`; that path must not have moved."""
    values, dummies = _panel()
    a = xs_project_out(values, [], dummies)
    b = xs_project_out(values, [], dummies, min_group_size=None)
    pd.testing.assert_frame_equal(a, b)
    print("\n=== SANITY CHECK: default is a no-op ===")
    print(f"  min_group_size unset vs None -> bit-identical over "
          f"{int(a.notna().to_numpy().sum())} cells. Validated.")


# --------------------------------------------------------------------------- #
# end-to-end: the frozen run of identical labels                              #
# --------------------------------------------------------------------------- #
def test_build_targets_multi_no_longer_freezes_a_thin_group_name():
    """The direct regression test for the reported shape: `F`'s 3,731 consecutive identical
    `target_epsilon` values. A name alone in its industry_group for the first stretch of the
    sample must NOT come out of `build_targets_multi` as a frozen run."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-01", periods=200)
    thin = [SOLO, *LATE]
    tickers = [*thin, *OTHERS]
    ret = pd.DataFrame(rng.normal(0, 0.012, (len(dates), len(tickers))),
                       index=dates, columns=tickers)
    for k, t in enumerate(LATE):                    # THIN holds only SOLO for 120 days
        ret.loc[ret.index[: 120 + k * 5], t] = np.nan
    close = pd.DataFrame(100 * np.cumprod(1 + ret.fillna(0.0).to_numpy(), axis=0),
                         index=dates, columns=tickers)
    factor_panel = pd.DataFrame({"market": rng.normal(0, 0.01, len(dates))}, index=dates)
    betas = {t: pd.DataFrame({"beta_market": 1.0}, index=dates) for t in tickers}
    groups = {SOLO: "THIN", **{t: "THIN" for t in LATE}, **{t: "FAT" for t in OTHERS}}

    out = build_targets_multi(close, betas, factor_panel, macro_cols=[], horizons=(20,),
                              labels=("epsilon",), min_names=5,
                              sector_groups={"industry_group": groups}, stock_ret=ret)
    series = out[20]["epsilon"][SOLO].dropna()
    assert len(series) > 50, "not enough labels to judge a frozen run"
    longest = int((series.diff() != 0).cumsum().value_counts().max())
    n_zero = int((series.abs() < 1e-12).sum())

    assert n_zero == 0, f"{n_zero} labels are still identically zero for {SOLO}"
    assert longest <= 2, f"{SOLO} still carries a frozen run of {longest} identical labels"

    print("\n=== SANITY CHECK: no frozen label run through build_targets_multi ===")
    print(f"  {SOLO} alone in industry_group THIN for the first ~120 of {len(dates)} days")
    print(f"  {len(series)} non-null h=20 epsilon labels; exactly-zero: {n_zero}; "
          f"longest identical run: {longest}")
    print(f"  floor in force: MIN_GROUP_SIZE_FOR_NEUTRALIZATION = "
          f"{MIN_GROUP_SIZE_FOR_NEUTRALIZATION}")
    print("  -> the multi-year frozen run is gone. Validated.")
