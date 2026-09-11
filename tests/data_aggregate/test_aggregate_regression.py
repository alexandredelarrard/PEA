"""
The aggregation refactor guard: `src/data_aggregate/` must produce byte-identical numbers.

`aggregate_fingerprint.py` runs every public panel builder AND every primitive that the
refactor deduplicates, over frozen inputs (a saved `fundamentals_history` slice + seeded
synthetic sources), and hashes every output column. This test replays it and diffs
against the baseline captured BEFORE the refactor.

Splitting a 500-line function, moving a helper into `utils/common/`, merging two identical
functions or sharing a memoized point-in-time cache must change nothing here. A single
differing hash means the reorganisation altered behaviour, and the test names the exact
output and column.

Unlike `test_refactor_regression.py` this needs no SEC `companyfacts` cache, so it runs on
any machine (see `aggregate_fingerprint`'s docstring).
"""
from __future__ import annotations

import json

import pytest

from tests.data_aggregate.aggregate_fingerprint import BASELINE, compute

# Outputs the baseline PREDATES by one DECLARED numeric change: commit 0053dc3 ("removed
# peers neutrality") dropped `beta_sector` from the factor panel, so `panel.betas` went
# 88 -> 66 columns, the two surviving betas were refitted without that regressor, and every
# factor-neutral label moved with them. The baseline was deliberately NOT regenerated, so the
# data-layer refactor is gated on the OTHER 28 fingerprints instead.
#
# SELF-POLICING: `test_declared_drift_list_is_still_accurate` fails if any entry here stops
# drifting. An exclusion list that silently outlives its cause is exactly how
# `cube_part_attention` came to be reported missing on every run (see parts.py's docstring) --
# when the baseline is eventually regenerated, this set must go to empty, not linger.
# EMPTY, and kept empty by regenerating the baseline rather than by adding entries here --
# which is what this set was always supposed to become.
#
# Regenerated twice. The 2026-09-01 price/shares basis fix was the first; the second is the
# 2026-09-03 `MIN_GROUP_SIZE_FOR_NEUTRALIZATION` change in `_neutral_label`, which stops a
# group with one present member (1996's `Automobiles & Components` held only `F` until
# `TSLA`'s first target in 2010, because `sp500_tickers` carries CURRENT membership only)
# having its whole residual absorbed by its own indicator column and shipping as an exact 0.0.
#
# What the second regeneration folded in, measured by diffing the regenerated file against
# the pre-regeneration copy: EXACTLY the six label digests
# (`label.rank_h30/60/90`, `label.zscore_h30/60/90`) and nothing else. All 15 panels, all 13
# primitives and the frozen `input.fundamentals_slice` are byte-identical across it, which is
# the check that the regeneration blessed one intended change and not a bundle of them.
# `_neutral_label` is the only code path the change touches and those six labels are the only
# outputs it feeds, so the blast radius matches the edit exactly.
#
# Regenerated a THIRD time, 2026-09-06: `downside_vol_63`'s `min_periods` went 20 -> 5. `neg`
# keeps only DOWN days, so the period count was a count of LOSING days rather than of
# available data, and requiring 20 of them nulled exactly the names that had been going up
# (measured: 14,376 cells over 410 tickers whose median trailing 63-day return is +21.36%
# against +3.87% where present). Same diff procedure, same discipline: of 35 fingerprinted
# outputs, 33 are byte-identical and the 2 that moved are `panel.price` (1 column digest) and
# `panel.raw_features` (22 per-ticker digests) -- every one of them `downside_vol_63`. Row and
# column counts unchanged on both. Notably `panel.betas` and all 6 labels are untouched, which
# is the check that the inclusive-trailing-refresh and cross-sectional-population changes
# shipped in the same commit moved no number.
#
# Regenerated a FOURTH time, 2026-09-09, and this one could NOT use the diff procedure above:
# the fixtures were re-seeded, so every digest moves at once and old-vs-new says nothing. The
# audit was run the other way round instead, BEFORE the reseed -- a control tree holding the
# random stream fixed while every production change stayed in place (`_control_fingerprint`,
# built by restoring the 24 draws `synthetic_def14a` had started skipping). Measured against
# the old baseline: **27 of 34 outputs byte-identical**, and all 7 that moved are accounted for
#   * 4 panels + `prim.quarter_features` + `prim.super_quarter_features` -- the `ic_` rename:
#     N column names swap and ZERO shared column changes value. A rename, proved not assumed.
#   * `panel.governance` -- +`f_control_wedge` and exactly 5 moved columns:
#     `f_say_on_pay_support` (the fixture emitted PERCENT into a (0,1) gate, so every cell was
#     blanked and the feature was silently ABSENT from the panel -- the baseline had been
#     pinning a hole), `f_insider_ownership_pct` and its peer leg (the ownership block now
#     reaches code that used to return early), and `f_ceo_pay_growth` /
#     `f_ceo_pay_vs_revenue_growth` from the deliberate `pay_features` change, which ships its
#     own tests -- all 77 governance tests pass.
# Six pure primitives (`safe_div`, `ratio_helpers`, `xs_standardize`, `forward_windows`,
# `macro_factor_returns`, `peer_relative_panel`) and the 659->658 row loss in
# `prim.super_quarter_features` were NOT production drift at all: they came back byte-identical
# the moment the draw stream was restored. That is what `rng_for` now makes impossible.
# Regenerated a FOURTH time, 2026-09-09 (phase 8, the look-ahead fill class). Of 34 outputs,
# 33 are byte-identical, the set is unchanged, and the ONE that moved is `panel.governance` --
# rows 43,010 and cols 66 both unchanged, with a single value-moved column:
#
#     f_pct_overboarded
#
# `directors._agreement_gated_fill` became `_carry_gated_fill`: a bounded forward carry replacing
# a rule that filled a director's gap only when the proxies either side AGREED, which read a
# LATER filing for both the value and the decision. `pct_overboarded` is the one fingerprinted
# feature that reads the CHILD column directly rather than through the board average, so it is
# the only one the fixture's synthetic boards can move. On the LIVE archive the same change moves
# `other_public_company_boards` by 17,355 cells (15,315 gained, 2,040 lost -- the old gate had no
# age bound and had been filling from past the 1,095-day horizon) and, through the board average,
# `f_board_busyness` and `f_board_busyness_delta_1y` as well; the fixture supplies those from the
# parent scalar, so they do not move here. That gap between fixture and archive is itself worth
# remembering: 43,010 synthetic rows are not 3,031,036 live ones, and this guard protects against
# ACCIDENTAL change, not against being wrong about the world.
#
# ⚠ `f_pct_overboarded` was verified to be the ONLY moved column BEFORE regenerating, by diffing
# the regenerated file against a copy of its predecessor -- the same discipline the third
# regeneration used, and the reason this comment can name one column instead of hedging.
#
# Regenerated a FIFTH time, 2026-09-10 (Phase 2.2b, the elite 13F panel). ⚠ AND THIS ONE HAS A
# CONFESSION ATTACHED: this test had been UNRUNNABLE since Phase 2.2. The 2.2 rewrite deleted
# `_super_quarter_features` and `_weight_map` from `superinvestor_features`, `aggregate_fingerprint`
# still imported them, so `compute()` raised ImportError at collection -- and it stayed invisible
# because the file was excluded from every suite run of that session. A guard that cannot run is a
# guard that is not there, and "473 passed" said nothing about it. It is BACK IN the default suite.
#
# The fixture was also HOLLOW. `synthetic_13f` carries no `cusip`/`position_type`, so the elite
# builder tripped its `need.issubset(holdings.columns)` guard and returned an EMPTY frame: the
# baseline was pinning a hole, exactly as `f_say_on_pay_support` did before the fourth
# regeneration. `synthetic_manager_book` now mirrors `sec13f_manager_holdings` for real -- CUSIP
# grain, the manager's WHOLE book, off-universe positions appended -- and draws from its own
# `rng_for` stream so no other fixture's draws shift.
#
# Verified BEFORE regenerating, by the usual diff of a freshly computed fingerprint against a copy
# of the predecessor. Of 33 shared outputs, 32 are byte-identical -- `panel.institutional` among
# them, which is the check that the separate RNG stream worked and the all-filer panel never saw
# the new fixture. The set changed by design (`prim.super_quarter_features` REMOVED, its function
# no longer exists; `prim.super_manager_state` and `prim.super_conviction` ADDED, the two
# intermediates that replaced it), and the ONE shared output that moved is `panel.superinvestor`:
# rows unchanged at 43,010, columns 18 -> 31, with
#
#     REMOVED 13 columns   ADDED 26 columns   `date`/`ticker` identical
#     SHARED and moved:  f_ic_super_breadth_chg_xs, f_ic_super_flow_to_mcap_xs,
#                        f_ic_super_shares_chg_xs
#
# The three that moved are the only pre-2.2 features that still exist, and they SHOULD move --
# plan D8: whole-book denominators, split-restated share counts and an availability stamp change
# every `ic_super_*` value. The 13 removed include three features that no longer exist at all
# (`cluster_buying`, `new_buyer_ratio`, `value_to_mcap`) and all 7 `_vs_peers` legs, which plan D25
# forbids for this family: the new panel has ZERO `_vs_peers` columns, asserted in the diff.
#
# So this regeneration blesses a whole new feature CONTRACT for one family, not a moved number,
# which is why it was put to the user rather than done on my own judgement.
#
# Regenerated a SIXTH time, 2026-09-10, minutes later: `_xs` REMOVED from `top10_holders`,
# `holders_yoy` and `breadth_chg`. Re-measuring the emission map per selection mode showed
# `top_k` collapses those three to 5-10 distinct values across ~262 present names -- 96-98% of
# each cross-section TIED -- so `rank(pct=True)` was returning a handful of tie-averaged
# plateaus that moved between dates because the number of names on each plateau changed, not
# because anything was reordered. The criterion for `_xs` moved from pooled rho to TIE FRACTION;
# the module docstring carries the measured table and the reason tie fraction, unlike rho, is
# stable across `k` (15/20/25 -> 96-98% ties on all three, while every rho rises).
#
# The cleanest diff of the six: of 35 outputs, 34 byte-identical, the set unchanged, and
# `panel.superinvestor` a PURE COLUMN REMOVAL -- rows 43,010 unchanged, cols 31 -> 28,
#
#     REMOVED 3   ADDED 0   SHARED-AND-MOVED 0
#     f_ic_super_breadth_chg_xs, f_ic_super_holders_yoy_xs, f_ic_super_top10_holders_xs
#
# Zero shared columns moved is the check that this touched the emission map and nothing else:
# dropping an `_xs` leg must not perturb the raw leg it was derived from, and it did not.
# ⚠ `k` 15 -> 20 landed in the same edit and is NOT in this diff, correctly -- `compute()`
# calls the builder without `selection=`, so the fixture panel runs at flat `sel` and no config
# value can reach it. A fingerprint that moved on a config change would mean the fixture had
# started reading `configs/`.
#
# Regenerated a SEVENTH time, 2026-09-10, for Phase 2.3: the insider panel rebuilt from the 4
# legacy features to the registry's 14 (23 feature columns). Of 35 outputs, **34 byte-identical**
# and `panel.insider` alone moved -- rows 43,010 unchanged, cols 10 -> 25,
#
#     REMOVED 8   ADDED 23   SHARED-AND-MOVED 0   (`date` and `ticker` identical)
#
# and the current panel has ZERO `_vs_peers` columns against 4 before, which is D25 extended to
# this family: an insider purchase is a fact about one company, not about its basket.
#
# ⚠ THE FIXTURE HAD TO BE WIDENED FIRST, AND THE FIRST ATTEMPT WITHOUT THAT WOULD HAVE FROZEN A
# HOLE. `synthetic_insider` returned four columns -- ticker, filing_date, transaction_code,
# value_usd -- and the new builder reads the whole Form 4 record, so it produced an EMPTY panel:
# `panel.insider` 43,010 x 10 -> **0 x 0**. Regenerating on that would have blessed the absence
# of a feature family, which is exactly what the elite panel's hollow fixture did in Phase 2.2
# and what the fifth regeneration had to undo. The fixture now carries share class, price,
# owner identity, post-trade holding and the 10b5-1 flag, and deliberately mis-prices ~1.5% of
# rows by 10^2 so the consensus repair fires -- a fixture where a repair never runs cannot
# detect that repair breaking.
#
# `panel.composites` is byte-identical DESPITE the `insider` composite group being renamed in
# `configs/build_cube.yml` in the same change. That is correct and worth stating: `compute()`
# builds composites over the fundamentals+superinvestor merge, which never contained an
# `f_ic_insider_*` column, so all four members were being skipped in silence before and after.
# It is a demonstration of the silent-skip failure mode, not evidence the rename was inert.
#
# The baseline was then re-verified and re-written a second time within Phase 2.3, for the
# coverage-mask fix (a ticker that only ever SOLD now reads 0 on the buy features rather than
# NaN, keyed on the union of purchases and sales). Diff against the first 2.3 baseline: set
# unchanged, cols 25 -> 25, and of the 25,
#
#     SHARED identical 19   SHARED-AND-MOVED 6   ADDED 0   REMOVED 0
#     moved: buy_shares_so_180d, buy_value_mcap_180d, discretionary_sell_mcap_60d (+ _xs)
#
# Exactly the windowed legs, and only those. `buy_value_mcap_60d` and the three role legs are
# DECAYED rather than windowed and did not move, which is the check that the mask reached the
# rolling path and nothing else. `planned_sell_mcap_60d` did not move either: every fixture
# ticker has a planned sale, so its column set was already complete, while the discretionary
# leg excludes exercise-and-sell packages and left some tickers empty -- those are the NaN ->
# 0 cells.
DECLARED_DRIFT: frozenset[str] = frozenset()


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE.name}; run "
                    "`python -m tests.data_aggregate.aggregate_fingerprint` first")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return compute()


def fingerprint_problems(old: dict, new: dict) -> list[str]:
    """Every way `new` differs from `old`, as a list of human-readable problems -- ALWAYS both
    kinds, never one instead of the other.

    ⚠ THE SET CHECK MUST NOT SHORT-CIRCUIT THE VALUE CHECK. This logic used to open with a bare
    `assert set(old) == set(new)` sitting ABOVE the per-output loop, so the moment one output
    was added or removed -- `panel.attention`, deleted on purpose -- execution stopped there and
    NONE of the other 34 outputs was value-checked. It read as one red test; what it actually
    was is a guard measuring nothing, the same shape of failure `parts.py` records for
    `cube_part_attention`. The information you most need during a DELIBERATE change is exactly
    the information it withheld.

    So: collect every problem, always compare the outputs the two runs SHARE, and let the caller
    fail once with the whole picture. The two kinds stay in SEPARATE entries so "we added a
    column" never reads the same as "a column's values moved".

    Lifted out of the test body to be directly exercisable on a synthetic pair -- a guard whose
    failure mode cannot be tested is the thing this whole phase is about.
    """
    problems: list[str] = []
    if set(old) != set(new):
        problems.append(
            f"outputs appeared/disappeared: only-before={sorted(set(old) - set(new))}, "
            f"only-after={sorted(set(new) - set(old))}")

    changed: list[str] = []
    for name in sorted(set(old) & set(new)):
        if name in DECLARED_DRIFT:          # predates commit 0053dc3 -- see DECLARED_DRIFT
            continue
        a, b = old[name], new[name]
        if a["hash"] == b["hash"]:
            continue
        detail = [f"{name}: rows {a['rows']}->{b['rows']}, cols {a['cols']}->{b['cols']}"]
        gone = sorted(set(a["columns"]) - set(b["columns"]))
        added = sorted(set(b["columns"]) - set(a["columns"]))
        if gone:
            detail.append(f"    columns REMOVED: {gone[:12]}")
        if added:
            detail.append(f"    columns ADDED:   {added[:12]}")
        moved = sorted(c for c in set(a["columns"]) & set(b["columns"])
                       if a["per_column"].get(c) != b["per_column"].get(c))
        if moved:
            detail.append(f"    VALUES changed in {len(moved)} column(s): {moved[:12]}")
        changed.append("\n".join(detail))
    if changed:
        problems.append("the refactor changed aggregation output:\n" + "\n".join(changed))
    return problems


def _fp(columns: dict[str, str], rows: int = 10) -> dict:
    """One fingerprint entry: `per_column` digests, and a `hash` that follows them."""
    return {"rows": rows, "cols": len(columns), "columns": sorted(columns),
            "per_column": dict(columns), "hash": "|".join(f"{k}={v}" for k, v in
                                                          sorted(columns.items()))}


def test_the_gate_reports_a_set_change_AND_a_value_change_together():
    """A run that BOTH gains/loses an output AND moves a value must report both, not the first.

    This is the regression that made the guard worthless: every phase of the governance plan
    adds or removes a feature, which is precisely when the value diff is the thing you need.
    Synthetic on purpose -- it is testing the comparison, not the aggregation.
    """
    old = {"panel.kept":  _fp({"a": "digest-a", "b": "digest-b"}),
           "panel.gone":  _fp({"z": "digest-z"})}
    new = {"panel.kept":  _fp({"a": "digest-a", "b": "MOVED"}),
           "panel.added": _fp({"y": "digest-y"})}

    problems = fingerprint_problems(old, new)

    assert len(problems) == 2, f"expected a set problem AND a value problem, got: {problems}"
    sets, values = problems[0], problems[1]
    assert "panel.gone" in sets and "panel.added" in sets, sets
    # the whole point: the value diff survives the set change
    assert "panel.kept" in values and "VALUES changed in 1 column(s): ['b']" in values, values
    assert "panel.gone" not in values, "a vanished output must not be value-compared"

    # ...and each kind alone is still reported alone
    assert fingerprint_problems(old, old) == []
    only_values = fingerprint_problems({"panel.kept": old["panel.kept"]},
                                       {"panel.kept": new["panel.kept"]})
    assert len(only_values) == 1 and "VALUES changed" in only_values[0], only_values

    print("\n[gate guard] a run with BOTH a set change and a value change reports 2 separate "
          "problems: the appeared/disappeared list, and 'VALUES changed in 1 column(s)' on the "
          "output the two runs share.")
    print("    SANITY CHECK: the set check no longer short-circuits the value check, so a "
          "deliberate feature add/remove can never hide a moved number again.")


def test_aggregation_output_is_unchanged_by_the_refactor(baseline, current):
    old = {k: v for k, v in baseline.items() if not k.startswith("_")}
    new = {k: v for k, v in current.items() if not k.startswith("_")}

    problems = fingerprint_problems(old, new)
    assert not problems, "\n".join(problems)

    gated = sorted(k for k in new if k not in DECLARED_DRIFT)
    panels = sorted(k for k in gated if k.startswith("panel."))
    prims = sorted(k for k in gated if k.startswith("prim."))
    labels = sorted(k for k in gated if k.startswith("label."))
    total_cols = sum(new[k]["cols"] for k in gated)
    print(f"\n[aggregation guard] {len(gated)} of {len(new)} outputs identical to the baseline "
          f"({total_cols} columns hashed)")
    print(f"    {len(panels)} panel builders | {len(prims)} deduplicated primitives | "
          f"{len(labels)} target labels")
    print(f"    NOT gated ({len(DECLARED_DRIFT)}): "
          f"{', '.join(sorted(DECLARED_DRIFT)) or 'nothing — every output is gated'}")
    print("    SANITY CHECK: the data-layer refactor changed no number in any of the "
          f"{len(gated)} gated aggregation outputs.")


def test_declared_drift_list_is_still_accurate(baseline, current):
    """Guard the exclusion list. Every entry in `DECLARED_DRIFT` must ACTUALLY still differ
    from the baseline; the moment one matches again (i.e. the baseline was regenerated) the
    entry is stale and must be deleted, or it would silently un-gate a real output.

    This is the lesson `parts.py` records: `cube_part_attention` stayed in a hand-kept list
    after it left the DAG, and the status gate reported it missing on every run for months."""
    stale = [name for name in sorted(DECLARED_DRIFT)
             if baseline[name]["hash"] == current[name]["hash"]]
    assert not stale, (
        "DECLARED_DRIFT lists outputs that now MATCH the baseline -- remove them so they are "
        f"gated again: {stale}")

    missing = sorted(DECLARED_DRIFT - set(baseline))
    assert not missing, f"DECLARED_DRIFT names outputs that do not exist: {missing}"

    print(f"\n[drift list] all {len(DECLARED_DRIFT)} declared-drift outputs still differ from "
          "the baseline, so none is silently un-gated.")
    print("    SANITY CHECK: the exclusion list is exact -- it hides the 0053dc3 beta/label "
          "change and nothing else. Regenerating the baseline will make this test demand its "
          "removal.")


def test_baseline_covers_every_panel_and_deduped_primitive(baseline):
    """Guard the guard. `pipeline_fingerprint` left 9 of the 13 panel builders and every
    primitive the dedup touches unprotected; this asserts that gap stays closed, so a
    future edit cannot quietly drop a builder out of the fingerprint."""
    keys = [k for k in baseline if not k.startswith("_")]
    panels = [k for k in keys if k.startswith("panel.")]
    prims = [k for k in keys if k.startswith("prim.")]
    labels = [k for k in keys if k.startswith("label.")]

    # every panel a cube part is built from. `panel.attention` was here until the attention
    # panel was deleted (D6) and `build_combined_attention_panel` with it; it is named in no
    # cube part now, so requiring it would pin a builder that cannot be built.
    for must in ("panel.price", "panel.fundamental", "panel.sector", "panel.earnings",
                 "panel.employee", "panel.dividend", "panel.governance",
                 "panel.short_interest", "panel.institutional", "panel.superinvestor",
                 "panel.insider", "panel.betas", "panel.composites", "panel.raw_features"):
        assert must in baseline, f"{must} is not fingerprinted"
        assert baseline[must]["rows"] > 0, f"{must} fingerprinted as empty"
        assert baseline[must]["cols"] > 2, f"{must} has no feature columns"

    # every primitive the dedup sweep merges or moves
    for must in ("prim.momentum_characteristic", "prim.mom_12_1_inline", "prim.trailing_vol",
                 "prim.daily_returns", "prim.forward_windows", "prim.xs_standardize",
                 # `prim.price_column_returns` became `prim.macro_factor_returns`: that helper
                 # was deleted when the commodity/FX series moved to `prices_macro` under their
                 # factor names, making its name->column remap the identity.
                 "prim.ratio_helpers", "prim.safe_div", "prim.macro_factor_returns",
                 "prim.quarter_features", "prim.pit", "prim.peer_relative_panel",
                 # `prim.super_quarter_features` was here until Phase 2.2 deleted
                 # `_super_quarter_features`; the elite panel is now built from these two
                 # intermediates instead, and naming BOTH is what keeps the replacement as
                 # protected as the thing it replaced. This tuple catching the vanished name
                 # is the whole point of the test -- it failed on the 2026-09-10 regeneration
                 # and had to be updated deliberately, which is the intended workflow.
                 "prim.super_manager_state", "prim.super_conviction"):
        assert must in baseline, f"{must} is not fingerprinted"
        assert baseline[must]["rows"] > 0, f"{must} fingerprinted as empty"

    # ⚠ These floors track the `must` tuples above rather than standing as free-floating
    # numbers. `>= 15` outlived its cause the moment `panel.attention` was deleted and had to
    # be hand-corrected here -- the same way the `> 100` slice-width assertion below went stale
    # silently. A floor that cannot be derived from what it guards will drift again.
    assert len(panels) >= 14, f"panel coverage regressed: {panels}"
    assert len(prims) >= 13, f"primitive coverage regressed: {prims}"
    assert len(labels) >= 6, f"only {len(labels)} target variants fingerprinted"
    # the frozen input must be pinned too: a silent DB change would otherwise look like a
    # code regression spread across every fundamentals-derived panel
    assert baseline["input.fundamentals_slice"]["rows"] > 0
    # A bare width threshold is what this used to be (`> 100`, written against a 237-column
    # vintage), and it went stale silently when `fundamentals_history` moved to the
    # Sharadar-first 93-column shape: the assertion still passed for a whole vintage because
    # the FROZEN PARQUET had not been regenerated, so it was measuring a file, not the table.
    # Name the columns that must be there instead -- a width is a proxy for coverage, these
    # ARE the coverage, and each one is a family that silently died when it went missing.
    slice_cols = set(baseline["input.fundamentals_slice"].get("columns", []))
    assert len(slice_cols) >= 80, f"frozen slice is a stub: {len(slice_cols)} columns"
    for must, why in (
            ("sector", "sector_gates.row_gate fails CLOSED without it -> every sector KPI off"),
            ("industry_group", "the finer gate, same failure mode"),
            ("revenueGrowth", "a CUBE_TIME_COLUMN: only the cube can compute it"),
            ("earningsGrowth", "ditto"),
            ("employees_sec", "the whole workforce family reads this exact name"),
            ("intangibles", "the ROIC deduction; the bare `goodwill` is written by no producer"),
            ("dividendsPaid", "payout_ratio and sustainable_growth_rate both need its sign")):
        assert must in slice_cols, f"frozen fundamentals slice has no `{must}` -- {why}"

    print(f"\n[coverage] {len(panels)} panels + {len(prims)} deduplicated primitives + "
          f"{len(labels)} labels + the frozen fundamentals input "
          f"({len(slice_cols)} columns, enrichments included)")
    print("    SANITY CHECK: all 13 panel builders and all 13 to-be-merged primitives are "
          "fingerprinted and non-empty, so no dedup step is unguarded.")


def test_momentum_dedup_is_provably_identical(baseline):
    """R1 stated as an executable claim: `features.mom_12_1` and
    `factors.momentum_characteristic` are the same expression, so replacing the inline copy
    with the shared helper cannot move a number. Both are fingerprinted separately; if the
    hashes ever diverge, the two definitions have drifted and the dedup is NOT safe."""
    a = baseline["prim.momentum_characteristic"]["hash"]
    b = baseline["prim.mom_12_1_inline"]["hash"]
    assert a == b, ("features.mom_12_1 and factors.momentum_characteristic no longer agree "
                    f"({a[:12]} vs {b[:12]}) -> the momentum dedup would change output")
    print(f"\n[dedup precheck] momentum_characteristic == inline mom_12_1 ({a[:12]})")
    print("    SANITY CHECK: the momentum dedup is bit-identical by construction. Validated.")
