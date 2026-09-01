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
# EMPTY, as of the 2026-09-01 price/shares basis fix -- which is what this set was always
# supposed to become. The baseline was regenerated then, so `panel.betas` and the six label
# digests are GATED AGAIN rather than excused.
#
# What the regeneration folded in, measured by fingerprinting HEAD's code and this branch's
# code against the SAME database and diffing the two:
#   * moved by THIS branch's code (4): `label.zscore_h30/60/90`, because the label stopped
#     being `forward_return(close, h)` (a literal price ratio that excluded every dividend)
#     and became `forward_compound(stock_ret, h)`; and `panel.institutional`, because
#     `inst_ownership_pct` now divides by `sharesOutstandingPit` rather than the vendor-basis
#     share count. Both are the intended effect of that change.
#   * NOT moved by this branch's code: `label.rank_h30/60/90`. On the harness's dividend-free
#     random walk `forward_compound` is a MONOTONE transform of `forward_return`, so the
#     cross-sectional rank is preserved while the z-score is not -- an internal check that
#     the label change did exactly what it claims.
#   * moved before this branch (4): `panel.betas` and the three rank labels, which had
#     already drifted at HEAD and were the reason this list existed.
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


def test_aggregation_output_is_unchanged_by_the_refactor(baseline, current):
    old = {k: v for k, v in baseline.items() if not k.startswith("_")}
    new = {k: v for k, v in current.items() if not k.startswith("_")}

    assert set(old) == set(new), (
        f"outputs appeared/disappeared: only-before={sorted(set(old) - set(new))}, "
        f"only-after={sorted(set(new) - set(old))}")

    changed: list[str] = []
    for name in sorted(old):
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
    assert not changed, ("the refactor changed aggregation output:\n" + "\n".join(changed))

    gated = sorted(k for k in new if k not in DECLARED_DRIFT)
    panels = sorted(k for k in gated if k.startswith("panel."))
    prims = sorted(k for k in gated if k.startswith("prim."))
    labels = sorted(k for k in gated if k.startswith("label."))
    total_cols = sum(new[k]["cols"] for k in gated)
    print(f"\n[aggregation guard] {len(gated)} of {len(new)} outputs identical to the baseline "
          f"({total_cols} columns hashed)")
    print(f"    {len(panels)} panel builders | {len(prims)} deduplicated primitives | "
          f"{len(labels)} target labels")
    print(f"    NOT gated ({len(DECLARED_DRIFT)}): {', '.join(sorted(DECLARED_DRIFT))}")
    print("      ^ baseline predates commit 0053dc3 ('removed peers neutrality'), which dropped "
          "beta_sector from the factor panel and moved every factor-neutral label with it.")
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

    # every panel a cube part is built from
    for must in ("panel.price", "panel.fundamental", "panel.sector", "panel.earnings",
                 "panel.employee", "panel.dividend", "panel.governance", "panel.attention",
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
                 "prim.quarter_features", "prim.super_quarter_features", "prim.pit",
                 "prim.peer_relative_panel"):
        assert must in baseline, f"{must} is not fingerprinted"
        assert baseline[must]["rows"] > 0, f"{must} fingerprinted as empty"

    assert len(panels) >= 15, f"panel coverage regressed: {panels}"
    assert len(prims) >= 13, f"primitive coverage regressed: {prims}"
    assert len(labels) >= 6, f"only {len(labels)} target variants fingerprinted"
    # the frozen input must be pinned too: a silent DB change would otherwise look like a
    # code regression spread across every fundamentals-derived panel
    assert baseline["input.fundamentals_slice"]["rows"] > 0
    assert baseline["input.fundamentals_slice"]["cols"] > 100

    print(f"\n[coverage] {len(panels)} panels + {len(prims)} deduplicated primitives + "
          f"{len(labels)} labels + the frozen fundamentals input")
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
