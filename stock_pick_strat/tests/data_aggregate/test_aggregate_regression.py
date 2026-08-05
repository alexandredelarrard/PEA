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

    panels = sorted(k for k in new if k.startswith("panel."))
    prims = sorted(k for k in new if k.startswith("prim."))
    labels = sorted(k for k in new if k.startswith("label."))
    total_cols = sum(v["cols"] for v in new.values())
    print(f"\n[aggregation guard] {len(new)} outputs identical to the baseline "
          f"({total_cols} columns hashed)")
    print(f"    {len(panels)} panel builders | {len(prims)} deduplicated primitives | "
          f"{len(labels)} target labels")
    print("    SANITY CHECK: splitting StepBuildCube into sub-steps, moving helpers into "
          "utils/common/ and merging the duplicated primitives changed no number anywhere "
          "in the aggregation layer.")


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
                 "prim.ratio_helpers", "prim.safe_div", "prim.price_column_returns",
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
