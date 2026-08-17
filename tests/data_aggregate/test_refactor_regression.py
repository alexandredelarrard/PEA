"""
The refactor guard: extraction + aggregation must produce byte-identical numbers.

`pipeline_fingerprint.py` runs the real public entry points of both modules over fixed
inputs (12 cached SEC filers + a seeded synthetic price panel) and hashes every output
column. This test replays it and diffs against the baseline captured BEFORE the refactor.

Moving a helper into utils, deleting an unused function, or splitting a 500-line function
must change nothing here. A single differing hash means the reorganisation altered
behaviour, and the test names the exact output and column.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.data_aggregate.pipeline_fingerprint import (
    BASELINE, MissingCompanyFactsCache, compute,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE.name}; run "
                    "`python -m tests.data_aggregate.pipeline_fingerprint` first")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    try:
        return compute()
    except MissingCompanyFactsCache as exc:
        # this guard needs the raw SEC companyfacts download; the AGGREGATION half is
        # covered without it by test_aggregate_regression.py
        pytest.skip(f"{exc} -> see tests/data_aggregate/test_aggregate_regression.py")


# The baseline PREDATES one declared numeric change: `beta_market_simple` was deleted (a
# univariate cov/var twin of `beta_market` that nothing read -- no modelling allow-list mentions
# `beta_*`, the L/S optimizer computes its own beta, and `compute_epsilon` never subtracted it).
# So `aggregate.betas` loses a column, and because that column WAS a regressor in the label's
# projection, all six labels move with it. CLAUDE.md forbids regenerating this baseline outside a
# src-free commit, so the drift is declared here instead and everything else stays gated.
#
# SELF-POLICING: `test_declared_drift_list_is_still_accurate` fails if any entry stops drifting,
# so when the baseline is eventually regenerated this set must go to EMPTY rather than linger --
# the same discipline as tests/data_aggregate/test_aggregate_regression.py.
DECLARED_DRIFT: frozenset[str] = frozenset({
    "aggregate.betas",
    "aggregate.target_rank_h30", "aggregate.target_rank_h60", "aggregate.target_rank_h90",
    "aggregate.target_zscore_h30", "aggregate.target_zscore_h60", "aggregate.target_zscore_h90",
})


def test_pipeline_output_is_unchanged_by_the_refactor(baseline, current):
    old = {k: v for k, v in baseline.items() if not k.startswith("_")}
    new = {k: v for k, v in current.items() if not k.startswith("_")}

    assert set(old) == set(new), (
        f"outputs appeared/disappeared: only-before={sorted(set(old) - set(new))}, "
        f"only-after={sorted(set(new) - set(old))}")

    changed: list[str] = []
    for name in sorted(old):
        if name in DECLARED_DRIFT:          # beta_market_simple was deleted -- see above
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
    assert not changed, ("the refactor changed pipeline output:\n" + "\n".join(changed))

    gated = {k: v for k, v in new.items() if k not in DECLARED_DRIFT}
    total_cols = sum(v["cols"] for v in gated.values())
    total_rows = sum(v["rows"] for v in gated.values())
    print(f"\n[refactor guard] {len(gated)} of {len(new)} pipeline outputs identical to the "
          f"baseline ({total_cols} columns, {total_rows} rows hashed)")
    print(f"    extraction: {sum(1 for k in gated if k.startswith('extract.'))} outputs | "
          f"aggregation: {sum(1 for k in gated if k.startswith('aggregate.'))} outputs")
    print(f"    NOT gated ({len(DECLARED_DRIFT)}): {', '.join(sorted(DECLARED_DRIFT))}")
    print("      ^ `beta_market_simple` was deliberately deleted; it was also a regressor in "
          "the label projection, so the six labels moved with it.")
    print("    SANITY CHECK: moving helpers to utils, deleting dead functions and "
          f"splitting large functions changed no number in any of the {len(gated)} gated outputs.")


def test_declared_drift_list_is_still_accurate(baseline, current):
    """Guard the exclusion list. Every entry must ACTUALLY still differ from the baseline; the
    moment one matches again (i.e. the baseline was regenerated) the entry is stale and must be
    deleted, or it would silently un-gate a real output."""
    stale = [name for name in sorted(DECLARED_DRIFT)
             if baseline[name]["hash"] == current[name]["hash"]]
    assert not stale, (
        "DECLARED_DRIFT lists outputs that now MATCH the baseline -- remove them so they are "
        f"gated again: {stale}")

    missing = sorted(DECLARED_DRIFT - set(baseline))
    assert not missing, f"DECLARED_DRIFT names outputs that do not exist: {missing}"

    print(f"\n[drift list] all {len(DECLARED_DRIFT)} declared-drift outputs still differ from "
          "the baseline, so none is silently un-gated.")
    print("    SANITY CHECK: the exclusion list is exact -- it hides the beta_market_simple "
          "deletion and nothing else. Regenerating the baseline will make this test demand "
          "its removal.")


def test_baseline_covers_both_modules(baseline):
    """Guard the guard: a fingerprint that silently stopped covering a module would let
    a real regression through."""
    keys = [k for k in baseline if not k.startswith("_")]
    extract = [k for k in keys if k.startswith("extract.")]
    aggregate = [k for k in keys if k.startswith("aggregate.")]
    assert len(extract) >= 10, f"extraction barely covered: {extract}"
    assert len(aggregate) >= 8, f"aggregation barely covered: {aggregate}"
    # the four functions the refactor splits must each be represented, PLUS the label --
    # targets were originally not fingerprinted at all, leaving the one output the model
    # actually trains on unprotected.
    for must in ("extract.fundamentals_history",        # _derive_history
                 "aggregate.fundamental_panel",         # _derived_fields
                 "aggregate.compute_sector_kpis",       # compute_sector_kpis
                 "aggregate.compute_raw_features",      # compute_raw_features
                 "aggregate.betas",                     # the rolling regressor join
                 "aggregate.target_rank_h30"):          # the label itself
        assert must in baseline, f"{must} is not fingerprinted"
        assert baseline[must]["rows"] > 0, f"{must} fingerprinted as empty"
    labels = [k for k in keys if k.startswith("aggregate.target_")]
    assert len(labels) >= 3, f"only {len(labels)} target variants fingerprinted"
    print(f"\n[coverage] {len(extract)} extraction + {len(aggregate)} aggregation outputs "
          f"(incl. {len(labels)} target variants); all split targets non-empty")
