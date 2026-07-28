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

from tests.data_aggregate.pipeline_fingerprint import BASELINE, compute


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE.name}; run "
                    "`python -m tests.data_aggregate.pipeline_fingerprint` first")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return compute()


def test_pipeline_output_is_unchanged_by_the_refactor(baseline, current):
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
    assert not changed, ("the refactor changed pipeline output:\n" + "\n".join(changed))

    total_cols = sum(v["cols"] for v in new.values())
    total_rows = sum(v["rows"] for v in new.values())
    print(f"\n[refactor guard] {len(new)} pipeline outputs identical to the baseline "
          f"({total_cols} columns, {total_rows} rows hashed)")
    print(f"    extraction: {sum(1 for k in new if k.startswith('extract.'))} outputs | "
          f"aggregation: {sum(1 for k in new if k.startswith('aggregate.'))} outputs")
    print("    SANITY CHECK: moving helpers to utils, deleting dead functions and "
          "splitting large functions changed no number anywhere in the pipeline.")


def test_baseline_covers_both_modules(baseline):
    """Guard the guard: a fingerprint that silently stopped covering a module would let
    a real regression through."""
    keys = [k for k in baseline if not k.startswith("_")]
    extract = [k for k in keys if k.startswith("extract.")]
    aggregate = [k for k in keys if k.startswith("aggregate.")]
    assert len(extract) >= 10, f"extraction barely covered: {extract}"
    assert len(aggregate) >= 8, f"aggregation barely covered: {aggregate}"
    # the four functions the refactor splits must each be represented
    for must in ("extract.fundamentals_history",        # _derive_history
                 "aggregate.fundamental_panel",         # _derived_fields
                 "aggregate.compute_sector_kpis",       # compute_sector_kpis
                 "aggregate.compute_raw_features"):     # compute_raw_features
        assert must in baseline, f"{must} is not fingerprinted"
        assert baseline[must]["rows"] > 0, f"{must} fingerprinted as empty"
    print(f"\n[coverage] {len(extract)} extraction + {len(aggregate)} aggregation outputs; "
          "all four split targets are fingerprinted and non-empty")
