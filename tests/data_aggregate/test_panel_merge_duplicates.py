"""`PanelMerger` is the cross-part combine's `validate="one_to_one"`.

`to_long()` is a single `concat(axis=1)` on (date, ticker)-indexed frames, and `concat` takes
no `validate=`. A repeated key there does NOT fail cleanly: it either multiplies the row count
on the cross-product or raises an opaque reindex error hundreds of columns away from the panel
that caused it. `add()` therefore checks the grain itself, before indexing.

That guard is the only thing standing between a duplicated part and a silently multiplied
cube, so it gets its own test rather than living as an untested code path. These fixtures are
synthetic on purpose -- this is key arithmetic with a known truth.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data_aggregate.utils.common.panel_merge import (
    DuplicateKeyError, FeatureCollisionError, PanelMerger,
)

_LOG = logging.getLogger("panel_merge_test")
_DATES = pd.to_datetime(["2023-01-02", "2023-01-03"])


def _panel(feature: str, rows=None) -> pd.DataFrame:
    rows = rows or [(d, t) for d in _DATES for t in ("AAA", "BBB")]
    return pd.DataFrame(rows, columns=["date", "ticker"]).assign(
        **{feature: [float(i) for i in range(len(rows))]})


def test_duplicate_key_raises_and_names_the_panel():
    """The message must name the LABEL, not just the count: with six parts merging ~570
    columns, 'duplicate key' alone does not say which build step to re-run."""
    dup_rows = [(_DATES[0], "AAA"), (_DATES[0], "AAA"), (_DATES[1], "BBB")]
    merger = PanelMerger(_LOG)
    merger.add(_panel("f_mom"), "momentum")

    with pytest.raises(DuplicateKeyError) as ei:
        merger.add(_panel("f_val", rows=dup_rows), "fundamentals")

    msg = str(ei.value)
    assert "fundamentals" in msg, "the offending panel is not named"
    assert "1 duplicate" in msg, msg
    assert "AAA" in msg, "the message should show the offending key(s)"

    print("\n=== SANITY CHECK: duplicate (date, ticker) raises, naming the panel ===")
    print(f"  {msg.splitlines()[0]}")
    print("  -> the build step to re-run is named, not just the count. Validated.")


def test_the_guard_replaces_an_opaque_pandas_error_and_leaves_no_trace():
    """Why the guard exists, measured rather than asserted from the docstring.

    Sent straight to `concat(axis=1)`, a repeated key raises `InvalidIndexError:
    Reindexing only valid with uniquely valued Index objects` -- which names neither the
    panel, the key, nor the count, and surfaces from inside the accumulator hundreds of
    columns after the part that caused it. The guard's job is to turn that into an
    actionable message AND to leave the accumulator usable."""
    dup_rows = [(_DATES[0], "AAA"), (_DATES[0], "AAA"), (_DATES[1], "BBB")]
    clean, dirty = _panel("f_mom"), _panel("f_val", rows=dup_rows)

    with pytest.raises(pd.errors.InvalidIndexError) as raw:
        pd.concat([clean.set_index(["date", "ticker"]),
                   dirty.set_index(["date", "ticker"])], axis=1)
    raw_msg = str(raw.value)
    assert "fundamentals" not in raw_msg and "AAA" not in raw_msg   # says nothing useful

    merger = PanelMerger(_LOG)
    merger.add(clean, "momentum")
    with pytest.raises(DuplicateKeyError) as guarded:
        merger.add(dirty, "fundamentals")
    # a REFUSED panel must not half-register: no feature owned, no frame accumulated
    assert merger.feature_columns == ["f_mom"], "the refused panel left a trace"
    assert len(merger.to_long()) == len(clean) == 4
    assert "f_val" not in merger.to_long().columns

    print("\n=== SANITY CHECK: the guard replaces an opaque pandas error ===")
    print(f"  bare concat(axis=1) -> InvalidIndexError: {raw_msg!r}")
    print(f"     ^ names no panel, no key, no count")
    print(f"  through PanelMerger  -> {str(guarded.value).splitlines()[0]}")
    print(f"  and the accumulator still yields {len(merger.to_long())} clean rows without "
          f"f_val -- a refused add contributes nothing. Validated.")


def test_clean_panels_merge_one_to_one():
    """The positive case, so the guard cannot be satisfied by simply rejecting everything."""
    merger = PanelMerger(_LOG)
    assert merger.add(_panel("f_mom"), "momentum") == 1
    assert merger.add(_panel("f_val"), "fundamentals") == 1

    out = merger.to_long()

    assert len(out) == 4 and not out.duplicated(["date", "ticker"]).any()
    assert set(out.columns) == {"date", "ticker", "f_mom", "f_val"}
    assert merger.feature_columns == ["f_mom", "f_val"]

    print("\n=== SANITY CHECK: clean one-to-one merge ===")
    print(f"  2 panels x 4 unique keys -> {len(out)} rows, columns {sorted(out.columns)}, "
          f"0 duplicates. Validated.")


def test_two_panels_owning_the_same_feature_name_still_raise():
    """The merger's other guarantee, pinned alongside the key guard because both protect the
    same `concat`: a name owned by two parts would become `_x`/`_y` under a bare merge."""
    merger = PanelMerger(_LOG)
    merger.add(_panel("f_shared"), "momentum")

    with pytest.raises(FeatureCollisionError) as ei:
        merger.add(_panel("f_shared"), "fundamentals")
    msg = str(ei.value)
    assert "f_shared" in msg and "momentum" in msg and "fundamentals" in msg

    print("\n=== SANITY CHECK: duplicate feature NAME raises ===")
    print("  'f_shared' emitted by two panels -> FeatureCollisionError naming the current "
          "owner and the newcomer, instead of a silent _x/_y split. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
