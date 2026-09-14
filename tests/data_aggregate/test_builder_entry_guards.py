"""D5: the `.empty` boundary rule, stated once and asserted at all seven public builders.

The rule: at the public `build_*_panel` entry, check `None`, check `.empty`, check the
required columns with `issubset`, and return the empty `["date", "ticker"]` panel. Inside, do
not re-check a frame the entry already validated and nothing has mutated.

⚠ THIS FILE EXISTS BECAUSE A BLANKET `.empty` SWEEP PRODUCED A LIVE CRASH. The entry guard on
`build_superinvestor_feature_panel` was removed as "redundant" while `attach_tickers` still
opens with `holdings.copy()` — so an absent or cold `sec13f_manager_holdings` was an
`AttributeError` two frames deep, on the one code path nothing in the suite covered. A guard
is only redundant when NOTHING between it and the previous check can have changed the frame;
`ownership_features._level_state` checks `e.empty`, does a `dropna(subset=["_grid_date"])`,
then checks again, and both of those are load-bearing.

Every test here prints its own sanity conclusion, because "returns empty instead of raising"
is the sort of assertion that passes for the wrong reason when a builder starts raising a
DIFFERENT exception during collection.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.cross_source_features import (
    build_cross_source_panel)
from src.data_aggregate.utils.institutionals.insider_features import (
    build_insider_feature_panel)
from src.data_aggregate.utils.institutionals.institutional_features import (
    build_institutional_feature_panel)
from src.data_aggregate.utils.institutionals.ownership_features import (
    build_ownership_feature_panel)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    build_short_flow_feature_panel)
from src.data_aggregate.utils.institutionals.signal_conditioning import (
    build_signal_conditioning_panel)
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_aggregate.utils.institutionals.superinvestor_features import (
    build_superinvestor_feature_panel)
from tests.conftest import make_frames

IDX = pd.bdate_range("2024-01-01", "2024-06-28")
TICKERS = ("AAA", "BBB")
PEERS = {t: {o: 1.0 for o in TICKERS if o != t} for t in TICKERS}
ROSTER = {"0000000001": "A Manager"}

#: `(label, callable taking ONE source frame)`. The source each builder cannot run without is
#: the one varied; everything else is left at whatever an absent optional means.
CASES = [
    ("institutional",
     lambda f, src: build_institutional_feature_panel(f, src, min_prior_holders=0)),
    ("superinvestor",
     lambda f, src: build_superinvestor_feature_panel(f, src, ROSTER)),
    ("insider", lambda f, src: build_insider_feature_panel(f, src)),
    ("short_flow", lambda f, src: build_short_flow_feature_panel(f, src)),
    ("ownership", lambda f, src: build_ownership_feature_panel(f, src, None)),
    ("signal_conditioning", lambda f, src: build_signal_conditioning_panel(f, src)),
]


def _frames():
    close = pd.DataFrame(100.0, index=IDX, columns=list(TICKERS))
    return make_frames(IDX, PEERS, universe=TICKERS, close_split=close, close_total=close,
                       volume=close, level_factor=close.copy() * 0 + 1.0)


def _is_empty_panel(panel) -> bool:
    return (isinstance(panel, pd.DataFrame) and panel.empty
            and list(panel.columns) == ["date", "ticker"])


@pytest.mark.parametrize("label,build", CASES, ids=[c[0] for c in CASES])
def test_an_absent_source_returns_the_empty_panel_and_does_not_raise(label, build):
    """`None` and an empty frame are the two shapes a cold table actually arrives in."""
    frames = _frames()
    # `signal_conditioning` takes an event MAP, not a frame; `{}` is its empty
    empty_src = {} if label == "signal_conditioning" else pd.DataFrame()
    got = {}
    for shape, src in (("None", None), ("empty", empty_src)):
        panel = build(frames, src)
        assert _is_empty_panel(panel), (
            f"{label} on a {shape} source returned {type(panel).__name__} "
            f"{getattr(panel, 'columns', None)}, not the empty ['date','ticker'] panel")
        got[shape] = list(panel.columns)

    print()
    print(f"=== SANITY: {label} on an absent source ===")
    for shape, cols in got.items():
        print(f"    {shape:<6} -> empty frame, columns {cols}")
    print("    CONCLUSION: the cold-table path returns the empty panel instead of raising, "
          "so `PanelMerger.add` reports 'nothing built'. Validated.")


def test_the_cross_source_builder_tolerates_an_unfilled_sink():
    """`build_cross_source_panel` reads a SINK, not a frame, so its empty case is its own."""
    panel = build_cross_source_panel(_frames(), ConditioningSink())
    assert _is_empty_panel(panel), list(panel.columns)
    print()
    print("=== SANITY: cross-source on an unfilled sink ===")
    print(f"    empty ConditioningSink -> empty frame, columns {list(panel.columns)}")
    print("    CONCLUSION: when no source panel contributed a signal or an actor, the derived "
          "panel is empty rather than a raise. Validated.")


@pytest.mark.parametrize("label,build", CASES, ids=[c[0] for c in CASES])
def test_a_source_missing_a_required_column_returns_empty_rather_than_raising(label, build):
    """The third leg of the D5 guard, and the one a `None`/`.empty` check alone misses.

    A projection that drops a column the builder dereferences unconditionally produces a
    frame that is neither None nor empty — it just cannot be used. Without the `issubset`
    leg that lands as a `KeyError` deep in the arithmetic instead of an empty panel.
    """
    if label == "signal_conditioning":
        pytest.skip("takes an event map keyed by family, not a columnar source frame")
    junk = pd.DataFrame({"not_a_column_any_builder_reads": [1, 2, 3]})
    panel = build(_frames(), junk)
    assert _is_empty_panel(panel), (
        f"{label} on a frame with no usable column returned {list(panel.columns)}")
    print()
    print(f"=== SANITY: {label} on a frame with no usable column ===")
    print(f"    columns in: {list(junk.columns)} -> empty panel, no KeyError")
    print("    CONCLUSION: a narrowed projection degrades to an empty family instead of "
          "killing the build. Validated.")


if __name__ == "__main__":
    for lbl, fn in CASES:
        test_an_absent_source_returns_the_empty_panel_and_does_not_raise(lbl, fn)
    test_the_cross_source_builder_tolerates_an_unfilled_sink()
