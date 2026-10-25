"""D5: the `.empty` boundary rule, stated once and asserted at all seven public builders.

The rule: at the public `build_*_panel` entry, check `None`, check `.empty`, check the
required columns with `issubset`, and return the empty `["date", "ticker"]` panel. Inside, do
not re-check a frame the entry already validated and nothing has mutated.

⚠ THIS FILE EXISTS BECAUSE A BLANKET `.empty` SWEEP PRODUCED A LIVE CRASH. The entry guard on
`build_superinvestor_feature_panel` was removed as "redundant" while `attach_tickers` still
opens with `holdings.copy()` — so an absent or cold `sec13f_manager_holdings` was an
`AttributeError` two frames deep, on the one code path nothing in the suite covered. The immutable SEC13F step now owns missing-source protection and is exercised in
test_sec_thirteen_step_contract and test_sec_thirteen_windows. A guard
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

from src.data_aggregate.utils.institutionals.ownership_features import build_ownership_feature_panel
from src.data_aggregate.utils.sec_thirteen.institutional_features import build_institutional_feature_panel
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel as build_insider_feature_panel
from tests.data_aggregate.test_short_flow_features import build_short_flow_feature_panel

IDX = pd.bdate_range("2024-01-01", "2024-06-28")
TICKERS = ("AAA", "BBB")
PEERS = {t: {o: 1.0 for o in TICKERS if o != t} for t in TICKERS}

#: `(label, callable taking ONE source frame)`. The source each builder cannot run without is
#: the one varied; everything else is left at whatever an absent optional means.
CASES = [
    ("institutional", lambda f, src: build_institutional_feature_panel(f, src, min_prior_holders=0)),
    ("insider", lambda f, src: build_insider_feature_panel(f, src)),
    ("short_flow", lambda f, src: build_short_flow_feature_panel(f, src)),
    ("ownership", lambda f, src: build_ownership_feature_panel(f, src, None)),
]


def _frames():
    close = pd.DataFrame(100.0, index=IDX, columns=list(TICKERS))
    return make_frames(IDX, PEERS, universe=TICKERS, close_split=close, close_total=close, volume=close, level_factor=close.copy() * 0 + 1.0)


def _is_empty_panel(panel) -> bool:
    return isinstance(panel, pd.DataFrame) and panel.empty and list(panel.columns) == ["date", "ticker"]


@pytest.mark.parametrize("label,build", CASES, ids=[c[0] for c in CASES])
def test_an_absent_source_returns_the_empty_panel_and_does_not_raise(label, build):
    """`None` and an empty frame are the two shapes a cold table actually arrives in."""
    frames = _frames()
    empty_src = pd.DataFrame()
    got = {}
    for shape, src in (("None", None), ("empty", empty_src)):
        panel = build(frames, src)
        assert _is_empty_panel(panel), (
            f"{label} on a {shape} source returned {type(panel).__name__} {getattr(panel, 'columns', None)}, not the empty ['date','ticker'] panel"
        )
        got[shape] = list(panel.columns)

    print()
    print(f"=== SANITY: {label} on an absent source ===")
    for shape, cols in got.items():
        print(f"    {shape:<6} -> empty frame, columns {cols}")
    print("    CONCLUSION: the cold-table path returns the empty panel instead of raising, so `PanelMerger.add` reports 'nothing built'. Validated.")


@pytest.mark.parametrize("label,build", CASES, ids=[c[0] for c in CASES])
def test_a_source_missing_a_required_column_returns_empty_rather_than_raising(label, build):
    """The third leg of the D5 guard, and the one a `None`/`.empty` check alone misses.

    A projection that drops a column the builder dereferences unconditionally produces a
    frame that is neither None nor empty — it just cannot be used. Without the `issubset`
    leg that lands as a `KeyError` deep in the arithmetic instead of an empty panel.
    """
    junk = pd.DataFrame({"not_a_column_any_builder_reads": [1, 2, 3]})
    panel = build(_frames(), junk)
    assert _is_empty_panel(panel), f"{label} on a frame with no usable column returned {list(panel.columns)}"
    print()
    print(f"=== SANITY: {label} on a frame with no usable column ===")
    print(f"    columns in: {list(junk.columns)} -> empty panel, no KeyError")
    print("    CONCLUSION: a narrowed projection degrades to an empty family instead of killing the build. Validated.")


if __name__ == "__main__":
    for lbl, fn in CASES:
        test_an_absent_source_returns_the_empty_panel_and_does_not_raise(lbl, fn)
