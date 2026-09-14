"""Cross-source agreement (`ic_xs_*`, registry section 8).

The two rules the registry is explicit about are the two that are easy to get wrong, so each
has its own test: ten filings from one actor are ONE opinion (#84), and the absence of buying
is not a short signal (#85). Plus the conflict flag's magnitude, the refusal to build a
"family count" out of one family, and declared-vs-emitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.cross_source_features import (
    ACTOR_WINDOW, EMISSION, PERCENTILE, build_cross_source_panel,
)
from src.data_aggregate.utils.institutionals.sink import (
    BEARISH_INPUTS, BULLISH_INPUTS, NEGATED_INPUTS, ConditioningSink,
)
from tests.conftest import make_frames

IDX = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=300))
TICKERS = [f"T{i}" for i in range(10)]
PEERS = {t: {p: 1.0 for p in TICKERS if p != t} for t in TICKERS}


def _ramp(top: list[str]) -> pd.DataFrame:
    """A wide frame where the named tickers alone sit above the flag percentile.

    ⚠ THE NON-TOP TICKERS ARE ALL EQUAL, deliberately. A 0..9 ramp over ten names puts the
    SECOND-highest at the 0.9 percentile, which is also above the 0.8 threshold, so every
    count came out one too high and the test was measuring the fixture. Nine tied values rank
    at 0.5 (`rank` averages ties), which is unambiguously below the flag.
    """
    base = pd.DataFrame({t: 1.0 for t in TICKERS}, index=IDX, dtype="float64")
    for t in top:
        base[t] = 100.0
    return base


def _bearish_signal(role: str, top: list[str]) -> pd.DataFrame:
    """The bearish input for `role` with `top` flagging.

    ⚠ `activist_reduction` IS NEGATED BY THE BUILDER -- a stake REDUCTION is a negative change
    in percent-of-class -- so its flagging ticker must carry the most negative value, not the
    largest. The first version of this fixture ignored that and read as a missing family.
    """
    frame = _ramp(top)
    return -frame if role in {r for r, n in BEARISH_INPUTS.items()
                              if r in NEGATED_INPUTS} else frame


def test_a_family_count_counts_FAMILIES_above_the_percentile():
    sink = ConditioningSink()
    roles = list(BULLISH_INPUTS)
    for role in roles[:3]:                       # T0 tops three of the four bullish families
        sink.signals[BULLISH_INPUTS[role]] = _ramp(["T0"])
    sink.signals[BULLISH_INPUTS[roles[3]]] = _ramp(["T9"])
    for role, name in BEARISH_INPUTS.items():
        sink.signals[name] = _bearish_signal(role, ["T9"])

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    assert day.loc["T0", "f_ic_xs_bullish_family_count"] == 3.0
    assert day.loc["T9", "f_ic_xs_bullish_family_count"] == 1.0
    # T0 is in none of the bearish families, T9 in all four
    assert day.loc["T0", "f_ic_xs_bearish_family_count"] == 0.0
    assert day.loc["T9", "f_ic_xs_bearish_family_count"] == 4.0
    # the flag is a PER-DATE percentile, so the count is bounded by the number of families
    counts = panel["f_ic_xs_bullish_family_count"].dropna()
    assert counts.between(0, len(BULLISH_INPUTS)).all()
    print("\n=== SANITY CHECK: ic_xs_* family counts ===")
    print(f"  T0 tops 3 of 4 bullish families -> 3, and 0 bearish; T9 the reverse "
          f"(1 bullish, 4 bearish). Threshold is the per-date {PERCENTILE:.0%} percentile, so "
          f"the count is bounded by 4. Validated.")


def test_absence_of_evidence_is_never_bearish_and_a_count_of_nothing_is_nan():
    """#85 (report section 3.6.2). A ticker no bearish family has an opinion about must read
    NaN, not 0 -- and certainly not a positive count. 'Nobody sold' is not a sale."""
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        sink.signals[name] = _ramp(["T0"])
    for role, name in BEARISH_INPUTS.items():
        frame = _bearish_signal(role, ["T9"])
        frame.loc[:, "T5"] = np.nan              # T5 is unknown to every bearish family
        sink.signals[name] = frame

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    assert np.isnan(day.loc["T5", "f_ic_xs_bearish_family_count"]), \
        "a ticker with no bearish input read as a count instead of NaN"
    assert day.loc["T1", "f_ic_xs_bearish_family_count"] == 0.0, \
        "a ticker with inputs but no flag must read a real 0"
    assert np.isnan(day.loc["T5", "f_ic_xs_conflict"])
    print("\n=== SANITY CHECK: absence of evidence vs a measured zero ===")
    print("  T5 (no bearish family has a value) reads NaN; T1 (all four have values, none "
          "flags) reads 0.0. The conflict flag inherits the NaN. Validated.")


def test_distinct_actors_not_rows_and_the_window_releases():
    """#84 (report section 3.1.6). Ten Form 4s from ONE officer are one vote; three officers
    are three. And the count releases once the window passes, so it is a trailing distinct
    count rather than an ever-growing one."""
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        sink.signals[name] = _ramp(["T0"])
    rows = [{"ticker": "T0", "date": IDX[10 + i], "actor": "OWNER-1"} for i in range(10)]
    rows += [{"ticker": "T1", "date": IDX[10], "actor": f"OWNER-{k}"} for k in (1, 2, 3)]
    sink.add_actors("insider", pd.DataFrame(rows))
    # the same actor id in ANOTHER family is a different actor -- an owner CIK and a manager
    # CIK are both 10 digits, so the family namespaces the id
    sink.add_actors("super", pd.DataFrame(
        [{"ticker": "T1", "date": IDX[10], "actor": "OWNER-1"}]))

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    col = "f_ic_xs_bullish_actor_count"
    at = panel[panel["date"] == IDX[25]].set_index("ticker")[col]
    assert at["T0"] == 1.0, f"ten filings from one owner counted as {at['T0']}"
    assert at["T1"] == 4.0, "three owners + one manager should be four distinct actors"
    # NaN before the ticker's first act, never 0
    before = panel[panel["date"] == IDX[5]].set_index("ticker")[col]
    assert np.isnan(before["T0"]) and np.isnan(before["T1"])
    # and released after the window
    after = panel[panel["date"] == IDX[10 + ACTOR_WINDOW + 5]].set_index("ticker")[col]
    assert after["T1"] == 0.0, "the trailing window never released"
    print("\n=== SANITY CHECK: distinct ACTORS, not rows ===")
    print(f"  T0: 10 Form 4s from one owner -> {at['T0']:.0f}; T1: 3 owners + 1 manager "
          f"-> {at['T1']:.0f} (the family namespaces a repeated id); NaN before the first act; "
          f"back to 0 once the {ACTOR_WINDOW}-day window passes. Validated.")


def test_the_conflict_flag_is_the_weaker_side():
    sink = ConditioningSink()
    roles_b, roles_s = list(BULLISH_INPUTS), list(BEARISH_INPUTS)
    for role in roles_b[:3]:
        sink.signals[BULLISH_INPUTS[role]] = _ramp(["T0"])
    sink.signals[BULLISH_INPUTS[roles_b[3]]] = _ramp(["T9"])
    for role in roles_s[:2]:
        sink.signals[BEARISH_INPUTS[role]] = _bearish_signal(role, ["T0"])
    for role in roles_s[2:]:
        sink.signals[BEARISH_INPUTS[role]] = _bearish_signal(role, ["T9"])

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    # T0: 3 bullish families and 2 bearish -> a conflict of strength 2, not 3
    assert day.loc["T0", "f_ic_xs_bullish_family_count"] == 3.0
    assert day.loc["T0", "f_ic_xs_bearish_family_count"] == 2.0
    assert day.loc["T0", "f_ic_xs_conflict"] == 2.0
    # T9: 1 bullish and 2 bearish -> below the 2-and-2 condition, so a real 0
    assert day.loc["T9", "f_ic_xs_conflict"] == 0.0
    print("\n=== SANITY CHECK: ic_xs_conflict ===")
    print("  T0 with 3 bullish and 2 bearish families reads 2 (the weaker side); T9 with "
          "1 and 2 reads 0 -- the flag needs >= 2 on BOTH sides. Validated.")


def test_it_refuses_to_build_a_family_count_from_one_family():
    """A rename that the sink's declaration does not follow would silently leave a 'family
    count' measuring one family. One resolved input is not a count of families, and the
    builder drops the column rather than shipping a misnamed one."""
    sink = ConditioningSink()
    sink.signals[BULLISH_INPUTS[list(BULLISH_INPUTS)[0]]] = _ramp(["T0"])
    for role, name in BEARISH_INPUTS.items():
        sink.signals[name] = _bearish_signal(role, ["T9"])
    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    assert "f_ic_xs_bullish_family_count" not in panel.columns
    assert "f_ic_xs_conflict" not in panel.columns, "the conflict flag needs both sides"
    assert "f_ic_xs_bearish_family_count" in panel.columns
    print("\n=== SANITY CHECK: the family count refuses to measure one family ===")
    print("  1 of 4 bullish inputs resolved -> the bullish count and the conflict flag are "
          "NOT emitted; the bearish count (4 of 4) still is. Validated.")


def test_declared_vs_emitted():
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        sink.signals[name] = _ramp(["T0"])
    for role, name in BEARISH_INPUTS.items():
        sink.signals[name] = _bearish_signal(role, ["T9"])
    sink.add_actors("insider", pd.DataFrame(
        [{"ticker": t, "date": IDX[20], "actor": "O1"} for t in TICKERS]))
    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    assert emitted == expected, (f"missing {sorted(expected - emitted)}; "
                                 f"undeclared {sorted(emitted - expected)}")
    for col in sorted(emitted):
        assert panel[col].notna().any(), f"{col} is declared, emitted and ALL-NaN"
    print("\n=== SANITY CHECK: ic_xs_* declared vs emitted ===")
    print(f"  {len(EMISSION)} features -> {len(emitted)} legs, exact match, all non-empty. "
          "Validated.")
